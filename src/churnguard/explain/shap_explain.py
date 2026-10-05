"""SHAP explainability engine for ChurnGuard (T4.3).

Implements:
- TreeExplainer for LightGBM champion model.
- Global SHAP feature importance & summary plots.
- Top-3 plain-language reason code generation for frontline retention agents.
- Batch customer reason extraction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from churnguard.config import CFG
from churnguard.models.calibrate import fit_champion_pipeline


class ChurnExplainer:
    """SHAP-based explainability engine that translates feature contributions into plain-language reason codes."""

    def __init__(self, fitted_pipeline: Pipeline) -> None:
        """Initialize with a fitted end-to-end sklearn Pipeline."""
        self.pipeline = fitted_pipeline
        self.fe = fitted_pipeline.named_steps["feature_engineer"]
        self.preprocessor = fitted_pipeline.named_steps["preprocessor"]
        self.model = fitted_pipeline.named_steps["model"]

        # If wrapped in calibrator or CalibratedClassifierCV
        if hasattr(self.model, "estimator"):
            self.model = self.model.estimator
        if hasattr(self.model, "calibrated_classifiers_"):
            # If CalibratedClassifierCV was passed
            self.model = self.model.calibrated_classifiers_[0].estimator

        self.feature_names = list(self.preprocessor.get_feature_names_out())

        from sklearn.linear_model import LogisticRegression

        if isinstance(self.model, LogisticRegression):
            self.explainer = shap.LinearExplainer(
                self.model,
                masker=shap.maskers.Independent(np.zeros((1, len(self.feature_names)))),
            )
        else:
            self.explainer = shap.TreeExplainer(self.model)

    def transform_data(self, df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
        """Run feature engineering and preprocessing on raw input DataFrame."""
        X = df.drop(columns=["Churn", "customerID"], errors="ignore")
        X_fe = self.fe.transform(X)
        X_trans = self.preprocessor.transform(X_fe)
        X_trans_df = pd.DataFrame(X_trans, columns=self.feature_names, index=df.index)
        return X_trans, X_trans_df

    def compute_shap_values(self, df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
        """Calculate SHAP values for the input DataFrame."""
        X_trans, X_trans_df = self.transform_data(df)
        shap_vals = self.explainer.shap_values(X_trans)

        # For LightGBM binary classification, shap_values can be a list [class0, class1] or (n, m)
        if isinstance(shap_vals, list):
            shap_array = np.array(shap_vals[1])
        elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
            shap_array = shap_vals[:, :, 1]
        else:
            shap_array = np.array(shap_vals)

        return shap_array, X_trans_df

    def plot_global_summary(
        self,
        df: pd.DataFrame,
        save_path: Path | str | None = None,
        max_display: int = 15,
    ) -> plt.Figure:
        """Generate and save publication-ready SHAP summary and importance plots."""
        shap_array, X_trans_df = self.compute_shap_values(df)

        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16, 7))

        # 1. Bar plot: Mean |SHAP value|
        mean_abs_shap = np.mean(np.abs(shap_array), axis=0)
        sorted_idx = np.argsort(mean_abs_shap)[::-1][:max_display]
        top_names = [self.feature_names[i] for i in sorted_idx][::-1]
        top_vals = mean_abs_shap[sorted_idx][::-1]

        ax1.barh(top_names, top_vals, color="#2980b9", edgecolor="black", alpha=0.85)
        ax1.set_xlabel("Mean |SHAP Value| (Impact on Log-Odds)", fontsize=11, fontweight="bold")
        ax1.set_title(
            f"Top {max_display} Global Drivers of Churn Risk",
            fontsize=12,
            fontweight="bold",
            pad=12,
        )
        ax1.grid(True, linestyle="--", alpha=0.5, axis="x")

        # 2. SHAP summary beeswarm plot on ax2
        plt.sca(ax2)
        shap.summary_plot(
            shap_array,
            X_trans_df,
            max_display=max_display,
            show=False,
            plot_size=None,
        )
        ax2.set_title(
            f"SHAP Value Distribution for Top {max_display} Features",
            fontsize=12,
            fontweight="bold",
            pad=12,
        )

        plt.tight_layout()

        if save_path:
            out_path = Path(save_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=300, bbox_inches="tight")

        return fig

    @staticmethod
    def _format_single_reason(
        feature: str,
        shap_val: float,
        row_dict: dict[str, Any],
    ) -> tuple[str, str]:
        """Convert a feature name and customer record into (group_key, plain_explanation)."""
        clean_feat = feature.replace("cat__", "").replace("num__", "")
        tenure = row_dict.get("tenure", "N/A")
        monthly = row_dict.get("MonthlyCharges", "N/A")

        # Contract
        if "Contract_Month-to-month" in clean_feat or clean_feat == "is_month_to_month_1":
            return "contract", "Month-to-month contract increases cancellation flexibility"
        if "Contract_Two year" in clean_feat:
            return (
                "contract",
                "Two-year long-term contract provides retention lock-in"
                if shap_val < 0
                else "Long-term contract nearing term completion",
            )
        if "Contract_One year" in clean_feat:
            return (
                "contract",
                "One-year contract provides retention stability"
                if shap_val < 0
                else "One-year contract nearing annual renewal",
            )
        if clean_feat == "is_month_to_month_0":
            return "contract", "Committed fixed-term contract"

        # Tenure
        if clean_feat in ("tenure", "tenure_bucket_0-6m") or "0-6m" in clean_feat:
            if isinstance(tenure, int | float) and tenure <= 6:
                return (
                    "tenure",
                    f"Early lifecycle risk: Short tenure ({int(tenure)} month{'s' if tenure != 1 else ''})",
                )
            return "tenure", f"Customer tenure ({tenure} months) elevates churn sensitivity"
        if "tenure_bucket_49-72m" in clean_feat:
            return "tenure", f"High loyalty: Established tenure ({tenure} months)"
        if "tenure_bucket" in clean_feat:
            return "tenure", f"Customer tenure stage ({tenure} months)"

        # Payment & Billing
        if "PaymentMethod_Electronic check" in clean_feat:
            return "payment", "Payment via Electronic Check (high manual friction)"
        if "is_auto_pay_0" in clean_feat or clean_feat in ("PaymentMethod_Mailed check",):
            return "payment", "Manual non-automated payment method increases friction"
        if "is_auto_pay_1" in clean_feat or "automatic" in clean_feat.lower():
            return "payment", "Automated recurring payment method stabilizes account"
        if "PaperlessBilling_Yes" in clean_feat:
            return "billing", "Paperless billing active without automated payment"
        if "PaperlessBilling_No" in clean_feat:
            return "billing", "Traditional paper billing receipt"

        # Broadband & Support Services
        if clean_feat == "fiber_no_support_1":
            return "fiber_support", "Fiber optic broadband without dedicated TechSupport"
        if "InternetService_Fiber optic" in clean_feat:
            return "internet", "Fiber optic broadband with elevated market switching rate"
        if "InternetService_DSL" in clean_feat:
            return "internet", "Standard DSL broadband service"
        if "InternetService_No" in clean_feat:
            return "internet", "Phone-only subscriber without broadband services"
        if clean_feat == "has_protection_bundle_0":
            return "bundle", "No security protection bundle (lacks TechSupport & OnlineSecurity)"
        if clean_feat == "has_protection_bundle_1":
            return "bundle", "Active security bundle (OnlineSecurity / TechSupport)"
        if clean_feat == "TechSupport_No":
            return "tech_support", "No dedicated technical support assistance"
        if clean_feat == "OnlineSecurity_No":
            return "security", "No online security protection service enabled"
        if clean_feat == "OnlineBackup_No":
            return "backup", "No cloud backup service enabled"
        if clean_feat == "DeviceProtection_No":
            return "device", "No device protection plan active"

        # Financial charges
        if clean_feat == "MonthlyCharges":
            if isinstance(monthly, int | float) and monthly > 70:
                return "charges", f"High monthly charges (RM{float(monthly):.2f}/month)"
            return "charges", f"Monthly bill amount (RM{float(monthly):.2f}/month)"
        if clean_feat == "TotalCharges":
            return (
                "total_charges",
                f"Cumulative spend to date (RM{float(row_dict.get('TotalCharges', 0)):.2f})",
            )
        if clean_feat == "charge_increase_ratio":
            return "charge_trend", "Recent monthly bill is higher than historical average spend"
        if clean_feat == "avg_monthly_spend":
            return "avg_spend", "Historical average monthly spend level"

        # Demographics / Account
        if "Partner_No" in clean_feat:
            return "partner", "Single account holder without family bundle"
        if "Dependents_No" in clean_feat:
            return "dependents", "No registered dependents on account"
        if "SeniorCitizen" in clean_feat:
            return "demographic", "Senior citizen customer segment"

        # Clean fallback
        clean_name = clean_feat.replace("_", " ").title()
        return (
            clean_feat,
            f"{clean_name} contributes to {'elevated risk' if shap_val > 0 else 'account retention'}",
        )

    def explain_customer(
        self,
        customer_row: pd.Series | dict[str, Any],
        top_k: int = 3,
    ) -> list[str]:
        """Generate top-k distinct plain language reason codes for a single customer.

        Args:
            customer_row: Series or Dict of customer features.
            top_k: Number of top reasons to return (default: 3).

        Returns:
            List of top-k human-readable reason strings.
        """
        if isinstance(customer_row, dict):
            row_df = pd.DataFrame([customer_row])
            row_dict = customer_row
        else:
            row_df = pd.DataFrame([customer_row.to_dict()])
            row_dict = customer_row.to_dict()

        shap_array, _ = self.compute_shap_values(row_df)
        shap_row = shap_array[0]

        # Rank all features by SHAP value (highest positive first)
        ranked_indices = np.argsort(shap_row)[::-1]

        reasons: list[str] = []
        seen_groups: set[str] = set()

        for idx in ranked_indices:
            feat_name = self.feature_names[idx]
            feat_shap = float(shap_row[idx])
            group_key, reason = self._format_single_reason(feat_name, feat_shap, row_dict)

            if group_key not in seen_groups:
                seen_groups.add(group_key)
                reasons.append(reason)
                if len(reasons) >= top_k:
                    break

        return reasons


def run_explainability_pipeline(
    train_path: Path | str | None = None,
    val_path: Path | str | None = None,
    save_artifacts: bool = True,
) -> tuple[ChurnExplainer, list[dict[str, Any]]]:
    """Execute complete explainability workflow and test sample customers."""
    tr_path = Path(train_path) if train_path else CFG["paths"]["processed_dir"] / "train.parquet"
    va_path = Path(val_path) if val_path else CFG["paths"]["processed_dir"] / "val.parquet"

    train_df = pd.read_parquet(tr_path)
    val_df = pd.read_parquet(va_path)

    # 1. Fit pipeline & instantiate explainer
    pipeline = fit_champion_pipeline(train_df)
    explainer = ChurnExplainer(fitted_pipeline=pipeline)

    # 2. Global summary plot
    if save_artifacts:
        fig_dir = CFG["paths"]["figures_dir"]
        fig_path1 = fig_dir / "09_shap_summary.png"

        # Use subset of val_df for clean plot
        sample_df = val_df.sample(n=min(500, len(val_df)), random_state=42)
        fig = explainer.plot_global_summary(sample_df, save_path=fig_path1)
        plt.close(fig)

    # 3. Explain 5 sample customers
    sample_customers = val_df.head(5)
    sample_explanations = []

    for _, cust in sample_customers.iterrows():
        reasons = explainer.explain_customer(cust, top_k=3)
        sample_explanations.append(
            {
                "customerID": cust.get("customerID", "UNKNOWN"),
                "Churn": int(cust.get("Churn", 0)),
                "Contract": cust.get("Contract", ""),
                "tenure": int(cust.get("tenure", 0)),
                "MonthlyCharges": float(cust.get("MonthlyCharges", 0.0)),
                "top_reasons": reasons,
            }
        )

    return explainer, sample_explanations


if __name__ == "__main__":
    explainer, sample_reasons = run_explainability_pipeline()
    print("=" * 60)
    print("Task T4.3: SHAP Explainability Engine & Sample Reasons")
    print("=" * 60)
    for item in sample_reasons:
        print(f"\nCustomer ID: {item['customerID']} (Actual Churn: {item['Churn']})")
        print(
            f"Contract: {item['Contract']} | Tenure: {item['tenure']}m | Monthly: RM{item['MonthlyCharges']:.2f}"
        )
        print("Top 3 Reason Codes:")
        for idx, r in enumerate(item["top_reasons"], 1):
            print(f"  {idx}. {r}")
