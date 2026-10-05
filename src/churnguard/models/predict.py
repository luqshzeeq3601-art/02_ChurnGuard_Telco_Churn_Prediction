"""Standalone inference engine and batch scoring CLI for ChurnGuard (T5.1 / T5.3).

Implements:
- ChurnPredictor: loads serialized model pipeline and metadata.
- Single customer scoring with calibrated probability, risk tier, and top-3 SHAP reasons.
- Batch scoring with risk ranking, summary aggregations, and CSV output.
- CLI interface for `make score FILE=...`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd

from churnguard.config import CFG
from churnguard.explain.shap_explain import ChurnExplainer


class ChurnPredictor:
    """Production inference engine for single-record and batch customer churn predictions."""

    def __init__(
        self,
        model_path: Optional[Path | str] = None,
        meta_path: Optional[Path | str] = None,
    ) -> None:
        """Initialize predictor by loading serialized pipeline and metadata.

        Args:
            model_path: Path to models/model.joblib.
            meta_path: Path to models/model_meta.json.
        """
        m_path = Path(model_path) if model_path else Path(CFG["paths"]["models_dir"]) / "model.joblib"
        mt_path = Path(meta_path) if meta_path else Path(CFG["paths"]["models_dir"]) / "model_meta.json"

        if not m_path.exists():
            raise FileNotFoundError(f"Model artifact not found at {m_path}. Run training & serialization first.")
        if not mt_path.exists():
            raise FileNotFoundError(f"Model metadata not found at {mt_path}.")

        self.model = joblib.load(m_path)
        with open(mt_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        self.model_version = self.meta.get("model_version", "1.0.0")
        self.optimal_threshold = float(self.meta.get("optimal_threshold", 0.18))
        self.medium_threshold = float(self.meta.get("risk_tiers", {}).get("medium", {}).get("min_prob", 0.09))

        # Initialize SHAP explainer for reason code generation
        # Extract underlying fitted pipeline from CalibratedClassifierCV if calibrated
        if hasattr(self.model, "estimator"):
            base_pipeline = self.model.estimator
        elif hasattr(self.model, "calibrated_classifiers_"):
            base_pipeline = self.model.calibrated_classifiers_[0].estimator
        else:
            base_pipeline = self.model

        self.explainer = ChurnExplainer(fitted_pipeline=base_pipeline)

    def _assign_risk_tier(self, probability: float) -> str:
        """Assign risk tier based on optimized threshold cutoffs."""
        if probability >= self.optimal_threshold:
            return "High"
        elif probability >= self.medium_threshold:
            return "Medium"
        return "Low"

    def predict_single(self, customer: Dict[str, Any] | pd.Series) -> Dict[str, Any]:
        """Generate prediction, risk tier, and plain-language reasons for a single customer.

        Args:
            customer: Dictionary or Series with customer features.

        Returns:
            Dictionary matching API response contract:
            {
                "customerID": str,
                "churn_probability": float,
                "risk_tier": "High" | "Medium" | "Low",
                "top_reasons": list[str],
                "model_version": str,
            }
        """
        cust_dict = customer.to_dict() if isinstance(customer, pd.Series) else dict(customer)
        cust_id = str(cust_dict.get("customerID", "UNKNOWN"))

        df = pd.DataFrame([cust_dict])
        prob = float(self.model.predict_proba(df)[:, 1][0])
        prob_rounded = round(prob, 4)
        tier = self._assign_risk_tier(prob_rounded)
        reasons = self.explainer.explain_customer(cust_dict, top_k=3)

        return {
            "customerID": cust_id,
            "churn_probability": prob_rounded,
            "risk_tier": tier,
            "top_reasons": reasons,
            "model_version": self.model_version,
        }

    def predict_batch(
        self,
        data: pd.DataFrame | List[Dict[str, Any]],
        include_reasons: bool = True,
    ) -> pd.DataFrame:
        """Score multiple customers and return a DataFrame ranked by churn risk descending.

        Args:
            data: DataFrame or list of customer dictionaries.
            include_reasons: Whether to generate top-3 reasons for each customer.

        Returns:
            DataFrame with predictions, ranked descending by churn_probability.
        """
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data.copy()

        if len(df) == 0:
            return pd.DataFrame(columns=["customerID", "churn_probability", "risk_tier", "top_reasons", "rank"])

        probs = self.model.predict_proba(df)[:, 1]
        probs_rounded = np.round(probs, 4)

        result_df = df.copy()
        result_df["churn_probability"] = probs_rounded
        result_df["risk_tier"] = [self._assign_risk_tier(p) for p in probs_rounded]

        if include_reasons:
            reasons_list = []
            for _, row in df.iterrows():
                reasons = self.explainer.explain_customer(row, top_k=3)
                reasons_list.append(reasons)
            result_df["top_reasons"] = reasons_list

        result_df["model_version"] = self.model_version

        # Sort descending by risk
        result_df = result_df.sort_values(by="churn_probability", ascending=False).reset_index(drop=True)
        result_df["rank"] = np.arange(1, len(result_df) + 1)

        return result_df


def score_file_cli(
    input_file: Path | str,
    output_file: Optional[Path | str] = None,
    top_n: int = 5,
) -> pd.DataFrame:
    """Read a batch file (CSV or Parquet), score customers, and export ranked CSV."""
    in_path = Path(input_file)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found at: {in_path}")

    if in_path.suffix == ".parquet":
        df = pd.read_parquet(in_path)
    else:
        df = pd.read_csv(in_path)

    predictor = ChurnPredictor()
    scored_df = predictor.predict_batch(df, include_reasons=True)

    out_path = Path(output_file) if output_file else Path(CFG["paths"]["reports_dir"]) / "scored.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Flatten reasons list for clean CSV export
    export_df = scored_df.copy()
    if "top_reasons" in export_df.columns:
        export_df["reason_1"] = export_df["top_reasons"].apply(lambda r: r[0] if len(r) > 0 else "")
        export_df["reason_2"] = export_df["top_reasons"].apply(lambda r: r[1] if len(r) > 1 else "")
        export_df["reason_3"] = export_df["top_reasons"].apply(lambda r: r[2] if len(r) > 2 else "")
        export_df = export_df.drop(columns=["top_reasons"])

    export_df.to_csv(out_path, index=False)

    print("\n" + "=" * 65)
    print(f"BATCH SCORING COMPLETED: {len(scored_df)} Customers Scored")
    print(f"Output exported to: {out_path}")
    print("=" * 65)

    tier_counts = scored_df["risk_tier"].value_counts()
    print("\nRisk Tier Distribution:")
    for tier in ["High", "Medium", "Low"]:
        count = tier_counts.get(tier, 0)
        pct = (count / len(scored_df)) * 100 if len(scored_df) > 0 else 0.0
        print(f"  - {tier:<8} Risk: {count:>5} ({pct:>5.1f}%)")

    print(f"\nTop {top_n} Highest Risk Customers:")
    display_cols = ["rank", "customerID", "churn_probability", "risk_tier", "Contract", "MonthlyCharges"]
    cols_present = [c for c in display_cols if c in scored_df.columns]
    print(scored_df[cols_present].head(top_n).to_string(index=False))

    return scored_df


def main() -> None:
    """CLI Entry point for batch scoring."""
    parser = argparse.ArgumentParser(description="ChurnGuard Batch Customer Scoring CLI")
    parser.add_argument("--file", "-f", type=str, required=True, help="Path to input CSV or Parquet file")
    parser.add_argument("--output", "-o", type=str, default=None, help="Path to output ranked CSV")
    parser.add_argument("--top-n", type=int, default=5, help="Number of top risk records to print")

    args = parser.parse_args()
    score_file_cli(input_file=args.file, output_file=args.output, top_n=args.top_n)


if __name__ == "__main__":
    main()
