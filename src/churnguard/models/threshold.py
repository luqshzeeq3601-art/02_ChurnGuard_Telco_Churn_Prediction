"""Profit curve calculation, optimal threshold optimization, and sensitivity analysis.

Implements:
- Profit curve across decision thresholds [0.01, 0.99] using calibrated probabilities.
- Optimal threshold search maximising campaign profit on validation set.
- Comparison against baseline strategies: Contact All, Contact None, Default (0.5).
- Expected profit normalized per 1,000 customers.
- Multi-parameter sensitivity analysis (success rate in {0.2, 0.3, 0.4}, offer cost in {40, 50, 60}).
- Publication-ready profit curve visualisation in reports/figures/.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from churnguard.config import CFG
from churnguard.models.calibrate import run_calibration_experiment


def compute_profit_for_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    offer_cost: float = 50.0,
    success_rate: float = 0.30,
    clv_values: np.ndarray | None = None,
    default_clv: float = 780.0,
) -> dict[str, float]:
    """Calculate detailed financial and operational metrics for a specific decision threshold.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_prob: Calibrated churn probabilities.
        threshold: Decision threshold for contacting customers.
        offer_cost: Cost per contacted customer in RM (default: 50.0).
        success_rate: Probability that a contacted churner accepts offer (default: 0.30).
        clv_values: Customer-specific CLV values in RM. If None, uses default_clv.
        default_clv: Default customer lifetime value preserved in RM (default: 780.0).

    Returns:
        Dictionary of threshold performance metrics.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    n = len(y_true)
    clvs = np.full(n, default_clv) if clv_values is None else np.asarray(clv_values).astype(float)

    contacted_mask = y_prob >= threshold
    n_contacted = int(np.sum(contacted_mask))
    pct_contacted = float(n_contacted / n) if n > 0 else 0.0

    true_churners_contacted = int(np.sum(contacted_mask & (y_true == 1)))
    total_churners = int(np.sum(y_true))
    churner_capture_rate = (
        float(true_churners_contacted / total_churners) if total_churners > 0 else 0.0
    )

    # Campaign Financials
    total_cost = n_contacted * offer_cost
    # Revenue preserved: for each true churner contacted, success_rate * individual CLV
    revenue_preserved = (
        float(
            np.sum(contacted_mask & (y_true == 1))
            * success_rate
            * np.mean(
                clvs[contacted_mask & (y_true == 1)]
                if true_churners_contacted > 0
                else default_clv
            )
        )
        if true_churners_contacted > 0
        else 0.0
    )
    net_profit = revenue_preserved - total_cost

    # Profit per 1,000 customers
    profit_per_1k = float((net_profit / n) * 1000.0) if n > 0 else 0.0

    return {
        "threshold": float(threshold),
        "n_contacted": n_contacted,
        "pct_contacted": pct_contacted,
        "true_churners_contacted": true_churners_contacted,
        "churner_capture_rate": churner_capture_rate,
        "total_cost_rm": float(total_cost),
        "revenue_preserved_rm": float(revenue_preserved),
        "net_profit_rm": float(net_profit),
        "profit_per_1k_customers_rm": float(profit_per_1k),
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    offer_cost: float = 50.0,
    success_rate: float = 0.30,
    clv_values: np.ndarray | None = None,
    default_clv: float = 780.0,
    thresholds: np.ndarray | None = None,
) -> dict[str, Any]:
    """Find the threshold that maximises net campaign profit.

    Args:
        y_true: True binary labels.
        y_prob: Calibrated probabilities.
        offer_cost: Offer cost in RM.
        success_rate: Retention success rate.
        clv_values: Optional customer-specific CLV values.
        default_clv: Default CLV.
        thresholds: Array of candidate thresholds (default: 100 points in [0.01, 0.99]).

    Returns:
        Dictionary containing optimal threshold details and the full threshold curve.
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 100)

    curve_records = []
    for t in thresholds:
        res = compute_profit_for_threshold(
            y_true=y_true,
            y_prob=y_prob,
            threshold=t,
            offer_cost=offer_cost,
            success_rate=success_rate,
            clv_values=clv_values,
            default_clv=default_clv,
        )
        curve_records.append(res)

    curve_df = pd.DataFrame(curve_records)
    best_idx = curve_df["net_profit_rm"].idxmax()
    best_row = curve_df.loc[best_idx].to_dict()

    # Compare against baseline benchmarks
    contact_all = compute_profit_for_threshold(
        y_true=y_true,
        y_prob=y_prob,
        threshold=0.0,
        offer_cost=offer_cost,
        success_rate=success_rate,
        clv_values=clv_values,
        default_clv=default_clv,
    )
    contact_none = compute_profit_for_threshold(
        y_true=y_true,
        y_prob=y_prob,
        threshold=1.01,
        offer_cost=offer_cost,
        success_rate=success_rate,
        clv_values=clv_values,
        default_clv=default_clv,
    )
    contact_default_05 = compute_profit_for_threshold(
        y_true=y_true,
        y_prob=y_prob,
        threshold=0.5,
        offer_cost=offer_cost,
        success_rate=success_rate,
        clv_values=clv_values,
        default_clv=default_clv,
    )

    return {
        "optimal_threshold": float(best_row["threshold"]),
        "max_profit_rm": float(best_row["net_profit_rm"]),
        "optimal_metrics": best_row,
        "curve_df": curve_df,
        "benchmarks": {
            "contact_all_profit_per_1k": contact_all["profit_per_1k_customers_rm"],
            "contact_none_profit_per_1k": contact_none["profit_per_1k_customers_rm"],
            "default_05_profit_per_1k": contact_default_05["profit_per_1k_customers_rm"],
            "optimal_profit_per_1k": best_row["profit_per_1k_customers_rm"],
        },
    }


def run_sensitivity_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    success_rates: list[float] | None = None,
    offer_costs: list[float] | None = None,
    clv_values: np.ndarray | None = None,
    default_clv: float = 780.0,
) -> pd.DataFrame:
    """Evaluate optimal threshold and profit across varying business assumptions."""
    if success_rates is None:
        success_rates = [0.20, 0.30, 0.40]
    if offer_costs is None:
        offer_costs = [40.0, 50.0, 60.0]

    records = []
    for sr in success_rates:
        for oc in offer_costs:
            opt = find_optimal_threshold(
                y_true=y_true,
                y_prob=y_prob,
                offer_cost=oc,
                success_rate=sr,
                clv_values=clv_values,
                default_clv=default_clv,
            )
            records.append(
                {
                    "success_rate": sr,
                    "offer_cost_rm": oc,
                    "optimal_threshold": opt["optimal_threshold"],
                    "pct_contacted": opt["optimal_metrics"]["pct_contacted"],
                    "churner_capture_rate": opt["optimal_metrics"]["churner_capture_rate"],
                    "net_profit_rm": opt["optimal_metrics"]["net_profit_rm"],
                    "profit_per_1k_rm": opt["optimal_metrics"]["profit_per_1k_customers_rm"],
                }
            )

    return pd.DataFrame(records)


def plot_profit_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    optimal_res: dict[str, Any],
    sensitivity_df: pd.DataFrame,
    clv_values: np.ndarray | None = None,
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Generate publication-ready profit curve and sensitivity chart."""
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

    # Left: Profit Curve across Thresholds for 20%, 30%, 40% Success Rates
    thresholds = np.linspace(0.01, 0.99, 100)
    colors = {0.20: "#e67e22", 0.30: "#2980b9", 0.40: "#27ae60"}

    for sr, color in colors.items():
        curve_data = [
            compute_profit_for_threshold(
                y_true=y_true,
                y_prob=y_prob,
                threshold=t,
                offer_cost=50.0,
                success_rate=sr,
                clv_values=clv_values,
            )["profit_per_1k_customers_rm"]
            for t in thresholds
        ]
        ax1.plot(
            thresholds,
            curve_data,
            label=f"Success Rate: {int(sr*100)}%",
            color=color,
            linewidth=2.5,
        )

    # Highlight optimal threshold for default case (30% success rate)
    opt_t = optimal_res["optimal_threshold"]
    opt_profit = optimal_res["optimal_metrics"]["profit_per_1k_customers_rm"]
    ax1.axvline(
        opt_t,
        color="#c0392b",
        linestyle="--",
        linewidth=1.5,
        label=f"Optimal Threshold (tau* = {opt_t:.2f})",
    )
    ax1.scatter([opt_t], [opt_profit], color="#c0392b", s=100, zorder=5)
    ax1.annotate(
        f"Max Profit: RM{opt_profit:,.0f} / 1k\n@ tau* = {opt_t:.2f}",
        xy=(opt_t, opt_profit),
        xytext=(opt_t + 0.08, opt_profit - 5000),
        arrowprops=dict(facecolor="#c0392b", shrink=0.08, width=1.5, headwidth=6),
        fontweight="bold",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="#fbeee6", ec="#c0392b", lw=1),
    )

    ax1.axhline(0, color="gray", linestyle=":", alpha=0.7)
    ax1.set_xlabel("Decision Threshold (tau)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Expected Profit per 1,000 Customers (RM)", fontsize=11, fontweight="bold")
    ax1.set_title(
        "Retention Campaign Profit Curve by Decision Threshold",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax1.legend(loc="lower right", frameon=True, fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Right: Sensitivity Matrix (Bar chart by Offer Cost and Success Rate)
    piv = sensitivity_df.pivot(
        index="offer_cost_rm", columns="success_rate", values="profit_per_1k_rm"
    )
    piv.plot(kind="bar", ax=ax2, colormap="viridis", width=0.75, edgecolor="black", alpha=0.85)

    ax2.set_xlabel("Retention Offer Cost (RM)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Expected Profit per 1,000 Customers (RM)", fontsize=11, fontweight="bold")
    ax2.set_title(
        "Profit Sensitivity: Offer Cost vs Success Rate", fontsize=12, fontweight="bold", pad=12
    )
    ax2.legend(title="Success Rate", labels=["20%", "30%", "40%"], loc="upper right", frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.5, axis="y")
    plt.xticks(rotation=0)

    plt.tight_layout()

    if save_path:
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")

    return fig


def run_threshold_optimization(
    val_path: Path | str | None = None,
    oof_path: Path | str | None = None,
    save_artifacts: bool = True,
) -> dict[str, Any]:
    """Execute complete threshold optimization and sensitivity workflow on OOF or validation set."""
    models_dir = CFG["paths"]["models_dir"]
    target_oof = Path(oof_path) if oof_path else models_dir / "oof_train_val_preds.parquet"

    best_calibrator = None
    if val_path is not None:
        va_path = Path(val_path)
        val_df = pd.read_parquet(va_path)
        cal_res = run_calibration_experiment(log_to_mlflow=False)
        best_calibrator = cal_res["best_calibrator"]
        X_val = val_df.drop(columns=["Churn"])
        y_val = val_df["Churn"].values
        probs_val = best_calibrator.predict_proba(X_val)[:, 1]
        clv_val = 12.0 * val_df["MonthlyCharges"].values
        source_label = "val"
    elif target_oof.exists():
        oof_df = pd.read_parquet(target_oof)
        y_val = oof_df["y_true"].values
        prob_col = (
            "prob_selected"
            if "prob_selected" in oof_df.columns
            else ("prob_isotonic" if "prob_isotonic" in oof_df.columns else oof_df.columns[1])
        )
        probs_val = oof_df[prob_col].values
        clv_val = 12.0 * oof_df["MonthlyCharges"].values
        source_label = "oof_train_val"
    else:
        va_path = CFG["paths"]["processed_dir"] / "val.parquet"
        val_df = pd.read_parquet(va_path)
        cal_res = run_calibration_experiment(log_to_mlflow=False)
        best_calibrator = cal_res["best_calibrator"]
        X_val = val_df.drop(columns=["Churn"])
        y_val = val_df["Churn"].values
        probs_val = best_calibrator.predict_proba(X_val)[:, 1]
        clv_val = 12.0 * val_df["MonthlyCharges"].values
        source_label = "val"

    # Find optimal threshold
    opt_res = find_optimal_threshold(
        y_true=y_val,
        y_prob=probs_val,
        offer_cost=CFG["cost"]["retention_offer_cost"],
        success_rate=0.30,
        clv_values=clv_val,
    )

    # Sensitivity analysis
    sens_df = run_sensitivity_analysis(
        y_true=y_val,
        y_prob=probs_val,
        clv_values=clv_val,
    )

    if save_artifacts:
        fig_dir = CFG["paths"]["figures_dir"]
        fig_path1 = fig_dir / "08_profit_curve.png"

        fig = plot_profit_curves(
            y_true=y_val,
            y_prob=probs_val,
            optimal_res=opt_res,
            sensitivity_df=sens_df,
            clv_values=clv_val,
            save_path=fig_path1,
        )
        plt.close(fig)

        # Compute budget capacity cutoffs (top 30% and top 20%)
        tau_30 = float(np.percentile(probs_val, 70))
        tau_20 = float(np.percentile(probs_val, 80))
        res_30 = compute_profit_for_threshold(
            y_true=y_val, y_prob=probs_val, threshold=tau_30, clv_values=clv_val
        )
        res_20 = compute_profit_for_threshold(
            y_true=y_val, y_prob=probs_val, threshold=tau_20, clv_values=clv_val
        )

        # Save optimal threshold summary to models/optimal_threshold.json
        threshold_meta = {
            "source": source_label,
            "n_samples": int(len(y_val)),
            "optimal_threshold": round(opt_res["optimal_threshold"], 4),
            "expected_profit_per_1k_rm": round(
                opt_res["optimal_metrics"]["profit_per_1k_customers_rm"], 2
            ),
            "pct_customers_contacted": round(opt_res["optimal_metrics"]["pct_contacted"] * 100, 2),
            "churner_capture_rate": round(
                opt_res["optimal_metrics"]["churner_capture_rate"] * 100, 2
            ),
            "strategies": {
                "profit_optimal": {
                    "threshold": round(opt_res["optimal_threshold"], 4),
                    "name": "Profit-Optimal (Unconstrained)",
                    "description": "Maximizes net campaign financial return",
                    "target_pct": round(opt_res["optimal_metrics"]["pct_contacted"] * 100, 2),
                    "expected_churner_recall": round(
                        opt_res["optimal_metrics"]["churner_capture_rate"] * 100, 2
                    ),
                },
                "budget_top30": {
                    "threshold": round(tau_30, 4),
                    "name": "Balanced Capacity (Top 30% Budget Cap)",
                    "description": "Targets highest-risk 30% to conserve retention voucher outlay",
                    "target_pct": round(res_30["pct_contacted"] * 100, 2),
                    "expected_churner_recall": round(res_30["churner_capture_rate"] * 100, 2),
                },
                "budget_top20": {
                    "threshold": round(tau_20, 4),
                    "name": "Strict Budget (Top 20% Call-Center Cap)",
                    "description": "Focuses frontline retention agents on the top quintile risk group",
                    "target_pct": round(res_20["pct_contacted"] * 100, 2),
                    "expected_churner_recall": round(res_20["churner_capture_rate"] * 100, 2),
                },
            },
            "benchmarks": {
                "contact_all_profit_per_1k_rm": round(
                    opt_res["benchmarks"]["contact_all_profit_per_1k"], 2
                ),
                "contact_none_profit_per_1k_rm": round(
                    opt_res["benchmarks"]["contact_none_profit_per_1k"], 2
                ),
                "default_05_profit_per_1k_rm": round(
                    opt_res["benchmarks"]["default_05_profit_per_1k"], 2
                ),
                "optimal_profit_per_1k_rm": round(
                    opt_res["benchmarks"]["optimal_profit_per_1k"], 2
                ),
            },
            "risk_tiers": {
                "high_threshold": round(opt_res["optimal_threshold"], 4),
                "medium_threshold": round(0.5 * opt_res["optimal_threshold"], 4),
                "low_threshold": 0.0,
            },
        }

        with open(models_dir / "optimal_threshold.json", "w", encoding="utf-8") as f:
            json.dump(threshold_meta, f, indent=2)

    return {
        "optimal_res": opt_res,
        "sensitivity_df": sens_df,
        "best_calibrator": best_calibrator,
    }


if __name__ == "__main__":
    out = run_threshold_optimization()
    opt = out["optimal_res"]
    print("=" * 60)
    print("Task T4.2: Optimal Threshold Search on Validation Set")
    print("=" * 60)
    print(f"Optimal Threshold (tau*): {opt['optimal_threshold']:.2f}")
    print(f"Contact Rate at tau*:      {opt['optimal_metrics']['pct_contacted']*100:.1f}%")
    print(f"Churner Capture at tau*:   {opt['optimal_metrics']['churner_capture_rate']*100:.1f}%")
    print(
        f"Profit per 1k (tau*):      RM{opt['optimal_metrics']['profit_per_1k_customers_rm']:,.2f}"
    )
    print(f"Profit per 1k (tau=0.5):   RM{opt['benchmarks']['default_05_profit_per_1k']:,.2f}")
    print(f"Profit per 1k (Contact All): RM{opt['benchmarks']['contact_all_profit_per_1k']:,.2f}")
    print(
        f"Profit per 1k (Contact None): RM{opt['benchmarks']['contact_none_profit_per_1k']:,.2f}"
    )
