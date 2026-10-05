"""Final evaluation on held-out test split with 1,000x bootstrap confidence intervals (T4.4).

Evaluates the champion calibrated pipeline ONCE on test.parquet.
Computes point estimates and 95% Bootstrap Confidence Intervals for:
- ROC-AUC, PR-AUC, Brier Score
- Lift@10%, Recall@20%
- Precision, Recall, F1 at optimal threshold tau*
- Expected campaign profit per 1,000 customers (RM)
- Saves full test metrics to reports/final_metrics.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from churnguard.config import CFG, SEED
from churnguard.models.calibrate import run_calibration_experiment
from churnguard.models.evaluate import compute_lift_at_k, compute_recall_at_k
from churnguard.models.threshold import compute_profit_for_threshold, run_threshold_optimization


def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    clv_values: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = SEED,
) -> Dict[str, Dict[str, float]]:
    """Compute 95% percentile bootstrap confidence intervals for all test metrics.

    Args:
        y_true: Ground truth binary labels.
        y_prob: Calibrated predicted probabilities.
        threshold: Decision threshold tau*.
        clv_values: Customer-specific CLV in RM.
        n_bootstrap: Number of bootstrap resamples (default: 1000).
        seed: Random seed for reproducibility.

    Returns:
        Dictionary mapping metric names to {'point': float, 'ci_lower': float, 'ci_upper': float}.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)

    boot_metrics: Dict[str, list[float]] = {
        "roc_auc": [],
        "pr_auc": [],
        "brier_score": [],
        "lift_at_10": [],
        "recall_at_20": [],
        "precision_at_tau": [],
        "recall_at_tau": [],
        "f1_at_tau": [],
        "profit_per_1k_customers_rm": [],
    }

    for _ in range(n_bootstrap):
        boot_idx = rng.choice(n, size=n, replace=True)
        y_tr_b = y_true[boot_idx]
        y_pr_b = y_prob[boot_idx]
        clv_b = clv_values[boot_idx]

        # Check for single-class edge case in resample
        if len(np.unique(y_tr_b)) < 2:
            continue

        y_pred_b = (y_pr_b >= threshold).astype(int)

        boot_metrics["roc_auc"].append(float(roc_auc_score(y_tr_b, y_pr_b)))
        boot_metrics["pr_auc"].append(float(average_precision_score(y_tr_b, y_pr_b)))
        boot_metrics["brier_score"].append(float(brier_score_loss(y_tr_b, y_pr_b)))
        boot_metrics["lift_at_10"].append(float(compute_lift_at_k(y_tr_b, y_pr_b, k=0.10)))
        boot_metrics["recall_at_20"].append(float(compute_recall_at_k(y_tr_b, y_pr_b, k=0.20)))
        boot_metrics["precision_at_tau"].append(float(precision_score(y_tr_b, y_pred_b, zero_division=0)))
        boot_metrics["recall_at_tau"].append(float(recall_score(y_tr_b, y_pred_b, zero_division=0)))
        boot_metrics["f1_at_tau"].append(float(f1_score(y_tr_b, y_pred_b, zero_division=0)))

        profit_res = compute_profit_for_threshold(
            y_true=y_tr_b,
            y_prob=y_pr_b,
            threshold=threshold,
            clv_values=clv_b,
        )
        boot_metrics["profit_per_1k_customers_rm"].append(profit_res["profit_per_1k_customers_rm"])

    ci_results = {}
    for metric_name, values in boot_metrics.items():
        arr = np.array(values)
        ci_lower = float(np.percentile(arr, 2.5))
        ci_upper = float(np.percentile(arr, 97.5))
        ci_results[metric_name] = {
            "ci_lower": round(ci_lower, 4),
            "ci_upper": round(ci_upper, 4),
        }

    return ci_results


def run_final_test_evaluation(
    test_path: Optional[Path | str] = None,
    save_json: bool = True,
) -> Dict[str, Any]:
    """Execute final evaluation on the test set (touched strictly once)."""
    # 1. Fit & calibrate champion model on train and val
    opt_output = run_threshold_optimization(save_artifacts=True)
    best_calibrator = opt_output["best_calibrator"]
    optimal_tau = opt_output["optimal_res"]["optimal_threshold"]

    # 2. Load held-out test data
    te_path = Path(test_path) if test_path else CFG["paths"]["processed_dir"] / "test.parquet"
    test_df = pd.read_parquet(te_path)

    X_test = test_df.drop(columns=["Churn"])
    y_test = test_df["Churn"].values
    clv_test = 12.0 * test_df["MonthlyCharges"].values

    # 3. Predict calibrated probabilities
    y_prob = best_calibrator.predict_proba(X_test)[:, 1]
    y_pred_tau = (y_prob >= optimal_tau).astype(int)

    # 4. Point Estimates
    profit_res = compute_profit_for_threshold(
        y_true=y_test,
        y_prob=y_prob,
        threshold=optimal_tau,
        clv_values=clv_test,
    )

    point_estimates = {
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
        "pr_auc": round(float(average_precision_score(y_test, y_prob)), 4),
        "brier_score": round(float(brier_score_loss(y_test, y_prob)), 4),
        "lift_at_10": round(float(compute_lift_at_k(y_test, y_prob, k=0.10)), 4),
        "recall_at_20": round(float(compute_recall_at_k(y_test, y_prob, k=0.20)), 4),
        "optimal_threshold": round(float(optimal_tau), 4),
        "precision_at_tau": round(float(precision_score(y_test, y_pred_tau, zero_division=0)), 4),
        "recall_at_tau": round(float(recall_score(y_test, y_pred_tau, zero_division=0)), 4),
        "f1_at_tau": round(float(f1_score(y_test, y_pred_tau, zero_division=0)), 4),
        "pct_contacted_at_tau": round(float(profit_res["pct_contacted"] * 100), 2),
        "churner_capture_rate_at_tau": round(float(profit_res["churner_capture_rate"] * 100), 2),
        "net_profit_rm": round(float(profit_res["net_profit_rm"]), 2),
        "profit_per_1k_customers_rm": round(float(profit_res["profit_per_1k_customers_rm"]), 2),
    }

    # 5. Bootstrap CIs (1,000x)
    ci_results = compute_bootstrap_ci(
        y_true=y_test,
        y_prob=y_prob,
        threshold=optimal_tau,
        clv_values=clv_test,
        n_bootstrap=1000,
        seed=SEED,
    )

    # Combine into comprehensive report
    final_report = {
        "dataset": "IBM Telco Churn (Held-out Test Split)",
        "n_samples": len(test_df),
        "n_churners": int(np.sum(y_test)),
        "churn_rate": round(float(np.mean(y_test) * 100), 2),
        "champion_model": "LightGBM + Isotonic Calibration",
        "optimal_threshold": point_estimates["optimal_threshold"],
        "metrics": {
            k: {
                "point_estimate": point_estimates[k],
                "ci_95_lower": ci_results[k]["ci_lower"] if k in ci_results else None,
                "ci_95_upper": ci_results[k]["ci_upper"] if k in ci_results else None,
            }
            for k in [
                "roc_auc",
                "pr_auc",
                "brier_score",
                "lift_at_10",
                "recall_at_20",
                "precision_at_tau",
                "recall_at_tau",
                "f1_at_tau",
                "profit_per_1k_customers_rm",
            ]
        },
        "operational_summary": {
            "pct_contacted": point_estimates["pct_contacted_at_tau"],
            "churner_capture_rate": point_estimates["churner_capture_rate_at_tau"],
            "net_profit_rm": point_estimates["net_profit_rm"],
            "profit_per_1k_rm": point_estimates["profit_per_1k_customers_rm"],
        },
        "objectives_verification": {
            "MO1_beat_baseline": bool(point_estimates["pr_auc"] >= 0.6587),
            "MO2_roc_auc_ge_084": bool(point_estimates["roc_auc"] >= 0.84),
            "MO2_pr_auc_ge_062": bool(point_estimates["pr_auc"] >= 0.62),
            "MO2_lift_ge_25": bool(point_estimates["lift_at_10"] >= 2.5),
            "MO3_brier_calibrated": bool(point_estimates["brier_score"] < 0.1542),
            "BO1_recall_at_20_ge_50": bool(point_estimates["recall_at_20"] >= 0.50),
            "BO2_profit_beats_all_and_none": bool(point_estimates["profit_per_1k_customers_rm"] > 21614.0),
        },
    }

    if save_json:
        out_path = Path(CFG["paths"]["reports_dir"]) / "final_metrics.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(final_report, f, indent=2)

    return final_report


if __name__ == "__main__":
    report = run_final_test_evaluation()
    print("=" * 65)
    print("Task T4.4: Final Held-out Test Evaluation with 1,000x Bootstrap CI")
    print("=" * 65)
    print(f"Test Samples: {report['n_samples']} | Churners: {report['n_churners']} ({report['churn_rate']}%)")
    print(f"Optimal Threshold: {report['optimal_threshold']:.2f}")
    print("-" * 65)
    for k, v in report["metrics"].items():
        ci_str = f"[{v['ci_95_lower']:.4f}, {v['ci_95_upper']:.4f}]" if v["ci_95_lower"] is not None else ""
        print(f"  {k:<28}: {v['point_estimate']:<10} 95% CI: {ci_str}")
    print("-" * 65)
    print("SMART Objectives Verification:")
    for obj_name, passed in report["objectives_verification"].items():
        print(f"  {obj_name:<32}: {'PASS [x]' if passed else 'FAIL [ ]'}")
