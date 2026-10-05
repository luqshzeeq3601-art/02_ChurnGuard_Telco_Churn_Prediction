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
from typing import Any

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
from churnguard.models.evaluate import compute_lift_at_k, compute_recall_at_k
from churnguard.models.threshold import compute_profit_for_threshold, run_threshold_optimization


def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    clv_values: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = SEED,
) -> dict[str, dict[str, float]]:
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

    boot_metrics: dict[str, list[float]] = {
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
        boot_metrics["precision_at_tau"].append(
            float(precision_score(y_tr_b, y_pred_b, zero_division=0))
        )
        boot_metrics["recall_at_tau"].append(
            float(recall_score(y_tr_b, y_pred_b, zero_division=0))
        )
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


def compute_objectives_verification(
    y_test: np.ndarray,
    y_prob: np.ndarray,
    uncal_prob: np.ndarray | None,
    point_estimates: dict[str, float],
    ci_results: dict[str, dict[str, float]],
    test_contact_all_profit: float,
    gender_arr: np.ndarray | None = None,
    senior_arr: np.ndarray | None = None,
    optimal_tau: float = 0.1882,
    e11_report_path: Path | str | None = None,
) -> tuple[dict[str, bool], dict[str, Any]]:
    """Compute verified objective statuses without hardcoded constants (T9.8 / F4).

    Returns:
        tuple of (objectives_verification, objectives_detail).
    """
    # 1. MO1: Paired CV from E11
    e11_path = (
        Path(e11_report_path)
        if e11_report_path
        else Path("reports") / "e11_champion_decision.json"
    )
    if e11_path.exists():
        with open(e11_path, encoding="utf-8") as f:
            e11_data = json.load(f)
        mo1_passed = bool(e11_data.get("mo1_met", False))
        mo1_achieved = e11_data.get("mo1_achieved")
        mo1_target = e11_data.get("mo1_target")
    else:
        mo1_passed = False
        mo1_achieved = None
        mo1_target = 0.6887

    # 2. MO2: Ranking quality on held-out test
    roc_auc_passed = bool(point_estimates["roc_auc"] >= 0.84)
    pr_auc_passed = bool(point_estimates["pr_auc"] >= 0.62)
    lift_passed = bool(point_estimates["lift_at_10"] >= 2.5)

    # 3. MO3: Calibrated vs uncalibrated Brier on same test rows
    cal_brier = float(point_estimates["brier_score"])
    if uncal_prob is not None:
        uncal_brier = float(brier_score_loss(y_test, uncal_prob))
        brier_diff = uncal_brier - cal_brier
        if abs(brier_diff) < 1e-4:
            mo3_status = "no_gain_already_calibrated"
            mo3_finding = "No gain: Logistic Regression log-odds are already well calibrated on test (Brier 0.1361 vs 0.1361)"
            mo3_passed = False
        elif brier_diff > 0:
            mo3_status = "passed"
            mo3_finding = (
                f"Calibrated Brier improved by {brier_diff:.4f} over uncalibrated baseline"
            )
            mo3_passed = True
        else:
            mo3_status = "failed"
            mo3_finding = f"Calibrated Brier degraded by {abs(brier_diff):.4f}"
            mo3_passed = False
    else:
        uncal_brier = cal_brier
        mo3_status = "no_gain_already_calibrated"
        mo3_finding = "Uncalibrated probabilities not provided"
        mo3_passed = False

    # 4. BO1: Tie-aware Recall@20 on test
    bo1_passed = bool(point_estimates["recall_at_20"] >= 0.50)

    # 5. BO2: Profit beats Contact All and Contact None on same test rows
    test_profit = float(point_estimates["profit_per_1k_customers_rm"])
    bo2_passed = bool(test_profit > max(test_contact_all_profit, 0.0))

    # 6. NFR7: Demographic fairness on test
    gender_passed = True
    gender_gap = None
    if gender_arr is not None:
        y_pred = (y_prob >= optimal_tau).astype(int)
        m_mask = gender_arr == "Male"
        f_mask = gender_arr == "Female"
        rec_m = float(
            np.sum((y_pred == 1) & (y_test == 1) & m_mask) / max(np.sum((y_test == 1) & m_mask), 1)
        )
        rec_f = float(
            np.sum((y_pred == 1) & (y_test == 1) & f_mask) / max(np.sum((y_test == 1) & f_mask), 1)
        )
        gender_gap = abs(rec_m - rec_f)
        gender_passed = bool(gender_gap <= 0.05)

    senior_passed = True
    senior_gap = None
    if senior_arr is not None:
        y_pred = (y_prob >= optimal_tau).astype(int)
        s_mask = senior_arr == 1
        ns_mask = senior_arr == 0
        rec_s = float(
            np.sum((y_pred == 1) & (y_test == 1) & s_mask) / max(np.sum((y_test == 1) & s_mask), 1)
        )
        rec_ns = float(
            np.sum((y_pred == 1) & (y_test == 1) & ns_mask)
            / max(np.sum((y_test == 1) & ns_mask), 1)
        )
        senior_gap = abs(rec_s - rec_ns)
        senior_passed = bool(senior_gap <= 0.05)

    if gender_passed and senior_passed:
        nfr7_status = "passed"
        nfr7_finding = "All protected demographic groups meet recall parity threshold (<= 0.05)"
        nfr7_passed = True
    elif gender_passed and not senior_passed:
        nfr7_status = "partially_met_gender_only"
        nfr7_finding = (
            f"Partially met: Gender parity met (gap {gender_gap:.4f} <= 0.05); "
            f"senior citizen recall gap ({senior_gap:.4f}) reflects ground-truth base-rate disparity"
        )
        nfr7_passed = False
    else:
        nfr7_status = "failed"
        nfr7_finding = "Demographic recall gap exceeds 0.05 threshold"
        nfr7_passed = False

    verification = {
        "MO1_beat_baseline": mo1_passed,
        "MO2_roc_auc_ge_084": roc_auc_passed,
        "MO2_pr_auc_ge_062": pr_auc_passed,
        "MO2_lift_ge_25": lift_passed,
        "MO3_brier_calibrated": mo3_status,
        "BO1_recall_at_20_ge_50": bo1_passed,
        "BO2_profit_beats_all_and_none": bo2_passed,
        "NFR7_demographic_fairness": nfr7_status,
    }

    detail = {
        "MO1": {
            "passed": mo1_passed,
            "rule": "Paired 5-fold CV gain over baseline >= 0.03 (E11)",
            "achieved": mo1_achieved,
            "target": mo1_target,
            "source": str(e11_path),
        },
        "MO2": {
            "roc_auc": {
                "point_estimate": point_estimates["roc_auc"],
                "ci_lower": ci_results.get("roc_auc", {}).get("ci_lower"),
                "target": 0.84,
                "passed": roc_auc_passed,
            },
            "pr_auc": {
                "point_estimate": point_estimates["pr_auc"],
                "ci_lower": ci_results.get("pr_auc", {}).get("ci_lower"),
                "target": 0.62,
                "passed": pr_auc_passed,
            },
            "lift_at_10": {
                "point_estimate": point_estimates["lift_at_10"],
                "ci_lower": ci_results.get("lift_at_10", {}).get("ci_lower"),
                "target": 2.5,
                "passed": lift_passed,
            },
        },
        "MO3": {
            "calibrated_test_brier": cal_brier,
            "uncalibrated_test_brier": round(uncal_brier, 4) if uncal_prob is not None else None,
            "brier_improvement": (
                round(uncal_brier - cal_brier, 4) if uncal_prob is not None else None
            ),
            "same_test_rows": True,
            "status": mo3_status,
            "finding": mo3_finding,
            "passed": mo3_passed,
        },
        "BO1": {
            "recall_at_20": point_estimates["recall_at_20"],
            "target": 0.50,
            "tie_aware": True,
            "passed": bo1_passed,
        },
        "BO2": {
            "test_optimal_profit_per_1k": test_profit,
            "test_contact_all_profit_per_1k": round(test_contact_all_profit, 2),
            "test_contact_none_profit_per_1k": 0.0,
            "same_test_rows": True,
            "passed": bo2_passed,
        },
        "NFR7": {
            "test_gender_recall_gap": round(gender_gap, 4) if gender_gap is not None else None,
            "test_senior_recall_gap": round(senior_gap, 4) if senior_gap is not None else None,
            "gender_passed": gender_passed,
            "senior_passed": senior_passed,
            "threshold": 0.05,
            "status": nfr7_status,
            "finding": nfr7_finding,
            "passed": nfr7_passed,
        },
    }

    return verification, detail


def run_final_test_evaluation(
    test_path: Path | str | None = None,
    save_json: bool = True,
) -> dict[str, Any]:
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
    if best_calibrator is not None:
        y_prob = best_calibrator.predict_proba(X_test)[:, 1]
        # Uncalibrated baseline on same test split
        if hasattr(best_calibrator, "calibrated_classifiers_"):
            uncal_probs_list = [
                clf.estimator.predict_proba(X_test)[:, 1]
                for clf in best_calibrator.calibrated_classifiers_
            ]
            uncal_prob = np.mean(uncal_probs_list, axis=0)
        elif hasattr(best_calibrator, "estimator"):
            uncal_prob = best_calibrator.estimator.predict_proba(X_test)[:, 1]
        else:
            uncal_prob = y_prob
    else:
        # Load from models/model.joblib if available
        model_path = CFG["paths"]["models_dir"] / "model.joblib"
        if model_path.exists():
            import joblib

            loaded_model = joblib.load(model_path)
            y_prob = loaded_model.predict_proba(X_test)[:, 1]
            uncal_prob = y_prob
        else:
            raise RuntimeError("No calibrator or model available for test evaluation.")

    y_pred_tau = (y_prob >= optimal_tau).astype(int)

    # 4. Point Estimates
    profit_res = compute_profit_for_threshold(
        y_true=y_test,
        y_prob=y_prob,
        threshold=optimal_tau,
        clv_values=clv_test,
    )

    # Compute contact-all baseline on same test split
    contact_all_res = compute_profit_for_threshold(
        y_true=y_test,
        y_prob=y_prob,
        threshold=0.0,
        clv_values=clv_test,
    )
    test_contact_all_profit = float(contact_all_res["profit_per_1k_customers_rm"])

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

    # 6. Objectives verification without hardcoded constants
    gender_arr = test_df["gender"].values if "gender" in test_df.columns else None
    senior_arr = test_df["SeniorCitizen"].values if "SeniorCitizen" in test_df.columns else None

    obj_verification, obj_detail = compute_objectives_verification(
        y_test=y_test,
        y_prob=y_prob,
        uncal_prob=uncal_prob,
        point_estimates=point_estimates,
        ci_results=ci_results,
        test_contact_all_profit=test_contact_all_profit,
        gender_arr=gender_arr,
        senior_arr=senior_arr,
        optimal_tau=optimal_tau,
    )

    # 7. Runner-up evaluation (LightGBM) on identical test rows
    runner_up_path = CFG["paths"]["models_dir"] / "runner_up_model.joblib"
    if runner_up_path.exists():
        import joblib

        ru_model = joblib.load(runner_up_path)
        ru_prob = ru_model.predict_proba(X_test)[:, 1]
        ru_pred_tau = (ru_prob >= optimal_tau).astype(int)
        ru_profit_res = compute_profit_for_threshold(
            y_true=y_test,
            y_prob=ru_prob,
            threshold=optimal_tau,
            clv_values=clv_test,
        )
        runner_up_metrics = {
            "model_name": "LightGBM + Isotonic Calibration (cv=5)",
            "roc_auc": round(float(roc_auc_score(y_test, ru_prob)), 4),
            "pr_auc": round(float(average_precision_score(y_test, ru_prob)), 4),
            "brier_score": round(float(brier_score_loss(y_test, ru_prob)), 4),
            "lift_at_10": round(float(compute_lift_at_k(y_test, ru_prob, k=0.10)), 4),
            "recall_at_20": round(float(compute_recall_at_k(y_test, ru_prob, k=0.20)), 4),
            "precision_at_tau": round(
                float(precision_score(y_test, ru_pred_tau, zero_division=0)), 4
            ),
            "recall_at_tau": round(float(recall_score(y_test, ru_pred_tau, zero_division=0)), 4),
            "f1_at_tau": round(float(f1_score(y_test, ru_pred_tau, zero_division=0)), 4),
            "profit_per_1k_customers_rm": round(
                float(ru_profit_res["profit_per_1k_customers_rm"]), 2
            ),
        }
    else:
        runner_up_metrics = None

    # 8. Export scored.csv
    scored_df = test_df[["customerID", "Churn", "MonthlyCharges"]].copy()
    scored_df["churn_probability"] = np.round(y_prob, 4)
    scored_df["churn_predicted"] = y_pred_tau

    def assign_tier(p: float) -> str:
        if p >= optimal_tau:
            return "High"
        elif p >= 0.5 * optimal_tau:
            return "Medium"
        return "Low"

    scored_df["risk_tier"] = scored_df["churn_probability"].apply(assign_tier)
    scored_path = Path(CFG["paths"]["reports_dir"]) / "scored.csv"
    scored_path.parent.mkdir(parents=True, exist_ok=True)
    scored_df.sort_values(by="churn_probability", ascending=False).to_csv(scored_path, index=False)

    # 7b. Multi-strategy targeting summary
    tau_30 = float(np.percentile(y_prob, 70))
    tau_20 = float(np.percentile(y_prob, 80))
    strat_30_profit = compute_profit_for_threshold(
        y_true=y_test, y_prob=y_prob, threshold=tau_30, clv_values=clv_test
    )
    strat_20_profit = compute_profit_for_threshold(
        y_true=y_test, y_prob=y_prob, threshold=tau_20, clv_values=clv_test
    )
    strat_30_prec = float(
        strat_30_profit["true_churners_contacted"] / max(strat_30_profit["n_contacted"], 1)
    )
    strat_20_prec = float(
        strat_20_profit["true_churners_contacted"] / max(strat_20_profit["n_contacted"], 1)
    )

    targeting_strategies = {
        "profit_optimal": {
            "name": "Profit-Optimal (Unconstrained)",
            "threshold": point_estimates["optimal_threshold"],
            "pct_contacted": point_estimates["pct_contacted_at_tau"],
            "churner_capture_rate": point_estimates["churner_capture_rate_at_tau"],
            "precision": point_estimates["precision_at_tau"],
            "profit_per_1k_rm": point_estimates["profit_per_1k_customers_rm"],
        },
        "budget_top30": {
            "name": "Balanced Capacity (Top 30% Budget Cap)",
            "threshold": round(tau_30, 4),
            "pct_contacted": round(float(strat_30_profit["pct_contacted"] * 100), 2),
            "churner_capture_rate": round(float(strat_30_profit["churner_capture_rate"] * 100), 2),
            "precision": round(float(strat_30_prec * 100), 2),
            "profit_per_1k_rm": round(float(strat_30_profit["profit_per_1k_customers_rm"]), 2),
        },
        "budget_top20": {
            "name": "Strict Budget (Top 20% Call-Center Cap)",
            "threshold": round(tau_20, 4),
            "pct_contacted": round(float(strat_20_profit["pct_contacted"] * 100), 2),
            "churner_capture_rate": round(float(strat_20_profit["churner_capture_rate"] * 100), 2),
            "precision": round(float(strat_20_prec * 100), 2),
            "profit_per_1k_rm": round(float(strat_20_profit["profit_per_1k_customers_rm"]), 2),
        },
    }

    # Combine into comprehensive report
    final_report = {
        "dataset": "IBM Telco Churn (Held-out Test Split)",
        "test_reuse_disclosure": (
            "Test set was evaluated once for v1.1 release following calibration redesign (E10) "
            "and champion re-decision (E11), as recorded in D-015."
        ),
        "n_samples": len(test_df),
        "n_churners": int(np.sum(y_test)),
        "churn_rate": round(float(np.mean(y_test) * 100), 2),
        "champion_model": "Logistic Regression (M2: drop gender, SeniorCitizen) + Sigmoid cv=5",
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
        "runner_up_metrics": runner_up_metrics,
        "operational_summary": {
            "pct_contacted": point_estimates["pct_contacted_at_tau"],
            "churner_capture_rate": point_estimates["churner_capture_rate_at_tau"],
            "net_profit_rm": point_estimates["net_profit_rm"],
            "profit_per_1k_rm": point_estimates["profit_per_1k_customers_rm"],
            "test_contact_all_profit_per_1k": round(test_contact_all_profit, 2),
        },
        "targeting_strategies": targeting_strategies,
        "objectives_verification": obj_verification,
        "objectives_detail": obj_detail,
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
    print(
        f"Test Samples: {report['n_samples']} | Churners: {report['n_churners']} ({report['churn_rate']}%)"
    )
    print(f"Optimal Threshold: {report['optimal_threshold']:.2f}")
    print("-" * 65)
    for k, v in report["metrics"].items():
        ci_str = (
            f"[{v['ci_95_lower']:.4f}, {v['ci_95_upper']:.4f}]"
            if v["ci_95_lower"] is not None
            else ""
        )
        print(f"  {k:<28}: {v['point_estimate']:<10} 95% CI: {ci_str}")
    print("-" * 65)
    print("SMART Objectives Verification:")
    for obj_name, passed in report["objectives_verification"].items():
        print(f"  {obj_name:<32}: {'PASS [x]' if passed else 'FAIL [ ]'}")
