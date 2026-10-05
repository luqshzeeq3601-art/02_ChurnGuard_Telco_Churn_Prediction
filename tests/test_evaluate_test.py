"""Tests for final test evaluation module and bootstrap confidence intervals."""

import numpy as np

from churnguard.models.evaluate_test import compute_bootstrap_ci, run_final_test_evaluation


def test_compute_bootstrap_ci_toy():
    """Test bootstrap CI calculation on synthetic inputs."""
    np.random.seed(42)
    n = 100
    y_true = np.random.choice([0, 1], n, p=[0.75, 0.25])
    y_prob = np.random.uniform(0.0, 1.0, n)
    clvs = np.full(n, 780.0)

    ci_res = compute_bootstrap_ci(
        y_true=y_true,
        y_prob=y_prob,
        threshold=0.5,
        clv_values=clvs,
        n_bootstrap=50,
        seed=42,
    )
    assert "roc_auc" in ci_res
    assert "pr_auc" in ci_res
    assert "brier_score" in ci_res
    assert ci_res["roc_auc"]["ci_lower"] <= ci_res["roc_auc"]["ci_upper"]


def test_run_final_test_evaluation_integration(tmp_path):
    """Test full integration evaluation on processed test parquet."""
    report = run_final_test_evaluation(save_json=True)
    assert "metrics" in report
    assert "objectives_verification" in report
    assert "objectives_detail" in report
    assert report["n_samples"] == 1057
    assert report["metrics"]["roc_auc"]["point_estimate"] > 0.80
    assert report["metrics"]["pr_auc"]["point_estimate"] > 0.60


def test_compute_objectives_verification_same_split():
    """Verify objective checks compare calibrated vs uncalibrated on same split and no hardcoded constants (T9.8 / F4)."""
    from churnguard.models.evaluate_test import compute_objectives_verification

    y_test = np.array([1, 1, 0, 0, 1, 0, 0, 1])
    cal_prob = np.array([0.8, 0.7, 0.2, 0.1, 0.6, 0.3, 0.2, 0.9])
    uncal_prob = np.array([0.95, 0.85, 0.05, 0.02, 0.9, 0.4, 0.1, 0.99])
    gender_arr = np.array(["Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female"])
    senior_arr = np.array([0, 1, 0, 0, 1, 0, 0, 1])

    point_estimates = {
        "roc_auc": 0.88,
        "pr_auc": 0.75,
        "brier_score": 0.12,
        "lift_at_10": 2.8,
        "recall_at_20": 0.55,
        "profit_per_1k_customers_rm": 35000.0,
    }
    ci_results = {
        "roc_auc": {"ci_lower": 0.82, "ci_upper": 0.94},
        "pr_auc": {"ci_lower": 0.68, "ci_upper": 0.82},
        "lift_at_10": {"ci_lower": 2.4, "ci_upper": 3.2},
    }

    verif, detail = compute_objectives_verification(
        y_test=y_test,
        y_prob=cal_prob,
        uncal_prob=uncal_prob,
        point_estimates=point_estimates,
        ci_results=ci_results,
        test_contact_all_profit=20000.0,
        gender_arr=gender_arr,
        senior_arr=senior_arr,
        optimal_tau=0.5,
    )

    assert "MO1_beat_baseline" in verif
    assert "MO2_roc_auc_ge_084" in verif
    assert "MO3_brier_calibrated" in verif
    assert "BO1_recall_at_20_ge_50" in verif
    assert "BO2_profit_beats_all_and_none" in verif
    assert "NFR7_demographic_fairness" in verif

    # Check detail structure guarantees same-split comparisons
    assert detail["MO3"]["same_test_rows"] is True
    assert detail["MO3"]["uncalibrated_test_brier"] is not None
    assert detail["BO2"]["same_test_rows"] is True
    assert detail["BO2"]["test_contact_all_profit_per_1k"] == 20000.0
