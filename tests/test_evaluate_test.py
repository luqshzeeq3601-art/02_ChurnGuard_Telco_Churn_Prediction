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
    assert report["n_samples"] == 1057
    assert report["metrics"]["roc_auc"]["point_estimate"] > 0.80
    assert report["metrics"]["pr_auc"]["point_estimate"] > 0.60
