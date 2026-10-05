"""Unit tests for hyperparameter tuning module."""

from churnguard.models.tune import tune_lightgbm


def test_tune_lightgbm_smoke(tmp_path):
    """Smoke test for Optuna tuning with 2 trials."""
    out_file = tmp_path / "test_best_params.json"
    res = tune_lightgbm(n_trials=2, output_path=out_file, log_to_mlflow=False)

    assert "best_params" in res
    assert "best_pr_auc" in res
    assert out_file.exists()
    assert res["best_pr_auc"] > 0.50
