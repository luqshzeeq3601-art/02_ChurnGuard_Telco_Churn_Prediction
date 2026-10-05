"""Unit tests for model training and CV pipeline."""

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from churnguard.models.train import run_cv_experiment


def test_run_cv_experiment_dummy(tmp_path):
    """Test CV with dummy model."""
    res = run_cv_experiment(
        exp_id="TEST_E00",
        run_name="test_dummy",
        model=DummyClassifier(strategy="most_frequent"),
        include_engineered=False,
        scale_numeric=False,
        n_splits=2,
        log_to_mlflow=False,
    )
    assert res["exp_id"] == "TEST_E00"
    assert "cv_pr_auc_mean" in res["mean_metrics"]
    assert "cv_roc_auc_mean" in res["mean_metrics"]


def test_run_cv_experiment_logistic_with_mlflow(tmp_path):
    """Test CV with LogisticRegression and MLflow logging enabled."""
    res = run_cv_experiment(
        exp_id="TEST_E01",
        run_name="test_logistic",
        model=LogisticRegression(solver="liblinear", max_iter=100),
        include_engineered=True,
        scale_numeric=True,
        n_splits=2,
        log_to_mlflow=True,
    )
    assert res["exp_id"] == "TEST_E01"
    assert res["mean_metrics"]["cv_roc_auc_mean"] > 0.60
    assert "oof_pr_auc" in res["oof_metrics"]
