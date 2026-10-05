"""Pytest configuration and global fixtures."""

import matplotlib
import mlflow
import pytest

# Set non-interactive backend for headless test runs
matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def isolate_test_mlflow(tmp_path, monkeypatch):
    """Ensure test runs log MLflow artifacts to a temporary directory."""
    test_mlruns = tmp_path / "test_mlruns"
    test_mlruns.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", str(test_mlruns))
    mlflow.set_tracking_uri(str(test_mlruns))
