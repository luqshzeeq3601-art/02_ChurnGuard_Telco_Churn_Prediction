"""Tests for probability calibration module."""

import numpy as np
import pandas as pd
import pytest

from churnguard.models.calibrate import (
    calibrate_pipeline,
    compute_ece,
    fit_champion_pipeline,
    plot_calibration_curves,
    run_calibration_experiment,
)


@pytest.fixture
def dummy_train_val():
    """Create dummy synthetic dataset with telco columns."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame(
        {
            "customerID": [f"ID_{i}" for i in range(n)],
            "gender": np.random.choice(["Male", "Female"], n),
            "SeniorCitizen": np.random.choice([0, 1], n),
            "Partner": np.random.choice(["Yes", "No"], n),
            "Dependents": np.random.choice(["Yes", "No"], n),
            "tenure": np.random.randint(0, 72, n),
            "PhoneService": np.random.choice(["Yes", "No"], n),
            "MultipleLines": np.random.choice(["Yes", "No", "No phone service"], n),
            "InternetService": np.random.choice(["DSL", "Fiber optic", "No"], n),
            "OnlineSecurity": np.random.choice(["Yes", "No", "No internet service"], n),
            "OnlineBackup": np.random.choice(["Yes", "No", "No internet service"], n),
            "DeviceProtection": np.random.choice(["Yes", "No", "No internet service"], n),
            "TechSupport": np.random.choice(["Yes", "No", "No internet service"], n),
            "StreamingTV": np.random.choice(["Yes", "No", "No internet service"], n),
            "StreamingMovies": np.random.choice(["Yes", "No", "No internet service"], n),
            "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], n),
            "PaperlessBilling": np.random.choice(["Yes", "No"], n),
            "PaymentMethod": np.random.choice(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                n,
            ),
            "MonthlyCharges": np.random.uniform(20.0, 100.0, n),
            "TotalCharges": np.random.uniform(20.0, 5000.0, n),
            "Churn": np.random.choice([0, 1], n, p=[0.73, 0.27]),
        }
    )
    train_df = df.iloc[:70].copy()
    val_df = df.iloc[70:].copy()
    return train_df, val_df


def test_compute_ece():
    """Test Expected Calibration Error calculation."""
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    ece = compute_ece(y_true, y_prob, n_bins=5)
    assert 0.0 <= ece <= 1.0


def test_calibrate_pipeline_and_plot(dummy_train_val, tmp_path):
    """Test calibration fitting and plotting on dummy data."""
    train_df, val_df = dummy_train_val
    small_params = {
        "n_estimators": 5,
        "max_depth": 3,
        "num_leaves": 8,
        "random_state": 42,
        "verbose": -1,
    }

    pipeline = fit_champion_pipeline(train_df, params=small_params)
    calibrator = calibrate_pipeline(pipeline, val_df, method="sigmoid")

    X_val = val_df.drop(columns=["Churn"])
    probs = calibrator.predict_proba(X_val)[:, 1]

    assert len(probs) == len(val_df)
    assert np.all((probs >= 0.0) & (probs <= 1.0))

    # Test plot
    fig_path = tmp_path / "test_cal.png"
    fig = plot_calibration_curves(
        val_df["Churn"].values,
        {"Sigmoid": probs},
        save_path=fig_path,
    )
    assert fig_path.exists()
    assert fig is not None


def test_run_calibration_experiment_integration():
    """Test running E09 experiment with real processed files."""
    res = run_calibration_experiment(log_to_mlflow=False)
    assert "uncalibrated" in res
    assert "sigmoid" in res
    assert "isotonic" in res
    assert "best_method" in res
    assert res["best_method"] in ["sigmoid", "isotonic"]
    assert res["sigmoid"]["brier_score"] >= 0.0
