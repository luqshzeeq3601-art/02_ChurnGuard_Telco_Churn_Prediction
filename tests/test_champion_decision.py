"""Tests for E11 champion re-decision module."""

import numpy as np
import pandas as pd
import pytest

from churnguard.models.champion_decision import run_e11_experiment


@pytest.fixture
def synthetic_data():
    """Create a minimal synthetic dataframe for fast testing."""
    np.random.seed(42)
    n = 60
    return pd.DataFrame(
        {
            "customerID": [f"ID_{i}" for i in range(n)],
            "gender": np.random.choice(["Male", "Female"], n),
            "SeniorCitizen": np.random.choice([0, 1], n),
            "Partner": np.random.choice(["Yes", "No"], n),
            "Dependents": np.random.choice(["Yes", "No"], n),
            "tenure": np.random.randint(1, 72, n),
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
            "Churn": np.random.choice([0, 1], n, p=[0.7, 0.3]),
        }
    )


def test_e11_experiment_synthetic(synthetic_data, tmp_path):
    """Test E11 experiment execution on synthetic data."""
    out_file = tmp_path / "e11_test.json"
    res = run_e11_experiment(data_df=synthetic_data, log_to_mlflow=False, save_path=out_file)

    assert "selected_champion" in res
    assert res["selected_champion"] in ["LogisticRegression", "LightGBM"]
    assert "paired_diff_mean" in res
    assert "paired_diff_std" in res
    assert out_file.exists()


def test_e11_report_exists():
    """Verify generated E11 report exists in reports/."""
    from pathlib import Path

    p = Path("reports/e11_champion_decision.json")
    assert p.exists()
