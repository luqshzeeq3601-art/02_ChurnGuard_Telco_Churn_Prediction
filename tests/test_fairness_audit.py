"""Tests for E12 fairness audit module."""

import numpy as np
import pandas as pd
import pytest

from churnguard.explain.fairness_audit import run_e12_fairness_audit


@pytest.fixture
def synthetic_data():
    """Create minimal synthetic dataset for testing."""
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


def test_fairness_audit_synthetic(synthetic_data, tmp_path):
    """Test fairness audit execution on synthetic data."""
    out_file = tmp_path / "fairness_test.json"
    res = run_e12_fairness_audit(
        data_df=synthetic_data, tau=0.2, log_to_mlflow=False, save_path=out_file
    )
    assert "selected_option" in res
    assert res["selected_option"] in ["M0", "M1", "M2"]
    assert "trade_off_table" in res
    assert len(res["trade_off_table"]) == 3
    assert out_file.exists()


def test_fairness_audit_report_exists():
    """Verify generated report exists in reports/."""
    from pathlib import Path

    p = Path("reports/e12_fairness_audit.json")
    assert p.exists()
