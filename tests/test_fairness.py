"""Tests for fairness auditing and error diagnostics."""

import numpy as np
import pandas as pd
import pytest

from churnguard.explain.fairness import (
    check_nfr7_fairness,
    compute_subgroup_metrics,
    diagnose_errors,
)


@pytest.fixture
def mock_evaluation_df():
    """Create mock DataFrame with demographic and segment features."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame(
        {
            "customerID": [f"ID_{i}" for i in range(n)],
            "gender": np.random.choice(["Male", "Female"], n),
            "SeniorCitizen": np.random.choice([0, 1], n, p=[0.8, 0.2]),
            "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], n),
            "InternetService": np.random.choice(["DSL", "Fiber optic", "No"], n),
            "PaymentMethod": np.random.choice(
                ["Electronic check", "Mailed check", "Bank transfer (automatic)"], n
            ),
            "tenure": np.random.randint(1, 72, n),
            "MonthlyCharges": np.random.uniform(20.0, 100.0, n),
            "Churn": np.random.choice([0, 1], n, p=[0.73, 0.27]),
            "churn_prob": np.random.uniform(0.0, 1.0, n),
        }
    )
    return df


def test_compute_subgroup_metrics(mock_evaluation_df):
    """Test subgroup metrics calculation returns required columns."""
    res_df = compute_subgroup_metrics(mock_evaluation_df, threshold=0.2)
    assert "slice_type" in res_df.columns
    assert "subgroup" in res_df.columns
    assert "recall_tpr" in res_df.columns
    assert "selection_rate" in res_df.columns
    assert len(res_df) > 5


def test_check_nfr7_fairness(mock_evaluation_df):
    """Test NFR7 fairness audit function."""
    audit = check_nfr7_fairness(mock_evaluation_df, threshold=0.2, max_recall_gap=0.5)
    assert "gender_audit" in audit
    assert "senior_citizen_audit" in audit
    assert "overall_nfr7_passed" in audit
    assert isinstance(audit["overall_nfr7_passed"], bool)


def test_diagnose_errors(mock_evaluation_df):
    """Test error diagnostic separation into FP and FN."""
    diag = diagnose_errors(mock_evaluation_df, threshold=0.2, top_n=5)
    assert "top_false_positives" in diag
    assert "top_false_negatives" in diag
    assert len(diag["top_false_positives"]) <= 5
    assert len(diag["top_false_negatives"]) <= 5
