"""Tests for SHAP explainability engine and reason code generator."""

import pandas as pd
import pytest

from churnguard.explain.shap_explain import (
    ChurnExplainer,
    run_explainability_pipeline,
)
from churnguard.models.calibrate import fit_champion_pipeline


@pytest.fixture
def sample_customer_record():
    """Create a realistic single customer record."""
    return {
        "customerID": "7590-VHVEG",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85,
    }


def test_explainer_reason_generation(sample_customer_record):
    """Test generating plain-language top-3 reasons for a single customer."""
    # Build minimal training dataset
    train_df = pd.DataFrame([sample_customer_record] * 20)
    train_df["Churn"] = [1, 0] * 10
    # Add minor variations
    train_df.loc[0:5, "Contract"] = "Two year"
    train_df.loc[0:5, "tenure"] = 50

    small_params = {
        "n_estimators": 5,
        "max_depth": 3,
        "num_leaves": 8,
        "random_state": 42,
        "verbose": -1,
    }
    pipeline = fit_champion_pipeline(train_df, params=small_params)
    explainer = ChurnExplainer(fitted_pipeline=pipeline)

    reasons = explainer.explain_customer(sample_customer_record, top_k=3)
    assert len(reasons) == 3
    for r in reasons:
        assert isinstance(r, str)
        assert len(r) > 5


def test_run_explainability_pipeline_integration():
    """Test full integration run on processed validation dataset."""
    explainer, sample_reasons = run_explainability_pipeline(save_artifacts=True)
    assert explainer is not None
    assert len(sample_reasons) == 5
    for item in sample_reasons:
        assert "customerID" in item
        assert "top_reasons" in item
        assert len(item["top_reasons"]) == 3
