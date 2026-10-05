"""Tests for standalone ChurnPredictor and CLI batch scoring."""

import pandas as pd
import pytest

from churnguard.models.predict import ChurnPredictor, score_file_cli


@pytest.fixture
def valid_single_customer():
    """Valid customer dictionary matching PRD Section 6 contract."""
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


def test_predict_single_contract(valid_single_customer):
    """Test predict_single returns all fields required by PRD Section 6."""
    predictor = ChurnPredictor()
    pred = predictor.predict_single(valid_single_customer)

    assert "customerID" in pred
    assert pred["customerID"] == "7590-VHVEG"
    assert "churn_probability" in pred
    assert 0.0 <= pred["churn_probability"] <= 1.0
    assert "risk_tier" in pred
    assert pred["risk_tier"] in ["High", "Medium", "Low"]
    assert "top_reasons" in pred
    assert len(pred["top_reasons"]) == 3
    assert "model_version" in pred


def test_predict_batch_ranked(valid_single_customer):
    """Test predict_batch returns ranked predictions descending by risk."""
    predictor = ChurnPredictor()
    c1 = valid_single_customer.copy()
    c2 = valid_single_customer.copy()
    c2["customerID"] = "LOYAL-CUST"
    c2["tenure"] = 60
    c2["Contract"] = "Two year"
    c2["PaymentMethod"] = "Bank transfer (automatic)"

    df = pd.DataFrame([c1, c2])
    scored = predictor.predict_batch(df, include_reasons=True)

    assert len(scored) == 2
    assert "rank" in scored.columns
    assert scored["churn_probability"].iloc[0] >= scored["churn_probability"].iloc[1]
    assert scored["rank"].iloc[0] == 1
    assert scored["rank"].iloc[1] == 2


def test_score_file_cli(tmp_path, valid_single_customer):
    """Test score_file_cli writes expected ranked CSV output."""
    in_csv = tmp_path / "test_input.csv"
    out_csv = tmp_path / "test_output.csv"

    pd.DataFrame([valid_single_customer] * 3).to_csv(in_csv, index=False)
    scored = score_file_cli(input_file=in_csv, output_file=out_csv, top_n=2)

    assert out_csv.exists()
    assert len(scored) == 3
    loaded = pd.read_csv(out_csv)
    assert "churn_probability" in loaded.columns
    assert "risk_tier" in loaded.columns
    assert "reason_1" in loaded.columns
