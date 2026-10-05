"""Integration tests for FastAPI endpoints using TestClient (T5.4)."""

import pytest
from api.main import app
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create TestClient instance."""
    return TestClient(app)


@pytest.fixture
def valid_customer_payload():
    """Valid payload matching PRD Section 6 contract."""
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


def test_health_endpoint(client):
    """Test GET /health returns 200 and correct status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "timestamp" in data
    assert "model_version" in data


def test_model_info_endpoint(client):
    """Test GET /model-info returns complete metadata."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "churnguard-champion"
    assert "optimal_threshold" in data
    assert "risk_tiers" in data
    assert "features" in data
    assert "cost_parameters" in data


def test_predict_single_valid(client, valid_customer_payload):
    """Test POST /predict with valid customer payload returns 200 and schema."""
    response = client.post("/predict", json=valid_customer_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["customerID"] == "7590-VHVEG"
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["risk_tier"] in ["High", "Medium", "Low"]
    assert len(data["top_reasons"]) == 3
    assert data["model_version"] == "1.0.0"


def test_predict_single_invalid_inputs(client, valid_customer_payload):
    """Test POST /predict with invalid data returns 422 Unprocessable Entity."""
    # 1. Invalid PaymentMethod
    bad_payload1 = valid_customer_payload.copy()
    bad_payload1["PaymentMethod"] = "Invalid Payment"
    res1 = client.post("/predict", json=bad_payload1)
    assert res1.status_code == 422

    # 2. Negative tenure
    bad_payload2 = valid_customer_payload.copy()
    bad_payload2["tenure"] = -5
    res2 = client.post("/predict", json=bad_payload2)
    assert res2.status_code == 422

    # 3. Missing required field
    bad_payload3 = valid_customer_payload.copy()
    del bad_payload3["MonthlyCharges"]
    res3 = client.post("/predict", json=bad_payload3)
    assert res3.status_code == 422


def test_predict_batch_endpoint(client, valid_customer_payload):
    """Test POST /predict/batch scores and ranks multiple customers."""
    c1 = valid_customer_payload.copy()
    c2 = valid_customer_payload.copy()
    c2["customerID"] = "LOYAL-002"
    c2["tenure"] = 65
    c2["Contract"] = "Two year"
    c2["PaymentMethod"] = "Bank transfer (automatic)"

    batch_payload = {"customers": [c1, c2]}
    response = client.post("/predict/batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()

    assert data["total_customers"] == 2
    assert data["high_risk_count"] + data["medium_risk_count"] + data["low_risk_count"] == 2
    assert len(data["predictions"]) == 2
    # Ensure ranking: highest churn probability first
    assert (
        data["predictions"][0]["churn_probability"] >= data["predictions"][1]["churn_probability"]
    )


def test_predict_batch_empty(client):
    """Test POST /predict/batch with empty list returns 200 with 0 count."""
    response = client.post("/predict/batch", json={"customers": []})
    assert response.status_code == 200
    data = response.json()
    assert data["total_customers"] == 0
    assert len(data["predictions"]) == 0
