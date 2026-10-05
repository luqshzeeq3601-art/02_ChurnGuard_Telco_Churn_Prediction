"""Tests for model serialization and artifact reloading."""

import pandas as pd

from churnguard.models.serialize import load_model_and_predict, save_final_model_artifacts


def test_save_and_reload_final_model(tmp_path):
    """Test saving model artifacts and reloading for standalone prediction."""
    model_path, meta_path = save_final_model_artifacts(models_dir=tmp_path, log_to_mlflow=False)
    assert model_path.exists()
    assert meta_path.exists()

    # Create dummy sample
    sample_df = pd.DataFrame(
        [
            {
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
        ]
    )

    scored = load_model_and_predict(sample_df, models_dir=tmp_path)
    assert "churn_probability" in scored.columns
    assert "risk_tier" in scored.columns
    assert scored["risk_tier"].iloc[0] in ["High", "Medium", "Low"]
    assert 0.0 <= scored["churn_probability"].iloc[0] <= 1.0
