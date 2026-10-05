"""Unit tests for data cleaning module."""

import numpy as np
import pandas as pd
import pytest

from churnguard.data.clean import clean_data
from churnguard.data.load import load_telco


def test_clean_data_on_raw():
    """Full raw dataset cleans without nulls and with 0/1 target."""
    raw = load_telco()
    cleaned = clean_data(raw)

    assert cleaned.isnull().sum().sum() == 0
    assert cleaned["Churn"].isin([0, 1]).all()
    assert cleaned["TotalCharges"].dtype == np.float64
    assert cleaned.shape == raw.shape


def test_clean_blank_total_charges():
    """Verify blank / whitespace TotalCharges are converted to 0.0."""
    sample = pd.DataFrame(
        {
            "customerID": ["1111-TEST", "2222-TEST", "3333-TEST"],
            "gender": ["Female", "Male", "Female"],
            "SeniorCitizen": [0, 0, 1],
            "Partner": ["Yes", "No", "No"],
            "Dependents": ["No", "No", "No"],
            "tenure": [0, 12, 0],
            "PhoneService": ["Yes", "Yes", "No"],
            "MultipleLines": ["No", "Yes", "No phone service"],
            "InternetService": ["DSL", "Fiber optic", "No"],
            "OnlineSecurity": ["Yes", "No", "No internet service"],
            "OnlineBackup": ["No", "Yes", "No internet service"],
            "DeviceProtection": ["No", "No", "No internet service"],
            "TechSupport": ["Yes", "No", "No internet service"],
            "StreamingTV": ["No", "Yes", "No internet service"],
            "StreamingMovies": ["No", "No", "No internet service"],
            "Contract": ["Month-to-month", "One year", "Two year"],
            "PaperlessBilling": ["Yes", "No", "No"],
            "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)"],
            "MonthlyCharges": [29.85, 70.35, 20.00],
            "TotalCharges": [" ", "840.20", ""],
            "Churn": ["No", "Yes", "No"],
        }
    )

    cleaned = clean_data(sample)
    assert cleaned.loc[0, "TotalCharges"] == 0.0
    assert cleaned.loc[1, "TotalCharges"] == 840.20
    assert cleaned.loc[2, "TotalCharges"] == 0.0
    assert cleaned["TotalCharges"].dtype == np.float64
    assert list(cleaned["Churn"]) == [0, 1, 0]
    assert cleaned.isnull().sum().sum() == 0


def test_clean_data_raises_on_unhandled_null():
    """Verify error raised if a column contains unhandled nulls (e.g. invalid target)."""
    sample = pd.DataFrame(
        {
            "customerID": ["1111-TEST"],
            "Churn": ["UnknownValue"],  # mapping will produce NaN
            "TotalCharges": ["100.0"],
        }
    )
    with pytest.raises(ValueError, match="Unexpected null values found"):
        clean_data(sample)
