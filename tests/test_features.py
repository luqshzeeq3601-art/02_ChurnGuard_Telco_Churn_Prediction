"""Unit tests for feature engineering and preprocessing pipelines."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from churnguard.data.clean import clean_data
from churnguard.data.load import load_telco
from churnguard.features.build import (
    FeatureEngineer,
    build_full_pipeline,
    build_preprocessor,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "customerID": ["001-A", "002-B", "003-C", "004-D"],
            "gender": ["Female", "Male", "Female", "Male"],
            "SeniorCitizen": [0, 1, 0, 0],
            "Partner": ["Yes", "No", "Yes", "No"],
            "Dependents": ["No", "No", "Yes", "No"],
            "tenure": [1, 10, 24, 60],
            "PhoneService": ["Yes", "Yes", "No", "Yes"],
            "MultipleLines": ["No", "Yes", "No phone service", "Yes"],
            "InternetService": ["DSL", "Fiber optic", "DSL", "No"],
            "OnlineSecurity": ["Yes", "No", "No", "No internet service"],
            "OnlineBackup": ["Yes", "No", "Yes", "No internet service"],
            "DeviceProtection": ["No", "No", "Yes", "No internet service"],
            "TechSupport": ["No", "No", "Yes", "No internet service"],
            "StreamingTV": ["No", "Yes", "No", "No internet service"],
            "StreamingMovies": ["No", "Yes", "Yes", "No internet service"],
            "Contract": ["Month-to-month", "Month-to-month", "One year", "Two year"],
            "PaperlessBilling": ["Yes", "Yes", "No", "No"],
            "PaymentMethod": [
                "Electronic check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
                "Mailed check",
            ],
            "MonthlyCharges": [30.0, 95.0, 60.0, 20.0],
            "TotalCharges": [30.0, 950.0, 1440.0, 1200.0],
            "Churn": [1, 1, 0, 0],
        }
    )


def test_feature_engineer_tenure_bucket(sample_df):
    fe = FeatureEngineer()
    df_out = fe.fit_transform(sample_df)
    assert "tenure_bucket" in df_out.columns
    assert list(df_out["tenure_bucket"]) == ["0-6m", "7-12m", "13-24m", "49-72m"]


def test_feature_engineer_avg_monthly_spend(sample_df):
    fe = FeatureEngineer()
    df_out = fe.fit_transform(sample_df)
    assert "avg_monthly_spend" in df_out.columns
    expected = [30.0 / 1.0, 950.0 / 10.0, 1440.0 / 24.0, 1200.0 / 60.0]
    np.testing.assert_allclose(df_out["avg_monthly_spend"], expected)


def test_feature_engineer_charge_increase_ratio(sample_df):
    fe = FeatureEngineer()
    df_out = fe.fit_transform(sample_df)
    assert "charge_increase_ratio" in df_out.columns
    expected = [30.0 / 30.0, 95.0 / 95.0, 60.0 / 60.0, 20.0 / 20.0]
    np.testing.assert_allclose(df_out["charge_increase_ratio"], expected)


def test_feature_engineer_num_services(sample_df):
    fe = FeatureEngineer()
    df_out = fe.fit_transform(sample_df)
    assert "num_services" in df_out.columns
    # Row 0: PhoneService(Yes), OnlineSecurity(Yes), OnlineBackup(Yes) -> 3
    # Row 1: PhoneService(Yes), MultipleLines(Yes), StreamingTV(Yes), StreamingMovies(Yes) -> 4
    # Row 2: OnlineBackup(Yes), DeviceProtection(Yes), TechSupport(Yes), StreamingMovies(Yes) -> 4
    # Row 3: PhoneService(Yes), MultipleLines(Yes) -> 2
    assert list(df_out["num_services"]) == [3, 4, 4, 2]


def test_feature_engineer_has_protection_bundle(sample_df):
    fe = FeatureEngineer()
    df_out = fe.fit_transform(sample_df)
    assert "has_protection_bundle" in df_out.columns
    assert list(df_out["has_protection_bundle"]) == [1, 0, 1, 0]


def test_feature_engineer_is_auto_pay(sample_df):
    fe = FeatureEngineer()
    df_out = fe.fit_transform(sample_df)
    assert "is_auto_pay" in df_out.columns
    assert list(df_out["is_auto_pay"]) == [0, 1, 1, 0]


def test_feature_engineer_is_month_to_month(sample_df):
    fe = FeatureEngineer()
    df_out = fe.fit_transform(sample_df)
    assert "is_month_to_month" in df_out.columns
    assert list(df_out["is_month_to_month"]) == [1, 1, 0, 0]


def test_feature_engineer_fiber_no_support(sample_df):
    fe = FeatureEngineer()
    df_out = fe.fit_transform(sample_df)
    assert "fiber_no_support" in df_out.columns
    # Row 1 is Fiber optic and TechSupport == No -> 1
    assert list(df_out["fiber_no_support"]) == [0, 1, 0, 0]


def test_feature_engineer_disable_flag(sample_df):
    fe = FeatureEngineer(include_engineered=False)
    df_out = fe.fit_transform(sample_df)
    assert "tenure_bucket" not in df_out.columns
    assert "fiber_no_support" not in df_out.columns
    assert df_out.shape == sample_df.shape


def test_preprocessor_output_shape():
    """AC: Fit on train only; output shape test."""
    raw = load_telco()
    cleaned = clean_data(raw)
    train_df = cleaned.iloc[:1000].copy()
    test_df = cleaned.iloc[1000:1200].copy()

    # Preprocessor with engineered features
    fe = FeatureEngineer(include_engineered=True)
    train_fe = fe.fit_transform(train_df)
    test_fe = fe.transform(test_df)

    preprocessor = build_preprocessor(include_engineered=True, scale_numeric=True)
    X_train_trans = preprocessor.fit_transform(train_fe)
    X_test_trans = preprocessor.transform(test_fe)

    assert X_train_trans.shape[0] == 1000
    assert X_test_trans.shape[0] == 200
    assert X_train_trans.shape[1] == X_test_trans.shape[1]
    assert not np.isnan(X_train_trans).any()
    assert not np.isnan(X_test_trans).any()


def test_full_pipeline_end_to_end(sample_df):
    """Verify full Pipeline fits and predicts cleanly."""
    X = sample_df.drop(columns=["Churn"])
    y = sample_df["Churn"]

    pipe = build_full_pipeline(
        model=LogisticRegression(solver="liblinear"),
        include_engineered=True,
        scale_numeric=True,
    )
    pipe.fit(X, y)

    preds = pipe.predict(X)
    probs = pipe.predict_proba(X)

    assert len(preds) == len(X)
    assert probs.shape == (len(X), 2)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
