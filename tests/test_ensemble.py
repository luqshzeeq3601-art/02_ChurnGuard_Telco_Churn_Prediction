"""Unit tests for CalibratedEnsembleClassifier."""

import numpy as np
import pandas as pd
import pytest

from churnguard.models.ensemble import CalibratedEnsembleClassifier


@pytest.fixture
def mini_df():
    np.random.seed(42)
    n = 120
    contracts = ["Month-to-month", "One year", "Two year"]
    payments = [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    internets = ["DSL", "Fiber optic", "No"]

    return pd.DataFrame(
        {
            "customerID": [f"ID-{i}" for i in range(n)],
            "gender": np.random.choice(["Male", "Female"], n),
            "SeniorCitizen": np.random.choice([0, 1], n),
            "Partner": np.random.choice(["Yes", "No"], n),
            "Dependents": np.random.choice(["Yes", "No"], n),
            "tenure": np.random.randint(1, 72, n),
            "PhoneService": np.random.choice(["Yes", "No"], n),
            "MultipleLines": np.random.choice(["Yes", "No", "No phone service"], n),
            "InternetService": np.random.choice(internets, n),
            "OnlineSecurity": np.random.choice(["Yes", "No", "No internet service"], n),
            "OnlineBackup": np.random.choice(["Yes", "No", "No internet service"], n),
            "DeviceProtection": np.random.choice(["Yes", "No", "No internet service"], n),
            "TechSupport": np.random.choice(["Yes", "No", "No internet service"], n),
            "StreamingTV": np.random.choice(["Yes", "No", "No internet service"], n),
            "StreamingMovies": np.random.choice(["Yes", "No", "No internet service"], n),
            "Contract": np.random.choice(contracts, n),
            "PaperlessBilling": np.random.choice(["Yes", "No"], n),
            "PaymentMethod": np.random.choice(payments, n),
            "MonthlyCharges": np.random.uniform(20.0, 110.0, n),
            "TotalCharges": np.random.uniform(50.0, 5000.0, n),
            "Churn": np.random.choice([0, 1], n, p=[0.73, 0.27]),
        }
    )


def test_ensemble_fit_and_predict(mini_df):
    X = mini_df.drop(columns=["Churn"])
    y = mini_df["Churn"].values

    ensemble = CalibratedEnsembleClassifier(
        weights=[0.34, 0.33, 0.33],
        drop_cols=["gender", "SeniorCitizen"],
        include_interactions=True,
        cv_splits=3,
        random_state=42,
    )
    ensemble.fit(X, y)

    probs = ensemble.predict_proba(X)
    preds = ensemble.predict(X, threshold=0.3)

    assert probs.shape == (len(X), 2)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(len(X)), atol=1e-5)

    assert len(preds) == len(X)
    assert set(np.unique(preds)).issubset({0, 1})
    assert len(ensemble.weights_) == 3
