"""Feature engineering and preprocessing pipelines for ChurnGuard.

Implements:
- FeatureEngineer transformer per docs/05_DATA_SPEC.md section 5.
- Scikit-learn ColumnTransformer preprocessing pipelines.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SERVICE_COLUMNS = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom Scikit-Learn transformer to engineer telco churn domain features.

    Engineered features (per 05_DATA_SPEC.md section 5):
    - tenure_bucket: (0-6m, 7-12m, 13-24m, 25-48m, 49-72m)
    - avg_monthly_spend: TotalCharges / max(tenure, 1)
    - charge_increase_ratio: MonthlyCharges / avg_monthly_spend
    - num_services: count of 'Yes' across 8 service columns
    - has_protection_bundle: 1 if OnlineSecurity or TechSupport is 'Yes' else 0
    - is_auto_pay: 1 if PaymentMethod contains 'automatic' else 0
    - is_month_to_month: 1 if Contract == 'Month-to-month' else 0
    - fiber_no_support: 1 if InternetService == 'Fiber optic' and TechSupport == 'No' else 0
    """

    def __init__(self, include_engineered: bool = True) -> None:
        self.include_engineered = include_engineered

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> FeatureEngineer:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.include_engineered:
            return X.copy()

        df = X.copy()

        # 1. tenure_bucket
        if "tenure" in df.columns:
            bins = [-1, 6, 12, 24, 48, 72]
            labels = ["0-6m", "7-12m", "13-24m", "25-48m", "49-72m"]
            df["tenure_bucket"] = pd.cut(df["tenure"], bins=bins, labels=labels).astype(str)

        # 2. avg_monthly_spend
        if "TotalCharges" in df.columns and "tenure" in df.columns:
            tenure_safe = np.maximum(df["tenure"].values, 1.0)
            df["avg_monthly_spend"] = df["TotalCharges"].values / tenure_safe

        # 3. charge_increase_ratio
        if "MonthlyCharges" in df.columns and "avg_monthly_spend" in df.columns:
            spend_safe = np.maximum(df["avg_monthly_spend"].values, 1.0)
            df["charge_increase_ratio"] = df["MonthlyCharges"].values / spend_safe

        # 4. num_services
        present_service_cols = [c for c in SERVICE_COLUMNS if c in df.columns]
        if present_service_cols:
            df["num_services"] = (df[present_service_cols] == "Yes").sum(axis=1)

        # 5. has_protection_bundle
        sec = (df["OnlineSecurity"] == "Yes") if "OnlineSecurity" in df.columns else False
        tech = (df["TechSupport"] == "Yes") if "TechSupport" in df.columns else False
        df["has_protection_bundle"] = (sec | tech).astype(int)

        # 6. is_auto_pay
        if "PaymentMethod" in df.columns:
            df["is_auto_pay"] = (
                df["PaymentMethod"].astype(str).str.contains("automatic", case=False).astype(int)
            )

        # 7. is_month_to_month
        if "Contract" in df.columns:
            df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)

        # 8. fiber_no_support
        if "InternetService" in df.columns and "TechSupport" in df.columns:
            df["fiber_no_support"] = (
                (df["InternetService"] == "Fiber optic") & (df["TechSupport"] == "No")
            ).astype(int)

        return df


def get_feature_lists(
    include_engineered: bool = True,
    drop_cols: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Return numeric and categorical feature column names."""
    base_numeric = ["tenure", "MonthlyCharges", "TotalCharges"]
    base_categorical = [
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
    ]

    if not include_engineered:
        num = base_numeric
        cat = base_categorical
    else:
        engineered_numeric = [
            "avg_monthly_spend",
            "charge_increase_ratio",
            "num_services",
        ]
        engineered_categorical = [
            "tenure_bucket",
            "has_protection_bundle",
            "is_auto_pay",
            "is_month_to_month",
            "fiber_no_support",
        ]
        num = base_numeric + engineered_numeric
        cat = base_categorical + engineered_categorical

    if drop_cols:
        num = [c for c in num if c not in drop_cols]
        cat = [c for c in cat if c not in drop_cols]

    return num, cat


def build_preprocessor(
    include_engineered: bool = True,
    scale_numeric: bool = False,
    drop_cols: list[str] | None = None,
) -> ColumnTransformer:
    """Build scikit-learn ColumnTransformer for preprocessing.

    Args:
        include_engineered: Whether to include engineered feature columns.
        scale_numeric: If True, applies StandardScaler to numeric features (for Logistic Regression).
        drop_cols: Optional list of columns to exclude (e.g. for fairness ablation).

    Returns:
        ColumnTransformer instance.
    """
    numeric_cols, categorical_cols = get_feature_lists(
        include_engineered=include_engineered,
        drop_cols=drop_cols,
    )

    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(steps=num_steps)

    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", cat_pipeline, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def build_full_pipeline(
    model: BaseEstimator,
    include_engineered: bool = True,
    scale_numeric: bool = False,
    drop_cols: list[str] | None = None,
) -> Pipeline:
    """Combine FeatureEngineer, ColumnTransformer, and estimator into one end-to-end Pipeline.

    Args:
        model: Scikit-learn compatible classifier.
        include_engineered: Whether to use engineered features.
        scale_numeric: Whether to scale numeric features.
        drop_cols: Optional columns to exclude.

    Returns:
        End-to-end Pipeline.
    """
    fe = FeatureEngineer(include_engineered=include_engineered)
    preprocessor = build_preprocessor(
        include_engineered=include_engineered,
        scale_numeric=scale_numeric,
        drop_cols=drop_cols,
    )

    return Pipeline(
        steps=[
            ("feature_engineer", fe),
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
