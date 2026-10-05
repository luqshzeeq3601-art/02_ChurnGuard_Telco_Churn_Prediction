"""Pandera validation schemas for raw and cleaned Telco Churn data.

Validates data types, ranges, allowed categorical values, and row counts
as specified in docs/05_DATA_SPEC.md.
"""

from __future__ import annotations

import pandas as pd
import pandera as pa
from pandera import Check, Column, DataFrameSchema

# Allowed categorical values per 05_DATA_SPEC.md section 2
ALLOWED_GENDER = ["Male", "Female"]
ALLOWED_SENIOR_CITIZEN = [0, 1]
ALLOWED_BINARY_YES_NO = ["Yes", "No"]
ALLOWED_MULTIPLE_LINES = ["Yes", "No", "No phone service"]
ALLOWED_INTERNET_SERVICE = ["DSL", "Fiber optic", "No"]
ALLOWED_INTERNET_ADDON = ["Yes", "No", "No internet service"]
ALLOWED_CONTRACT = ["Month-to-month", "One year", "Two year"]
ALLOWED_PAYMENT_METHOD = [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]
ALLOWED_CHURN = ["Yes", "No"]


def get_raw_schema() -> DataFrameSchema:
    """Return pandera schema for the raw Telco dataset."""
    return DataFrameSchema(
        columns={
            "customerID": Column(
                pa.String,
                nullable=False,
                unique=True,
                description="Unique customer identifier",
            ),
            "gender": Column(
                pa.String,
                checks=Check.isin(ALLOWED_GENDER),
                nullable=False,
            ),
            "SeniorCitizen": Column(
                pa.Int,
                checks=Check.isin(ALLOWED_SENIOR_CITIZEN),
                nullable=False,
            ),
            "Partner": Column(
                pa.String,
                checks=Check.isin(ALLOWED_BINARY_YES_NO),
                nullable=False,
            ),
            "Dependents": Column(
                pa.String,
                checks=Check.isin(ALLOWED_BINARY_YES_NO),
                nullable=False,
            ),
            "tenure": Column(
                pa.Int,
                checks=[Check.ge(0), Check.le(72)],
                nullable=False,
            ),
            "PhoneService": Column(
                pa.String,
                checks=Check.isin(ALLOWED_BINARY_YES_NO),
                nullable=False,
            ),
            "MultipleLines": Column(
                pa.String,
                checks=Check.isin(ALLOWED_MULTIPLE_LINES),
                nullable=False,
            ),
            "InternetService": Column(
                pa.String,
                checks=Check.isin(ALLOWED_INTERNET_SERVICE),
                nullable=False,
            ),
            "OnlineSecurity": Column(
                pa.String,
                checks=Check.isin(ALLOWED_INTERNET_ADDON),
                nullable=False,
            ),
            "OnlineBackup": Column(
                pa.String,
                checks=Check.isin(ALLOWED_INTERNET_ADDON),
                nullable=False,
            ),
            "DeviceProtection": Column(
                pa.String,
                checks=Check.isin(ALLOWED_INTERNET_ADDON),
                nullable=False,
            ),
            "TechSupport": Column(
                pa.String,
                checks=Check.isin(ALLOWED_INTERNET_ADDON),
                nullable=False,
            ),
            "StreamingTV": Column(
                pa.String,
                checks=Check.isin(ALLOWED_INTERNET_ADDON),
                nullable=False,
            ),
            "StreamingMovies": Column(
                pa.String,
                checks=Check.isin(ALLOWED_INTERNET_ADDON),
                nullable=False,
            ),
            "Contract": Column(
                pa.String,
                checks=Check.isin(ALLOWED_CONTRACT),
                nullable=False,
            ),
            "PaperlessBilling": Column(
                pa.String,
                checks=Check.isin(ALLOWED_BINARY_YES_NO),
                nullable=False,
            ),
            "PaymentMethod": Column(
                pa.String,
                checks=Check.isin(ALLOWED_PAYMENT_METHOD),
                nullable=False,
            ),
            "MonthlyCharges": Column(
                pa.Float,
                checks=Check.gt(0),
                nullable=False,
            ),
            "TotalCharges": Column(
                pa.Object,
                nullable=False,
            ),
            "Churn": Column(
                pa.String,
                checks=Check.isin(ALLOWED_CHURN),
                nullable=False,
            ),
        },
        checks=[
            Check(
                lambda df: len(df) >= 5000,
                error="Dataset must contain at least 5000 rows (guard against truncated files).",
            )
        ],
        strict=False,
        coerce=False,
    )


def validate_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a raw Telco DataFrame against the schema.

    Args:
        df: Raw Telco DataFrame.

    Returns:
        The validated DataFrame if valid.

    Raises:
        pandera.errors.SchemaError: If validation fails.
    """
    schema = get_raw_schema()
    return schema.validate(df)
