"""Data cleaning module for Telco Churn dataset.

Implements decision D-003:
- Converts TotalCharges to numeric, replacing blanks (tenure == 0) with 0.0.
- Encodes Churn target ('Yes' -> 1, 'No' -> 0).
- Ensures no null values remain in the cleaned dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw Telco dataset.

    Performs the following transformations:
    1. Creates a copy of the DataFrame.
    2. Converts TotalCharges to float; blank or whitespace strings become 0.0 (D-003).
    3. Maps the binary target 'Churn' from {'Yes', 'No'} to {1, 0} (if present).
    4. Casts numeric columns to standard types.
    5. Validates that no missing values remain.

    Args:
        df: Raw or partially processed Telco DataFrame.

    Returns:
        Cleaned pandas DataFrame.

    Raises:
        ValueError: If unexpected nulls exist after cleaning.
    """
    df_clean = df.copy()

    # 1. TotalCharges cleaning (11 blanks where tenure = 0 -> 0.0)
    if "TotalCharges" in df_clean.columns:
        if df_clean["TotalCharges"].dtype == object:
            # Replace whitespace-only strings with NaN, then fillna with 0.0
            total_charges_num = pd.to_numeric(
                df_clean["TotalCharges"].astype(str).str.strip(),
                errors="coerce",
            )
            df_clean["TotalCharges"] = total_charges_num.fillna(0.0)
        else:
            df_clean["TotalCharges"] = df_clean["TotalCharges"].fillna(0.0).astype(float)

    # 2. Target encoding (Yes -> 1, No -> 0)
    if "Churn" in df_clean.columns:
        if df_clean["Churn"].dtype == object:
            churn_mapping = {"Yes": 1, "No": 0}
            df_clean["Churn"] = df_clean["Churn"].map(churn_mapping)

    # 3. Numeric column casting
    if "tenure" in df_clean.columns:
        df_clean["tenure"] = df_clean["tenure"].astype(int)
    if "SeniorCitizen" in df_clean.columns:
        df_clean["SeniorCitizen"] = df_clean["SeniorCitizen"].astype(int)
    if "MonthlyCharges" in df_clean.columns:
        df_clean["MonthlyCharges"] = df_clean["MonthlyCharges"].astype(float)

    # 4. Check for unexpected nulls
    null_counts = df_clean.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if not cols_with_nulls.empty:
        raise ValueError(
            f"Unexpected null values found after cleaning in columns: {cols_with_nulls.to_dict()}"
        )

    return df_clean
