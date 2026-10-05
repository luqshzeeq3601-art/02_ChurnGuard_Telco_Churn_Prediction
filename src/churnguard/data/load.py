"""Data loading utilities for ChurnGuard.

Downloads and loads the IBM Telco Customer Churn dataset and the
Malaysian cellular subscribers dataset.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from churnguard.config import CFG


def load_telco(path: Path | str | None = None) -> pd.DataFrame:
    """Load the raw Telco Customer Churn CSV.

    Args:
        path: Override path. Defaults to ``config.yaml`` → ``paths.raw_data``.

    Returns:
        Raw DataFrame (7,043 rows × 21 columns).

    Raises:
        FileNotFoundError: If the CSV is missing. Download it from Kaggle:
            https://www.kaggle.com/datasets/blastchar/telco-customer-churn
    """
    fpath = Path(path) if path else CFG["paths"]["raw_data"]

    if not fpath.exists():
        raise FileNotFoundError(
            f"Raw data not found at {fpath}. "
            "Download from https://www.kaggle.com/datasets/blastchar/telco-customer-churn "
            "and save as data/raw/telco_churn.csv"
        )

    df = pd.read_csv(fpath)
    return df


def load_malaysia_subscribers(path: Path | str | None = None) -> pd.DataFrame:
    """Load the Malaysian cellular subscribers dataset (data.gov.my).

    Args:
        path: Override path. Defaults to ``config.yaml`` → ``paths.raw_my_data``.

    Returns:
        DataFrame with columns: date, plan_type, subscriptions.
    """
    fpath = Path(path) if path else CFG["paths"]["raw_my_data"]

    if not fpath.exists():
        raise FileNotFoundError(
            f"Malaysia data not found at {fpath}. "
            "Download from https://storage.data.gov.my/communications/cellular_subscribers.csv "
            "and save as data/raw/my_cellular_subscribers.csv"
        )

    df = pd.read_csv(fpath)
    return df


if __name__ == "__main__":
    # Quick check: load and print shape
    telco = load_telco()
    print(f"Telco dataset: {telco.shape[0]} rows × {telco.shape[1]} columns")

    try:
        my_subs = load_malaysia_subscribers()
        print(f"Malaysia subscribers: {my_subs.shape[0]} rows × {my_subs.shape[1]} columns")
    except FileNotFoundError as e:
        print(f"[SKIP] {e}")
