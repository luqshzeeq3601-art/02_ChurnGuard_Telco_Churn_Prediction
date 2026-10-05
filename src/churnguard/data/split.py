"""Data splitting module for ChurnGuard.

Performs stratified train/val/test splitting (70% / 15% / 15%) on the Churn target
with fixed seed=42, and saves splits to data/processed/.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from churnguard.config import CFG, SEED
from churnguard.data.clean import clean_data
from churnguard.data.load import load_telco


def split_data(
    df: pd.DataFrame,
    train_ratio: float | None = None,
    val_ratio: float | None = None,
    test_ratio: float | None = None,
    stratify_col: str | None = None,
    seed: int | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset into stratified train, validation, and test subsets.

    Args:
        df: Cleaned Telco DataFrame.
        train_ratio: Proportion of data for training (default: 0.70 from config).
        val_ratio: Proportion of data for validation (default: 0.15 from config).
        test_ratio: Proportion of data for testing (default: 0.15 from config).
        stratify_col: Column name to stratify on (default: 'Churn').
        seed: Random seed for reproducibility (default: 42).

    Returns:
        Tuple of (train_df, val_df, test_df).

    Raises:
        ValueError: If split ratios do not sum to 1.0.
    """
    split_cfg = CFG["split"]
    train_r = train_ratio if train_ratio is not None else split_cfg["train_ratio"]
    val_r = val_ratio if val_ratio is not None else split_cfg["val_ratio"]
    test_r = test_ratio if test_ratio is not None else split_cfg["test_ratio"]
    strat_col = stratify_col if stratify_col is not None else split_cfg["stratify_col"]
    random_state = seed if seed is not None else SEED

    total_ratio = train_r + val_r + test_r
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(
            f"Split ratios must sum to 1.0, got {train_r} + {val_r} + {test_r} = {total_ratio}"
        )

    # Step 1: Split into train (70%) and temp (30% = val + test)
    temp_ratio = val_r + test_r
    stratify_labels = df[strat_col] if strat_col in df.columns else None

    train_df, temp_df = train_test_split(
        df,
        test_size=temp_ratio,
        random_state=random_state,
        stratify=stratify_labels,
    )

    # Step 2: Split temp into val (50% of 30% = 15%) and test (50% of 30% = 15%)
    val_share_of_temp = val_r / temp_ratio
    stratify_temp = temp_df[strat_col] if strat_col in temp_df.columns else None

    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_share_of_temp,
        random_state=random_state,
        stratify=stratify_temp,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path | str | None = None,
) -> dict[str, Path]:
    """Save split DataFrames to disk in both CSV and Parquet formats.

    Args:
        train_df: Training set DataFrame.
        val_df: Validation set DataFrame.
        test_df: Test set DataFrame.
        output_dir: Target directory path (default: data/processed/).

    Returns:
        Dictionary mapping split names to saved CSV file paths.
    """
    target_dir = Path(output_dir) if output_dir else CFG["paths"]["processed_dir"]
    target_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: dict[str, Path] = {}
    splits = {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }

    for name, df in splits.items():
        csv_path = target_dir / f"{name}.csv"
        parquet_path = target_dir / f"{name}.parquet"

        df.to_csv(csv_path, index=False)
        df.to_parquet(parquet_path, index=False)
        saved_paths[name] = csv_path

    return saved_paths


def split_and_save_data(output_dir: Path | str | None = None) -> dict[str, Path]:
    """Load raw data, clean it, split stratified, and save to processed directory.

    Returns:
        Dictionary mapping split names to file paths.
    """
    raw_df = load_telco()
    cleaned_df = clean_data(raw_df)
    train_df, val_df, test_df = split_data(cleaned_df)
    return save_splits(train_df, val_df, test_df, output_dir=output_dir)


if __name__ == "__main__":
    saved = split_and_save_data()
    print(f"Splits saved successfully: {saved}")
