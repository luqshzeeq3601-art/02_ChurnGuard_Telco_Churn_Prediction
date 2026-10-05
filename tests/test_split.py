"""Unit tests for data splitting module."""

import pytest

from churnguard.data.clean import clean_data
from churnguard.data.load import load_telco
from churnguard.data.split import save_splits, split_data


@pytest.fixture
def cleaned_telco_data():
    return clean_data(load_telco())


def test_split_shapes(cleaned_telco_data):
    """Verify split proportions 70 / 15 / 15 and exact total rows."""
    train_df, val_df, test_df = split_data(cleaned_telco_data)
    total_len = len(train_df) + len(val_df) + len(test_df)
    assert total_len == len(cleaned_telco_data) == 7043

    # Check approximate split sizes
    assert abs(len(train_df) / 7043 - 0.70) < 0.01
    assert abs(len(val_df) / 7043 - 0.15) < 0.01
    assert abs(len(test_df) / 7043 - 0.15) < 0.01


def test_split_stratification(cleaned_telco_data):
    """AC: Churn rate within plus or minus 1% across splits."""
    train_df, val_df, test_df = split_data(cleaned_telco_data)

    overall_churn_rate = cleaned_telco_data["Churn"].mean()
    train_churn_rate = train_df["Churn"].mean()
    val_churn_rate = val_df["Churn"].mean()
    test_churn_rate = test_df["Churn"].mean()

    assert abs(train_churn_rate - overall_churn_rate) < 0.01
    assert abs(val_churn_rate - overall_churn_rate) < 0.01
    assert abs(test_churn_rate - overall_churn_rate) < 0.01


def test_split_no_overlap(cleaned_telco_data):
    """Ensure zero customerID leakage across train, val, and test splits."""
    train_df, val_df, test_df = split_data(cleaned_telco_data)

    train_ids = set(train_df["customerID"])
    val_ids = set(val_df["customerID"])
    test_ids = set(test_df["customerID"])

    assert len(train_ids.intersection(val_ids)) == 0
    assert len(train_ids.intersection(test_ids)) == 0
    assert len(val_ids.intersection(test_ids)) == 0


def test_split_invalid_ratios(cleaned_telco_data):
    """Fails if ratios do not sum to 1.0."""
    with pytest.raises(ValueError, match="Split ratios must sum to 1.0"):
        split_data(cleaned_telco_data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.2)


def test_save_splits(tmp_path, cleaned_telco_data):
    """Test that save_splits writes valid CSV and Parquet files."""
    train_df, val_df, test_df = split_data(cleaned_telco_data)
    saved = save_splits(train_df, val_df, test_df, output_dir=tmp_path)

    for name in ["train", "val", "test"]:
        assert (tmp_path / f"{name}.csv").exists()
        assert (tmp_path / f"{name}.parquet").exists()
        assert name in saved
