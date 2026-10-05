"""Tests for data loading."""

import pytest

from churnguard.data.load import load_malaysia_subscribers, load_telco


def test_load_telco_shape():
    """Raw dataset should have 7,043 rows and 21 columns."""
    df = load_telco()
    assert df.shape == (7043, 21), f"Expected (7043, 21), got {df.shape}"


def test_load_telco_columns():
    """Spot-check key columns exist."""
    df = load_telco()
    expected = {"customerID", "Churn", "tenure", "MonthlyCharges", "TotalCharges"}
    assert expected.issubset(set(df.columns))


def test_load_telco_file_not_found(tmp_path):
    """Test FileNotFoundError when path does not exist."""
    non_existent = tmp_path / "missing.csv"
    with pytest.raises(FileNotFoundError):
        load_telco(path=non_existent)


def test_load_malaysia_subscribers():
    """Test loading Malaysia subscribers data."""
    df = load_malaysia_subscribers()
    assert not df.empty
    expected = {"date", "plan", "subscriptions"}
    assert expected.issubset(set(df.columns))


def test_load_malaysia_subscribers_file_not_found(tmp_path):
    """Test FileNotFoundError when Malaysia data does not exist."""
    non_existent = tmp_path / "missing_my.csv"
    with pytest.raises(FileNotFoundError):
        load_malaysia_subscribers(path=non_existent)
