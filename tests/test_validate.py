"""Unit tests for data validation module."""

import pandera.errors
import pytest

from churnguard.data.load import load_telco
from churnguard.data.validate import validate_raw_data


def test_validate_raw_data_success():
    """Real raw telco data passes pandera schema validation."""
    df = load_telco()
    validated = validate_raw_data(df)
    assert validated is not None
    assert len(validated) == 7043


def test_validate_bad_gender():
    """Fails when invalid gender is introduced."""
    df = load_telco().head(5100).copy()
    df.loc[0, "gender"] = "Other"
    with pytest.raises(pandera.errors.SchemaError):
        validate_raw_data(df)


def test_validate_bad_tenure():
    """Fails when tenure is outside [0, 72]."""
    df = load_telco().head(5100).copy()
    df.loc[0, "tenure"] = -1
    with pytest.raises(pandera.errors.SchemaError):
        validate_raw_data(df)

    df.loc[0, "tenure"] = 100
    with pytest.raises(pandera.errors.SchemaError):
        validate_raw_data(df)


def test_validate_duplicate_customer_id():
    """Fails when customerID is not unique."""
    df = load_telco().head(5100).copy()
    df.loc[1, "customerID"] = df.loc[0, "customerID"]
    with pytest.raises(pandera.errors.SchemaError):
        validate_raw_data(df)


def test_validate_min_row_count():
    """Fails when dataset has fewer than 5000 rows."""
    df = load_telco().head(100).copy()
    with pytest.raises(pandera.errors.SchemaError):
        validate_raw_data(df)
