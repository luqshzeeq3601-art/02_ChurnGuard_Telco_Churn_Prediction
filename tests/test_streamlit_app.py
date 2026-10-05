"""Unit tests for Streamlit app helpers and logic."""

import pandas as pd
from app.streamlit_app import get_predictor, load_metrics_meta, load_sample_data


def test_streamlit_helpers():
    """Verify Streamlit app helper functions execute cleanly."""
    predictor = get_predictor()
    assert predictor is not None
    assert hasattr(predictor, "predict_single")

    meta = load_metrics_meta()
    assert isinstance(meta, dict)
    assert meta.get("model_name") == "churnguard-champion"

    data = load_sample_data()
    assert isinstance(data, pd.DataFrame)
    assert len(data) > 0
