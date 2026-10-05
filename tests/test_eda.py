"""Unit tests for EDA and reporting utilities."""

from churnguard.data.generate_eda_figures import compute_cramers_v, generate_all_eda_figures
from churnguard.data.generate_notebook import create_eda_notebook
from churnguard.data.load import load_telco


def test_compute_cramers_v():
    """Verify Cramer's V calculation."""
    df = load_telco().head(500)
    val = compute_cramers_v(df["Contract"], df["Churn"])
    assert 0.0 <= val <= 1.0


def test_generate_all_eda_figures(tmp_path):
    """Test generating all EDA figures to temporary directory."""
    saved_figures = generate_all_eda_figures(output_dir=tmp_path)
    assert len(saved_figures) == 6
    for f in saved_figures:
        assert f.exists()
        assert f.stat().st_size > 0


def test_create_eda_notebook(tmp_path, monkeypatch):
    """Test creating EDA notebook."""
    monkeypatch.chdir(tmp_path)
    create_eda_notebook()
    nb_file = tmp_path / "notebooks" / "01_eda.ipynb"
    assert nb_file.exists()
    assert nb_file.stat().st_size > 0
