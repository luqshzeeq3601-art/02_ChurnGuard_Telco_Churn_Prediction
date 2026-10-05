"""Root entrypoint for Streamlit dashboard (compatible with Hugging Face Spaces)."""

import runpy
from pathlib import Path

app_path = Path(__file__).resolve().parent / "app" / "streamlit_app.py"
runpy.run_path(str(app_path), run_name="__main__")
