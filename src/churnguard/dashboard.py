"""Read recorded dashboard facts without inventing old metric defaults."""

import json
from pathlib import Path
from typing import Any


def load_dashboard_summary(metadata_path: Path, evaluation_path: Path) -> dict[str, Any]:
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    )
    metadata["evaluation_metrics"] = {}
    if evaluation_path.exists() and metadata.get("optimal_threshold") is not None:
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        if evaluation.get("optimal_threshold") == metadata["optimal_threshold"]:
            metadata["evaluation_metrics"] = evaluation.get("metrics", {})
    return metadata
