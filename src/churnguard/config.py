"""Configuration loader for ChurnGuard.

Reads configs/config.yaml and exposes settings as a typed dictionary.
All paths are resolved relative to the project root.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load YAML config and resolve relative paths to absolute.

    Args:
        path: Optional override for the config file location.
              Defaults to ``configs/config.yaml`` at the project root.

    Returns:
        Parsed config dictionary with path values resolved to ``pathlib.Path``.
    """
    cfg_path = Path(path) if path else _DEFAULT_CONFIG_PATH

    with open(cfg_path, encoding="utf-8") as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh)

    # Resolve all entries under "paths" relative to project root
    for key, val in cfg.get("paths", {}).items():
        cfg["paths"][key] = PROJECT_ROOT / val

    return cfg


# Module-level convenience: import once, use everywhere
CFG = load_config()
SEED = CFG["project"]["seed"]
