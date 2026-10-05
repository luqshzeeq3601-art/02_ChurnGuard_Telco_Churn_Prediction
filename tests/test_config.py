"""Tests for config loader."""

from pathlib import Path

from churnguard.config import CFG, SEED, load_config


def test_load_config_returns_dict():
    cfg = load_config()
    assert isinstance(cfg, dict)
    assert "project" in cfg
    assert "paths" in cfg


def test_seed_is_42():
    assert SEED == 42


def test_paths_are_resolved():
    """All path values should be absolute Path objects."""
    for key, val in CFG["paths"].items():
        assert isinstance(val, Path), f"paths.{key} should be Path, got {type(val)}"
        assert val.is_absolute(), f"paths.{key} should be absolute"


def test_split_ratios_sum_to_one():
    split = CFG["split"]
    total = split["train_ratio"] + split["val_ratio"] + split["test_ratio"]
    assert abs(total - 1.0) < 1e-9


def test_cost_assumptions_present():
    cost = CFG["cost"]
    assert cost["retention_offer_cost"] > 0
    assert cost["profit_per_tp"] > 0
