"""Tests for threshold optimization and profit calculation module."""

import numpy as np
import pytest

from churnguard.models.threshold import (
    compute_profit_for_threshold,
    find_optimal_threshold,
    run_sensitivity_analysis,
    run_threshold_optimization,
)


def test_compute_profit_for_threshold_toy():
    """Test profit computation on known toy inputs."""
    # 4 customers: 2 churners, 2 non-churners
    y_true = np.array([1, 1, 0, 0])
    y_prob = np.array([0.9, 0.7, 0.4, 0.1])
    # At threshold 0.5: contacts index 0 and 1 (both are true churners)
    # Revenue = 2 * 0.3 * 780 = 468
    # Cost = 2 * 50 = 100
    # Net profit = 368
    # Profit per 1k = (368 / 4) * 1000 = 92000
    res = compute_profit_for_threshold(
        y_true=y_true,
        y_prob=y_prob,
        threshold=0.5,
        offer_cost=50.0,
        success_rate=0.30,
        default_clv=780.0,
    )
    assert res["n_contacted"] == 2
    assert res["true_churners_contacted"] == 2
    assert res["churner_capture_rate"] == 1.0
    assert pytest.approx(res["net_profit_rm"], 0.01) == 368.0
    assert pytest.approx(res["profit_per_1k_customers_rm"], 0.01) == 92000.0


def test_find_optimal_threshold_toy():
    """Test optimal threshold finder identifies the highest profit threshold."""
    y_true = np.array([1, 1, 0, 0])
    y_prob = np.array([0.9, 0.7, 0.4, 0.1])
    opt = find_optimal_threshold(
        y_true=y_true,
        y_prob=y_prob,
        offer_cost=50.0,
        success_rate=0.30,
        default_clv=780.0,
    )
    assert 0.4 < opt["optimal_threshold"] <= 0.7
    assert opt["max_profit_rm"] > 0


def test_sensitivity_analysis_shape():
    """Test sensitivity analysis outputs expected shape and columns."""
    y_true = np.array([1, 1, 0, 0, 1, 0])
    y_prob = np.array([0.8, 0.6, 0.2, 0.1, 0.7, 0.3])
    sens_df = run_sensitivity_analysis(y_true, y_prob)
    assert len(sens_df) == 9  # 3 success rates x 3 offer costs
    assert "profit_per_1k_rm" in sens_df.columns
    assert "optimal_threshold" in sens_df.columns


def test_run_threshold_optimization_integration(tmp_path):
    """Test full integration run on processed validation dataset."""
    res = run_threshold_optimization(save_artifacts=True)
    assert "optimal_res" in res
    assert "sensitivity_df" in res
    opt = res["optimal_res"]
    assert 0.0 < opt["optimal_threshold"] < 1.0
    assert opt["optimal_metrics"]["profit_per_1k_customers_rm"] > 0
