"""Unit tests for model evaluation metrics on toy arrays."""

import numpy as np
import pytest

from churnguard.models.evaluate import (
    compute_all_metrics,
    compute_expected_profit,
    compute_lift_at_k,
    compute_recall_at_k,
)


def test_lift_at_10_perfect_ranking():
    """Toy array: 10 items, 2 positives (indices 0 and 1). Ranked perfectly."""
    # Baseline positive rate = 2 / 10 = 0.20
    # Top 10% is 1 item. If top item is positive (y=1), precision = 1.0.
    # Lift = 1.0 / 0.20 = 5.0
    y_true = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    y_prob = np.array([0.95, 0.85, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.05])

    lift = compute_lift_at_k(y_true, y_prob, k=0.10)
    assert lift == pytest.approx(5.0)


def test_recall_at_20_toy():
    """Toy array: 10 items, 4 positives. Top 20% = top 2 items."""
    # If top 2 items have 2 positives, recall@20% = 2 / 4 = 0.50
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])

    recall = compute_recall_at_k(y_true, y_prob, k=0.20)
    assert recall == pytest.approx(0.50)


def test_expected_profit_calculation():
    """Toy profit check.

    Cost per offer = RM50
    CLV = RM780
    Success rate = 0.30 -> Gain per true churner saved = 0.30 * 780 = RM234
    If threshold = 0.5:
      Customer 1: prob 0.8 (>= 0.5, contacted), true 1 (churner) -> net = 234 - 50 = +184
      Customer 2: prob 0.6 (>= 0.5, contacted), true 0 (not churner) -> net = 0 - 50 = -50
      Customer 3: prob 0.2 (< 0.5, not contacted), true 1 -> net = 0
    Total expected profit = 184 - 50 = RM134
    """
    y_true = np.array([1, 0, 1])
    y_prob = np.array([0.8, 0.6, 0.2])

    profit = compute_expected_profit(
        y_true=y_true,
        y_prob=y_prob,
        threshold=0.5,
        retention_offer_cost=50.0,
        offer_success_rate=0.30,
        clv_saved=780.0,
    )
    assert profit == pytest.approx(134.0)


def test_compute_all_metrics():
    """Verify compute_all_metrics returns complete dictionary with bounded values."""
    y_true = np.array([1, 0, 1, 0, 1, 0, 0, 1, 0, 0])
    y_prob = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.4, 0.85, 0.15, 0.25])

    metrics = compute_all_metrics(y_true, y_prob, threshold=0.5)

    expected_keys = {
        "pr_auc",
        "roc_auc",
        "brier_score",
        "lift_at_10",
        "recall_at_20",
        "precision",
        "recall",
        "f1",
        "expected_profit_rm",
    }
    assert expected_keys.issubset(set(metrics.keys()))
    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["brier_score"] <= 1.0
    assert metrics["lift_at_10"] >= 0.0
    assert 0.0 <= metrics["recall_at_20"] <= 1.0
