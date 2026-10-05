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
        "n_unique_probs",
    }
    assert expected_keys.issubset(set(metrics.keys()))
    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["brier_score"] <= 1.0
    assert metrics["lift_at_10"] >= 0.0
    assert 0.0 <= metrics["recall_at_20"] <= 1.0
    assert metrics["n_unique_probs"] == 10


def test_tie_aware_analytical_fraction():
    """Verify exact analytical expected value when ties straddle top-k boundary."""
    # 5 items, top 40% = top 2 items
    # item 0: prob 0.9, y 1 -> strictly top 1
    # items 1, 2, 3: prob 0.5 with y in [1, 0, 1] (2 positives among 3 tied)
    # item 4: prob 0.1, y 0
    # Expected TP in top 2 = 1 (item 0) + (1 slot / 3 tied) * 2 = 5/3
    # Total positives = 3
    # Expected Recall@40 = (5/3) / 3 = 5/9
    y_true = np.array([1, 1, 0, 1, 0])
    y_prob = np.array([0.9, 0.5, 0.5, 0.5, 0.1])

    recall_at_40 = compute_recall_at_k(y_true, y_prob, k=0.40)
    assert recall_at_40 == pytest.approx(5.0 / 9.0)

    # Base rate = 3/5 = 0.6. Precision = (5/3) / 2 = 5/6. Lift = (5/6) / (3/5) = 25/18
    lift_at_40 = compute_lift_at_k(y_true, y_prob, k=0.40)
    assert lift_at_40 == pytest.approx(25.0 / 18.0)


def test_tie_aware_row_order_invariance():
    """Test AC for T9.3: tied scores give identical metric values regardless of row order."""
    rng = np.random.default_rng(42)
    # Construct array with a large block of tied scores around the top 20% cutoff
    y_true = np.array([1, 1, 0, 1, 0, 1, 0, 0, 1, 0] * 5)  # 50 items, 20 positives
    # Probabilities: 5 items high (0.9), 20 items tied at cutoff (0.4), 25 items low (0.1)
    y_prob = np.array([0.9] * 5 + [0.4] * 20 + [0.1] * 25)

    base_recall = compute_recall_at_k(y_true, y_prob, k=0.20)
    base_lift = compute_lift_at_k(y_true, y_prob, k=0.10)

    # Test across 20 random row permutations
    for seed in range(20):
        perm = rng.permutation(len(y_true))
        perm_y_true = y_true[perm]
        perm_y_prob = y_prob[perm]

        perm_recall = compute_recall_at_k(perm_y_true, perm_y_prob, k=0.20)
        perm_lift = compute_lift_at_k(perm_y_true, perm_y_prob, k=0.10)

        assert perm_recall == pytest.approx(base_recall, abs=1e-12), (
            f"Recall@20 changed under permutation seed {seed}: {perm_recall} vs {base_recall}"
        )
        assert perm_lift == pytest.approx(base_lift, abs=1e-12), (
            f"Lift@10 changed under permutation seed {seed}: {perm_lift} vs {base_lift}"
        )
