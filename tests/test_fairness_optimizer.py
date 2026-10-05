"""Unit tests for global fairness disparity auditor (ADR D-014 compliant)."""

import numpy as np

from churnguard.explain.fairness_optimizer import (
    compute_demographic_disparities,
    find_fairest_global_threshold,
)


def test_compute_demographic_disparities():
    y_true = np.array([1, 1, 0, 0])
    y_prob = np.array([0.8, 0.4, 0.2, 0.1])
    is_protected = np.array([True, False, True, False])

    # Threshold 0.5: protected churner (idx 0, prob 0.8) recalled; non-protected churner (idx 1, prob 0.4) not recalled
    res = compute_demographic_disparities(y_true, y_prob, is_protected, threshold=0.5)
    assert res["recall_protected"] == 1.0
    assert res["recall_reference"] == 0.0
    assert res["recall_gap"] == 1.0

    # Threshold 0.3: both churners recalled
    res2 = compute_demographic_disparities(y_true, y_prob, is_protected, threshold=0.3)
    assert res2["recall_protected"] == 1.0
    assert res2["recall_reference"] == 1.0
    assert res2["recall_gap"] == 0.0


def test_find_fairest_global_threshold():
    np.random.seed(42)
    y_true = np.random.binomial(1, 0.3, 100)
    y_prob = np.random.uniform(0, 1, 100)
    group_mask = np.random.choice([True, False], 100)

    res = find_fairest_global_threshold(y_true, y_prob, group_mask)
    assert "best_threshold" in res
    assert "min_recall_gap" in res
    assert 0.0 <= res["min_recall_gap"] <= 1.0
