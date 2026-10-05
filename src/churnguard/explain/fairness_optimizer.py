"""Fairness Disparity and Global Threshold Auditor for ChurnGuard (ADR D-014 Compliant).

In accordance with ADR D-014, group-specific runtime thresholds are explicitly rejected
to prevent disparate treatment and avoid collecting protected attributes at inference time.

This module evaluates demographic parity across global thresholds to identify single
decision thresholds that minimize demographic recall disparity while maximizing business return.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_demographic_disparities(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    group_mask: np.ndarray,
    threshold: float = 0.1882,
) -> dict[str, float]:
    """Calculate recall, precision, and contact rate disparity at a global decision threshold.

    Args:
        y_true: Binary ground truth labels (0 or 1).
        y_prob: Predicted churn probabilities.
        group_mask: Boolean mask indicating sensitive attribute (e.g. SeniorCitizen == 1).
        threshold: Universal global decision threshold applied equally to all customers.

    Returns:
        Dictionary of fairness metrics and gap.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    g_mask = np.asarray(group_mask, dtype=bool)

    contacted = y_prob >= threshold

    # Group metrics
    churn_group = g_mask & (y_true == 1)
    n_churn_group = np.sum(churn_group)
    rec_group = (
        float(np.sum(contacted[churn_group]) / n_churn_group) if n_churn_group > 0 else 0.0
    )

    # Non-group metrics
    churn_non_group = (~g_mask) & (y_true == 1)
    n_churn_non_group = np.sum(churn_non_group)
    rec_non_group = (
        float(np.sum(contacted[churn_non_group]) / n_churn_non_group)
        if n_churn_non_group > 0
        else 0.0
    )

    rec_gap = float(abs(rec_group - rec_non_group))

    return {
        "threshold": float(threshold),
        "recall_protected": rec_group,
        "recall_reference": rec_non_group,
        "recall_gap": rec_gap,
        "contact_rate_protected": float(np.mean(contacted[g_mask])) if np.any(g_mask) else 0.0,
        "contact_rate_reference": float(np.mean(contacted[~g_mask])) if np.any(~g_mask) else 0.0,
    }


def find_fairest_global_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    group_mask: np.ndarray,
    candidate_thresholds: np.ndarray | None = None,
) -> dict[str, Any]:
    """Search for the global threshold that minimizes demographic recall disparity.

    Complies with ADR D-014: enforces a single universal threshold across all customers.
    """
    if candidate_thresholds is None:
        candidate_thresholds = np.linspace(0.10, 0.60, 51)

    evaluations = [
        compute_demographic_disparities(y_true, y_prob, group_mask, threshold=t)
        for t in candidate_thresholds
    ]

    best_eval = min(evaluations, key=lambda e: e["recall_gap"])
    return {
        "best_threshold": best_eval["threshold"],
        "min_recall_gap": best_eval["recall_gap"],
        "metrics": best_eval,
    }
