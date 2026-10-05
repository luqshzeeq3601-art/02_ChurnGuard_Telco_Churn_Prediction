"""Evaluation metrics for churn prediction models.

Implements all primary, secondary, business, and calibration metrics per docs/06_EXPERIMENT_PLAN.md:
- PR-AUC (Average Precision)
- ROC-AUC
- Brier Score
- Lift@top k% (e.g. Lift@10%)
- Recall@top k% (e.g. Recall@20%)
- Classification metrics (Precision, Recall, F1 at threshold)
- Expected retention profit (RM)
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from churnguard.config import CFG


def _expected_positives_at_k(y_true: np.ndarray, y_prob: np.ndarray, top_n: int) -> float:
    """Compute expected true positives in top_n under uniform random tie-breaking.

    When multiple items share the cutoff probability, fractional weighting is applied
    so the metric value is deterministic and strictly row-order invariant.
    """
    n = len(y_true)
    if n == 0 or top_n <= 0:
        return 0.0
    top_n = min(top_n, n)

    # Sort descending
    sorted_probs = np.sort(y_prob)[::-1]
    cutoff_prob = sorted_probs[top_n - 1]

    strictly_higher_mask = y_prob > cutoff_prob
    tied_mask = y_prob == cutoff_prob

    n_higher = int(np.sum(strictly_higher_mask))
    n_tied = int(np.sum(tied_mask))

    tp_higher = float(np.sum(y_true[strictly_higher_mask]))
    tp_tied = float(np.sum(y_true[tied_mask]))

    if n_tied == 0:
        return tp_higher

    slots_for_tied = top_n - n_higher
    frac = slots_for_tied / n_tied
    return tp_higher + (frac * tp_tied)


def compute_lift_at_k(y_true: np.ndarray, y_prob: np.ndarray, k: float = 0.10) -> float:
    """Calculate tie-aware Lift in the top k fraction of ranked predictions.

    Lift@k = (Precision in top k%) / (Baseline positive rate).
    When ties occur at the boundary, expected precision under random tie-breaking is computed.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_prob: Predicted positive class probabilities.
        k: Top fraction of predictions to evaluate (default 0.10 = top 10%).

    Returns:
        Lift value (float >= 0).
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    n = len(y_true)

    if n == 0:
        return 0.0

    base_rate = float(np.mean(y_true))
    if base_rate == 0:
        return 0.0

    top_n = max(1, int(np.ceil(k * n)))
    expected_tp = _expected_positives_at_k(y_true, y_prob, top_n)
    top_precision = expected_tp / top_n

    return float(top_precision / base_rate)


def compute_recall_at_k(y_true: np.ndarray, y_prob: np.ndarray, k: float = 0.20) -> float:
    """Calculate tie-aware Recall in the top k fraction of ranked predictions.

    Recall@k = (Expected True Positives in top k%) / (Total true positives).
    When ties occur at the boundary, fractional weighting is applied for row-order invariance.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_prob: Predicted positive class probabilities.
        k: Top fraction of predictions to evaluate (default 0.20 = top 20%).

    Returns:
        Recall fraction (float between 0.0 and 1.0).
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    n = len(y_true)

    total_positives = float(np.sum(y_true))
    if total_positives == 0 or n == 0:
        return 0.0

    top_n = max(1, int(np.ceil(k * n)))
    expected_tp = _expected_positives_at_k(y_true, y_prob, top_n)

    return float(expected_tp / total_positives)


def compute_n_unique_probs(y_prob: np.ndarray) -> int:
    """Count the number of unique predicted probabilities."""
    y_prob = np.asarray(y_prob).astype(float)
    return int(len(np.unique(y_prob)))


def compute_expected_profit(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    retention_offer_cost: float | None = None,
    offer_success_rate: float = 0.30,
    clv_saved: float | None = None,
) -> float:
    """Calculate total expected retention campaign profit (RM).

    For every customer with y_prob >= threshold, we contact them at retention_offer_cost.
    If they are an actual churner (y_true == 1), with offer_success_rate they are retained,
    saving clv_saved.

    Expected profit = sum_{contacted} (y_true * offer_success_rate * clv_saved - retention_offer_cost)

    Args:
        y_true: Ground truth binary labels.
        y_prob: Predicted probabilities.
        threshold: Decision threshold.
        retention_offer_cost: Cost per contact (default from config: RM50).
        offer_success_rate: Probability a contacted churner accepts offer (default: 0.30).
        clv_saved: Customer value preserved if retained (default from config: RM780).

    Returns:
        Net expected profit in RM.
    """
    cost_cfg = CFG["cost"]
    offer_cost = (
        retention_offer_cost
        if retention_offer_cost is not None
        else cost_cfg["retention_offer_cost"]
    )
    clv = clv_saved if clv_saved is not None else cost_cfg["clv_if_retained"]

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    contacted = (y_prob >= threshold).astype(int)
    true_churners_contacted = contacted * y_true

    total_cost = np.sum(contacted) * offer_cost
    total_gain = np.sum(true_churners_contacted) * offer_success_rate * clv

    return float(total_gain - total_cost)


def compute_all_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Calculate all standard and business metrics for model evaluation.

    Args:
        y_true: Ground truth binary labels.
        y_prob: Predicted probabilities for positive class.
        threshold: Binary decision threshold for precision/recall/F1/profit.

    Returns:
        Dictionary of computed metric names and values.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "lift_at_10": float(compute_lift_at_k(y_true, y_prob, k=0.10)),
        "recall_at_20": float(compute_recall_at_k(y_true, y_prob, k=0.20)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "expected_profit_rm": float(compute_expected_profit(y_true, y_prob, threshold=threshold)),
        "n_unique_probs": compute_n_unique_probs(y_prob),
    }

    return metrics
