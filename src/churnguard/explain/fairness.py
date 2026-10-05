"""Fairness auditing, subgroup performance slicing, and error analysis (T4.5).

Implements:
- Subgroup performance slicing across demographics (gender, SeniorCitizen) and business segments.
- Demographic parity, Equal Opportunity (Recall/TPR), and False Positive Rate (FPR) auditing.
- NFR7 fairness verification: Flagging subgroup recall gaps > 0.05.
- False Positive and False Negative profile diagnostics.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_subgroup_metrics(
    df: pd.DataFrame,
    y_true_col: str = "Churn",
    y_prob_col: str = "churn_prob",
    threshold: float = 0.18,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Calculate performance and fairness metrics across demographic and business subgroups.

    Args:
        df: DataFrame containing features, target, and predicted probabilities.
        y_true_col: Name of true target column (default: 'Churn').
        y_prob_col: Name of predicted probability column (default: 'churn_prob').
        threshold: Decision threshold for classification (default: 0.18).
        group_cols: List of column names to slice by (default: ['gender', 'SeniorCitizen']).

    Returns:
        DataFrame summarizing subgroup sample size, positive rate, TPR, FPR, Precision, PR-AUC, ROC-AUC.
    """
    if group_cols is None:
        group_cols = ["gender", "SeniorCitizen", "Contract", "InternetService", "PaymentMethod"]

    rows = []
    overall_y_true = df[y_true_col].values.astype(int)
    overall_y_prob = df[y_prob_col].values.astype(float)
    overall_y_pred = (overall_y_prob >= threshold).astype(int)

    # Base overall row
    rows.append(
        {
            "slice_type": "Overall",
            "subgroup": "All Customers",
            "n_samples": len(df),
            "n_positives": int(np.sum(overall_y_true)),
            "base_rate": float(np.mean(overall_y_true)),
            "selection_rate": float(np.mean(overall_y_pred)),
            "recall_tpr": float(recall_score(overall_y_true, overall_y_pred, zero_division=0)),
            "false_positive_rate": float(np.mean(overall_y_pred[overall_y_true == 0]))
            if np.sum(overall_y_true == 0) > 0
            else 0.0,
            "precision": float(precision_score(overall_y_true, overall_y_pred, zero_division=0)),
            "f1": float(f1_score(overall_y_true, overall_y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(overall_y_true, overall_y_prob)),
            "pr_auc": float(average_precision_score(overall_y_true, overall_y_prob)),
        }
    )

    for col in group_cols:
        if col not in df.columns:
            continue
        for val in sorted(df[col].unique(), key=lambda x: str(x)):
            sub_df = df[df[col] == val]
            if len(sub_df) == 0:
                continue
            y_t = sub_df[y_true_col].values.astype(int)
            y_p = sub_df[y_prob_col].values.astype(float)
            y_hat = (y_p >= threshold).astype(int)

            n_pos = int(np.sum(y_t))
            n_neg = int(np.sum(y_t == 0))

            roc = float(roc_auc_score(y_t, y_p)) if len(np.unique(y_t)) > 1 else np.nan
            pr = float(average_precision_score(y_t, y_p)) if len(np.unique(y_t)) > 1 else np.nan

            rows.append(
                {
                    "slice_type": col,
                    "subgroup": str(val),
                    "n_samples": len(sub_df),
                    "n_positives": n_pos,
                    "base_rate": float(np.mean(y_t)),
                    "selection_rate": float(np.mean(y_hat)),
                    "recall_tpr": float(recall_score(y_t, y_hat, zero_division=0)),
                    "false_positive_rate": float(np.mean(y_hat[y_t == 0])) if n_neg > 0 else 0.0,
                    "precision": float(precision_score(y_t, y_hat, zero_division=0)),
                    "f1": float(f1_score(y_t, y_hat, zero_division=0)),
                    "roc_auc": roc,
                    "pr_auc": pr,
                }
            )

    return pd.DataFrame(rows)


def check_nfr7_fairness(
    df: pd.DataFrame,
    y_true_col: str = "Churn",
    y_prob_col: str = "churn_prob",
    threshold: float = 0.18,
    max_recall_gap: float = 0.05,
) -> dict[str, Any]:
    """Audit model against Non-Functional Requirement 7 (NFR7).

    Audits:
    - gender (Female vs Male): Recall gap <= 0.05.
    - SeniorCitizen (Non-Senior 0 vs Senior 1): Recall gap <= 0.05.

    Returns:
        Dictionary summarizing fairness audit results and pass/fail statuses.
    """
    sub_df = compute_subgroup_metrics(
        df=df,
        y_true_col=y_true_col,
        y_prob_col=y_prob_col,
        threshold=threshold,
        group_cols=["gender", "SeniorCitizen"],
    )

    # 1. Gender Audit
    female_rec = sub_df[(sub_df["slice_type"] == "gender") & (sub_df["subgroup"] == "Female")][
        "recall_tpr"
    ].values[0]
    male_rec = sub_df[(sub_df["slice_type"] == "gender") & (sub_df["subgroup"] == "Male")][
        "recall_tpr"
    ].values[0]
    gender_gap = abs(female_rec - male_rec)
    gender_passed = bool(gender_gap <= max_recall_gap)

    # 2. SeniorCitizen Audit
    non_senior_rec = sub_df[
        (sub_df["slice_type"] == "SeniorCitizen") & (sub_df["subgroup"] == "0")
    ]["recall_tpr"].values[0]
    senior_rec = sub_df[(sub_df["slice_type"] == "SeniorCitizen") & (sub_df["subgroup"] == "1")][
        "recall_tpr"
    ].values[0]
    senior_gap = abs(non_senior_rec - senior_rec)
    senior_passed = bool(senior_gap <= max_recall_gap)

    overall_passed = bool(gender_passed and senior_passed)

    return {
        "nfr7_target_max_gap": max_recall_gap,
        "gender_audit": {
            "female_recall": round(float(female_rec), 4),
            "male_recall": round(float(male_rec), 4),
            "recall_gap": round(float(gender_gap), 4),
            "passed": gender_passed,
        },
        "senior_citizen_audit": {
            "non_senior_recall": round(float(non_senior_rec), 4),
            "senior_recall": round(float(senior_rec), 4),
            "recall_gap": round(float(senior_gap), 4),
            "passed": senior_passed,
        },
        "overall_nfr7_passed": overall_passed,
        "subgroup_summary_df": sub_df,
    }


def diagnose_errors(
    df: pd.DataFrame,
    y_true_col: str = "Churn",
    y_prob_col: str = "churn_prob",
    threshold: float = 0.18,
    top_n: int = 10,
) -> dict[str, pd.DataFrame]:
    """Diagnose False Positive (FP) and False Negative (FN) customer cohorts."""
    y_true = df[y_true_col].values.astype(int)
    y_prob = df[y_prob_col].values.astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    analysis_df = df.copy()
    analysis_df["prediction"] = y_pred
    analysis_df["error_type"] = "Correct"
    analysis_df.loc[(y_true == 0) & (y_pred == 1), "error_type"] = "False Positive"
    analysis_df.loc[(y_true == 1) & (y_pred == 0), "error_type"] = "False Negative"

    fps = analysis_df[analysis_df["error_type"] == "False Positive"].sort_values(
        by=y_prob_col, ascending=False
    )
    fns = analysis_df[analysis_df["error_type"] == "False Negative"].sort_values(
        by=y_prob_col, ascending=True
    )

    return {
        "top_false_positives": fps.head(top_n),
        "top_false_negatives": fns.head(top_n),
        "all_errors_df": analysis_df,
    }
