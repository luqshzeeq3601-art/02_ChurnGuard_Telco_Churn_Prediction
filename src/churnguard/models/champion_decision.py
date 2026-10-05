"""E11 Champion Re-Decision Experiment.

Compares E02 (Logistic Regression, engineered features) and E07 (Optuna-tuned LightGBM)
on identical 5 folds of train + val data. Computes per-fold paired difference in PR-AUC
and applies the pre-registered decision rule from docs/14_IMPROVEMENT_PLAN.md section 4.4.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from churnguard.config import CFG, SEED
from churnguard.models.train import build_pipeline_with_options


def run_e11_experiment(
    data_df: pd.DataFrame | None = None,
    log_to_mlflow: bool = True,
    save_path: Path | str | None = None,
) -> dict[str, Any]:
    """Execute paired 5-fold CV comparison between E02 and E07.

    Pre-registered rule:
      If mean paired PR-AUC difference >= 1 std AND >= 0.01:
          LightGBM is champion
      Else:
          Logistic Regression is champion (simpler, interpretable),
          LightGBM is runner-up.
    """
    if data_df is None:
        train_df = pd.read_parquet(CFG["paths"]["processed_dir"] / "train.parquet")
        val_df = pd.read_parquet(CFG["paths"]["processed_dir"] / "val.parquet")
        df = pd.concat([train_df, val_df], ignore_index=True)
    else:
        df = data_df.copy()

    X = df.drop(columns=["Churn"])
    y = df["Churn"].values

    params_path = CFG["paths"]["models_dir"] / "best_params.json"
    if params_path.exists():
        with open(params_path, "r", encoding="utf-8") as f:
            best_params = json.load(f)
    else:
        best_params = {
            "n_estimators": 100,
            "random_state": SEED,
            "n_jobs": -1,
            "verbose": -1,
        }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    fold_results = []
    pr_auc_lr = []
    pr_auc_lgb = []
    roc_auc_lr = []
    roc_auc_lgb = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
        X_va, y_va = X.iloc[val_idx], y[val_idx]

        # E02: Logistic Regression (engineered, scaled numeric)
        pipe_lr = build_pipeline_with_options(
            model=LogisticRegression(max_iter=1000, random_state=SEED, class_weight="balanced"),
            include_engineered=True,
            scale_numeric=True,
        )
        pipe_lr.fit(X_tr, y_tr)
        prob_lr = pipe_lr.predict_proba(X_va)[:, 1]
        score_pr_lr = float(average_precision_score(y_va, prob_lr))
        score_roc_lr = float(roc_auc_score(y_va, prob_lr))

        # E07: Tuned LightGBM
        pipe_lgb = build_pipeline_with_options(
            model=LGBMClassifier(**best_params),
            include_engineered=True,
            scale_numeric=False,
        )
        pipe_lgb.fit(X_tr, y_tr)
        prob_lgb = pipe_lgb.predict_proba(X_va)[:, 1]
        score_pr_lgb = float(average_precision_score(y_va, prob_lgb))
        score_roc_lgb = float(roc_auc_score(y_va, prob_lgb))

        pr_auc_lr.append(score_pr_lr)
        pr_auc_lgb.append(score_pr_lgb)
        roc_auc_lr.append(score_roc_lr)
        roc_auc_lgb.append(score_roc_lgb)

        fold_results.append(
            {
                "fold": fold + 1,
                "lr_pr_auc": score_pr_lr,
                "lgb_pr_auc": score_pr_lgb,
                "diff_pr_auc": score_pr_lgb - score_pr_lr,
                "lr_roc_auc": score_roc_lr,
                "lgb_roc_auc": score_roc_lgb,
            }
        )

    pr_auc_lr_arr = np.array(pr_auc_lr)
    pr_auc_lgb_arr = np.array(pr_auc_lgb)
    diffs = pr_auc_lgb_arr - pr_auc_lr_arr

    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs))

    # Pre-registered rule check
    rule_condition_met = bool((mean_diff >= std_diff) and (mean_diff >= 0.01))
    champion = "LightGBM" if rule_condition_met else "LogisticRegression"
    runner_up = "LogisticRegression" if rule_condition_met else "LightGBM"

    # MO1 assessment: Baseline E01 PR-AUC = 0.6587, target improvement >= 0.03
    mo1_target = 0.6587 + 0.03
    mo1_met = bool(float(np.mean(pr_auc_lgb_arr)) >= mo1_target)

    decision_summary = {
        "experiment": "E11_Champion_ReDecision",
        "n_samples": len(df),
        "n_folds": 5,
        "lr_mean_pr_auc": float(np.mean(pr_auc_lr_arr)),
        "lr_std_pr_auc": float(np.std(pr_auc_lr_arr)),
        "lgb_mean_pr_auc": float(np.mean(pr_auc_lgb_arr)),
        "lgb_std_pr_auc": float(np.std(pr_auc_lgb_arr)),
        "paired_diff_mean": mean_diff,
        "paired_diff_std": std_diff,
        "rule_thresholds": {
            "required_mean_ge_std": bool(mean_diff >= std_diff),
            "required_mean_ge_0_01": bool(mean_diff >= 0.01),
        },
        "selected_champion": champion,
        "runner_up": runner_up,
        "mo1_target": mo1_target,
        "mo1_achieved": float(np.mean(pr_auc_lgb_arr)),
        "mo1_met": mo1_met,
        "fold_details": fold_results,
    }

    if save_path:
        out_path = Path(save_path)
    else:
        out_path = Path("reports") / "e11_champion_decision.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(decision_summary, f, indent=2)

    if log_to_mlflow:
        mlflow_cfg = CFG["mlflow"]
        mlflow.set_tracking_uri(str(mlflow_cfg["tracking_uri"]))
        mlflow.set_experiment(mlflow_cfg["experiment_name"])

        with mlflow.start_run(run_name="E11_Champion_ReDecision"):
            mlflow.log_params(
                {
                    "n_folds": 5,
                    "dataset": "train_val_5986",
                    "rule": "mean_gain >= 1_std AND mean_gain >= 0.01",
                }
            )
            mlflow.log_metrics(
                {
                    "lr_mean_pr_auc": decision_summary["lr_mean_pr_auc"],
                    "lgb_mean_pr_auc": decision_summary["lgb_mean_pr_auc"],
                    "paired_diff_mean": mean_diff,
                    "paired_diff_std": std_diff,
                }
            )
            mlflow.log_artifact(str(out_path))

    return decision_summary


if __name__ == "__main__":
    res = run_e11_experiment(log_to_mlflow=True)
    print("=" * 60)
    print("E11 Champion Re-Decision Results (Paired 5-Fold CV)")
    print("=" * 60)
    print(f"LR PR-AUC:       {res['lr_mean_pr_auc']:.4f} +/- {res['lr_std_pr_auc']:.4f}")
    print(f"LightGBM PR-AUC: {res['lgb_mean_pr_auc']:.4f} +/- {res['lgb_std_pr_auc']:.4f}")
    print(f"Paired Diff:     {res['paired_diff_mean']:+.4f} +/- {res['paired_diff_std']:.4f}")
    print(f"Rule Condition:  {res['rule_thresholds']}")
    print(f"Champion:        {res['selected_champion']}")
    print(f"Runner-up:       {res['runner_up']}")
    print(f"MO1 Status:      {'MET' if res['mo1_met'] else 'NOT MET'} (LGBM {res['mo1_achieved']:.4f} vs target {res['mo1_target']:.4f})")
