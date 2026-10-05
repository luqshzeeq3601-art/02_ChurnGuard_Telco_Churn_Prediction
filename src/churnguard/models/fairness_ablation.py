"""Fairness ablation experiment (E08).

Evaluates the impact on performance and fairness when dropping protected
demographic attributes ('gender' and 'SeniorCitizen') from the feature set.
Updates Decision D-006 per docs/08_DECISIONS_LOG.md.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import lightgbm as lgb

from churnguard.config import CFG
from churnguard.models.train import run_cv_experiment


def run_fairness_ablation() -> dict[str, Any]:
    """Execute E08: Best tuned LightGBM without gender and SeniorCitizen."""
    params_file = CFG["paths"]["models_dir"] / "best_params.json"
    if not params_file.exists():
        raise FileNotFoundError(f"Best params file not found at {params_file}. Run tune.py first.")

    with open(params_file, encoding="utf-8") as fh:
        best_params = json.load(fh)

    model = lgb.LGBMClassifier(**best_params)

    print("\n" + "=" * 60)
    print("Running Experiment E08: Fairness Ablation (Dropping gender & SeniorCitizen)")
    print("=" * 60)

    # Drop gender and SeniorCitizen
    drop_cols = ["gender", "SeniorCitizen"]
    res_e08 = run_cv_experiment(
        exp_id="E08",
        run_name="E08_lgbm_fairness_ablation",
        model=model,
        include_engineered=True,
        scale_numeric=False,
        drop_cols=drop_cols,
        log_to_mlflow=True,
    )

    print(
        f"E08 -> CV PR-AUC: {res_e08['mean_metrics']['cv_pr_auc_mean']:.4f} ± {res_e08['std_metrics']['cv_pr_auc_std']:.4f} | "
        f"ROC-AUC: {res_e08['mean_metrics']['cv_roc_auc_mean']:.4f} | "
        f"Lift@10: {res_e08['mean_metrics']['cv_lift_at_10_mean']:.2f} | "
        f"Recall@20: {res_e08['mean_metrics']['cv_recall_at_20_mean']:.2%}"
    )

    return res_e08


if __name__ == "__main__":
    run_fairness_ablation()
