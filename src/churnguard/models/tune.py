"""Hyperparameter tuning with Optuna and MLflow integration.

Implements:
- Bayesian hyperparameter optimization for LightGBM/XGBoost over 5-fold Stratified CV.
- Objective: maximize mean CV PR-AUC on training split.
- Logging of all trial metrics and best parameters to MLflow and disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import lightgbm as lgb
import mlflow
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from churnguard.config import CFG, SEED
from churnguard.features.build import build_full_pipeline
from churnguard.models.evaluate import compute_all_metrics
from churnguard.models.train import run_cv_experiment

# Suppress verbose Optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective(trial: optuna.Trial, X: pd.DataFrame, y: np.ndarray, n_splits: int = 5) -> float:
    """Optuna objective function for LightGBM hyperparameter search."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=25),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "num_leaves": trial.suggest_int("num_leaves", 7, 63),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 15, 80),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.5, 3.2),
        "random_state": SEED,
        "n_jobs": -1,
        "verbose": -1,
    }

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    fold_pr_aucs: list[float] = []

    for train_idx, val_idx in skf.split(X, y):
        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
        X_va, y_val = X.iloc[val_idx], y[val_idx]

        model = lgb.LGBMClassifier(**params)
        pipeline = build_full_pipeline(model=model, include_engineered=True, scale_numeric=False)

        pipeline.fit(X_tr, y_tr)
        probs_val = pipeline.predict_proba(X_va)[:, 1]
        metrics = compute_all_metrics(y_val, probs_val)
        fold_pr_aucs.append(metrics["pr_auc"])

    return float(np.mean(fold_pr_aucs))


def tune_lightgbm(
    n_trials: int = 60,
    output_path: Path | str | None = None,
    log_to_mlflow: bool = True,
) -> dict[str, Any]:
    """Run Optuna tuning for LightGBM and save best parameters."""
    train_path = CFG["paths"]["processed_dir"] / "train.parquet"
    train_df = pd.read_parquet(train_path)

    X = train_df.drop(columns=["Churn"])
    y = train_df["Churn"].values

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    print(f"\n[Optuna] Starting hyperparameter optimization with {n_trials} trials...")
    study.optimize(lambda trial: objective(trial, X, y), n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_pr_auc = study.best_value
    print(f"\n[Optuna] Best Trial CV PR-AUC: {best_pr_auc:.4f}")
    print(f"[Optuna] Best Parameters: {json.dumps(best_params, indent=2)}")

    # Add non-tuned fixed parameters
    full_best_params = {
        **best_params,
        "subsample_freq": 1,
        "random_state": SEED,
        "n_jobs": -1,
        "verbose": -1,
    }

    # Save to models/best_params.json
    out_file = Path(output_path) if output_path else CFG["paths"]["models_dir"] / "best_params.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as fh:
        json.dump(full_best_params, fh, indent=2)
    print(f"[Optuna] Saved best parameters to {out_file}")

    # Now evaluate best model across full CV with all metrics and log as E07
    best_model = lgb.LGBMClassifier(**full_best_params)
    e07_results = run_cv_experiment(
        exp_id="E07",
        run_name="E07_lgbm_optuna_tuned",
        model=best_model,
        include_engineered=True,
        scale_numeric=False,
        log_to_mlflow=log_to_mlflow,
    )

    return {
        "best_params": full_best_params,
        "best_pr_auc": best_pr_auc,
        "e07_results": e07_results,
    }


if __name__ == "__main__":
    tune_lightgbm(n_trials=60)
