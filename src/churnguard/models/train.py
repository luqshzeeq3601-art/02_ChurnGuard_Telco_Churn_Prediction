"""Model training and cross-validation with MLflow experiment tracking.

Implements:
- 5-fold Stratified K-Fold CV on training split.
- Support for imblearn Pipeline (SMOTE inside CV folds).
- Support for fairness ablation (dropping protected attributes).
- Experiment tracking in MLflow with full parameter, metric, and tag logging.
- Phase 2 (E00-E02) and Phase 3 (E03-E08) model runners.
"""

from __future__ import annotations

from typing import Any, Optional
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
import xgboost as xgb

from churnguard.config import CFG, SEED
from churnguard.features.build import FeatureEngineer, build_preprocessor
from churnguard.models.evaluate import compute_all_metrics


def build_pipeline_with_options(
    model: BaseEstimator,
    include_engineered: bool = True,
    scale_numeric: bool = False,
    use_smote: bool = False,
    drop_cols: Optional[list[str]] = None,
) -> Pipeline | ImbPipeline:
    """Construct an end-to-end sklearn or imblearn Pipeline with preprocessing and model."""
    fe = FeatureEngineer(include_engineered=include_engineered)
    preprocessor = build_preprocessor(
        include_engineered=include_engineered,
        scale_numeric=scale_numeric,
        drop_cols=drop_cols,
    )

    steps = [
        ("feature_engineer", fe),
        ("preprocessor", preprocessor),
    ]

    if use_smote:
        steps.append(("smote", SMOTE(random_state=SEED)))
        steps.append(("model", model))
        return ImbPipeline(steps=steps)

    steps.append(("model", model))
    return Pipeline(steps=steps)


def run_cv_experiment(
    exp_id: str,
    run_name: str,
    model: BaseEstimator,
    include_engineered: bool = True,
    scale_numeric: bool = False,
    use_smote: bool = False,
    drop_cols: Optional[list[str]] = None,
    n_splits: int = 5,
    train_path: Optional[str] = None,
    log_to_mlflow: bool = True,
) -> dict[str, Any]:
    """Run cross-validation for a given model and feature configuration, logging to MLflow."""
    data_path = train_path if train_path else CFG["paths"]["processed_dir"] / "train.parquet"
    train_df = pd.read_parquet(data_path)

    if drop_cols:
        train_df = train_df.drop(columns=drop_cols, errors="ignore")

    X = train_df.drop(columns=["Churn"])
    y = train_df["Churn"].values

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    oof_probs = np.zeros(len(train_df))
    fold_metrics: list[dict[str, float]] = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
        X_va, y_val = X.iloc[val_idx], y[val_idx]

        pipeline = build_pipeline_with_options(
            model=model,
            include_engineered=include_engineered,
            scale_numeric=scale_numeric,
            use_smote=use_smote,
            drop_cols=drop_cols,
        )

        pipeline.fit(X_tr, y_tr)
        probs_val = pipeline.predict_proba(X_va)[:, 1]
        oof_probs[val_idx] = probs_val

        f_metrics = compute_all_metrics(y_val, probs_val)
        fold_metrics.append(f_metrics)

    # Compute mean and std across folds
    metric_keys = fold_metrics[0].keys()
    mean_metrics = {f"cv_{k}_mean": float(np.mean([m[k] for m in fold_metrics])) for k in metric_keys}
    std_metrics = {f"cv_{k}_std": float(np.std([m[k] for m in fold_metrics])) for k in metric_keys}

    # Compute out-of-fold pooled metrics
    oof_metrics = {f"oof_{k}": float(v) for k, v in compute_all_metrics(y, oof_probs).items()}

    results = {
        "exp_id": exp_id,
        "run_name": run_name,
        "mean_metrics": mean_metrics,
        "std_metrics": std_metrics,
        "oof_metrics": oof_metrics,
    }

    if log_to_mlflow:
        mlflow_cfg = CFG["mlflow"]
        mlflow.set_tracking_uri(str(mlflow_cfg["tracking_uri"]))
        mlflow.set_experiment(mlflow_cfg["experiment_name"])

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(
                {
                    "exp_id": exp_id,
                    "model_class": model.__class__.__name__,
                    "include_engineered": include_engineered,
                    "scale_numeric": scale_numeric,
                    "use_smote": use_smote,
                    "drop_cols": str(drop_cols) if drop_cols else "None",
                    "n_splits": n_splits,
                    "seed": SEED,
                }
            )

            # Log model-specific hyperparameters
            for param_name, param_val in model.get_params().items():
                mlflow.log_param(f"model__{param_name}", str(param_val))

            mlflow.log_metrics(mean_metrics)
            mlflow.log_metrics(std_metrics)
            mlflow.log_metrics(oof_metrics)

            mlflow.set_tags(
                {
                    "exp_id": exp_id,
                    "model": model.__class__.__name__,
                }
            )

    return results


def run_phase_3_models() -> dict[str, dict[str, Any]]:
    """Execute Phase 3 experiments: E03 (RF), E04 (XGBoost), E05 (LightGBM), E06 (SMOTE)."""
    results = {}

    # E03: Random Forest (Bagging reference)
    print("\n" + "=" * 60)
    print("Running Experiment E03: Random Forest Classifier")
    print("=" * 60)
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )
    res_e03 = run_cv_experiment(
        exp_id="E03",
        run_name="E03_random_forest_engineered",
        model=rf_model,
        include_engineered=True,
        scale_numeric=False,
    )
    results["E03"] = res_e03
    print(
        f"E03 -> CV PR-AUC: {res_e03['mean_metrics']['cv_pr_auc_mean']:.4f} ± {res_e03['std_metrics']['cv_pr_auc_std']:.4f} | "
        f"ROC-AUC: {res_e03['mean_metrics']['cv_roc_auc_mean']:.4f} | "
        f"Lift@10: {res_e03['mean_metrics']['cv_lift_at_10_mean']:.2f}"
    )

    # E04: XGBoost (Gradient Boosting)
    print("\n" + "=" * 60)
    print("Running Experiment E04: XGBoost Classifier")
    print("=" * 60)
    pos_weight = (1.0 - 0.2653) / 0.2653  # ~2.77
    xgb_model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=pos_weight,
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=-1,
    )
    res_e04 = run_cv_experiment(
        exp_id="E04",
        run_name="E04_xgboost_engineered",
        model=xgb_model,
        include_engineered=True,
        scale_numeric=False,
    )
    results["E04"] = res_e04
    print(
        f"E04 -> CV PR-AUC: {res_e04['mean_metrics']['cv_pr_auc_mean']:.4f} ± {res_e04['std_metrics']['cv_pr_auc_std']:.4f} | "
        f"ROC-AUC: {res_e04['mean_metrics']['cv_roc_auc_mean']:.4f} | "
        f"Lift@10: {res_e04['mean_metrics']['cv_lift_at_10_mean']:.2f}"
    )

    # E05: LightGBM (Gradient Boosting)
    print("\n" + "=" * 60)
    print("Running Experiment E05: LightGBM Classifier")
    print("=" * 60)
    lgb_model = lgb.LGBMClassifier(
        n_estimators=150,
        max_depth=5,
        num_leaves=20,
        learning_rate=0.05,
        scale_pos_weight=pos_weight,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    )
    res_e05 = run_cv_experiment(
        exp_id="E05",
        run_name="E05_lgbm_engineered",
        model=lgb_model,
        include_engineered=True,
        scale_numeric=False,
    )
    results["E05"] = res_e05
    print(
        f"E05 -> CV PR-AUC: {res_e05['mean_metrics']['cv_pr_auc_mean']:.4f} ± {res_e05['std_metrics']['cv_pr_auc_std']:.4f} | "
        f"ROC-AUC: {res_e05['mean_metrics']['cv_roc_auc_mean']:.4f} | "
        f"Lift@10: {res_e05['mean_metrics']['cv_lift_at_10_mean']:.2f}"
    )

    # E06: SMOTE inside CV on the best booster (LightGBM)
    print("\n" + "=" * 60)
    print("Running Experiment E06: LightGBM with SMOTE Resampling inside CV")
    print("=" * 60)
    lgb_smote_model = lgb.LGBMClassifier(
        n_estimators=150,
        max_depth=5,
        num_leaves=20,
        learning_rate=0.05,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    )
    res_e06 = run_cv_experiment(
        exp_id="E06",
        run_name="E06_lgbm_smote",
        model=lgb_smote_model,
        include_engineered=True,
        scale_numeric=False,
        use_smote=True,
    )
    results["E06"] = res_e06
    print(
        f"E06 -> CV PR-AUC: {res_e06['mean_metrics']['cv_pr_auc_mean']:.4f} ± {res_e06['std_metrics']['cv_pr_auc_std']:.4f} | "
        f"ROC-AUC: {res_e06['mean_metrics']['cv_roc_auc_mean']:.4f} | "
        f"Lift@10: {res_e06['mean_metrics']['cv_lift_at_10_mean']:.2f}"
    )

    return results


if __name__ == "__main__":
    run_phase_3_models()
