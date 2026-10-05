"""Model serialization, artifact export, and MLflow model registry logging (T4.6).

Implements:
- Training and calibrating the final production pipeline.
- Serializing full end-to-end artifact to models/model.joblib.
- Exporting production metadata to models/model_meta.json.
- Logging and registering the model in MLflow Model Registry as 'churnguard-model'.
- Verification of artifact loading and standalone inference.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd

from churnguard.config import CFG, SEED
from churnguard.features.build import get_feature_lists
from churnguard.models.calibrate import (
    calibrate_pipeline,
    fit_champion_pipeline,
    load_champion_params,
)
from churnguard.models.threshold import run_threshold_optimization


def save_final_model_artifacts(
    models_dir: Path | str | None = None,
    log_to_mlflow: bool = True,
) -> tuple[Path, Path]:
    """Train, calibrate, serialize, and register final production model (v1.1).

    Returns:
        Tuple of (model_joblib_path, meta_json_path).
    """
    from lightgbm import LGBMClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression

    from churnguard.models.train import build_pipeline_with_options

    out_dir = Path(models_dir) if models_dir else Path(CFG["paths"]["models_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data splits and combine train + val (5,986 samples)
    train_df = pd.read_parquet(CFG["paths"]["processed_dir"] / "train.parquet")
    val_df = pd.read_parquet(CFG["paths"]["processed_dir"] / "val.parquet")
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    X_train_val = train_val_df.drop(columns=["Churn"])
    y_train_val = train_val_df["Churn"].values

    # 2. Fit Champion pipeline (Logistic Regression M2: drop gender & SeniorCitizen, sigmoid cv=5)
    champion_base = build_pipeline_with_options(
        model=LogisticRegression(max_iter=1000, random_state=SEED, class_weight="balanced"),
        include_engineered=True,
        scale_numeric=True,
        drop_cols=["gender", "SeniorCitizen"],
    )
    champion_calibrated = CalibratedClassifierCV(estimator=champion_base, method="sigmoid", cv=5)
    champion_calibrated.fit(X_train_val, y_train_val)

    # 3. Fit Runner-up pipeline (LightGBM Optuna-tuned, isotonic cv=5)
    params = load_champion_params()
    runner_up_base = build_pipeline_with_options(
        model=LGBMClassifier(**params),
        include_engineered=True,
        scale_numeric=False,
    )
    runner_up_calibrated = CalibratedClassifierCV(estimator=runner_up_base, method="isotonic", cv=5)
    runner_up_calibrated.fit(X_train_val, y_train_val)

    # 4. Save joblib artifacts
    model_path = out_dir / "model.joblib"
    joblib.dump(champion_calibrated, model_path)

    runner_up_path = out_dir / "runner_up_model.joblib"
    joblib.dump(runner_up_calibrated, runner_up_path)

    # 5. Load threshold and test metrics metadata
    threshold_file = out_dir / "optimal_threshold.json"
    if threshold_file.exists():
        with open(threshold_file, encoding="utf-8") as f:
            thresh_data = json.load(f)
            optimal_tau = thresh_data.get("optimal_threshold", 0.1882)
    else:
        opt_res = run_threshold_optimization(save_artifacts=False)
        optimal_tau = opt_res["optimal_res"]["optimal_threshold"]

    metrics_file = Path(CFG["paths"]["reports_dir"]) / "final_metrics.json"
    if metrics_file.exists():
        with open(metrics_file, encoding="utf-8") as f:
            test_metrics_data = json.load(f)
    else:
        test_metrics_data = {}

    num_cols, cat_cols = get_feature_lists(include_engineered=True)

    # 6. Construct metadata dictionary
    meta = {
        "model_name": "churnguard-champion",
        "model_version": "1.1.0",
        "model_class": "CalibratedClassifierCV(LogisticRegression, method='sigmoid', cv=5)",
        "runner_up_model": "CalibratedClassifierCV(LGBMClassifier, method='isotonic', cv=5)",
        "fairness_mitigation": "Option M2 (dropped gender, SeniorCitizen)",
        "test_reuse_disclosure": (
            "Test set was reused exactly once for v1.1 evaluation after calibration redesign (E10) "
            "and champion re-decision (E11), as recorded in D-015."
        ),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "seed": SEED,
        "optimal_threshold": float(optimal_tau),
        "risk_tiers": {
            "high": {"min_prob": float(optimal_tau), "max_prob": 1.0, "label": "High Risk"},
            "medium": {
                "min_prob": float(round(0.5 * optimal_tau, 4)),
                "max_prob": float(optimal_tau),
                "label": "Medium Risk",
            },
            "low": {
                "min_prob": 0.0,
                "max_prob": float(round(0.5 * optimal_tau, 4)),
                "label": "Low Risk",
            },
        },
        "features": {
            "numeric_features": num_cols,
            "categorical_features": cat_cols,
            "total_feature_count": len(num_cols) + len(cat_cols),
        },
        "cost_parameters": CFG["cost"],
        "test_performance": test_metrics_data.get("metrics", {}),
        "operational_summary": test_metrics_data.get("operational_summary", {}),
    }

    meta_path = out_dir / "model_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # 7. MLflow registration
    if log_to_mlflow:
        mlflow_cfg = CFG["mlflow"]
        mlflow.set_tracking_uri(str(mlflow_cfg["tracking_uri"]))
        mlflow.set_experiment(mlflow_cfg["experiment_name"])

        with mlflow.start_run(run_name="Final_Model_Registration_v1.1"):
            mlflow.log_params(
                {
                    "model_version": "1.1.0",
                    "champion": "Logistic Regression + Sigmoid cv=5 (M2)",
                    "runner_up": "LightGBM + Isotonic cv=5",
                    "optimal_threshold": optimal_tau,
                    "seed": SEED,
                }
            )
            mlflow.log_artifact(str(model_path))
            mlflow.log_artifact(str(runner_up_path))
            mlflow.log_artifact(str(meta_path))
            mlflow.set_tags({"stage": "production", "model": "churnguard-champion", "version": "1.1.0"})

    return model_path, meta_path


def load_model_and_predict(
    sample_df: pd.DataFrame,
    models_dir: Path | str | None = None,
) -> pd.DataFrame:
    """Load serialized model and metadata, returning predicted probabilities and risk tiers."""
    out_dir = Path(models_dir) if models_dir else Path(CFG["paths"]["models_dir"])
    model_path = out_dir / "model.joblib"
    meta_path = out_dir / "model_meta.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {model_path}. Run save_final_model_artifacts first."
        )

    model = joblib.load(model_path)
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    threshold = meta["optimal_threshold"]
    med_threshold = meta["risk_tiers"]["medium"]["min_prob"]

    probs = model.predict_proba(sample_df)[:, 1]

    res_df = sample_df.copy()
    res_df["churn_probability"] = np.round(probs, 4)

    def assign_tier(p: float) -> str:
        if p >= threshold:
            return "High"
        elif p >= med_threshold:
            return "Medium"
        return "Low"

    res_df["risk_tier"] = res_df["churn_probability"].apply(assign_tier)
    res_df["model_version"] = meta["model_version"]

    return res_df


if __name__ == "__main__":
    m_path, meta_path = save_final_model_artifacts()
    print("=" * 60)
    print("Task T4.6: Model Serialization & Registration Completed")
    print("=" * 60)
    print(f"Model artifact saved to:    {m_path}")
    print(f"Metadata artifact saved to: {meta_path}")

    # Test standalone reload and scoring
    test_sample = pd.read_parquet(CFG["paths"]["processed_dir"] / "test.parquet").head(3)
    scored = load_model_and_predict(test_sample)
    print("\nStandalone Reload & Scoring Smoke Test (First 3 Customers):")
    for _, row in scored.iterrows():
        print(
            f"  ID: {row.get('customerID', 'N/A')} -> Prob: {row['churn_probability']:.4f} | Tier: {row['risk_tier']}"
        )
