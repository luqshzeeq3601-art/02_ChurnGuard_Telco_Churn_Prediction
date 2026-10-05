"""Probability calibration for churn prediction models (Experiment E09).

Implements:
- Training champion LightGBM on training split.
- Post-hoc calibration on validation split via Platt Scaling (Sigmoid) and Isotonic Regression.
- Expected Calibration Error (ECE) and Brier Score evaluation.
- Reliability diagram plotting and saving.
- MLflow logging for Experiment E09.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import lightgbm as lgb
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, log_loss

from churnguard.config import CFG, SEED
from churnguard.features.build import build_full_pipeline
from churnguard.models.evaluate import compute_all_metrics


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE) using equal-width bins.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_prob: Predicted positive class probabilities.
        n_bins: Number of probability bins (default: 10).

    Returns:
        Expected Calibration Error (float between 0 and 1).
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        bin_mask = (y_prob >= bin_edges[i]) & (
            y_prob < bin_edges[i + 1] if i < n_bins - 1 else y_prob <= bin_edges[i + 1]
        )
        bin_count = np.sum(bin_mask)
        if bin_count > 0:
            bin_acc = np.mean(y_true[bin_mask])
            bin_conf = np.mean(y_prob[bin_mask])
            ece += (bin_count / n) * np.abs(bin_acc - bin_conf)

    return float(ece)


def load_champion_params(params_path: Path | str | None = None) -> dict[str, Any]:
    """Load tuned hyperparameters from models/best_params.json."""
    path = Path(params_path) if params_path else CFG["paths"]["models_dir"] / "best_params.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    # Fallback to sensible defaults
    return {
        "n_estimators": 150,
        "max_depth": 5,
        "num_leaves": 20,
        "learning_rate": 0.05,
        "scale_pos_weight": 2.77,
        "random_state": SEED,
        "n_jobs": -1,
        "verbose": -1,
    }


def fit_champion_pipeline(
    train_df: pd.DataFrame,
    params: dict[str, Any] | None = None,
) -> Any:
    """Fit full champion LightGBM pipeline on training split."""
    if params is None:
        params = load_champion_params()

    model = lgb.LGBMClassifier(**params)
    pipeline = build_full_pipeline(
        model=model,
        include_engineered=True,
        scale_numeric=False,
    )

    X_train = train_df.drop(columns=["Churn"])
    y_train = train_df["Churn"].values

    pipeline.fit(X_train, y_train)
    return pipeline


def calibrate_pipeline(
    fitted_pipeline: Any,
    val_df: pd.DataFrame,
    method: str = "sigmoid",
) -> CalibratedClassifierCV:
    """Fit a calibrator (sigmoid or isotonic) on validation data using a pre-fitted pipeline.

    Args:
        fitted_pipeline: Pre-fitted sklearn Pipeline.
        val_df: Validation DataFrame with features and target 'Churn'.
        method: Calibration method ('sigmoid' or 'isotonic').

    Returns:
        Fitted CalibratedClassifierCV model.
    """
    X_val = val_df.drop(columns=["Churn"])
    y_val = val_df["Churn"].values

    calibrator = CalibratedClassifierCV(
        estimator=fitted_pipeline,
        method=method,
        cv="prefit",
    )
    calibrator.fit(X_val, y_val)
    return calibrator


def plot_calibration_curves(
    y_true: np.ndarray,
    prob_dict: dict[str, np.ndarray],
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Generate and save publication-ready reliability diagram and probability histograms.

    Args:
        y_true: True binary target values.
        prob_dict: Dictionary mapping model/method name to predicted probabilities.
        save_path: Optional path to save figure.

    Returns:
        Matplotlib Figure object.
    """
    fig, (ax1, ax2) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(8, 10),
        gridspec_kw={"height_ratios": [2, 1]},
    )

    ax1.plot([0, 1], [0, 1], "k--", label="Perfect Calibration (y = x)", alpha=0.7)

    colors = {
        "Uncalibrated (LightGBM)": "#e74c3c",
        "Calibrated (Sigmoid / Platt)": "#2980b9",
        "Calibrated (Isotonic)": "#27ae60",
    }

    for name, probs in prob_dict.items():
        prob_true, prob_pred = calibration_curve(y_true, probs, n_bins=10, strategy="uniform")
        brier = brier_score_loss(y_true, probs)
        ece = compute_ece(y_true, probs)
        color = colors.get(name)
        ax1.plot(
            prob_pred,
            prob_true,
            marker="o",
            linewidth=2,
            label=f"{name} (Brier: {brier:.4f}, ECE: {ece:.4f})",
            color=color,
        )
        ax2.hist(
            probs,
            bins=20,
            range=(0, 1),
            histtype="step",
            linewidth=1.5,
            label=name,
            color=color,
            density=True,
        )

    ax1.set_xlabel("Mean Predicted Probability", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Fraction of True Churners", fontsize=11, fontweight="bold")
    ax1.set_title(
        "Probability Calibration Reliability Diagram (Validation Set)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax1.legend(loc="upper left", frameon=True, fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.0])

    ax2.set_xlabel("Predicted Probability", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Density", fontsize=11, fontweight="bold")
    ax2.set_title("Probability Distribution Comparison", fontsize=12, fontweight="bold", pad=10)
    ax2.legend(loc="upper right", frameon=True, fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.set_xlim([0.0, 1.0])

    plt.tight_layout()

    if save_path:
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")

    return fig


def run_calibration_experiment(
    train_path: Path | str | None = None,
    val_path: Path | str | None = None,
    log_to_mlflow: bool = True,
) -> dict[str, Any]:
    """Run Experiment E09: Train champion, calibrate on val, evaluate and log to MLflow.

    Returns:
        Dictionary with evaluation results and fitted calibrators.
    """
    tr_path = Path(train_path) if train_path else CFG["paths"]["processed_dir"] / "train.parquet"
    va_path = Path(val_path) if val_path else CFG["paths"]["processed_dir"] / "val.parquet"

    train_df = pd.read_parquet(tr_path)
    val_df = pd.read_parquet(va_path)

    X_val = val_df.drop(columns=["Churn"])
    y_val = val_df["Churn"].values

    # 1. Fit uncalibrated champion pipeline
    params = load_champion_params()
    pipeline = fit_champion_pipeline(train_df, params=params)
    probs_uncal = pipeline.predict_proba(X_val)[:, 1]

    # 2. Fit Sigmoid (Platt Scaling)
    cal_sigmoid = calibrate_pipeline(pipeline, val_df, method="sigmoid")
    probs_sigmoid = cal_sigmoid.predict_proba(X_val)[:, 1]

    # 3. Fit Isotonic Regression
    cal_isotonic = calibrate_pipeline(pipeline, val_df, method="isotonic")
    probs_isotonic = cal_isotonic.predict_proba(X_val)[:, 1]

    # 4. Compute metrics
    metrics_uncal = compute_all_metrics(y_val, probs_uncal)
    metrics_uncal["ece"] = compute_ece(y_val, probs_uncal)
    metrics_uncal["log_loss"] = float(log_loss(y_val, probs_uncal))

    metrics_sigmoid = compute_all_metrics(y_val, probs_sigmoid)
    metrics_sigmoid["ece"] = compute_ece(y_val, probs_sigmoid)
    metrics_sigmoid["log_loss"] = float(log_loss(y_val, probs_sigmoid))

    metrics_isotonic = compute_all_metrics(y_val, probs_isotonic)
    metrics_isotonic["ece"] = compute_ece(y_val, probs_isotonic)
    metrics_isotonic["log_loss"] = float(log_loss(y_val, probs_isotonic))

    # 5. Generate and save figure
    fig_dir = CFG["paths"]["figures_dir"]
    fig_path1 = fig_dir / "07_calibration_curve.png"

    prob_dict = {
        "Uncalibrated (LightGBM)": probs_uncal,
        "Calibrated (Sigmoid / Platt)": probs_sigmoid,
        "Calibrated (Isotonic)": probs_isotonic,
    }

    fig = plot_calibration_curves(y_val, prob_dict, save_path=fig_path1)
    plt.close(fig)

    # Choose best calibration method (lowest Brier score)
    best_method = (
        "sigmoid"
        if metrics_sigmoid["brier_score"] <= metrics_isotonic["brier_score"]
        else "isotonic"
    )
    best_calibrator = cal_sigmoid if best_method == "sigmoid" else cal_isotonic

    results = {
        "uncalibrated": metrics_uncal,
        "sigmoid": metrics_sigmoid,
        "isotonic": metrics_isotonic,
        "best_method": best_method,
        "best_calibrator": best_calibrator,
        "uncalibrated_pipeline": pipeline,
        "brier_improvement": metrics_uncal["brier_score"]
        - min(metrics_sigmoid["brier_score"], metrics_isotonic["brier_score"]),
    }

    if log_to_mlflow:
        mlflow_cfg = CFG["mlflow"]
        mlflow.set_tracking_uri(str(mlflow_cfg["tracking_uri"]))
        mlflow.set_experiment(mlflow_cfg["experiment_name"])

        with mlflow.start_run(run_name="E09_lgbm_calibrated"):
            mlflow.log_params(
                {
                    "exp_id": "E09",
                    "base_model": "LightGBM_Tuned",
                    "best_method": best_method,
                    "val_samples": len(val_df),
                    "seed": SEED,
                }
            )

            mlflow.log_metrics({f"uncal_{k}": v for k, v in metrics_uncal.items()})
            mlflow.log_metrics({f"sigmoid_{k}": v for k, v in metrics_sigmoid.items()})
            mlflow.log_metrics({f"isotonic_{k}": v for k, v in metrics_isotonic.items()})
            mlflow.log_metric("brier_improvement", results["brier_improvement"])

            mlflow.log_artifact(str(fig_path1))
            mlflow.set_tags(
                {"exp_id": "E09", "model": "LightGBM_Calibrated", "stage": "candidate"}
            )

    return results


if __name__ == "__main__":
    res = run_calibration_experiment()
    print("=" * 60)
    print("Experiment E09: Probability Calibration on Validation Set")
    print("=" * 60)
    print(
        f"Uncalibrated Brier Score: {res['uncalibrated']['brier_score']:.4f} | ECE: {res['uncalibrated']['ece']:.4f}"
    )
    print(
        f"Sigmoid Brier Score:      {res['sigmoid']['brier_score']:.4f} | ECE: {res['sigmoid']['ece']:.4f}"
    )
    print(
        f"Isotonic Brier Score:     {res['isotonic']['brier_score']:.4f} | ECE: {res['isotonic']['ece']:.4f}"
    )
    print(
        f"Best Method:              {res['best_method']} (Brier Improvement: +{res['brier_improvement']:.4f})"
    )
