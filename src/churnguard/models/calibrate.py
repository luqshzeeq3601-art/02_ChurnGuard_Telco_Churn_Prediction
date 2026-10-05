"""Probability calibration redesign for churn prediction models (Experiment E10).

Adheres to docs/14_IMPROVEMENT_PLAN.md section 4.2 (fixing F2 and F3):
- Evaluates on combined train + val (5,986 rows) using 5-fold Stratified K-Fold.
- Compares:
  1. Uncalibrated LightGBM (5-fold OOF)
  2. Sigmoid CalibratedClassifierCV(cv=5) (5-fold OOF)
  3. Isotonic CalibratedClassifierCV(cv=5) (5-fold OOF)
- Pre-registered selection rule: Lowest OOF Brier score, subject to:
  * OOF PR-AUC drop <= 0.005 vs uncalibrated
  * At least 200 unique predicted probabilities
- Reports out-of-fold ECE (strictly zero in-sample calibration claims).
- Exports OOF predictions to models/oof_train_val_preds.parquet for threshold search (T9.5).
- Logs Experiment E10 to MLflow.
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
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from churnguard.config import CFG, SEED
from churnguard.features.build import build_full_pipeline
from churnguard.models.evaluate import compute_all_metrics, compute_n_unique_probs


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
    """Fit full champion LightGBM pipeline on input dataset."""
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
    pipeline: Any,
    cal_df: pd.DataFrame,
    method: str = "isotonic",
) -> Any:
    """Fit a CalibratedClassifierCV on a calibration dataset."""
    X_cal = cal_df.drop(columns=["Churn"])
    y_cal = cal_df["Churn"].values
    calibrator = CalibratedClassifierCV(pipeline, method=method, cv=5)
    calibrator.fit(X_cal, y_cal)
    return calibrator


def plot_calibration_curves(
    y_true: np.ndarray,
    prob_dict: dict[str, np.ndarray],
    n_bins: int = 10,
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Plot reliability diagram comparing multiple calibration methods."""
    fig, (ax1, ax2) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(8, 8),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    ax1.plot([0, 1], [0, 1], "k--", label="Perfect Calibration (Ideal)")
    colors = ["#94A3B8", "#3B82F6", "#10B981", "#F59E0B"]

    for idx, (label, probs) in enumerate(prob_dict.items()):
        color = colors[idx % len(colors)]
        prob_true, prob_pred = calibration_curve(y_true, probs, n_bins=n_bins, strategy="uniform")
        brier = brier_score_loss(y_true, probs)
        ece = compute_ece(y_true, probs, n_bins=n_bins)
        n_uniq = compute_n_unique_probs(probs)

        ax1.plot(
            prob_pred,
            prob_true,
            marker="o",
            linewidth=2,
            color=color,
            label=f"{label} (Brier={brier:.4f}, ECE={ece:.4f}, Uniq={n_uniq})",
        )

        ax2.hist(
            probs,
            range=(0, 1),
            bins=n_bins,
            histtype="step",
            lw=1.5,
            color=color,
            density=True,
        )

    ax1.set_ylabel("Empirical True Probability (Fraction of Positives)")
    ax1.set_title("OOF Reliability Curves (Experiment E10: 5-Fold on Train+Val)")
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.set_ylim([-0.05, 1.05])

    ax2.set_xlabel("Mean Predicted Probability")
    ax2.set_ylabel("Density")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.set_xlim([0.0, 1.0])

    plt.tight_layout()

    if save_path:
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")

    return fig


def run_calibration_redesign(
    train_path: Path | str | None = None,
    val_path: Path | str | None = None,
    log_to_mlflow: bool = True,
) -> dict[str, Any]:
    """Execute Experiment E10: 5-Fold OOF calibration evaluation on train + val.

    Returns:
        Dictionary containing OOF metrics, winning method, fitted calibrator, and paths.
    """
    tr_path = Path(train_path) if train_path else CFG["paths"]["processed_dir"] / "train.parquet"
    va_path = Path(val_path) if val_path else CFG["paths"]["processed_dir"] / "val.parquet"

    train_df = pd.read_parquet(tr_path)
    val_df = pd.read_parquet(va_path)
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)

    X = train_val_df.drop(columns=["Churn"])
    y = train_val_df["Churn"].values

    params = load_champion_params()
    base_model = lgb.LGBMClassifier(**params)
    pipeline = build_full_pipeline(base_model, include_engineered=True, scale_numeric=False)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # 1. Uncalibrated LightGBM (5-Fold OOF)
    oof_uncal = cross_val_predict(pipeline, X, y, cv=skf, method="predict_proba")[:, 1]
    metrics_uncal = compute_all_metrics(y, oof_uncal)
    metrics_uncal["ece"] = compute_ece(y, oof_uncal)
    metrics_uncal["log_loss"] = float(log_loss(y, oof_uncal))

    # 2. Sigmoid CalibratedClassifierCV(cv=5) (5-Fold OOF)
    cal_sig_proto = CalibratedClassifierCV(pipeline, cv=5, method="sigmoid")
    oof_sigmoid = cross_val_predict(cal_sig_proto, X, y, cv=skf, method="predict_proba")[:, 1]
    metrics_sigmoid = compute_all_metrics(y, oof_sigmoid)
    metrics_sigmoid["ece"] = compute_ece(y, oof_sigmoid)
    metrics_sigmoid["log_loss"] = float(log_loss(y, oof_sigmoid))

    # 3. Isotonic CalibratedClassifierCV(cv=5) (5-Fold OOF)
    cal_iso_proto = CalibratedClassifierCV(pipeline, cv=5, method="isotonic")
    oof_isotonic = cross_val_predict(cal_iso_proto, X, y, cv=skf, method="predict_proba")[:, 1]
    metrics_isotonic = compute_all_metrics(y, oof_isotonic)
    metrics_isotonic["ece"] = compute_ece(y, oof_isotonic)
    metrics_isotonic["log_loss"] = float(log_loss(y, oof_isotonic))

    # 4. Pre-registered selection rule
    candidates = {}
    for name, m, p in [("isotonic", metrics_isotonic, oof_isotonic), ("sigmoid", metrics_sigmoid, oof_sigmoid)]:
        pr_drop = metrics_uncal["pr_auc"] - m["pr_auc"]
        unique_count = m["n_unique_probs"]
        if pr_drop <= 0.005 and unique_count >= 200:
            candidates[name] = (m["brier_score"], m, p)

    if candidates:
        best_method = min(candidates.keys(), key=lambda k: candidates[k][0])
    else:
        best_method = "uncalibrated"

    # 5. Fit production calibrator on all train_val_df
    if best_method in ("sigmoid", "isotonic"):
        prod_calibrator = CalibratedClassifierCV(pipeline, cv=5, method=best_method)
        prod_calibrator.fit(X, y)
    else:
        prod_calibrator = fit_champion_pipeline(train_val_df, params=params)

    # 6. Save OOF predictions for threshold search (T9.5)
    oof_df = pd.DataFrame(
        {
            "y_true": y,
            "prob_uncal": oof_uncal,
            "prob_sigmoid": oof_sigmoid,
            "prob_isotonic": oof_isotonic,
            "prob_selected": oof_isotonic if best_method == "isotonic" else (oof_sigmoid if best_method == "sigmoid" else oof_uncal),
            "MonthlyCharges": train_val_df["MonthlyCharges"].values,
        }
    )
    models_dir = Path(CFG["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    oof_out = models_dir / "oof_train_val_preds.parquet"
    oof_df.to_parquet(oof_out, index=False)

    # 7. Plot reliability curves and save figure
    fig_path = Path(CFG["paths"]["figures_dir"]) / "07_calibration_curve.png"
    prob_dict = {
        "Uncalibrated (LightGBM)": oof_uncal,
        "Sigmoid (5-Fold CV)": oof_sigmoid,
        "Isotonic (5-Fold CV)": oof_isotonic,
    }
    fig = plot_calibration_curves(y, prob_dict, save_path=fig_path)
    plt.close(fig)

    results = {
        "uncalibrated": metrics_uncal,
        "sigmoid": metrics_sigmoid,
        "isotonic": metrics_isotonic,
        "best_method": best_method,
        "prod_calibrator": prod_calibrator,
        "oof_path": oof_out,
        "fig_path": fig_path,
        "brier_improvement": metrics_uncal["brier_score"] - (
            metrics_isotonic["brier_score"] if best_method == "isotonic" else metrics_sigmoid["brier_score"]
        ),
    }

    # 8. Log to MLflow
    if log_to_mlflow:
        mlflow_cfg = CFG["mlflow"]
        mlflow.set_tracking_uri(str(mlflow_cfg["tracking_uri"]))
        mlflow.set_experiment(mlflow_cfg["experiment_name"])

        with mlflow.start_run(run_name="E10_Calibration_Redesign"):
            mlflow.log_params(
                {
                    "exp_id": "E10",
                    "dataset": "train_val_combined",
                    "total_samples": len(train_val_df),
                    "best_method": best_method,
                    "cv_folds": 5,
                    "seed": SEED,
                }
            )
            mlflow.log_metrics({f"uncal_{k}": v for k, v in metrics_uncal.items()})
            mlflow.log_metrics({f"sigmoid_{k}": v for k, v in metrics_sigmoid.items()})
            mlflow.log_metrics({f"isotonic_{k}": v for k, v in metrics_isotonic.items()})
            mlflow.log_metric("brier_improvement", results["brier_improvement"])
            mlflow.log_artifact(str(fig_path))
            mlflow.set_tags(
                {
                    "exp_id": "E10",
                    "model": f"LightGBM_{best_method.capitalize()}",
                    "stage": "hardened_candidate",
                    "oof_evaluated": "true",
                }
            )

    return results


# Backward compatibility alias
def run_calibration_experiment(
    train_path: Path | str | None = None,
    val_path: Path | str | None = None,
    log_to_mlflow: bool = True,
) -> dict[str, Any]:
    """Execute calibration experiment (for backward compatibility)."""
    return run_calibration_redesign(train_path=train_path, val_path=val_path, log_to_mlflow=log_to_mlflow)


if __name__ == "__main__":
    res = run_calibration_redesign()
    print("=" * 60)
    print("Experiment E10: Calibration Redesign (5-Fold OOF on Train+Val)")
    print("=" * 60)
    for name in ["uncalibrated", "sigmoid", "isotonic"]:
        m = res[name]
        print(
            f"{name.capitalize():14s} | Brier: {m['brier_score']:.4f} | PR-AUC: {m['pr_auc']:.4f} | "
            f"ECE: {m['ece']:.4f} | Unique: {m['n_unique_probs']}"
        )
    print(f"\nWinning Method: {res['best_method'].upper()} (Brier Gain: +{res['brier_improvement']:.4f})")
    print(f"OOF Predictions Saved: {res['oof_path']}")
