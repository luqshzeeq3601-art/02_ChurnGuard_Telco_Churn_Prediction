"""E12 Fairness Audit and Mitigation Experiment.

Evaluates demographic fairness (recall, precision, contact rate) across gender
and SeniorCitizen for M0 (all features), M1 (drop gender), and M2 (drop gender + SeniorCitizen).
Applies the pre-registered selection rule from docs/14_IMPROVEMENT_PLAN.md section 4.5.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from churnguard.config import CFG, SEED
from churnguard.models.threshold import compute_profit_for_threshold
from churnguard.models.train import build_pipeline_with_options


def run_e12_fairness_audit(
    data_df: pd.DataFrame | None = None,
    tau: float = 0.1882,
    log_to_mlflow: bool = True,
    save_path: Path | str | None = None,
) -> dict[str, Any]:
    """Execute complete E12 fairness audit on out-of-fold calibrated predictions."""
    if data_df is None:
        train_df = pd.read_parquet(CFG["paths"]["processed_dir"] / "train.parquet")
        val_df = pd.read_parquet(CFG["paths"]["processed_dir"] / "val.parquet")
        df = pd.concat([train_df, val_df], ignore_index=True)
    else:
        df = data_df.copy()

    y = df["Churn"].values
    clv = 12.0 * df["MonthlyCharges"].values

    options = {
        "M0": {"desc": "Current features", "drop_cols": None},
        "M1": {"desc": "Drop gender", "drop_cols": ["gender"]},
        "M2": {"desc": "Drop gender + SeniorCitizen", "drop_cols": ["gender", "SeniorCitizen"]},
    }

    results = {}
    trade_off_rows = []

    for opt_name, spec in options.items():
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        oof_probs = np.zeros(len(df))

        for train_idx, val_idx in skf.split(df, y):
            X_tr = df.iloc[train_idx].drop(columns=["Churn"])
            y_tr = y[train_idx]
            X_va = df.iloc[val_idx].drop(columns=["Churn"])

            base_pipe = build_pipeline_with_options(
                model=LogisticRegression(
                    max_iter=1000, random_state=SEED, class_weight="balanced"
                ),
                include_engineered=True,
                scale_numeric=True,
                drop_cols=spec["drop_cols"],
            )
            cal = CalibratedClassifierCV(estimator=base_pipe, method="sigmoid", cv=3)
            cal.fit(X_tr, y_tr)
            oof_probs[val_idx] = cal.predict_proba(X_va)[:, 1]

        preds = (oof_probs >= tau).astype(int)

        # Financial metric
        prof_res = compute_profit_for_threshold(
            y_true=y, y_prob=oof_probs, threshold=tau, clv_values=clv
        )
        prof_1k = float(prof_res["profit_per_1k_customers_rm"])

        # Gender fairness
        m_mask = (df["gender"] == "Male").values
        f_mask = (df["gender"] == "Female").values
        rec_m = float(np.sum((preds == 1) & (y == 1) & m_mask) / np.sum((y == 1) & m_mask))
        rec_f = float(np.sum((preds == 1) & (y == 1) & f_mask) / np.sum((y == 1) & f_mask))
        prec_m = float(
            np.sum((preds == 1) & (y == 1) & m_mask) / max(np.sum((preds == 1) & m_mask), 1)
        )
        prec_f = float(
            np.sum((preds == 1) & (y == 1) & f_mask) / max(np.sum((preds == 1) & f_mask), 1)
        )
        cr_m = float(np.sum((preds == 1) & m_mask) / np.sum(m_mask))
        cr_f = float(np.sum((preds == 1) & f_mask) / np.sum(f_mask))
        gender_gap = abs(rec_m - rec_f)

        # Senior citizen fairness
        s_mask = (df["SeniorCitizen"] == 1).values
        ns_mask = (df["SeniorCitizen"] == 0).values
        rec_s = float(np.sum((preds == 1) & (y == 1) & s_mask) / np.sum((y == 1) & s_mask))
        rec_ns = float(np.sum((preds == 1) & (y == 1) & ns_mask) / np.sum((y == 1) & ns_mask))
        prec_s = float(
            np.sum((preds == 1) & (y == 1) & s_mask) / max(np.sum((preds == 1) & s_mask), 1)
        )
        prec_ns = float(
            np.sum((preds == 1) & (y == 1) & ns_mask) / max(np.sum((preds == 1) & ns_mask), 1)
        )
        cr_s = float(np.sum((preds == 1) & s_mask) / np.sum(s_mask))
        cr_ns = float(np.sum((preds == 1) & ns_mask) / np.sum(ns_mask))
        senior_gap = abs(rec_s - rec_ns)

        max_gap = max(gender_gap, senior_gap)

        results[opt_name] = {
            "description": spec["desc"],
            "profit_per_1k_rm": prof_1k,
            "gender": {
                "recall_male": rec_m,
                "recall_female": rec_f,
                "precision_male": prec_m,
                "precision_female": prec_f,
                "contact_rate_male": cr_m,
                "contact_rate_female": cr_f,
                "recall_gap": gender_gap,
            },
            "senior": {
                "recall_senior": rec_s,
                "recall_non_senior": rec_ns,
                "precision_senior": prec_s,
                "precision_non_senior": prec_ns,
                "contact_rate_senior": cr_s,
                "contact_rate_non_senior": cr_ns,
                "recall_gap": senior_gap,
            },
            "max_recall_gap": max_gap,
        }

        trade_off_rows.append(
            {
                "Option": opt_name,
                "Description": spec["desc"],
                "Profit/1k (RM)": f"RM{prof_1k:,.2f}",
                "Gender Recall Gap": f"{gender_gap:.4f}",
                "Senior Recall Gap": f"{senior_gap:.4f}",
                "Max Recall Gap": f"{max_gap:.4f}",
            }
        )

    # Selection rule: smallest max recall gap with profit loss at most 5% vs M0
    m0_prof = results["M0"]["profit_per_1k_rm"]
    valid_candidates = []
    for opt_name, opt_data in results.items():
        loss_pct = (m0_prof - opt_data["profit_per_1k_rm"]) / m0_prof
        if loss_pct <= 0.05:
            valid_candidates.append((opt_data["max_recall_gap"], opt_name))

    valid_candidates.sort()
    selected_option = valid_candidates[0][1]

    output_payload = {
        "experiment": "E12_Fairness_Audit",
        "threshold": tau,
        "selected_option": selected_option,
        "selected_details": results[selected_option],
        "all_options": results,
        "trade_off_table": trade_off_rows,
        "base_rates": {
            "female": float(np.mean(df.loc[df["gender"] == "Female", "Churn"])),
            "male": float(np.mean(df.loc[df["gender"] == "Male", "Churn"])),
            "senior": float(np.mean(df.loc[df["SeniorCitizen"] == 1, "Churn"])),
            "non_senior": float(np.mean(df.loc[df["SeniorCitizen"] == 0, "Churn"])),
        },
    }

    out_path = Path(save_path) if save_path else Path("reports") / "e12_fairness_audit.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    if log_to_mlflow:
        mlflow_cfg = CFG["mlflow"]
        mlflow.set_tracking_uri(str(mlflow_cfg["tracking_uri"]))
        mlflow.set_experiment(mlflow_cfg["experiment_name"])

        with mlflow.start_run(run_name="E12_Fairness_Audit"):
            mlflow.log_params({"threshold": tau, "selected_option": selected_option})
            mlflow.log_metrics(
                {
                    "selected_max_gap": results[selected_option]["max_recall_gap"],
                    "selected_profit_per_1k": results[selected_option]["profit_per_1k_rm"],
                }
            )
            mlflow.log_artifact(str(out_path))

    return output_payload


if __name__ == "__main__":
    res = run_e12_fairness_audit()
    print("=" * 60)
    print("E12 Fairness Audit Trade-Off Table")
    print("=" * 60)
    print(pd.DataFrame(res["trade_off_table"]).to_string(index=False))
    print(f"\nSelected Option: {res['selected_option']}")
    print(f"Max Recall Gap:  {res['selected_details']['max_recall_gap']:.4f}")
