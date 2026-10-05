"""Evidently AI Data & Prediction Drift Monitoring Engine for ChurnGuard.

Adheres to docs/04_TECHNICAL_DESIGN.md section 10 and PRD EO5:
- Reference dataset: Training split (data/processed/train.parquet)
- Current dataset: Production/simulated batch (data/processed/drifted_batch.parquet)
- Produces full interactive HTML report at reports/drift/drift_report.html
- Extracts JSON summary and evaluates retrain trigger rules (drift share > 30%).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.report import Report

from churnguard.config import CFG
from churnguard.models.predict import ChurnPredictor


def run_drift_analysis(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    predictor: ChurnPredictor | None = None,
    output_html_path: Path | None = None,
    output_json_path: Path | None = None,
) -> tuple[dict[str, Any], Path, Path]:
    """Execute data and prediction drift analysis using Evidently.

    Args:
        reference_df: Baseline reference dataframe (training split).
        current_df: Current production batch dataframe to monitor.
        predictor: Optional pre-loaded ChurnPredictor instance.
        output_html_path: Filepath to save interactive HTML report.
        output_json_path: Filepath to save structured JSON summary.

    Returns:
        Tuple of (drift_summary_dict, html_report_path, json_summary_path).
    """
    reports_dir = Path(CFG["paths"]["reports_dir"]) / "drift"
    reports_dir.mkdir(parents=True, exist_ok=True)

    html_path = output_html_path or (reports_dir / "drift_report.html")
    json_path = output_json_path or (reports_dir / "drift_summary.json")

    # Load predictor if not provided
    if predictor is None:
        predictor = ChurnPredictor()

    ref = reference_df.copy()
    curr = current_df.copy()

    # Generate model prediction probabilities for prediction drift tracking
    ref["prediction"] = predictor.model.predict_proba(ref)[:, 1]
    curr["prediction"] = predictor.model.predict_proba(curr)[:, 1]

    # Setup column mapping
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    exclude_cols = {"customerID", "Churn", "prediction"} | set(num_cols)
    cat_cols = [c for c in ref.columns if c not in exclude_cols]

    col_mapping = ColumnMapping(
        target="Churn" if "Churn" in ref.columns and "Churn" in curr.columns else None,
        prediction="prediction",
        id="customerID" if "customerID" in ref.columns else None,
        numerical_features=num_cols,
        categorical_features=cat_cols,
    )

    # Initialize Evidently Report
    presets = [DataDriftPreset(), TargetDriftPreset()]
    report = Report(metrics=presets)
    report.run(reference_data=ref, current_data=curr, column_mapping=col_mapping)

    # Save HTML report
    report.save_html(str(html_path))

    # Parse metrics dictionary
    raw_dict = report.as_dict()
    metrics_list = raw_dict.get("metrics", [])

    # Extract Data Drift Metrics
    data_drift_res = {}
    for m in metrics_list:
        metric_name = m.get("metric", "")
        if "DataDriftTable" in metric_name or "DataDrift" in metric_name:
            data_drift_res = m.get("result", {})

    share_drifted = float(data_drift_res.get("share_of_drifted_columns", 0.0))
    n_drifted = int(data_drift_res.get("number_of_drifted_columns", 0))
    n_cols = int(data_drift_res.get("number_of_columns", len(num_cols) + len(cat_cols)))
    dataset_drift = bool(data_drift_res.get("dataset_drift", False))

    drift_by_columns = data_drift_res.get("drift_by_columns", {})
    drifted_feature_names: list[str] = [
        col for col, details in drift_by_columns.items() if details.get("drift_detected", False)
    ]

    # Retrain trigger rule: drift share > 30% or dataset drift flag
    retrain_recommended = (share_drifted >= 0.30) or dataset_drift

    summary: dict[str, Any] = {
        "dataset_drift_detected": dataset_drift,
        "share_of_drifted_columns": round(share_drifted, 4),
        "number_of_drifted_columns": n_drifted,
        "total_columns_analyzed": n_cols,
        "drifted_features": drifted_feature_names,
        "retrain_recommended": retrain_recommended,
        "retrain_rule_description": "Trigger retrain if share of drifted features >= 30% or dataset drift detected.",
        "reference_rows": len(ref),
        "current_rows": len(curr),
        "prediction_drift": {
            "drift_detected": drift_by_columns.get("prediction", {}).get("drift_detected", False),
            "drift_score": drift_by_columns.get("prediction", {}).get("drift_score"),
            "stat_test": drift_by_columns.get("prediction", {}).get("stat_test_name"),
        }
        if "prediction" in drift_by_columns
        else {},
    }

    # Save JSON summary
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary, html_path, json_path


def main(
    reference_path: Path | None = None,
    current_path: Path | None = None,
    html_out: Path | None = None,
    json_out: Path | None = None,
) -> None:
    """CLI runner to generate and log drift reports."""
    ref_p = reference_path or (CFG["paths"]["processed_dir"] / "train.parquet")
    curr_p = current_path or (CFG["paths"]["processed_dir"] / "drifted_batch.parquet")

    if not curr_p.exists():
        curr_p = CFG["paths"]["processed_dir"] / "test.parquet"

    print("Running drift analysis:")
    print(f" - Reference dataset: {ref_p}")
    print(f" - Current dataset:   {curr_p}")

    ref_df = pd.read_parquet(ref_p) if ref_p.suffix == ".parquet" else pd.read_csv(ref_p)
    curr_df = pd.read_parquet(curr_p) if curr_p.suffix == ".parquet" else pd.read_csv(curr_p)

    summary, html_path, json_path = run_drift_analysis(
        reference_df=ref_df,
        current_df=curr_df,
        output_html_path=html_out,
        output_json_path=json_out,
    )

    print("\n=== Evidently Drift Report Generated ===")
    print(f"Dataset Drift Detected:     {summary['dataset_drift_detected']}")
    print(f"Share of Drifted Columns:   {summary['share_of_drifted_columns']:.1%}")
    print(
        f"Drifted Features ({summary['number_of_drifted_columns']}/{summary['total_columns_analyzed']}): {summary['drifted_features']}"
    )
    print(f"Retrain Recommended:        {summary['retrain_recommended']}")
    print(f"HTML Report:                {html_path}")
    print(f"JSON Summary:               {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Evidently drift report.")
    parser.add_argument("--reference", type=Path, default=None, help="Reference dataset path.")
    parser.add_argument("--current", type=Path, default=None, help="Current dataset path.")
    parser.add_argument("--html-out", type=Path, default=None, help="Output HTML path.")
    parser.add_argument("--json-out", type=Path, default=None, help="Output JSON path.")
    args = parser.parse_args()

    main(
        reference_path=args.reference,
        current_path=args.current,
        html_out=args.html_out,
        json_out=args.json_out,
    )
