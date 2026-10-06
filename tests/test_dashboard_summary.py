"""Dashboard facts follow saved metadata/evaluation rather than fixed old values."""

import json

from churnguard.dashboard import load_dashboard_summary


def test_dashboard_summary_uses_actual_threshold_and_report(tmp_path):
    metadata = tmp_path / "meta.json"
    report = tmp_path / "evaluation.json"
    metadata.write_text(json.dumps({"optimal_threshold": 0.42, "model_version": "test"}))
    report.write_text(
        json.dumps(
            {
                "optimal_threshold": 0.42,
                "metrics": {
                    "roc_auc": {"point_estimate": 0.91},
                    "pr_auc": {"point_estimate": 0.76},
                },
            }
        )
    )
    summary = load_dashboard_summary(metadata, report)
    assert summary["optimal_threshold"] == 0.42
    assert summary["evaluation_metrics"]["roc_auc"]["point_estimate"] == 0.91
    assert summary["evaluation_metrics"]["pr_auc"]["point_estimate"] == 0.76
    report.write_text(
        json.dumps({"optimal_threshold": 0.18, "metrics": {"roc_auc": {"point_estimate": 0.99}}})
    )
    assert load_dashboard_summary(metadata, report)["evaluation_metrics"] == {}


def test_dashboard_summary_does_not_fabricate_missing_metrics(tmp_path):
    metadata = tmp_path / "meta.json"
    metadata.write_text(json.dumps({"optimal_threshold": 0.42}))
    assert load_dashboard_summary(metadata, tmp_path / "missing.json")["evaluation_metrics"] == {}
