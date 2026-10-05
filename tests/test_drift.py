"""Unit tests for Evidently drift monitoring module."""

import pandas as pd

from churnguard.monitoring.drift import run_drift_analysis


def test_run_drift_analysis(tmp_path):
    """Verify drift analysis generates HTML report and structured summary."""
    train_df = pd.DataFrame(
        {
            "customerID": ["C1", "C2", "C3", "C4", "C5", "C6"],
            "gender": ["Male", "Female", "Male", "Female", "Male", "Female"],
            "SeniorCitizen": [0, 0, 1, 0, 1, 0],
            "Partner": ["Yes", "No", "Yes", "No", "Yes", "No"],
            "Dependents": ["No", "No", "Yes", "No", "No", "Yes"],
            "tenure": [12, 24, 6, 48, 1, 36],
            "PhoneService": ["Yes", "Yes", "Yes", "No", "Yes", "Yes"],
            "MultipleLines": ["No", "Yes", "No", "No phone service", "Yes", "No"],
            "InternetService": ["DSL", "Fiber optic", "DSL", "DSL", "Fiber optic", "No"],
            "OnlineSecurity": ["Yes", "No", "No", "Yes", "No", "No internet service"],
            "OnlineBackup": ["No", "Yes", "No", "No", "No", "No internet service"],
            "DeviceProtection": ["No", "Yes", "No", "Yes", "No", "No internet service"],
            "TechSupport": ["Yes", "No", "No", "Yes", "No", "No internet service"],
            "StreamingTV": ["No", "Yes", "No", "No", "Yes", "No internet service"],
            "StreamingMovies": ["No", "Yes", "No", "No", "Yes", "No internet service"],
            "Contract": [
                "Month-to-month",
                "One year",
                "Month-to-month",
                "Two year",
                "Month-to-month",
                "Two year",
            ],
            "PaperlessBilling": ["Yes", "No", "Yes", "No", "Yes", "No"],
            "PaymentMethod": [
                "Electronic check",
                "Bank transfer (automatic)",
                "Electronic check",
                "Mailed check",
                "Electronic check",
                "Credit card (automatic)",
            ],
            "MonthlyCharges": [29.85, 89.10, 53.85, 42.30, 95.00, 20.25],
            "TotalCharges": [29.85, 2138.40, 323.10, 2030.40, 95.00, 729.00],
            "Churn": [0, 0, 1, 0, 1, 0],
        }
    )

    current_df = train_df.copy()
    # Apply synthetic drift to current_df
    current_df["MonthlyCharges"] = current_df["MonthlyCharges"] * 1.50
    current_df["Contract"] = "Month-to-month"

    html_file = tmp_path / "drift_report.html"
    json_file = tmp_path / "drift_summary.json"

    summary, h_out, j_out = run_drift_analysis(
        reference_df=train_df,
        current_df=current_df,
        output_html_path=html_file,
        output_json_path=json_file,
    )

    assert h_out.exists()
    assert j_out.exists()
    assert h_out.stat().st_size > 0
    assert "dataset_drift_detected" in summary
    assert "share_of_drifted_columns" in summary
    assert "drifted_features" in summary
    assert summary["reference_rows"] == 6
    assert summary["current_rows"] == 6
