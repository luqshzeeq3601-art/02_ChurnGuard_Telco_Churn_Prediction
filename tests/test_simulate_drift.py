"""Unit tests for drift simulation module."""

import pandas as pd

from churnguard.monitoring.simulate_drift import generate_and_save_drifted_batch, simulate_drift


def test_simulate_drift_modifications():
    """Verify simulate_drift introduces expected distributional changes."""
    df = pd.DataFrame(
        {
            "customerID": ["1", "2", "3", "4", "5"],
            "MonthlyCharges": [50.0, 70.0, 90.0, 60.0, 80.0],
            "TotalCharges": [500.0, 700.0, 900.0, 600.0, 800.0],
            "tenure": [10, 24, 36, 12, 48],
            "Contract": ["One year", "Two year", "Month-to-month", "One year", "Two year"],
            "InternetService": ["DSL", "DSL", "Fiber optic", "No", "DSL"],
            "PaymentMethod": [
                "Bank transfer (automatic)",
                "Credit card (automatic)",
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
            ],
            "Churn": [0, 0, 1, 0, 0],
        }
    )

    drifted = simulate_drift(
        df,
        seed=42,
        price_inflation=1.30,
        contract_shift_prob=1.0,  # force all 1/2-yr to month-to-month
        fiber_shift_prob=1.0,  # force DSL to Fiber optic
        electronic_check_shift_prob=1.0,
        tenure_shift_prob=1.0,
    )

    # 1. Price inflation
    assert drifted["MonthlyCharges"].mean() > df["MonthlyCharges"].mean()

    # 2. All 1/2 yr became Month-to-month
    assert (drifted["Contract"] == "Month-to-month").all()

    # 3. DSL shifted to Fiber optic
    assert (drifted["InternetService"] != "DSL").all()

    # 4. Electronic check shifted
    assert (drifted["PaymentMethod"] == "Electronic check").all()

    # 5. Tenure shortened
    assert (drifted["tenure"] <= 6).all()


def test_generate_and_save_drifted_batch(tmp_path):
    """Test loading and saving drifted batch to disk."""
    sample_df = pd.DataFrame(
        {
            "customerID": ["A", "B"],
            "MonthlyCharges": [40.0, 80.0],
            "TotalCharges": [100.0, 800.0],
            "tenure": [2, 10],
            "Contract": ["Month-to-month", "One year"],
            "InternetService": ["DSL", "Fiber optic"],
            "PaymentMethod": ["Electronic check", "Mailed check"],
            "Churn": [0, 1],
        }
    )
    src_file = tmp_path / "test.parquet"
    sample_df.to_parquet(src_file, index=False)

    out_p, out_c = generate_and_save_drifted_batch(
        source_path=src_file, output_dir=tmp_path, seed=42
    )
    assert out_p.exists()
    assert out_c.exists()
    loaded = pd.read_parquet(out_p)
    assert len(loaded) == 2
