"""Simulate macroeconomic and behavioral data drift on customer batches.

Adheres to docs/04_TECHNICAL_DESIGN.md section 10:
- Reference data: training split
- Current data: simulated batch with injected shifts (e.g. +20-25% MonthlyCharges,
  shift to month-to-month contracts, higher fiber adoption).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from churnguard.config import CFG


def simulate_drift(
    df: pd.DataFrame,
    seed: int = 42,
    price_inflation: float = 1.25,
    contract_shift_prob: float = 0.35,
    fiber_shift_prob: float = 0.25,
    electronic_check_shift_prob: float = 0.20,
    tenure_shift_prob: float = 0.30,
) -> pd.DataFrame:
    """Generate a realistic drifted customer batch.

    Simulates macroeconomic pressure, post-promotional contract churn,
    and inflation:
    1. MonthlyCharges inflated by `price_inflation` factor.
    2. Long-term contracts shift into Month-to-month.
    3. DSL users migrate to Fiber optic.
    4. Bank/Credit card autopay users shift to Electronic check.
    5. Tenure shortened for a proportion of users (influx of new users).

    Args:
        df: Input DataFrame (e.g., test or validation split).
        seed: Random seed for reproducibility.
        price_inflation: Multiplier on MonthlyCharges.
        contract_shift_prob: Probability of shifting 1/2-year contracts to Month-to-month.
        fiber_shift_prob: Probability of shifting DSL to Fiber optic.
        electronic_check_shift_prob: Probability of shifting auto-pay to Electronic check.
        tenure_shift_prob: Probability of compressing tenure.

    Returns:
        Drifted pandas DataFrame copy.
    """
    rng = np.random.default_rng(seed)
    drifted = df.copy()

    # 1. Price Inflation: scale MonthlyCharges
    if "MonthlyCharges" in drifted.columns:
        noise = rng.normal(1.0, 0.05, size=len(drifted))
        drifted["MonthlyCharges"] = np.round(
            drifted["MonthlyCharges"] * price_inflation * noise, 2
        )

    # 2. Contract Drift: shift 1-year and 2-year contracts to Month-to-month
    if "Contract" in drifted.columns:
        mask_contract = (drifted["Contract"] != "Month-to-month") & (
            rng.random(len(drifted)) < contract_shift_prob
        )
        drifted.loc[mask_contract, "Contract"] = "Month-to-month"

    # 3. Fiber Optic Adoption Drift: DSL users upgrade to Fiber optic
    if "InternetService" in drifted.columns:
        mask_dsl = (drifted["InternetService"] == "DSL") & (
            rng.random(len(drifted)) < fiber_shift_prob
        )
        drifted.loc[mask_dsl, "InternetService"] = "Fiber optic"

    # 4. Payment Method Drift: automated payment users switch to Electronic check
    if "PaymentMethod" in drifted.columns:
        mask_pay = (drifted["PaymentMethod"] != "Electronic check") & (
            rng.random(len(drifted)) < electronic_check_shift_prob
        )
        drifted.loc[mask_pay, "PaymentMethod"] = "Electronic check"

    # 5. Tenure Drift: simulate higher proportion of recent signups (tenure <= 6)
    if "tenure" in drifted.columns:
        mask_tenure = (drifted["tenure"] > 6) & (rng.random(len(drifted)) < tenure_shift_prob)
        drifted.loc[mask_tenure, "tenure"] = rng.integers(
            1, 7, size=mask_tenure.sum(), dtype=np.int64
        ).astype(drifted["tenure"].dtype)

    # 6. Recalculate TotalCharges consistently
    if (
        "TotalCharges" in drifted.columns
        and "MonthlyCharges" in drifted.columns
        and "tenure" in drifted.columns
    ):
        # Approximate updated TotalCharges based on revised tenure and monthly charges
        drifted["TotalCharges"] = np.round(
            np.maximum(
                0.0, drifted["MonthlyCharges"] * drifted["tenure"] + rng.normal(0, 5, len(drifted))
            ),
            2,
        )

    return drifted


def generate_and_save_drifted_batch(
    source_path: Path | None = None,
    output_dir: Path | None = None,
    seed: int = 42,
) -> tuple[Path, Path]:
    """Load test dataset, apply drift transformations, and save output files."""
    if source_path is None:
        source_path = CFG["paths"]["processed_dir"] / "test.parquet"
        if not source_path.exists():
            source_path = CFG["paths"]["processed_dir"] / "test.csv"

    if output_dir is None:
        output_dir = CFG["paths"]["processed_dir"]

    output_dir.mkdir(parents=True, exist_ok=True)

    if source_path.suffix == ".parquet":
        df = pd.read_parquet(source_path)
    else:
        df = pd.read_csv(source_path)

    drifted_df = simulate_drift(df, seed=seed)

    parquet_out = output_dir / "drifted_batch.parquet"
    csv_out = output_dir / "drifted_batch.csv"

    drifted_df.to_parquet(parquet_out, index=False)
    drifted_df.to_csv(csv_out, index=False)

    print(f"Saved drifted batch ({len(drifted_df)} rows) to:")
    print(f" - {parquet_out}")
    print(f" - {csv_out}")

    return parquet_out, csv_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate drifted customer data batch.")
    parser.add_argument("--source", type=Path, default=None, help="Path to input dataset.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    generate_and_save_drifted_batch(source_path=args.source, seed=args.seed)
