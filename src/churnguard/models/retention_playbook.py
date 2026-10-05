"""Dynamic CLV and Prescriptive Retention Playbooks for ChurnGuard (v2.0 Architecture).

Implements:
1. Dynamic survival-discounted CLV estimation based on customer tenure and contract lock-in.
2. Prescriptive frontline intervention recommender mapping risk drivers to targeted deals.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def compute_dynamic_clv(
    monthly_charges: float | np.ndarray,
    tenure: int | np.ndarray,
    contract: str | np.ndarray,
    base_horizon_months: int = 18,
    discount_rate_monthly: float = 0.008,
) -> float | np.ndarray:
    """Calculate survival-discounted Customer Lifetime Value (RM).

    Customers on longer contracts and with established tenure experience lower
    hazard rates, preserving greater expected remaining lifetime value.

    Args:
        monthly_charges: Customer monthly bill amount in RM.
        tenure: Existing customer tenure in months.
        contract: 'Month-to-month', 'One year', or 'Two year'.
        base_horizon_months: Forecast window (default: 18 months).
        discount_rate_monthly: Monthly financial discount factor.

    Returns:
        Expected preserved CLV in RM.
    """
    mc = np.asarray(monthly_charges, dtype=float)
    t = np.asarray(tenure, dtype=float)

    # Base expected retention multiplier in months
    if isinstance(contract, pd.Series | np.ndarray | list):
        c_arr = np.asarray(contract, dtype=str)
        multiplier = np.where(
            c_arr == "Two year",
            22.0,
            np.where(c_arr == "One year", 15.0, 9.0),
        )
    else:
        if contract == "Two year":
            multiplier = 22.0
        elif contract == "One year":
            multiplier = 15.0
        else:
            multiplier = 9.0

    # Tenure stability boost (mature customers stay longer)
    tenure_factor = 1.0 + np.minimum(t / 72.0, 0.5)
    effective_months = np.minimum(multiplier * tenure_factor, 24.0)

    # Discounted cash flow multiplier
    discounted_months = effective_months / (1.0 + (discount_rate_monthly * effective_months * 0.5))

    clv = mc * discounted_months
    return float(clv) if np.ndim(clv) == 0 else clv


def recommend_retention_playbook(
    customer: dict[str, Any] | pd.Series,
    churn_prob: float,
) -> dict[str, Any]:
    """Recommend optimal frontline retention intervention based on customer risk drivers.

    Args:
        customer: Dictionary or Series of customer features.
        churn_prob: Model calibrated churn probability.

    Returns:
        Structured recommendation dictionary.
    """
    c = dict(customer)
    monthly_charges = float(c.get("MonthlyCharges", 65.0))
    contract = str(c.get("Contract", "Month-to-month"))
    internet = str(c.get("InternetService", "DSL"))
    tech_support = str(c.get("TechSupport", "No"))
    online_security = str(c.get("OnlineSecurity", "No"))
    payment_method = str(c.get("PaymentMethod", "Electronic check"))

    # Determine risk category
    if churn_prob >= 0.35:
        urgency = "High Priority (Immediate Action)"
    elif churn_prob >= 0.1882:
        urgency = "Medium Priority (Next Billing Cycle)"
    else:
        urgency = "Low Priority (Monitor)"

    # Identify primary retention playbook
    if contract == "Month-to-month" and monthly_charges >= 60.0:
        offer_title = "Contract Commitment Upgrade"
        offer_details = (
            f"Offer 15% discount (RM{monthly_charges * 0.85:.2f}/mo) on a 1-year contract lock-in."
        )
        estimated_churn_reduction = 0.35
        action_cost_rm = 50.0
    elif internet == "Fiber optic" and (tech_support == "No" or online_security == "No"):
        offer_title = "Fiber Care Service Bundle"
        offer_details = "Provide free 6-month TechSupport and OnlineSecurity add-on package."
        estimated_churn_reduction = 0.28
        action_cost_rm = 40.0
    elif payment_method == "Electronic check":
        offer_title = "Auto-Pay Migration Incentive"
        offer_details = "Grant RM30 statement credit upon enabling Credit Card or Bank Auto-Pay."
        estimated_churn_reduction = 0.20
        action_cost_rm = 30.0
    else:
        offer_title = "Loyalty Appreciation Check-in"
        offer_details = (
            "Senior frontline agent call for service review and custom plan rightsizing."
        )
        estimated_churn_reduction = 0.15
        action_cost_rm = 25.0

    return {
        "churn_probability": float(round(churn_prob, 4)),
        "urgency": urgency,
        "primary_playbook": offer_title,
        "action_details": offer_details,
        "estimated_action_cost_rm": action_cost_rm,
        "estimated_churn_reduction_pct": int(estimated_churn_reduction * 100),
    }
