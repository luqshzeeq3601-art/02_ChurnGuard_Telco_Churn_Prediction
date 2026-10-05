"""Unit tests for dynamic CLV and retention playbooks."""

import numpy as np

from churnguard.models.retention_playbook import (
    compute_dynamic_clv,
    recommend_retention_playbook,
)


def test_compute_dynamic_clv_scalar_and_array():
    # Two year contract should yield higher remaining CLV than Month-to-month
    clv_2yr = compute_dynamic_clv(monthly_charges=100.0, tenure=24, contract="Two year")
    clv_m2m = compute_dynamic_clv(monthly_charges=100.0, tenure=2, contract="Month-to-month")

    assert clv_2yr > clv_m2m
    assert clv_2yr > 1000.0

    # Vectorized test
    mc = np.array([50.0, 80.0])
    t = np.array([5, 30])
    c = np.array(["Month-to-month", "One year"])
    clvs = compute_dynamic_clv(mc, t, c)
    assert len(clvs) == 2
    assert clvs[1] > clvs[0]


def test_recommend_retention_playbook():
    # Month to month high charge customer
    cust_m2m = {
        "MonthlyCharges": 85.0,
        "Contract": "Month-to-month",
        "InternetService": "DSL",
        "PaymentMethod": "Credit card (automatic)",
    }
    rec1 = recommend_retention_playbook(cust_m2m, churn_prob=0.55)
    assert rec1["urgency"] == "High Priority (Immediate Action)"
    assert rec1["primary_playbook"] == "Contract Commitment Upgrade"

    # Fiber optic customer without support
    cust_fiber = {
        "MonthlyCharges": 70.0,
        "Contract": "One year",
        "InternetService": "Fiber optic",
        "TechSupport": "No",
        "OnlineSecurity": "No",
        "PaymentMethod": "Bank transfer (automatic)",
    }
    rec2 = recommend_retention_playbook(cust_fiber, churn_prob=0.25)
    assert rec2["primary_playbook"] == "Fiber Care Service Bundle"

    # Electronic check user
    cust_echeck = {
        "MonthlyCharges": 40.0,
        "Contract": "One year",
        "InternetService": "DSL",
        "PaymentMethod": "Electronic check",
    }
    rec3 = recommend_retention_playbook(cust_echeck, churn_prob=0.10)
    assert rec3["primary_playbook"] == "Auto-Pay Migration Incentive"
