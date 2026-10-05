"""Pydantic schemas for ChurnGuard FastAPI service (T5.2).

Enforces data contracts for input validation, predictions, batch scoring, and health checks.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class CustomerInput(BaseModel):
    """Input payload for a single customer churn prediction."""

    customerID: str = Field(
        default="UNKNOWN", description="Unique customer identifier", examples=["7590-VHVEG"]
    )
    gender: Literal["Male", "Female"] = Field(description="Customer gender", examples=["Female"])
    SeniorCitizen: Literal[0, 1] = Field(
        description="Whether customer is a senior citizen (1) or not (0)", examples=[0]
    )
    Partner: Literal["Yes", "No"] = Field(
        description="Whether customer has a partner", examples=["Yes"]
    )
    Dependents: Literal["Yes", "No"] = Field(
        description="Whether customer has dependents", examples=["No"]
    )
    tenure: int = Field(
        ge=0, le=100, description="Number of months customer has stayed with company", examples=[1]
    )
    PhoneService: Literal["Yes", "No"] = Field(
        description="Whether customer has phone service", examples=["No"]
    )
    MultipleLines: Literal["Yes", "No", "No phone service"] = Field(
        description="Whether customer has multiple lines", examples=["No phone service"]
    )
    InternetService: Literal["DSL", "Fiber optic", "No"] = Field(
        description="Customer's internet service provider", examples=["DSL"]
    )
    OnlineSecurity: Literal["Yes", "No", "No internet service"] = Field(
        description="Whether customer has online security", examples=["No"]
    )
    OnlineBackup: Literal["Yes", "No", "No internet service"] = Field(
        description="Whether customer has online backup", examples=["Yes"]
    )
    DeviceProtection: Literal["Yes", "No", "No internet service"] = Field(
        description="Whether customer has device protection", examples=["No"]
    )
    TechSupport: Literal["Yes", "No", "No internet service"] = Field(
        description="Whether customer has tech support", examples=["No"]
    )
    StreamingTV: Literal["Yes", "No", "No internet service"] = Field(
        description="Whether customer has streaming TV", examples=["No"]
    )
    StreamingMovies: Literal["Yes", "No", "No internet service"] = Field(
        description="Whether customer has streaming movies", examples=["No"]
    )
    Contract: Literal["Month-to-month", "One year", "Two year"] = Field(
        description="Contract term of the customer", examples=["Month-to-month"]
    )
    PaperlessBilling: Literal["Yes", "No"] = Field(
        description="Whether customer has paperless billing", examples=["Yes"]
    )
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ] = Field(description="Customer payment method", examples=["Electronic check"])
    MonthlyCharges: float = Field(
        ge=0.0, description="Amount charged to customer monthly in RM", examples=[29.85]
    )
    TotalCharges: float = Field(
        ge=0.0, description="Total amount charged to customer in RM", examples=[29.85]
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "customerID": "7590-VHVEG",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "No",
                "MultipleLines": "No phone service",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85,
                "TotalCharges": 29.85,
            }
        }
    }


class CustomerPrediction(BaseModel):
    """Response payload for a single customer churn prediction."""

    customerID: str = Field(description="Unique customer identifier", examples=["7590-VHVEG"])
    churn_probability: float = Field(
        ge=0.0, le=1.0, description="Calibrated probability of customer churn", examples=[0.7123]
    )
    risk_tier: Literal["High", "Medium", "Low"] = Field(
        description="Categorical risk tier based on profit-optimal threshold", examples=["High"]
    )
    top_reasons: list[str] = Field(
        description="Top 3 plain-language explanations for frontline agents",
        examples=[
            "Month-to-month contract increases cancellation flexibility",
            "Early lifecycle risk: Short tenure (1 month)",
            "Payment via Electronic Check (high manual friction)",
        ],
    )
    model_version: str = Field(description="Active production model version", examples=["1.0.0"])


class BatchPredictionRequest(BaseModel):
    """Request payload for batch scoring multiple customers."""

    customers: list[CustomerInput] = Field(description="List of customer records to score")


class BatchPredictionResponse(BaseModel):
    """Response payload for batch prediction with rankings and summary counts."""

    total_customers: int = Field(description="Total number of customers scored", examples=[100])
    high_risk_count: int = Field(description="Count of High Risk customers", examples=[40])
    medium_risk_count: int = Field(description="Count of Medium Risk customers", examples=[25])
    low_risk_count: int = Field(description="Count of Low Risk customers", examples=[35])
    predictions: list[CustomerPrediction] = Field(
        description="Ranked list of scored customers (highest risk first)"
    )


class HealthResponse(BaseModel):
    """Service health check response."""

    status: str = Field(default="ok", examples=["ok"])
    timestamp: str = Field(description="Current UTC timestamp")
    model_version: str = Field(description="Loaded model version", examples=["1.0.0"])


class ModelInfoResponse(BaseModel):
    """Metadata response describing active champion model and business benchmarks."""

    model_name: str = Field(examples=["churnguard-champion"])
    model_version: str = Field(examples=["1.0.0"])
    model_class: str = Field(
        examples=["CalibratedClassifierCV(LGBMClassifier, method='isotonic')"]
    )
    created_at: str = Field(description="Model training timestamp")
    optimal_threshold: float = Field(examples=[0.18])
    risk_tiers: dict[str, Any] = Field(description="Configured risk tier threshold boundaries")
    features: dict[str, Any] = Field(description="Numeric and categorical feature lists")
    cost_parameters: dict[str, Any] = Field(
        description="Business cost & revenue assumptions in RM"
    )
    test_performance: dict[str, Any] = Field(
        description="Held-out test set metrics with 95% bootstrap CIs"
    )
    operational_summary: dict[str, Any] | None = Field(
        default=None, description="Test operational results (contact rate, capture rate, profit)"
    )
    targeting_strategies: dict[str, Any] | None = Field(
        default=None,
        description="Retention campaign budget targeting strategies (optimal, top30, top20)",
    )
