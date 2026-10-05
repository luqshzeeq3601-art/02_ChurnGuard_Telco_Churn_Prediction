"""FastAPI production service for ChurnGuard churn prediction (T5.2).

Endpoints:
- GET /health: Health check endpoint.
- GET /model-info: Model metadata, optimal threshold, risk tiers, and test performance.
- POST /predict: Single-customer prediction with calibrated probability and top-3 SHAP reasons.
- POST /predict/batch: Batch prediction returning ranked customers and risk summary.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    CustomerInput,
    CustomerPrediction,
    HealthResponse,
    ModelInfoResponse,
)
from churnguard.models.predict import ChurnPredictor

# Global predictor instance
predictor: ChurnPredictor | None = None


def get_predictor() -> ChurnPredictor:
    """Retrieve global ChurnPredictor instance, initializing if necessary."""
    global predictor
    if predictor is None:
        predictor = ChurnPredictor()
    return predictor


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager to warm up predictor on startup."""
    try:
        get_predictor()
    except Exception as exc:
        print(f"Warning: Model could not be preloaded on startup: {exc}")
    yield


app = FastAPI(
    title="ChurnGuard API",
    description=(
        "Production REST API for telco customer churn propensity scoring, "
        "profit-optimal budget targeting, and frontline SHAP reason codes."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for frontend dashboard and cross-origin clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check",
    status_code=status.HTTP_200_OK,
)
def health_check() -> HealthResponse:
    """Return API health status and loaded model version."""
    pred = get_predictor()
    return HealthResponse(
        status="ok",
        timestamp=datetime.utcnow().isoformat() + "Z",
        model_version=pred.model_version,
    )


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
    tags=["System"],
    summary="Model metadata and performance",
    status_code=status.HTTP_200_OK,
)
def get_model_info() -> ModelInfoResponse:
    """Return model version, optimal decision threshold, risk tiers, and test performance."""
    pred = get_predictor()
    meta = pred.meta
    return ModelInfoResponse(
        model_name=meta.get("model_name", "churnguard-champion"),
        model_version=meta.get("model_version", "1.0.0"),
        model_class=meta.get("model_class", "CalibratedClassifierCV(LGBMClassifier)"),
        created_at=meta.get("created_at", datetime.utcnow().isoformat() + "Z"),
        optimal_threshold=float(meta.get("optimal_threshold", 0.18)),
        risk_tiers=meta.get("risk_tiers", {}),
        features=meta.get("features", {}),
        cost_parameters=meta.get("cost_parameters", {}),
        test_performance=meta.get("test_performance", {}),
    )


@app.post(
    "/predict",
    response_model=CustomerPrediction,
    tags=["Predictions"],
    summary="Score a single customer",
    status_code=status.HTTP_200_OK,
)
def predict_single_customer(customer: CustomerInput) -> CustomerPrediction:
    """Generate calibrated churn probability, risk tier, and top 3 plain-language reasons."""
    pred = get_predictor()
    try:
        raw_dict = customer.model_dump()
        result = pred.predict_single(raw_dict)
        return CustomerPrediction(**result)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(exc)}",
        ) from exc


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Predictions"],
    summary="Score a batch of customers",
    status_code=status.HTTP_200_OK,
)
def predict_batch_customers(payload: BatchPredictionRequest) -> BatchPredictionResponse:
    """Score multiple customers, return ranked predictions descending by risk with tier counts."""
    pred = get_predictor()
    if len(payload.customers) == 0:
        return BatchPredictionResponse(
            total_customers=0,
            high_risk_count=0,
            medium_risk_count=0,
            low_risk_count=0,
            predictions=[],
        )

    try:
        records = [c.model_dump() for c in payload.customers]
        scored_df = pred.predict_batch(records, include_reasons=True)

        pred_objects = []
        for _, row in scored_df.iterrows():
            pred_objects.append(
                CustomerPrediction(
                    customerID=str(row["customerID"]),
                    churn_probability=float(row["churn_probability"]),
                    risk_tier=str(row["risk_tier"]),
                    top_reasons=list(row["top_reasons"]),
                    model_version=str(row["model_version"]),
                )
            )

        tier_counts = scored_df["risk_tier"].value_counts().to_dict()

        return BatchPredictionResponse(
            total_customers=len(scored_df),
            high_risk_count=int(tier_counts.get("High", 0)),
            medium_risk_count=int(tier_counts.get("Medium", 0)),
            low_risk_count=int(tier_counts.get("Low", 0)),
            predictions=pred_objects,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference error: {str(exc)}",
        ) from exc
