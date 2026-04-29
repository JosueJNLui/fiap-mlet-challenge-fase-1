from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Request

from ..application.predictor import ChurnPredictor
from .dependencies import get_predictor
from .schemas import HealthResponse, PredictRequest, PredictResponse

api_router = APIRouter()


@api_router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok",
        timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    )


@api_router.post("/predict", response_model=PredictResponse)
async def predict(
    payload: PredictRequest,
    request: Request,
    predictor: ChurnPredictor = Depends(get_predictor),
) -> PredictResponse:
    result = predictor.predict(payload.model_dump())
    return PredictResponse(
        churn_probability=result.probability,
        prediction=result.label,
        threshold=result.threshold,
        model_version=predictor.version,
        request_id=getattr(request.state, "request_id", ""),
    )
