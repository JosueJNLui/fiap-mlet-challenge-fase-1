from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Request, status

from ..application.predictor import ChurnPredictor
from .dependencies import get_predictor
from .schemas import (
    HealthResponse,
    PredictRequest,
    PredictResponse,
    ServiceUnavailableResponse,
    ValidationErrorResponse,
)

api_router = APIRouter()


@api_router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    tags=["Health"],
    summary="Liveness/readiness probe",
    description=(
        "Retorna `200 OK` assim que o lifespan termina de carregar o modelo. "
        "Use este endpoint em probes de Kubernetes (`livenessProbe` e "
        "`readinessProbe`) e em healthchecks de balanceadores."
    ),
    response_description="API saudável e pronta para servir predições.",
    operation_id="getHealth",
)
async def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok",
        timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    )


@api_router.post(
    "/predict",
    response_model=PredictResponse,
    status_code=status.HTTP_200_OK,
    tags=["Prediction"],
    summary="Prediz a probabilidade de churn de um cliente Telco",
    description=(
        "Recebe um payload bruto do dataset Telco Customer Churn (21 campos "
        "menos `customerID`) e devolve a probabilidade calibrada de churn, "
        "a decisão binária (aplicando o threshold otimizado) e metadados de "
        "rastreamento.\n\n"
        "**Pipeline interno**: validação Pydantic → encoding categórico → "
        "scaler → MLP (PyTorch) → sigmoid → comparação com threshold.\n\n"
        "Para correlacionar logs, envie o header `X-Request-ID` — ele é "
        "ecoado no body e nos headers da resposta."
    ),
    response_description="Probabilidade de churn e decisão aplicando o threshold.",
    operation_id="postPredict",
    responses={
        status.HTTP_422_UNPROCESSABLE_CONTENT: {
            "model": ValidationErrorResponse,
            "description": (
                "Payload inválido — campo fora do enum, range incorreto, "
                "tipo errado, etc."
            ),
        },
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": ServiceUnavailableResponse,
            "description": (
                "Modelo não carregado. Ocorre quando o lifespan falhou em "
                "buscar o artefato do MLflow no startup."
            ),
        },
    },
)
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
