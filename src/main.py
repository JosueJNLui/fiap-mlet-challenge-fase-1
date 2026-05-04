from __future__ import annotations

import faulthandler
import json
import logging
import signal
import sys
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from fastapi import FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.middleware.base import RequestResponseEndpoint

# Diagnóstico: habilita `kill -USR1 <pid>` para despejar tracebacks de todas
# as threads. Útil para debugar travas no startup (ex.: chamadas bloqueantes
# de MLflow/urllib3 dentro do lifespan).
faulthandler.enable()
if hasattr(signal, "SIGUSR1"):  # pragma: no cover - branch só em POSIX (sem SIGUSR1 no Windows)
    faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)

from .api.routes import api_router  # noqa: E402 - importado após o setup do faulthandler por design
from .config import Settings, get_settings  # noqa: E402
from .infrastructure.mlflow_loader import load_predictor  # noqa: E402

API_DESCRIPTION = """
API REST de **previsão de churn** para clientes Telco. Serve, por padrão, a
**Logistic Regression (sklearn)** registrada no MLflow do DagsHub; o **MLP
(PyTorch)** está disponível como alternativa A/B-testável via
`MODEL_FLAVOR=pytorch` (ver `docs/MODEL_CARD.md` §7.1).

## Como usar

1. Faça `GET /health` para confirmar que o modelo foi carregado.
2. Faça `POST /predict` com o payload Telco bruto. A API cuida do
   pré-processamento, encoding, scaler e aplicação do threshold de negócio.
3. Use o header `X-Request-ID` para correlacionar logs entre cliente e API.

## Pipeline interno

`payload Telco` → validação Pydantic → encoding categórico →
`scaler.joblib` (MLflow) → modelo carregado do Registry (sklearn
`predict_proba` ou MLP PyTorch + sigmoid) → threshold otimizado pela
curva de lucro de negócio → `prediction`.

## Observabilidade

- Cada resposta inclui os headers `X-Process-Time` (latência em ms) e
  `X-Request-ID`.
- Logs estruturados em JSON com `request_id` propagado por toda a chain.
""".strip()

OPENAPI_TAGS = [
    {
        "name": "Health",
        "description": (
            "Liveness/readiness probes. Use para healthchecks de Kubernetes "
            "e balanceadores antes de rotear tráfego para o pod."
        ),
    },
    {
        "name": "Prediction",
        "description": (
            "Inferência online do modelo de churn. Recebe um cliente Telco "
            "bruto e devolve probabilidade calibrada + decisão binária."
        ),
    },
]

HTTP_REQUESTS_TOTAL = Counter(
    "fiap_mlet_http_requests_total",
    "Total HTTP requests processed by the churn prediction API.",
    ("method", "path", "status_code"),
)
HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "fiap_mlet_http_request_duration_seconds",
    "HTTP request latency in seconds for the churn prediction API.",
    ("method", "path", "status_code"),
)


class JSONLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        record_dict = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC)
            .isoformat()
            .replace("+00:00", "Z"),
            "level": record.levelname,
            "message": record.getMessage(),
            **getattr(record, "extra", {}),
        }
        return json.dumps(record_dict, ensure_ascii=False)


def configure_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JSONLogFormatter())

    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(logging.INFO)

    access_logger = logging.getLogger("uvicorn.access")
    access_logger.handlers = []
    access_logger.propagate = False
    access_logger.disabled = True


def _is_suppressed_health_check(request: Request) -> bool:
    if request.url.path != "/health":
        return False

    user_agent = request.headers.get("user-agent", "").lower()
    client_host = request.client.host if request.client else ""
    suppressed_user_agents = (
        "kube-probe",
        "kube-proxy",
        "elb-healthchecker",
        "amazon-route53-health-check-service",
        "ecs-container-healthcheck",
    )
    is_known_probe = any(agent in user_agent for agent in suppressed_user_agents)
    is_local_ecs_probe = user_agent.startswith("python-urllib/") and client_host in (
        "127.0.0.1",
        "::1",
    )
    return is_known_probe or is_local_ecs_probe


def _build_lifespan(settings: Settings, *, load_model: bool):
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        logger = logging.getLogger("fiap-mlet-challenge-fase-1")
        if load_model and settings.load_model_on_startup:
            try:
                app.state.predictor = load_predictor(settings)
                logger.info(
                    "model.loaded",
                    extra={
                        "extra": {
                            "model_name": settings.model_name,
                            "model_version": settings.model_version,
                        }
                    },
                )
            except Exception as exc:
                logger.error(
                    "model.load.failed",
                    extra={"extra": {"error": str(exc), "type": type(exc).__name__}},
                )
                # Fail fast: o processo encerra e o orquestrador reinicia.
                raise
        yield

    return lifespan


def create_app(*, load_model: bool = True) -> FastAPI:
    configure_logging()
    settings = get_settings()
    app = FastAPI(
        title="FIAP MLET: Churn Prediction API",
        summary="Previsão de churn de clientes Telco com modelo registrado no MLflow (LogReg sklearn por padrão; MLP PyTorch alternativo).",
        description=API_DESCRIPTION,
        version="0.1.0",
        contact={
            "name": "FIAP MLET, Fase 1",
            "url": "https://github.com/JosueJNLui/fiap-mlet-challenge-fase-1",
        },
        license_info={
            "name": "MIT",
            "url": "https://opensource.org/licenses/MIT",
        },
        openapi_tags=OPENAPI_TAGS,
        docs_url=settings.docs_url or None,
        redoc_url=settings.redoc_url or None,
        openapi_url=settings.openapi_url or None,
        swagger_ui_parameters={
            "defaultModelsExpandDepth": 1,
            "displayRequestDuration": True,
            "filter": True,
            "tryItOutEnabled": True,
            "persistAuthorization": True,
        },
        lifespan=_build_lifespan(settings, load_model=load_model),
    )

    @app.middleware("http")
    async def latency_and_logging_middleware(
        request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        request.state.request_id = request_id

        started_at = time.perf_counter()
        response = await call_next(request)
        duration_seconds = time.perf_counter() - started_at
        duration_ms = round(duration_seconds * 1000, 2)
        response.headers["X-Process-Time"] = str(duration_ms)
        response.headers["X-Request-ID"] = request_id
        route = request.scope.get("route")
        labels = (
            request.method,
            route.path if route is not None else request.url.path,
            str(response.status_code),
        )
        HTTP_REQUESTS_TOTAL.labels(*labels).inc()
        HTTP_REQUEST_DURATION_SECONDS.labels(*labels).observe(duration_seconds)

        if not _is_suppressed_health_check(request):
            log_payload = {
                "path": request.url.path,
                "method": request.method,
                "status_code": response.status_code,
                "execution_time_ms": duration_ms,
                "request_id": request_id,
            }
            logging.getLogger("fiap-mlet-challenge-fase-1").info(
                "request.complete", extra={"extra": log_payload}
            )
        return response

    app.include_router(api_router)

    @app.get("/metrics", include_in_schema=False)
    async def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=False)
