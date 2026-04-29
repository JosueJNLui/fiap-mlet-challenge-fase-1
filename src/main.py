from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, Request, Response
from starlette.middleware.base import RequestResponseEndpoint

from .api.routes import api_router
from .config import Settings, get_settings
from .infrastructure.mlflow_loader import load_predictor


class JSONLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        record_dict = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc)
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
                # Fail fast: process exits, orchestrator restarts.
                raise
        yield

    return lifespan


def create_app(*, load_model: bool = True) -> FastAPI:
    configure_logging()
    settings = get_settings()
    app = FastAPI(
        title="fiap-mlet-challenge-fase-1",
        version="0.1.0",
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
        duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
        response.headers["X-Process-Time"] = str(duration_ms)
        response.headers["X-Request-ID"] = request_id

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
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=False)
