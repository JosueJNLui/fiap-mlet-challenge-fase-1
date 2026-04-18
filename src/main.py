from __future__ import annotations

import json
import logging
import time
from datetime import datetime

from fastapi import FastAPI, Request, Response
from pydantic import BaseModel
from starlette.middleware.base import RequestResponseEndpoint


class HealthResponse(BaseModel):
    status: str = "ok"
    timestamp: str


class JSONLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        record_dict = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
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


def create_app() -> FastAPI:
    configure_logging()
    app = FastAPI(title="fiap-mlet-challenge-fase-1", version="0.1.0")

    @app.middleware("http")
    async def latency_and_logging_middleware(
        request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        started_at = time.perf_counter()
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
        response.headers["X-Process-Time"] = str(duration_ms)

        log_payload = {
            "path": request.url.path,
            "method": request.method,
            "status_code": response.status_code,
            "execution_time_ms": duration_ms,
            "user_id": None,
        }
        logging.getLogger("fiap-mlet-challenge-fase-1").info(
            "request.complete", extra={"extra": log_payload}
        )
        return response

    @app.get("/health", response_model=HealthResponse)
    async def health_check() -> HealthResponse:
        return HealthResponse(
            status="ok", timestamp=datetime.utcnow().isoformat() + "Z"
        )

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
