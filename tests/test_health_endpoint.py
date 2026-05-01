import logging

import pytest
from fastapi.testclient import TestClient

from src.main import configure_logging


def test_health_endpoint_returns_ok(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert isinstance(payload["timestamp"], str)
    assert payload["timestamp"].endswith("Z")


def test_health_endpoint_includes_latency_header(client: TestClient) -> None:
    response = client.get("/health")
    assert "x-process-time" in {k.lower() for k in response.headers}
    assert "x-request-id" in {k.lower() for k in response.headers}


def test_configure_logging_disables_uvicorn_access_log() -> None:
    configure_logging()

    access_logger = logging.getLogger("uvicorn.access")

    assert access_logger.disabled is True
    assert access_logger.handlers == []
    assert access_logger.propagate is False


def test_health_endpoint_logs_non_probe_requests(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="fiap-mlet-challenge-fase-1")

    client.get("/health", headers={"user-agent": "curl/8.0"})

    assert any(
        record.name == "fiap-mlet-challenge-fase-1"
        and record.getMessage() == "request.complete"
        and getattr(record, "extra", {}).get("path") == "/health"
        for record in caplog.records
    )


@pytest.mark.parametrize(
    "user_agent",
    [
        "kube-probe/1.30",
        "kube-proxy/1.30",
        "ELB-HealthChecker/2.0",
        "Amazon-Route53-Health-Check-Service",
    ],
)
def test_health_endpoint_skips_probe_request_logs(
    client: TestClient, caplog: pytest.LogCaptureFixture, user_agent: str
) -> None:
    caplog.set_level(logging.INFO, logger="fiap-mlet-challenge-fase-1")

    client.get("/health", headers={"user-agent": user_agent})

    assert not any(
        record.name == "fiap-mlet-challenge-fase-1"
        and record.getMessage() == "request.complete"
        and getattr(record, "extra", {}).get("path") == "/health"
        for record in caplog.records
    )
