from fastapi.testclient import TestClient


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
