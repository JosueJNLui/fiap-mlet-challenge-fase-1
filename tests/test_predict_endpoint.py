from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient


def test_predict_returns_expected_shape(
    client: TestClient, sample_payload: dict[str, Any]
) -> None:
    response = client.post("/predict", json=sample_payload)

    assert response.status_code == 200, response.text
    body = response.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert isinstance(body["prediction"], bool)
    assert body["threshold"] == 0.20303030303030303
    assert body["model_version"] == "test-version"
    assert isinstance(body["request_id"], str) and body["request_id"]


def test_predict_propagates_provided_request_id(
    client: TestClient, sample_payload: dict[str, Any]
) -> None:
    response = client.post(
        "/predict",
        json=sample_payload,
        headers={"X-Request-ID": "abc-123"},
    )

    assert response.status_code == 200
    assert response.json()["request_id"] == "abc-123"
    assert response.headers["x-request-id"] == "abc-123"


def test_predict_rejects_invalid_enum(
    client: TestClient, sample_payload: dict[str, Any]
) -> None:
    bad = {**sample_payload, "Contract": "Daily"}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


def test_predict_rejects_negative_tenure(
    client: TestClient, sample_payload: dict[str, Any]
) -> None:
    bad = {**sample_payload, "tenure": -1}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422
