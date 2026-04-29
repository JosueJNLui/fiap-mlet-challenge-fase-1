from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.api.dependencies import get_predictor
from src.application.predictor import PredictionResult
from src.main import app


class FakePredictor:
    """Stand-in for ChurnPredictor used in API tests.

    Returns a fixed probability so we can assert response shape and
    middleware behavior without touching MLflow.
    """

    def __init__(
        self, probability: float = 0.42, threshold: float = 0.20303030303030303
    ) -> None:
        self.probability = probability
        self.threshold = threshold
        self.version = "test-version"

    def predict(self, payload: dict[str, Any]) -> PredictionResult:
        return PredictionResult(
            probability=self.probability,
            label=self.probability >= self.threshold,
            threshold=self.threshold,
        )


@pytest.fixture
def fake_predictor() -> FakePredictor:
    return FakePredictor()


@pytest.fixture
def client(fake_predictor: FakePredictor) -> Iterator[TestClient]:
    app.dependency_overrides[get_predictor] = lambda: fake_predictor
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def sample_payload() -> dict[str, Any]:
    """Realistic Telco payload for one customer."""
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 24,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "One year",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 75.5,
        "TotalCharges": 1850.0,
    }
