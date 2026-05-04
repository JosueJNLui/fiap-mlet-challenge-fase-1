"""Testes de integração contra o registry real do MLflow / DagsHub.

Estes testes são pulados por padrão (`addopts = "-m 'not integration'"` no
pyproject.toml). Rode com `pytest -m integration` e credenciais válidas do
DagsHub no ambiente para exercitar todo o caminho de carregamento.
"""

from __future__ import annotations

import os

import pytest

from src.application.predictor import ChurnPredictor, PredictionResult
from src.config import Settings
from src.infrastructure.mlflow_loader import load_predictor

pytestmark = pytest.mark.integration

_REQUIRED_ENV = ("MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD")


def _missing_creds() -> bool:
    return any(not os.environ.get(var) for var in _REQUIRED_ENV)


@pytest.fixture(scope="module")
def real_predictor() -> ChurnPredictor:
    if _missing_creds():
        pytest.skip(
            "Integration tests require MLFLOW_TRACKING_USERNAME and "
            "MLFLOW_TRACKING_PASSWORD; skipping."
        )
    settings = Settings()
    return load_predictor(settings)


def test_loads_real_model_from_registry(real_predictor: ChurnPredictor) -> None:
    assert real_predictor.version
    assert real_predictor.threshold > 0
    # O flavor sklearn (default) encapsula a Pipeline completa; o flavor
    # pytorch usa o caminho antigo, baseado em componentes separados.
    if real_predictor._pipeline is not None:
        assert real_predictor.model is None
        assert real_predictor.scaler is None
    else:
        assert real_predictor.model is not None
        assert real_predictor.scaler is not None


def test_real_model_predicts_known_payload(real_predictor: ChurnPredictor) -> None:
    payload = {
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
    result = real_predictor.predict(payload)
    assert isinstance(result, PredictionResult)
    assert 0.0 <= result.probability <= 1.0
    assert result.label is (result.probability >= result.threshold)
