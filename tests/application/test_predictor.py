from __future__ import annotations

from typing import Any

import numpy as np
import torch

from src.application.predictor import ChurnPredictor


class _IdentityScaler:
    """Stand-in for sklearn StandardScaler that returns input untouched."""

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X


class _ConstantLogitModel(torch.nn.Module):
    def __init__(self, logit: float) -> None:
        super().__init__()
        self._logit = logit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full((x.shape[0], 1), self._logit, dtype=torch.float32)


def _payload() -> dict[str, Any]:
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


def test_predictor_label_above_threshold() -> None:
    # logit=2 → sigmoid≈0.881, comfortably above default threshold
    predictor = ChurnPredictor(
        model=_ConstantLogitModel(2.0),
        scaler=_IdentityScaler(),
        threshold=0.5,
        version="test",
    )
    result = predictor.predict(_payload())
    assert 0.0 <= result.probability <= 1.0
    assert result.probability > 0.5
    assert result.label is True
    assert result.threshold == 0.5


def test_predictor_label_below_threshold() -> None:
    # logit=-2 → sigmoid≈0.119
    predictor = ChurnPredictor(
        model=_ConstantLogitModel(-2.0),
        scaler=_IdentityScaler(),
        threshold=0.5,
        version="test",
    )
    result = predictor.predict(_payload())
    assert result.probability < 0.5
    assert result.label is False
