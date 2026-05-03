from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch

from src.application.predictor import (
    ChurnPredictor,
    pytorch_inference,
    sklearn_inference,
)
from src.application.preprocessing import EXPECTED_FEATURE_ORDER


class _IdentityScaler:
    """Stand-in for sklearn StandardScaler that returns input untouched."""

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return X.to_numpy()


class _RecordingScaler:
    def __init__(self) -> None:
        self.received: pd.DataFrame | None = None

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        self.received = X
        return X.to_numpy()


class _ConstantLogitModel(torch.nn.Module):
    def __init__(self, logit: float) -> None:
        super().__init__()
        self._logit = logit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full((x.shape[0], 1), self._logit, dtype=torch.float32)


class _StubSklearnModel:
    """Minimal sklearn-like stub exposing ``predict_proba``."""

    def __init__(self, prob_positive: float) -> None:
        self._prob = prob_positive

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.tile([1.0 - self._prob, self._prob], (X.shape[0], 1))


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
        inference_fn=pytorch_inference,
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
        inference_fn=pytorch_inference,
    )
    result = predictor.predict(_payload())
    assert result.probability < 0.5
    assert result.label is False


def test_predictor_sends_named_dataframe_to_scaler() -> None:
    scaler = _RecordingScaler()
    predictor = ChurnPredictor(
        model=_ConstantLogitModel(0.0),
        scaler=scaler,
        threshold=0.5,
        version="test",
        inference_fn=pytorch_inference,
    )

    predictor.predict(_payload())

    assert scaler.received is not None
    assert isinstance(scaler.received, pd.DataFrame)
    assert list(scaler.received.columns) == EXPECTED_FEATURE_ORDER


def test_predictor_uses_sklearn_inference() -> None:
    # Threshold 0.2278 (LogReg em produção): prob 0.7 deve ser True; 0.1 False.
    predictor_high = ChurnPredictor(
        model=_StubSklearnModel(prob_positive=0.7),
        scaler=_IdentityScaler(),
        threshold=0.2278,
        version="1",
        inference_fn=sklearn_inference,
    )
    result_high = predictor_high.predict(_payload())
    assert result_high.probability == 0.7
    assert result_high.label is True

    predictor_low = ChurnPredictor(
        model=_StubSklearnModel(prob_positive=0.1),
        scaler=_IdentityScaler(),
        threshold=0.2278,
        version="1",
        inference_fn=sklearn_inference,
    )
    result_low = predictor_low.predict(_payload())
    assert result_low.probability == 0.1
    assert result_low.label is False
