from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd

from .preprocessing import preprocess_one


@dataclass(frozen=True)
class PredictionResult:
    probability: float
    label: bool
    threshold: float


class _Scaler(Protocol):
    def transform(self, X: pd.DataFrame) -> np.ndarray: ...  # pragma: no cover


InferenceFn = Callable[[Any, np.ndarray], float]


def sklearn_inference(model: Any, scaled: np.ndarray) -> float:
    """LogReg path: predict_proba já retorna probabilidade calibrada."""
    return float(model.predict_proba(scaled)[0, 1])


def pytorch_inference(model: Any, scaled: np.ndarray) -> float:
    """MLP path: logits → sigmoid. Mantido como fallback A/B-testável."""
    import torch

    tensor = torch.tensor(scaled, dtype=torch.float32)
    with torch.no_grad():
        logits = model(tensor)
    return float(torch.sigmoid(logits).item())


class ChurnPredictor:
    """Bundles the trained model with its scaler, business threshold and
    framework-specific inference function.

    The class is flavor-agnostic: ``inference_fn`` receives the loaded model
    and the scaled feature array, and returns a probability in ``[0, 1]``.
    Two helpers ship with this module — :func:`sklearn_inference` (LogReg
    served in production) and :func:`pytorch_inference` (MLP fallback).
    """

    def __init__(
        self,
        model: Any,
        scaler: _Scaler,
        threshold: float,
        version: str,
        inference_fn: InferenceFn,
    ) -> None:
        self.model = model
        self.scaler = scaler
        self.threshold = threshold
        self.version = version
        self.inference_fn = inference_fn

    def predict(self, payload: dict[str, Any]) -> PredictionResult:
        features = preprocess_one(payload)
        scaled = self.scaler.transform(features)
        prob = self.inference_fn(self.model, scaled)
        return PredictionResult(
            probability=prob,
            label=prob >= self.threshold,
            threshold=self.threshold,
        )
