from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd
import torch

from .preprocessing import preprocess_one


@dataclass(frozen=True)
class PredictionResult:
    probability: float
    label: bool
    threshold: float


class _Scaler(Protocol):
    def transform(self, X: pd.DataFrame) -> np.ndarray: ...


class ChurnPredictor:
    """Bundles the trained MLP with its scaler and business threshold.

    The model returns logits (BCEWithLogitsLoss in training); inference
    applies sigmoid, then compares against ``threshold`` to produce the
    final label. The scaler was fit on the full 28-column matrix so we
    apply it before tensor conversion.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        scaler: _Scaler,
        threshold: float,
        version: str,
    ) -> None:
        self.model = model.eval()
        self.scaler = scaler
        self.threshold = threshold
        self.version = version

    def predict(self, payload: dict[str, Any]) -> PredictionResult:
        features = preprocess_one(payload)
        scaled = self.scaler.transform(features)
        tensor = torch.tensor(scaled, dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(tensor)
            prob = float(torch.sigmoid(logits).item())
        return PredictionResult(
            probability=prob,
            label=prob >= self.threshold,
            threshold=self.threshold,
        )
