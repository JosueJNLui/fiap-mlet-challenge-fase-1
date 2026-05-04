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


class _Pipeline(Protocol):
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...  # pragma: no cover


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
    """Bundles the trained model with its threshold and the inference path.

    Two flavors share this class:

    * **Pipeline mode** (sklearn flavor in production) — built via
      :meth:`from_pipeline`. Wraps a fitted ``sklearn.Pipeline`` that owns
      feature engineering, scaling and classification end-to-end. The raw
      payload goes straight into ``pipeline.predict_proba``.
    * **Components mode** (PyTorch MLP fallback) — uses the legacy
      constructor ``__init__(model, scaler, ...)``. The MLP is a
      ``torch.nn.Module`` that cannot live inside a sklearn Pipeline, so the
      preprocessing + scaler steps stay external.

    Both modes expose the same ``predict(payload)`` contract used by the API.
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
        self._pipeline: _Pipeline | None = None

    @classmethod
    def from_pipeline(
        cls, pipeline: _Pipeline, threshold: float, version: str
    ) -> ChurnPredictor:
        """Build a predictor backed by a sklearn.Pipeline (production path).

        The Pipeline is the single artifact registered in MLflow as
        ``Churn_LogReg_Final_Production``; it embeds FeatureEngineer +
        StandardScaler + LogisticRegression and accepts a raw 1-row DataFrame.
        """
        instance = cls.__new__(cls)
        instance.model = None
        instance.scaler = None
        instance.threshold = threshold
        instance.version = version
        instance.inference_fn = None  # type: ignore[assignment]
        instance._pipeline = pipeline
        return instance

    def predict(self, payload: dict[str, Any]) -> PredictionResult:
        if self._pipeline is not None:
            df = pd.DataFrame([payload])
            prob = float(self._pipeline.predict_proba(df)[0, 1])
        else:
            features = preprocess_one(payload)
            scaled = self.scaler.transform(features)
            prob = self.inference_fn(self.model, scaled)
        return PredictionResult(
            probability=prob,
            label=prob >= self.threshold,
            threshold=self.threshold,
        )
