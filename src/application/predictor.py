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
    """Caminho LogReg: ``predict_proba`` já devolve probabilidade calibrada."""
    return float(model.predict_proba(scaled)[0, 1])


def pytorch_inference(model: Any, scaled: np.ndarray) -> float:
    """Caminho MLP: logits → sigmoid. Mantido como fallback A/B-testável."""
    import torch

    tensor = torch.tensor(scaled, dtype=torch.float32)
    with torch.no_grad():
        logits = model(tensor)
    return float(torch.sigmoid(logits).item())


class ChurnPredictor:
    """Empacota o modelo treinado com seu threshold e o caminho de inferência.

    Dois flavors compartilham esta classe:

    * **Modo Pipeline** (flavor sklearn em produção), construído via
      :meth:`from_pipeline`. Encapsula uma ``sklearn.Pipeline`` já ajustada que
      cuida de feature engineering, scaling e classificação ponta-a-ponta. O
      payload bruto vai direto para ``pipeline.predict_proba``.
    * **Modo componentes** (fallback MLP em PyTorch), usa o construtor antigo
      ``__init__(model, scaler, ...)``. O MLP é um ``torch.nn.Module`` que não
      cabe dentro de uma sklearn Pipeline, então pré-processamento e scaler
      permanecem externos.

    Ambos os modos expõem o mesmo contrato ``predict(payload)`` usado pela API.
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
        """Constrói um predictor apoiado em uma sklearn.Pipeline (caminho de produção).

        A Pipeline é o artefato único registrado no MLflow como
        ``Churn_LogReg_Final_Production``; embute FeatureEngineer +
        StandardScaler + LogisticRegression e aceita um DataFrame bruto de 1 linha.
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
