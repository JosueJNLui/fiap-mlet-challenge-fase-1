from __future__ import annotations

from typing import Any

from src.application.predictor import PredictionResult
from src.main import create_app


class E2EPredictor:
    """Predictor determinístico usado pelo servidor HTTP de E2E."""

    threshold = 0.20303030303030303
    version = "e2e-test-version"

    def predict(self, payload: dict[str, Any]) -> PredictionResult:
        probability = 0.91 if payload["Contract"] == "Month-to-month" else 0.12
        return PredictionResult(
            probability=probability,
            label=probability >= self.threshold,
            threshold=self.threshold,
        )


app = create_app(load_model=False)
app.state.predictor = E2EPredictor()
