from __future__ import annotations

from fastapi import Request

from ..application.predictor import ChurnPredictor


def get_predictor(request: Request) -> ChurnPredictor:
    """Returns the predictor singleton stored on `app.state` during lifespan.

    Tests override this via `app.dependency_overrides[get_predictor]` so they
    don't have to populate `app.state` or hit MLflow.
    """
    return request.app.state.predictor  # type: ignore[no-any-return]
