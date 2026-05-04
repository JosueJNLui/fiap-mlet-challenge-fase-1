from __future__ import annotations

from fastapi import Request

from ..application.predictor import ChurnPredictor


def get_predictor(request: Request) -> ChurnPredictor:
    """Devolve o predictor singleton guardado em `app.state` pelo lifespan.

    Os testes sobrescrevem isto via `app.dependency_overrides[get_predictor]`
    para não precisarem popular `app.state` nem acessar o MLflow.
    """
    return request.app.state.predictor  # type: ignore[no-any-return]
