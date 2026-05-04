"""Unit tests for `load_predictor` — wiring only, MLflow is fully mocked.

Os testes de integracao reais ficam em `tests/integration/test_mlflow_loader.py`
e exigem credenciais DagsHub. Aqui validamos apenas que o switch
sklearn/pytorch chama o `load_fn` correto e propaga `inference_fn` para o
`ChurnPredictor` resultante — o que e impossivel de pegar com mocks
intrusivos em integracao.
"""

from __future__ import annotations

from typing import Literal
from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from src.application.predictor import (
    ChurnPredictor,
    pytorch_inference,
    sklearn_inference,
)
from src.config import Settings
from src.infrastructure.mlflow_loader import load_predictor


def _make_settings(flavor: Literal["sklearn", "pytorch"]) -> Settings:
    return Settings(
        mlflow_tracking_uri="http://fake.invalid",
        mlflow_tracking_username="user",
        mlflow_tracking_password=SecretStr("pwd"),
        model_name="Churn_Test",
        model_version="1",
        scaler_artifact_path="model_components/scaler.joblib",
        prediction_threshold=0.5,
        model_flavor=flavor,
        load_model_on_startup=False,
    )


@pytest.fixture
def patched_mlflow():
    """Patch every MLflow boundary call so `load_predictor` runs offline."""
    with (
        patch("src.infrastructure.mlflow_loader.MlflowClient") as client_cls,
        patch("src.infrastructure.mlflow_loader.mlflow") as mlflow_mod,
        patch("src.infrastructure.mlflow_loader.joblib") as joblib_mod,
    ):
        version_obj = MagicMock(run_id="run-xyz")
        client_cls.return_value.get_model_version.return_value = version_obj

        sklearn_model = MagicMock(name="sklearn_model")
        pytorch_model = MagicMock(name="pytorch_model")
        mlflow_mod.sklearn.load_model.return_value = sklearn_model
        mlflow_mod.pytorch.load_model.return_value = pytorch_model
        mlflow_mod.artifacts.download_artifacts.return_value = "/tmp/scaler.joblib"

        scaler_obj = MagicMock(name="scaler")
        joblib_mod.load.return_value = scaler_obj

        # Re-bind _LOADERS so it points at the patched callables. The module
        # captured the originals at import time.
        with patch.dict(
            "src.infrastructure.mlflow_loader._LOADERS",
            {
                "sklearn": (mlflow_mod.sklearn.load_model, sklearn_inference),
                "pytorch": (mlflow_mod.pytorch.load_model, pytorch_inference),
            },
            clear=True,
        ):
            yield {
                "mlflow": mlflow_mod,
                "client_cls": client_cls,
                "joblib": joblib_mod,
                "sklearn_model": sklearn_model,
                "pytorch_model": pytorch_model,
                "scaler": scaler_obj,
            }


def test_load_predictor_sklearn_flavor_wires_sklearn_inference(patched_mlflow) -> None:
    settings = _make_settings("sklearn")

    predictor = load_predictor(settings)

    assert isinstance(predictor, ChurnPredictor)
    assert predictor.model is patched_mlflow["sklearn_model"]
    assert predictor.scaler is patched_mlflow["scaler"]
    assert predictor.threshold == settings.prediction_threshold
    assert predictor.version == settings.model_version
    assert predictor.inference_fn is sklearn_inference
    patched_mlflow["mlflow"].sklearn.load_model.assert_called_once_with(
        f"models:/{settings.model_name}/{settings.model_version}"
    )
    # sklearn nao deve disparar .eval() (atributo so existe em modulos torch).
    patched_mlflow["sklearn_model"].eval.assert_not_called()


def test_load_predictor_pytorch_flavor_calls_eval_and_wires_pytorch_inference(
    patched_mlflow,
) -> None:
    settings = _make_settings("pytorch")

    predictor = load_predictor(settings)

    assert predictor.model is patched_mlflow["pytorch_model"]
    assert predictor.inference_fn is pytorch_inference
    patched_mlflow["pytorch_model"].eval.assert_called_once()


def test_load_predictor_downloads_scaler_from_model_run(patched_mlflow) -> None:
    settings = _make_settings("sklearn")

    load_predictor(settings)

    patched_mlflow["mlflow"].artifacts.download_artifacts.assert_called_once_with(
        run_id="run-xyz",
        artifact_path=settings.scaler_artifact_path,
    )
    patched_mlflow["joblib"].load.assert_called_once_with("/tmp/scaler.joblib")
