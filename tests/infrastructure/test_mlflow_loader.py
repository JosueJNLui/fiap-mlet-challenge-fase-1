"""Unit tests for ``load_predictor`` — wiring only, MLflow is fully mocked.

Os testes de integração reais ficam em
``tests/integration/test_mlflow_loader.py`` e exigem credenciais DagsHub.
Aqui validamos apenas que o switch sklearn/pytorch dispatcha corretamente:

* sklearn → ``mlflow.sklearn.load_model`` retorna a Pipeline empacotada e o
  predictor é construído via ``ChurnPredictor.from_pipeline`` (sem scaler
  separado, sem download extra).
* pytorch → ``mlflow.pytorch.load_model`` + ``.eval()`` + download do scaler
  via ``mlflow.artifacts.download_artifacts`` + ``joblib.load``.
"""

from __future__ import annotations

from typing import Literal
from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from src.application.predictor import ChurnPredictor, pytorch_inference
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
    """Patch every MLflow boundary so ``load_predictor`` runs offline."""
    with (
        patch("src.infrastructure.mlflow_loader.MlflowClient") as client_cls,
        patch("src.infrastructure.mlflow_loader.mlflow") as mlflow_mod,
        patch("src.infrastructure.mlflow_loader.joblib") as joblib_mod,
    ):
        version_obj = MagicMock(run_id="run-xyz")
        client_cls.return_value.get_model_version.return_value = version_obj

        sklearn_pipeline = MagicMock(name="sklearn_pipeline")
        pytorch_model = MagicMock(name="pytorch_model")
        mlflow_mod.sklearn.load_model.return_value = sklearn_pipeline
        mlflow_mod.pytorch.load_model.return_value = pytorch_model
        mlflow_mod.artifacts.download_artifacts.return_value = "/tmp/scaler.joblib"

        scaler_obj = MagicMock(name="scaler")
        joblib_mod.load.return_value = scaler_obj

        yield {
            "mlflow": mlflow_mod,
            "client_cls": client_cls,
            "joblib": joblib_mod,
            "sklearn_pipeline": sklearn_pipeline,
            "pytorch_model": pytorch_model,
            "scaler": scaler_obj,
        }


def test_load_predictor_sklearn_flavor_wraps_pipeline(patched_mlflow) -> None:
    settings = _make_settings("sklearn")

    predictor = load_predictor(settings)

    assert isinstance(predictor, ChurnPredictor)
    # from_pipeline mode: components are intentionally None.
    assert predictor.model is None
    assert predictor.scaler is None
    assert predictor.inference_fn is None
    assert predictor._pipeline is patched_mlflow["sklearn_pipeline"]
    assert predictor.threshold == settings.prediction_threshold
    assert predictor.version == settings.model_version
    patched_mlflow["mlflow"].sklearn.load_model.assert_called_once_with(
        f"models:/{settings.model_name}/{settings.model_version}"
    )
    # No separate scaler download for sklearn path.
    patched_mlflow["mlflow"].artifacts.download_artifacts.assert_not_called()
    patched_mlflow["joblib"].load.assert_not_called()


def test_load_predictor_pytorch_flavor_calls_eval_and_downloads_scaler(
    patched_mlflow,
) -> None:
    settings = _make_settings("pytorch")

    predictor = load_predictor(settings)

    assert isinstance(predictor, ChurnPredictor)
    assert predictor.model is patched_mlflow["pytorch_model"]
    assert predictor.scaler is patched_mlflow["scaler"]
    assert predictor.inference_fn is pytorch_inference
    assert predictor._pipeline is None
    patched_mlflow["pytorch_model"].eval.assert_called_once()
    patched_mlflow["mlflow"].artifacts.download_artifacts.assert_called_once_with(
        run_id="run-xyz",
        artifact_path=settings.scaler_artifact_path,
    )
    patched_mlflow["joblib"].load.assert_called_once_with("/tmp/scaler.joblib")


def test_load_predictor_unsupported_flavor_raises() -> None:
    # Pydantic blocks unknown literals at validation; we simulate the runtime
    # branch via direct assignment to make sure the sentinel raises.
    settings = _make_settings("sklearn")
    object.__setattr__(settings, "model_flavor", "tensorflow")

    with (
        patch("src.infrastructure.mlflow_loader.MlflowClient"),
        patch("src.infrastructure.mlflow_loader.mlflow"),
        patch("src.infrastructure.mlflow_loader.joblib"),
        pytest.raises(ValueError, match="Unsupported model_flavor"),
    ):
        load_predictor(settings)
