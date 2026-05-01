from __future__ import annotations

import os

import joblib
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

from ..application.predictor import ChurnPredictor
from ..config import Settings


def load_predictor(settings: Settings) -> ChurnPredictor:
    """Loads the registered MLP and its scaler from the MLflow tracking server.

    Sets MLFLOW_TRACKING_USERNAME / PASSWORD env vars so the underlying client
    uses Basic Auth (DagsHub's auth model). Both the model and the scaler
    artifact live in the same run; we look up the run_id from the registry
    and download the scaler artifact directly.
    """
    os.environ["MLFLOW_TRACKING_USERNAME"] = settings.mlflow_tracking_username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = (
        settings.mlflow_tracking_password.get_secret_value()
    )
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    model_version = client.get_model_version(
        settings.model_name, settings.model_version
    )

    model_uri = f"models:/{settings.model_name}/{settings.model_version}"
    model = mlflow.pytorch.load_model(model_uri)

    scaler_local_path = mlflow.artifacts.download_artifacts(
        run_id=model_version.run_id,
        artifact_path=settings.scaler_artifact_path,
    )
    scaler = joblib.load(scaler_local_path)

    return ChurnPredictor(
        model=model,
        scaler=scaler,
        threshold=settings.prediction_threshold,
        version=settings.model_version,
    )
