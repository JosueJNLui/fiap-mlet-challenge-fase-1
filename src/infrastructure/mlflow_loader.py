"""Carrega o predictor a partir do MLflow Registry (DagsHub).

A API mantém DOIS caminhos de inferência (sklearn LogReg e PyTorch MLP)
selecionados via ``settings.model_flavor``. Ambos compartilham preprocessing
e scaler — só diferem em ``load_model`` e ``inference_fn``.

Por que manter os dois:

- **LogReg (default em produção):** parsimônia + interpretabilidade.
  Friedman + Nemenyi mostraram equivalência estatística com o MLP no K-Fold
  pareado (ver ``notebooks/models-comparison.ipynb``).
- **MLP (fallback A/B-testável):** já registrado como
  ``Churn_MLP_Final_Production`` v8. Trocar ``MODEL_FLAVOR`` / ``MODEL_NAME``
  / ``MODEL_VERSION`` / ``PREDICTION_THRESHOLD`` no ``.env`` alterna o
  caminho sem deploy de código (ver ``.env.example``).
"""

from __future__ import annotations

import os

import joblib
import mlflow
import mlflow.pytorch
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from ..application.predictor import (
    ChurnPredictor,
    pytorch_inference,
    sklearn_inference,
)
from ..config import Settings


_LOADERS = {
    "sklearn": (mlflow.sklearn.load_model, sklearn_inference),
    "pytorch": (mlflow.pytorch.load_model, pytorch_inference),
}


def load_predictor(settings: Settings) -> ChurnPredictor:
    """Loads the registered model and its scaler from the MLflow tracking server.

    Sets MLFLOW_TRACKING_USERNAME / PASSWORD env vars so the underlying client
    uses Basic Auth (DagsHub's auth model). The model is resolved via the
    Registry URI ``models:/<name>/<version>`` (which may point to a LoggedModel
    in modern MLflow) and the scaler is downloaded from
    ``model_version.run_id`` — both must come from the same training run, or
    the scaler download will 404 and trigger a urllib3 retry storm.
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

    load_fn, inference_fn = _LOADERS[settings.model_flavor]
    model_uri = f"models:/{settings.model_name}/{settings.model_version}"
    model = load_fn(model_uri)
    if settings.model_flavor == "pytorch":
        model.eval()

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
        inference_fn=inference_fn,
    )
