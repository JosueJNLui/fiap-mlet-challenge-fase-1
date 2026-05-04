"""Carrega o predictor a partir do MLflow Registry (DagsHub).

Dois caminhos coexistem, controlados por ``settings.model_flavor``:

* **sklearn (default em produção):** o artefato registrado é uma
  ``sklearn.Pipeline`` completa (FeatureEngineer + StandardScaler +
  LogisticRegression). Carregamento é uma única chamada
  ``mlflow.sklearn.load_model``; nada de scaler em arquivo separado.
* **pytorch (fallback A/B-testável):** o MLP não cabe em ``sklearn.Pipeline``.
  Continua sendo registrado como modelo PyTorch + scaler ``joblib`` em
  ``model_components/scaler.joblib`` no mesmo run; ambos são carregados aqui.

A escolha do flavor + versão acontece via variáveis de ambiente
(``MODEL_FLAVOR`` / ``MODEL_NAME`` / ``MODEL_VERSION`` /
``PREDICTION_THRESHOLD``) — alterná-las troca o caminho sem deploy de
código (ver ``.env.example``).
"""

from __future__ import annotations

import os

import joblib
import mlflow
import mlflow.pytorch
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from ..application.predictor import ChurnPredictor, pytorch_inference
from ..config import Settings


def load_predictor(settings: Settings) -> ChurnPredictor:
    """Carrega o modelo registrado no MLflow tracking server.

    Define as env vars MLFLOW_TRACKING_USERNAME / PASSWORD para que o cliente
    subjacente use Basic Auth (modelo de autenticação do DagsHub). A URI do
    modelo é resolvida como ``models:/<name>/<version>``; no caminho PyTorch,
    o scaler é baixado a partir do ``model_version.run_id`` e precisa vir
    do mesmo run em que o modelo foi logado.
    """
    os.environ["MLFLOW_TRACKING_USERNAME"] = settings.mlflow_tracking_username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = (
        settings.mlflow_tracking_password.get_secret_value()
    )
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    model_uri = f"models:/{settings.model_name}/{settings.model_version}"

    if settings.model_flavor == "sklearn":
        pipeline = mlflow.sklearn.load_model(model_uri)
        return ChurnPredictor.from_pipeline(
            pipeline=pipeline,
            threshold=settings.prediction_threshold,
            version=settings.model_version,
        )

    if settings.model_flavor == "pytorch":
        client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
        model_version = client.get_model_version(
            settings.model_name, settings.model_version
        )
        model = mlflow.pytorch.load_model(model_uri)
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
            inference_fn=pytorch_inference,
        )

    raise ValueError(f"Unsupported model_flavor: {settings.model_flavor!r}")
