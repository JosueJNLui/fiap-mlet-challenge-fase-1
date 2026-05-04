from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configurações de runtime da API de previsão de churn.

    Lidas das variáveis de ambiente (e de um `.env` local, se existir). O
    cliente do MLflow consome `MLFLOW_TRACKING_USERNAME` e
    `MLFLOW_TRACKING_PASSWORD` automaticamente. Populamos essas envs a partir
    das settings no loader, então quem faz o deploy pode usar qualquer um dos
    estilos.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    mlflow_tracking_uri: str = (
        "https://dagshub.com/JosueJNLui/fiap-mlet-challenge-fase-1.mlflow"
    )
    mlflow_tracking_username: str = "JosueJNLui"
    mlflow_tracking_password: SecretStr = SecretStr("")
    model_name: str = "Churn_LogReg_Final_Production"
    model_version: str = "3"
    scaler_artifact_path: str = "model_components/scaler.joblib"
    prediction_threshold: float = 0.2080
    model_flavor: Literal["sklearn", "pytorch"] = "sklearn"
    load_model_on_startup: bool = True

    # OpenAPI / Swagger UI. Defina qualquer um destes como string vazia para
    # desabilitar o endpoint correspondente em produção (ex.: `DOCS_URL=`).
    docs_url: str | None = "/docs"
    redoc_url: str | None = "/redoc"
    openapi_url: str | None = "/openapi.json"


@lru_cache
def get_settings() -> Settings:
    return Settings()
