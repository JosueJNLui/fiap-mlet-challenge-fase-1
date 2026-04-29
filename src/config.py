from __future__ import annotations

from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the churn prediction API.

    Reads from environment variables (and a local `.env` if present). The
    MLflow tracking client picks up `MLFLOW_TRACKING_USERNAME` and
    `MLFLOW_TRACKING_PASSWORD` automatically — we set them from settings in the
    loader, so deployers can supply either env style.
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
    model_name: str = "Churn_MLP_Final_Production"
    model_version: str = "8"
    scaler_artifact_path: str = "model_components/scaler.joblib"
    prediction_threshold: float = 0.20303030303030303
    load_model_on_startup: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
