from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.application.pipeline import (
    DEFAULT_LOGREG_PARAMS,
    build_logreg_pipeline,
)
from src.application.predictor import ChurnPredictor
from src.application.transformers import FeatureEngineer

DATASET_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "dataset"
    / "telco_customer_churn.csv"
)


@pytest.fixture
def raw_payload() -> dict[str, Any]:
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 24,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "One year",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 75.5,
        "TotalCharges": 1850.0,
    }


@pytest.fixture(scope="module")
def fitted_pipeline() -> Pipeline:
    df = pd.read_csv(DATASET_PATH)
    y = df["Churn"].map({"Yes": 1, "No": 0})
    X_raw = df.drop(columns=["Churn"])
    pipeline = build_logreg_pipeline()
    pipeline.fit(X_raw, y)
    return pipeline


def test_build_logreg_pipeline_has_three_steps() -> None:
    pipe = build_logreg_pipeline()
    assert [name for name, _ in pipe.steps] == ["features", "scaler", "clf"]
    assert isinstance(pipe.named_steps["features"], FeatureEngineer)
    assert isinstance(pipe.named_steps["scaler"], StandardScaler)
    assert isinstance(pipe.named_steps["clf"], LogisticRegression)


def test_build_logreg_pipeline_default_hyperparams() -> None:
    clf = build_logreg_pipeline().named_steps["clf"]
    for key, value in DEFAULT_LOGREG_PARAMS.items():
        assert getattr(clf, key) == value


def test_build_logreg_pipeline_accepts_overrides() -> None:
    clf = build_logreg_pipeline(C=0.5, max_iter=2000).named_steps["clf"]
    assert clf.C == 0.5
    assert clf.max_iter == 2000
    # Other defaults preserved
    assert clf.class_weight == "balanced"


def test_pipeline_predict_proba_runs_on_raw_dataframe(
    fitted_pipeline: Pipeline, raw_payload: dict[str, Any]
) -> None:
    df = pd.DataFrame([raw_payload])
    proba = fitted_pipeline.predict_proba(df)
    assert proba.shape == (1, 2)
    assert 0.0 <= proba[0, 1] <= 1.0


def test_predictor_from_pipeline_uses_pipeline_predict_proba(
    fitted_pipeline: Pipeline, raw_payload: dict[str, Any]
) -> None:
    predictor = ChurnPredictor.from_pipeline(
        pipeline=fitted_pipeline, threshold=0.2278, version="test-pipeline"
    )
    result = predictor.predict(raw_payload)
    assert 0.0 <= result.probability <= 1.0
    assert result.threshold == 0.2278
    expected = float(fitted_pipeline.predict_proba(pd.DataFrame([raw_payload]))[0, 1])
    assert result.probability == pytest.approx(expected)
    assert result.label is (result.probability >= 0.2278)


def test_predictor_from_pipeline_does_not_set_components(
    fitted_pipeline: Pipeline,
) -> None:
    predictor = ChurnPredictor.from_pipeline(
        pipeline=fitted_pipeline, threshold=0.2278, version="v1"
    )
    assert predictor.model is None
    assert predictor.scaler is None
    assert predictor.inference_fn is None
    assert predictor.version == "v1"
