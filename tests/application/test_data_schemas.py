from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pandera.pandas as pa
import pytest

from src.application.data_schemas import (
    PROCESSED_FEATURES_SCHEMA,
    RAW_TELCO_SCHEMA,
)
from src.application.preprocessing import EXPECTED_FEATURE_ORDER, preprocess_one

DATASET_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "dataset"
    / "telco_customer_churn.csv"
)


@pytest.fixture(scope="module")
def raw_dataset() -> pd.DataFrame:
    return pd.read_csv(DATASET_PATH)


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


def test_raw_schema_accepts_valid_dataset(raw_dataset: pd.DataFrame) -> None:
    validated = RAW_TELCO_SCHEMA.validate(raw_dataset, lazy=True)
    assert validated.shape == raw_dataset.shape


def test_raw_schema_rejects_unknown_column(raw_dataset: pd.DataFrame) -> None:
    df = raw_dataset.copy()
    df["unexpected"] = 0
    with pytest.raises(pa.errors.SchemaErrors):
        RAW_TELCO_SCHEMA.validate(df, lazy=True)


def test_raw_schema_rejects_invalid_gender(raw_dataset: pd.DataFrame) -> None:
    df = raw_dataset.copy()
    df.loc[0, "gender"] = "Other"
    with pytest.raises(pa.errors.SchemaErrors):
        RAW_TELCO_SCHEMA.validate(df, lazy=True)


def test_processed_schema_accepts_preprocess_one_output(
    raw_payload: dict[str, Any],
) -> None:
    out = preprocess_one(raw_payload)
    validated = PROCESSED_FEATURES_SCHEMA.validate(out, lazy=True)
    assert list(validated.columns) == EXPECTED_FEATURE_ORDER


def test_processed_schema_rejects_wrong_column_order(
    raw_payload: dict[str, Any],
) -> None:
    out = preprocess_one(raw_payload)
    swapped = out[[EXPECTED_FEATURE_ORDER[1], EXPECTED_FEATURE_ORDER[0], *EXPECTED_FEATURE_ORDER[2:]]]
    with pytest.raises(pa.errors.SchemaErrors):
        PROCESSED_FEATURES_SCHEMA.validate(swapped, lazy=True)


def test_processed_schema_rejects_missing_dummy(raw_payload: dict[str, Any]) -> None:
    out = preprocess_one(raw_payload).drop(columns=["Contract_One year"])
    with pytest.raises(pa.errors.SchemaErrors):
        PROCESSED_FEATURES_SCHEMA.validate(out, lazy=True)
