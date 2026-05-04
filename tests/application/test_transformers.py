from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.application.preprocessing import EXPECTED_FEATURE_ORDER, preprocess_one
from src.application.transformers import FeatureEngineer


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


def _alt_payload() -> dict[str, Any]:
    return {
        "gender": "Male",
        "SeniorCitizen": 1,
        "Partner": "No",
        "Dependents": "Yes",
        "tenure": 5,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 110.25,
        "TotalCharges": 551.25,
    }


def test_feature_engineer_fit_is_noop(raw_payload: dict[str, Any]) -> None:
    fe = FeatureEngineer()
    assert fe.fit(pd.DataFrame([raw_payload])) is fe


def test_feature_engineer_output_matches_preprocess_one_for_single_payload(
    raw_payload: dict[str, Any],
) -> None:
    expected = preprocess_one(raw_payload)
    actual = FeatureEngineer().transform(pd.DataFrame([raw_payload]))
    pd.testing.assert_frame_equal(
        actual.reset_index(drop=True), expected.reset_index(drop=True)
    )


def test_feature_engineer_output_matches_preprocess_one_for_alt_payload() -> None:
    payload = _alt_payload()
    expected = preprocess_one(payload)
    actual = FeatureEngineer().transform(pd.DataFrame([payload]))
    pd.testing.assert_frame_equal(
        actual.reset_index(drop=True), expected.reset_index(drop=True)
    )


def test_feature_engineer_handles_batch_of_two(raw_payload: dict[str, Any]) -> None:
    batch = pd.DataFrame([raw_payload, _alt_payload()])
    out = FeatureEngineer().transform(batch)
    assert out.shape == (2, 28)
    assert list(out.columns) == EXPECTED_FEATURE_ORDER
    # Cada linha deve ser igual a preprocess_one aplicado individualmente
    expected_row_0 = preprocess_one(raw_payload).iloc[0]
    expected_row_1 = preprocess_one(_alt_payload()).iloc[0]
    pd.testing.assert_series_equal(
        out.iloc[0].reset_index(drop=True),
        expected_row_0.reset_index(drop=True),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        out.iloc[1].reset_index(drop=True),
        expected_row_1.reset_index(drop=True),
        check_names=False,
    )


def test_feature_engineer_drops_customer_id() -> None:
    payload_with_id = {**_alt_payload(), "customerID": "1234-ABCDE"}
    out = FeatureEngineer().transform(pd.DataFrame([payload_with_id]))
    assert "customerID" not in out.columns
    assert list(out.columns) == EXPECTED_FEATURE_ORDER


def test_feature_engineer_get_feature_names_out() -> None:
    names = FeatureEngineer().get_feature_names_out()
    assert isinstance(names, np.ndarray)
    assert names.tolist() == EXPECTED_FEATURE_ORDER
