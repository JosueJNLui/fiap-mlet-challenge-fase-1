from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from src.application.preprocessing import (
    EXPECTED_FEATURE_ORDER,
    preprocess_one,
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


def test_preprocess_one_returns_expected_columns_in_order(
    raw_payload: dict[str, Any],
) -> None:
    out = preprocess_one(raw_payload)
    assert list(out.columns) == EXPECTED_FEATURE_ORDER
    assert out.shape == (1, 28)


def test_preprocess_one_binary_mappings(raw_payload: dict[str, Any]) -> None:
    out = preprocess_one(raw_payload).iloc[0]
    assert out["Partner"] == 1
    assert out["Dependents"] == 0
    assert out["PhoneService"] == 1
    assert out["OnlineSecurity"] == 1
    # InternetService=DSL → 1
    assert out["InternetService"] == 1
    # dummy gender_Male é 0 porque gender="Female"
    assert out["gender_Male"] == 0
    # Contract_One year é acionado
    assert out["Contract_One year"] == 1
    assert out["Contract_Two year"] == 0


def test_preprocess_one_log1p_applied_to_total_charges(
    raw_payload: dict[str, Any],
) -> None:
    out = preprocess_one(raw_payload).iloc[0]
    assert out["TotalCharges"] == pytest.approx(np.log1p(1850.0))


def test_preprocess_one_handles_string_total_charges(
    raw_payload: dict[str, Any],
) -> None:
    # Como no CSV bruto, TotalCharges chega como string " " para tenure==0.
    payload = {**raw_payload, "tenure": 0, "TotalCharges": " "}
    out = preprocess_one(payload).iloc[0]
    # Fallback: MonthlyCharges * tenure = 75.5 * 0 = 0; log1p(0) = 0
    assert out["TotalCharges"] == pytest.approx(0.0)


def test_preprocess_one_num_services_count(raw_payload: dict[str, Any]) -> None:
    # PhoneService=1, MultipleLines=0, InternetService=1, OnlineSecurity=1,
    # OnlineBackup=0, DeviceProtection=0, TechSupport=1, StreamingTV=0,
    # StreamingMovies=0  → total = 4 serviços ativos
    out = preprocess_one(raw_payload).iloc[0]
    assert out["num_services"] == 4


def test_preprocess_one_charge_vs_expected(raw_payload: dict[str, Any]) -> None:
    out = preprocess_one(raw_payload).iloc[0]
    # O notebook aplica log1p a TotalCharges ANTES de create_features,
    # então a média usa o valor já log-transformado. Replicamos essa peculiaridade.
    log_total = np.log1p(1850.0)
    expected_avg = log_total / 24.0
    assert out["avg_charges_per_month"] == pytest.approx(expected_avg)
    assert out["charge_vs_expected"] == pytest.approx(75.5 - expected_avg)
