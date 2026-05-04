"""Esquemas pandera para validação de DataFrames em pontos críticos do
pipeline de churn.

Dois esquemas são exportados:

* ``RAW_TELCO_SCHEMA`` valida o CSV bruto do dataset Telco
  (``data/dataset/telco_customer_churn.csv``) usado em ``notebooks/eda.ipynb``.
  É chamado como passo formal de *data readiness* antes de qualquer análise.
* ``PROCESSED_FEATURES_SCHEMA`` valida a saída de
  :func:`src.application.preprocessing.preprocess_one` — as 28 colunas float
  exatamente na ordem que o ``StandardScaler`` espera.

Ambos os esquemas só são usados em testes e notebooks. O hot-path da API
não dispara validação para evitar overhead em payloads de 1 linha.
"""

from __future__ import annotations

import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

from .preprocessing import EXPECTED_FEATURE_ORDER

_YES_NO = ["Yes", "No"]
_YES_NO_NO_INTERNET = ["Yes", "No", "No internet service"]
_YES_NO_NO_PHONE = ["Yes", "No", "No phone service"]


RAW_TELCO_SCHEMA: DataFrameSchema = DataFrameSchema(
    columns={
        "customerID": Column(str, Check.str_matches(r"^\d{4}-[A-Z]{5}$")),
        "gender": Column(str, Check.isin(["Male", "Female"])),
        "SeniorCitizen": Column(int, Check.isin([0, 1])),
        "Partner": Column(str, Check.isin(_YES_NO)),
        "Dependents": Column(str, Check.isin(_YES_NO)),
        "tenure": Column(int, Check.in_range(0, 120)),
        "PhoneService": Column(str, Check.isin(_YES_NO)),
        "MultipleLines": Column(str, Check.isin(_YES_NO_NO_PHONE)),
        "InternetService": Column(str, Check.isin(["DSL", "Fiber optic", "No"])),
        "OnlineSecurity": Column(str, Check.isin(_YES_NO_NO_INTERNET)),
        "OnlineBackup": Column(str, Check.isin(_YES_NO_NO_INTERNET)),
        "DeviceProtection": Column(str, Check.isin(_YES_NO_NO_INTERNET)),
        "TechSupport": Column(str, Check.isin(_YES_NO_NO_INTERNET)),
        "StreamingTV": Column(str, Check.isin(_YES_NO_NO_INTERNET)),
        "StreamingMovies": Column(str, Check.isin(_YES_NO_NO_INTERNET)),
        "Contract": Column(str, Check.isin(["Month-to-month", "One year", "Two year"])),
        "PaperlessBilling": Column(str, Check.isin(_YES_NO)),
        "PaymentMethod": Column(
            str,
            Check.isin(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ]
            ),
        ),
        "MonthlyCharges": Column(float, Check.greater_than_or_equal_to(0)),
        # TotalCharges arrives as object in the raw CSV (string with " " for
        # tenure==0 rows). Validation accepts the raw form; coercion to float
        # is responsibility of preprocess_one downstream.
        "TotalCharges": Column(object),
        "Churn": Column(str, Check.isin(_YES_NO)),
    },
    strict=True,
    coerce=False,
)


_BINARY_FLOAT = Check.isin([0.0, 1.0])

_processed_columns: dict[str, Column] = {}
for _col in EXPECTED_FEATURE_ORDER:
    if _col == "tenure":
        _processed_columns[_col] = Column(float, Check.in_range(0.0, 120.0))
    elif _col == "MonthlyCharges":
        _processed_columns[_col] = Column(float, Check.greater_than_or_equal_to(0.0))
    elif _col in {"TotalCharges", "avg_charges_per_month", "charge_vs_expected"}:
        # log1p-scale or derived-from-log1p; bounded by training-time range.
        _processed_columns[_col] = Column(float)
    elif _col == "num_services":
        _processed_columns[_col] = Column(float, Check.in_range(0.0, 9.0))
    elif _col == "SeniorCitizen":
        _processed_columns[_col] = Column(float, _BINARY_FLOAT)
    else:
        # Binary encoded originals (Partner, Dependents, etc.) and one-hot
        # dummies (gender_Male, Contract_*, PaymentMethod_*, tenure_bucket_*).
        _processed_columns[_col] = Column(float, _BINARY_FLOAT)

PROCESSED_FEATURES_SCHEMA: DataFrameSchema = DataFrameSchema(
    columns=_processed_columns,
    strict=True,
    ordered=True,
    coerce=False,
)


__all__ = ["PROCESSED_FEATURES_SCHEMA", "RAW_TELCO_SCHEMA", "pa"]
