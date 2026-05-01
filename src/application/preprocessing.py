"""Feature engineering replicating ``notebooks/modeling.ipynb`` exactly.

The MLP serialized in MLflow consumes already-encoded numeric features, so
the API has to recreate the same pipeline before scaling and inference.

Single source of truth: keep this module in lockstep with the preprocessing
cells of the notebook. Any change here that diverges from training is a
silent correctness bug.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

BINARY_COLS: list[str] = [
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "MultipleLines",
]
BINARY_MAP: dict[str, int] = {
    "Yes": 1,
    "No": 0,
    "No internet service": 0,
    "No phone service": 0,
}
INTERNET_SERVICE_MAP: dict[str, int] = {"DSL": 1, "Fiber optic": 1, "No": 0}
SERVICE_LIST: list[str] = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "InternetService",
]

# Ordem exata das 28 features do X_dev usado em modeling.ipynb.
# Extraído rodando o pipeline contra o CSV completo. Reorder em
# inference é load-bearing: get_dummies em 1 linha não produz as mesmas
# colunas, e o scaler espera essa ordem.
EXPECTED_FEATURE_ORDER: list[str] = [
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "PaperlessBilling",
    "MonthlyCharges",
    "TotalCharges",
    "avg_charges_per_month",
    "charge_vs_expected",
    "num_services",
    "gender_Male",
    "Contract_One year",
    "Contract_Two year",
    "PaymentMethod_Credit card (automatic)",
    "PaymentMethod_Electronic check",
    "PaymentMethod_Mailed check",
    "tenure_bucket_13-24",
    "tenure_bucket_25-48",
    "tenure_bucket_49+",
]


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    bins = [-1, 12, 24, 48, np.inf]
    labels = ["0-12", "13-24", "25-48", "49+"]
    df["tenure_bucket"] = pd.cut(df["tenure"], bins=bins, labels=labels)
    df["avg_charges_per_month"] = df["TotalCharges"] / df["tenure"].replace(0, 1)
    df["charge_vs_expected"] = df["MonthlyCharges"] - df["avg_charges_per_month"]
    df["num_services"] = df[SERVICE_LIST].sum(axis=1)
    return df


# One-hot encoding determinístico replicando `pd.get_dummies(drop_first=True)`
# do notebook. Hardcoded porque get_dummies em 1 linha colapsa para 1 coluna
# (a categoria observada vira "first" e é dropada), produzindo zeros silenciosos.
# Cada tupla: (coluna_origem, [valor_dropado_first, ...valores_que_viram_dummies])
ONE_HOT_SPECS: list[tuple[str, list[str]]] = [
    ("gender", ["Female", "Male"]),
    ("Contract", ["Month-to-month", "One year", "Two year"]),
    (
        "PaymentMethod",
        [
            "Bank transfer (automatic)",
            "Credit card (automatic)",
            "Electronic check",
            "Mailed check",
        ],
    ),
    ("tenure_bucket", ["0-12", "13-24", "25-48", "49+"]),
]


def _apply_one_hot(df: pd.DataFrame) -> pd.DataFrame:
    for source_col, ordered_values in ONE_HOT_SPECS:
        # drop_first=True ⇒ skip ordered_values[0]
        for value in ordered_values[1:]:
            df[f"{source_col}_{value}"] = (df[source_col] == value).astype(int)
        df = df.drop(columns=[source_col])
    return df


def preprocess_one(payload: dict[str, Any]) -> pd.DataFrame:
    """Turns a single raw Telco payload into a 1x28 DataFrame.

    Output columns are aligned to ``EXPECTED_FEATURE_ORDER`` (training-time
    feature order). One-hot encoding is applied manually because pandas'
    get_dummies on a single row produces incomplete dummy sets.
    """
    df = pd.DataFrame([payload])
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    for col in BINARY_COLS:
        df[col] = df[col].map(BINARY_MAP)

    df["InternetService"] = df["InternetService"].map(INTERNET_SERVICE_MAP)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    mask = df["TotalCharges"].isna()
    df.loc[mask, "TotalCharges"] = (
        df.loc[mask, "MonthlyCharges"] * df.loc[mask, "tenure"]
    )
    df["TotalCharges"] = np.log1p(df["TotalCharges"])

    df = create_features(df)
    # Convert the categorical bucket to its string label so the one-hot helper
    # can compare it directly.
    df["tenure_bucket"] = df["tenure_bucket"].astype(str)
    df = _apply_one_hot(df)

    return df.reindex(columns=EXPECTED_FEATURE_ORDER, fill_value=0).astype(float)
