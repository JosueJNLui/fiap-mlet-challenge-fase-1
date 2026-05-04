"""Transformadores sklearn-compatíveis usados pelo pipeline de churn.

A engenharia de features replica exatamente o que ``notebooks/modeling.ipynb``
faz. A versão original vive em :mod:`preprocessing` como ``preprocess_one``
(orientado a 1 payload), :class:`FeatureEngineer` reusa as mesmas constantes
e helpers, mas opera em DataFrames de qualquer tamanho, podendo entrar em
um ``sklearn.Pipeline`` no caminho de treino e inferência.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .preprocessing import (
    BINARY_COLS,
    BINARY_MAP,
    EXPECTED_FEATURE_ORDER,
    INTERNET_SERVICE_MAP,
    _apply_one_hot,
    create_features,
)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Feature engineering stateless para o dataset Telco Customer Churn.

    Produz a matriz float de 28 colunas que o StandardScaler espera, na ordem
    exata definida por ``EXPECTED_FEATURE_ORDER``. Stateless porque todos os
    encoders são determinísticos e baseados num vocabulário fixo; ``fit`` é
    no-op, mantido apenas por compatibilidade com sklearn.
    """

    def fit(self, X: pd.DataFrame, y: Any = None) -> FeatureEngineer:
        del X, y  # stateless; aceitos apenas por compatibilidade com sklearn
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
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
        df["tenure_bucket"] = df["tenure_bucket"].astype(str)
        df = _apply_one_hot(df)

        return df.reindex(columns=EXPECTED_FEATURE_ORDER, fill_value=0).astype(float)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        del input_features  # nomes de saída são fixados por EXPECTED_FEATURE_ORDER
        return np.array(EXPECTED_FEATURE_ORDER)
