"""Construtor canônico do ``sklearn.Pipeline`` usado pelo modelo de produção.

A Pipeline encapsula:

* :class:`FeatureEngineer` — engenharia de features determinística (one-hot
  fixo, mapeamentos binários, derivações tenure/charges) que parte do payload
  bruto do Telco.
* :class:`StandardScaler` — z-score das 28 colunas resultantes; ajustado no
  treino, reaplicado em inferência.
* :class:`LogisticRegression` — classificador final com
  ``class_weight='balanced'``.

Single source of truth: tanto ``notebooks/modeling.ipynb`` (treinamento
manual ao reexecutar o notebook) quanto ``scripts/train_pipeline.py``
(retreino operacional headless) chamam :func:`build_logreg_pipeline`. O
mesmo objeto é serializado no MLflow Registry como
``Churn_LogReg_Final_Production`` e carregado pela API em
``src/infrastructure/mlflow_loader.py``.
"""

from __future__ import annotations

from typing import Any

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .transformers import FeatureEngineer

DEFAULT_LOGREG_PARAMS: dict[str, Any] = {
    "max_iter": 1000,
    "random_state": 42,
    "class_weight": "balanced",
    "solver": "lbfgs",
}


def build_logreg_pipeline(**logreg_overrides: Any) -> Pipeline:
    """Build the production-grade LogReg Pipeline.

    Parameters mirror the baseline LogReg trained in ``notebooks/eda.ipynb``
    (``class_weight='balanced'``, ``solver='lbfgs'``, ``max_iter=1000``).
    Any keyword override here is forwarded to :class:`LogisticRegression`.
    """
    params = {**DEFAULT_LOGREG_PARAMS, **logreg_overrides}
    return Pipeline(
        steps=[
            ("features", FeatureEngineer()),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(**params)),
        ]
    )
