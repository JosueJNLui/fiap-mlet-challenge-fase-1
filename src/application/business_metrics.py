"""Business metrics for churn prediction — single source of truth.

Centraliza o cálculo de lucro líquido e a busca de threshold ótimo
compartilhados entre os notebooks de EDA, modelagem e comparação.

Os custos refletem a regra de negócio acordada com a área:
``Lucro = TP * (LTV - Custo_retencao) - FP * Custo_retencao - FN * LTV``.
Mantém este módulo em sincronia com `MODEL_CARD.md` §5.
"""

from __future__ import annotations

from typing import Literal, overload

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

VALOR_CLIENTE_LTV: int = 500
CUSTO_RETENCAO: int = 100


def _net_profit(tp: int, fp: int, fn: int) -> float:
    return float(
        tp * (VALOR_CLIENTE_LTV - CUSTO_RETENCAO)
        - fp * CUSTO_RETENCAO
        - fn * VALOR_CLIENTE_LTV
    )


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_points: int = 100,
) -> float:
    """Varre `n_points` thresholds em [0.01, 0.99] e devolve o que maximiza o lucro líquido."""
    best_threshold, max_lucro = 0.5, -float("inf")
    for t in np.linspace(0.01, 0.99, n_points):
        y_pred = (y_proba >= t).astype(int)
        try:
            _tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        except ValueError:
            continue
        lucro = _net_profit(tp, fp, fn)
        if lucro > max_lucro:
            max_lucro, best_threshold = lucro, float(t)
    return best_threshold


@overload
def calculate_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    return_confusion_matrix: Literal[False] = False,
) -> dict: ...


@overload
def calculate_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    return_confusion_matrix: Literal[True],
) -> tuple[dict, np.ndarray]: ...


def calculate_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    return_confusion_matrix: bool = False,
) -> dict | tuple[dict, np.ndarray]:
    """Devolve 6 métricas técnicas + 4 de negócio + threshold para um corte de probabilidade.

    Quando `return_confusion_matrix=True`, retorna `(dict, cm)` — preserva o uso atual
    do `notebooks/modeling.ipynb` que precisa da matriz para visualizações.
    """
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    _tn, fp, fn, tp = cm.ravel()

    pr_p, pr_r, _ = precision_recall_curve(y_true, y_proba)
    ganho_tp = tp * (VALOR_CLIENTE_LTV - CUSTO_RETENCAO)
    custo_fp = fp * CUSTO_RETENCAO
    perda_fn = fn * VALOR_CLIENTE_LTV

    metrics: dict = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred, pos_label=1),
        "precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=1),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": auc(pr_r, pr_p),
        "custo_churn_evitado_BRL": float(ganho_tp),
        "custo_falso_positivo_BRL": float(custo_fp),
        "custo_churn_perdido_BRL": float(perda_fn),
        "lucro_liquido_BRL": float(ganho_tp - custo_fp - perda_fn),
        "optimal_threshold": float(threshold),
    }
    if return_confusion_matrix:
        return metrics, cm
    return metrics
