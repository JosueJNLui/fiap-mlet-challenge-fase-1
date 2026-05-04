from __future__ import annotations

import numpy as np
import pytest

from src.application.business_metrics import (
    CUSTO_RETENCAO,
    VALOR_CLIENTE_LTV,
    calculate_metrics,
    find_optimal_threshold,
)


@pytest.fixture
def perfectly_separable() -> tuple[np.ndarray, np.ndarray]:
    """y_proba ordenada por classe — threshold ótimo deve dar lucro positivo máximo."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_proba = np.array([0.05, 0.1, 0.15, 0.2, 0.7, 0.8, 0.9, 0.95])
    return y_true, y_proba


def test_find_optimal_threshold_returns_value_in_search_range() -> None:
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=200)
    y_proba = rng.random(200)

    t = find_optimal_threshold(y_true, y_proba)

    assert isinstance(t, float)
    assert 0.01 <= t <= 0.99


def test_find_optimal_threshold_favors_recall_under_asymmetric_costs() -> None:
    # Como FN custa 5x mais que FP (LTV R$500 vs custo retencao R$100), a curva
    # de lucro deve ser maximizada em um threshold relativamente baixo. Aqui, com
    # positivos concentrados em proba 0.3-0.5 e negativos em 0.0-0.4, qualquer
    # threshold ≤ 0.5 captura praticamente todos os churns reais — o ótimo deve
    # ser claramente abaixo do default de 0.5.
    rng = np.random.default_rng(0)
    y_true = np.concatenate([np.zeros(800, dtype=int), np.ones(200, dtype=int)])
    y_proba = np.concatenate(
        [
            rng.uniform(0.0, 0.4, size=800),  # negativos
            rng.uniform(0.3, 0.5, size=200),  # positivos (sobreposição parcial)
        ]
    )

    t = find_optimal_threshold(y_true, y_proba)

    assert t < 0.5, f"esperado threshold < 0.5 sob custo assimétrico, obtido {t}"


def test_calculate_metrics_lucro_liquido_obeys_business_rule(
    perfectly_separable: tuple[np.ndarray, np.ndarray],
) -> None:
    y_true, y_proba = perfectly_separable
    out = calculate_metrics(y_true, y_proba, threshold=0.5)

    expected = (
        out["custo_churn_evitado_BRL"]
        - out["custo_falso_positivo_BRL"]
        - out["custo_churn_perdido_BRL"]
    )
    assert out["lucro_liquido_BRL"] == pytest.approx(expected)
    # Caso perfeitamente separavel: 4 TP, 0 FP, 0 FN -> lucro = 4 * (LTV - Custo)
    assert out["lucro_liquido_BRL"] == pytest.approx(4 * (VALOR_CLIENTE_LTV - CUSTO_RETENCAO))


def test_calculate_metrics_returns_confusion_matrix_when_requested(
    perfectly_separable: tuple[np.ndarray, np.ndarray],
) -> None:
    y_true, y_proba = perfectly_separable
    result = calculate_metrics(y_true, y_proba, threshold=0.5, return_confusion_matrix=True)

    assert isinstance(result, tuple)
    metrics, cm = result
    assert isinstance(metrics, dict)
    assert cm.shape == (2, 2)
    # 4 negativos corretos, 4 positivos corretos
    assert cm[0, 0] == 4 and cm[1, 1] == 4


def test_calculate_metrics_dummy_majority_baseline_is_unprofitable() -> None:
    # Dataset 73/27 (proporções do Telco) com classificador majoritário
    # (sempre 0): zera TP, gera FN para todos os positivos. Lucro tem que
    # ser negativo — sanity check do baseline da Etapa 1.
    y_true = np.concatenate([np.zeros(730, dtype=int), np.ones(270, dtype=int)])
    y_proba = np.zeros(1000)  # sempre prediz 0

    out = calculate_metrics(y_true, y_proba, threshold=0.5)

    assert out["lucro_liquido_BRL"] < 0
    assert out["custo_churn_perdido_BRL"] == 270 * VALOR_CLIENTE_LTV
    assert out["custo_falso_positivo_BRL"] == 0


def test_find_optimal_threshold_handles_single_class_data() -> None:
    # Dataset degenerado (so classe 0): a varredura nao deve falhar nem
    # emitir warnings do sklearn (`labels=[0, 1]` forca matriz 2x2). Como o
    # lucro e 0 em todos os thresholds, o primeiro da grade e selecionado.
    y_true = np.zeros(10, dtype=int)
    y_proba = np.zeros(10)

    t = find_optimal_threshold(y_true, y_proba)

    assert t == pytest.approx(0.01)
