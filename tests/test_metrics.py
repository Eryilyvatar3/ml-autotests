"""
Тесты на корректность вычисления метрик (accuracy, precision, recall).
"""

import numpy as np
import pytest

from src.metrics import calculate_metrics


# ---------- Точные значения на эталонных примерах ----------

def test_metrics_perfect_predictions():
    """Идеальные предсказания → все метрики = 1.0."""
    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 1, 1, 0])
    m = calculate_metrics(y_true, y_pred)
    assert m["accuracy"] == 1.0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0


def test_metrics_all_wrong_binary():
    """Полностью неверные предсказания на бинарных метках."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 1, 0, 0])
    m = calculate_metrics(y_true, y_pred)
    assert m["accuracy"] == 0.0
    # precision: TP=0, FP=2 → 0/2 = 0
    assert m["precision"] == 0.0
    # recall: TP=0, FN=2 → 0/2 = 0
    assert m["recall"] == 0.0


def test_metrics_hand_computed():
    """
    Ручной расчёт на конкретном примере.
    y_true = [1,0,1,1,0,0,1,0], y_pred = [1,0,1,0,0,1,1,0]
    accuracy  = 6/8 = 0.75
    precision = 3/(3+1) = 0.75
    recall    = 3/(3+1) = 0.75
    """
    y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0])
    m = calculate_metrics(y_true, y_pred)
    assert m["accuracy"] == pytest.approx(0.75)
    assert m["precision"] == pytest.approx(0.75)
    assert m["recall"] == pytest.approx(0.75)


def test_metrics_precision_vs_recall_different():
    """
    Пример, где precision != recall, чтобы убедиться, что метрики не перепутаны.
    y_true = [1, 1, 1, 1, 0]  (4 positive, 1 negative)
    y_pred = [1, 1, 0, 0, 0]  (2 предсказано positive)
    TP=2, FP=0, FN=2, TN=1
    precision = 2 / (2 + 0) = 1.0
    recall    = 2 / (2 + 2) = 0.5
    accuracy  = 3 / 5 = 0.6
    """
    y_true = np.array([1, 1, 1, 1, 0])
    y_pred = np.array([1, 1, 0, 0, 0])
    m = calculate_metrics(y_true, y_pred)
    assert m["accuracy"] == pytest.approx(0.6)
    assert m["precision"] == pytest.approx(1.0)
    assert m["recall"] == pytest.approx(0.5)


def test_metrics_zero_division_no_crash():
    """
    Если модель не предсказала ни одного positive, precision должен быть 0
    (а не падать/NaN). Аналогично recall=0, если нет реальных positive.
    """
    # Нет предсказанных positive
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([0, 0, 0, 0])
    m = calculate_metrics(y_true, y_pred)
    assert m["precision"] == 0.0  # zero_division=0

    # Нет реальных positive
    y_true2 = np.array([0, 0, 0, 0])
    y_pred2 = np.array([0, 1, 0, 1])
    m2 = calculate_metrics(y_true2, y_pred2)
    assert m2["recall"] == 0.0


def test_metrics_range():
    """Все метрики в диапазоне [0, 1]."""
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=200)
    y_pred = rng.integers(0, 2, size=200)
    m = calculate_metrics(y_true, y_pred)
    for name, value in m.items():
        assert 0.0 <= value <= 1.0, f"{name}={value} вне [0, 1]"


def test_metrics_keys():
    """calculate_metrics возвращает ровно 3 ключа: accuracy, precision, recall."""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0])
    m = calculate_metrics(y_true, y_pred)
    assert set(m.keys()) == {"accuracy", "precision", "recall"}


def test_metrics_returns_float():
    """Все значения — float (не np.float64), для сериализации в JSON."""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0])
    m = calculate_metrics(y_true, y_pred)
    for name, value in m.items():
        assert type(value) is float, f"{name} имеет тип {type(value)}, ожидался float"


def test_metrics_empty_raises():
    """Пустые массивы → ValueError."""
    with pytest.raises(ValueError, match="не должны быть пустыми"):
        calculate_metrics(np.array([]), np.array([]))


def test_metrics_mismatched_lengths_raises():
    """Разные длины y_true/y_pred → ValueError."""
    with pytest.raises(ValueError, match="Длины не совпадают"):
        calculate_metrics(np.array([0, 1, 1]), np.array([0, 1]))


# ---------- Смоук-проверка на реальной модели ----------

def test_metrics_on_trained_model(trained_model, split):
    """
    Метрики на реальной модели: accuracy/precision/recall > 0.9.
    Breast cancer — хорошо разделяемый датасет, ниже 0.9 — признак поломки.
    """
    _, X_test, _, y_test = split
    y_pred = trained_model.predict(X_test)
    m = calculate_metrics(y_test, y_pred)

    assert m["accuracy"] > 0.9, f"accuracy={m['accuracy']:.3f} < 0.9"
    assert m["precision"] > 0.9, f"precision={m['precision']:.3f} < 0.9"
    assert m["recall"] > 0.9, f"recall={m['recall']:.3f} < 0.9"