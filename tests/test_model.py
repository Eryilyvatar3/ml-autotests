"""
Тесты на корректность обучения модели.
"""

import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from src.model import train_model, predict


def test_train_returns_pipeline(trained_model):
    """train_model возвращает sklearn Pipeline (scaler + clf)."""
    assert isinstance(trained_model, Pipeline), f"Ожидался Pipeline, получен {type(trained_model)}"
    step_names = [name for name, _ in trained_model.steps]
    assert "scaler" in step_names, f"В Pipeline нет scaler: {step_names}"
    assert "clf" in step_names, f"В Pipeline нет clf: {step_names}"


def test_train_produces_fitted_model(trained_model, split):
    """Модель обучена: есть атрибут classes_ и она может предсказывать."""
    X_train, X_test, _, _ = split
    assert hasattr(trained_model, "classes_"), "У модели нет атрибута classes_"
    assert set(trained_model.classes_.tolist()) == {0, 1}, (
        f"Классы модели: {trained_model.classes_}, ожидались {{0, 1}}"
    )
    # predict не должен падать
    preds = trained_model.predict(X_test)
    assert preds.shape[0] == X_test.shape[0]


def test_train_reproducible(split):
    """При одинаковом random_state две модели дают идентичные предсказания."""
    X_train, X_test, y_train, _ = split
    m1 = train_model(X_train, y_train, random_state=42)
    m2 = train_model(X_train, y_train, random_state=42)

    preds1 = m1.predict(X_test)
    preds2 = m2.predict(X_test)
    assert np.array_equal(preds1, preds2), "Модели с одинаковым random_state дали разные предсказания"


def test_train_empty_data_raises():
    """Пустые данные → ValueError."""
    X_empty = np.empty((0, 30))
    y_empty = np.empty((0,))
    with pytest.raises(ValueError, match="пусты"):
        train_model(X_empty, y_empty)


def test_train_mismatched_lengths_raises():
    """Разные длины X_train и y_train → ValueError."""
    X_bad = np.random.rand(10, 30)
    y_bad = np.random.randint(0, 2, size=5)
    with pytest.raises(ValueError, match="Несовпадение размеров"):
        train_model(X_bad, y_bad)


def test_predict_returns_binary_labels(trained_model, split):
    """Предсказания — только 0 и 1, длина совпадает с тестовой выборкой."""
    _, X_test, _, _ = split
    preds = predict(trained_model, X_test)
    assert set(np.unique(preds).tolist()).issubset({0, 1}), (
        f"Предсказания содержат не 0/1: {np.unique(preds)}"
    )
    assert preds.shape[0] == X_test.shape[0]


def test_predict_empty_X_raises(trained_model):
    """Пустая матрица X → ValueError."""
    with pytest.raises(ValueError, match="пуста"):
        predict(trained_model, np.empty((0, 30)))


def test_predict_none_model_raises():
    """None вместо модели → ValueError."""
    with pytest.raises(ValueError, match="не обучена"):
        predict(None, np.random.rand(5, 30))


def test_predict_accuracy_baseline(trained_model, split):
    """
    Смоук-проверка качества: accuracy на тесте должна быть > 0.9.
    Это не «жёсткий» тест метрики (он в test_metrics.py), а проверка,
    что модель не сломана и не выдаёт случайный шум.
    """
    _, X_test, _, y_test = split
    preds = predict(trained_model, X_test)
    accuracy = (preds == y_test).mean()
    assert accuracy > 0.9, f"Слишком низкая accuracy: {accuracy:.3f} (ожидалось > 0.9)"