"""
Загрузка датасета и разделение на train/test.

Датасет: Breast Cancer Wisconsin (Diagnostic) — встроен в sklearn.
Задача: бинарная классификация (malignant = 0, benign = 1).
"""

from typing import Tuple

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def load_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """
    Загружает датасет Breast Cancer.

    Returns:
        X: матрица признаков (n_samples, n_features)
        y: вектор меток (n_samples,), значения 0/1

    Raises:
        RuntimeError: если данные некорректны (NaN, пустые).
    """
    data = load_breast_cancer()
    X, y = data.data, data.target

    if X.shape[0] == 0:
        raise RuntimeError("Датасет пустой")
    if np.isnan(X).any():
        raise RuntimeError("В признаках есть NaN")
    if not set(np.unique(y)).issubset({0, 1}):
        raise RuntimeError(f"Метки должны быть 0/1, получено: {np.unique(y)}")

    return X, y


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Делит данные на train/test со стратификацией по классам.

    Args:
        X, y: признаки и метки.
        test_size: доля тестовой выборки (0 < test_size < 1).
        random_state: seed для воспроизводимости.

    Returns:
        X_train, X_test, y_train, y_test

    Raises:
        ValueError: если test_size вне (0, 1).
    """
    if not 0 < test_size < 1:
        raise ValueError(f"test_size должен быть в (0, 1), получено {test_size}")

    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )