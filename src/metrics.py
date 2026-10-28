"""
Расчёт метрик качества бинарной классификации.

Метрики по требованиям ДЗ:
- accuracy  — доля верно предсказанных случаев;
- precision — доля верно предсказанных положительных среди всех предсказанных положительных;
- recall    — доля верно предсказанных положительных среди всех реальных положительных.
"""

from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Вычисляет accuracy, precision, recall для бинарной классификации.

    Args:
        y_true: истинные метки (0/1).
        y_pred: предсказанные метки (0/1).

    Returns:
        dict с ключами 'accuracy', 'precision', 'recall'.

    Raises:
        ValueError: если массивы пусты или разной длины.
    """
    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("Массивы y_true и y_pred не должны быть пустыми")
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Длины не совпадают: len(y_true)={len(y_true)}, len(y_pred)={len(y_pred)}"
        )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }