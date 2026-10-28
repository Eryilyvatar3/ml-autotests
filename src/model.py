"""
Обучение и сохранение/загрузка ML-модели.

Модель: Pipeline(StandardScaler → LogisticRegression).
Масштабирование признаков обязательно для корректной сходимости LogisticRegression.
"""

from pathlib import Path
from typing import Union

import joblib
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_MODEL_PATH = Path("artifacts/model.joblib")


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
    max_iter: int = 1000,
) -> Pipeline:
    """
    Обучает Pipeline(StandardScaler → LogisticRegression) на тренировочных данных.

    Args:
        X_train, y_train: тренировочные данные.
        random_state: seed для воспроизводимости.
        max_iter: максимальное число итераций solver'а.

    Returns:
        Обученный sklearn Pipeline.

    Raises:
        ValueError: если X_train или y_train пусты или размеры не совпадают.
    """
    if X_train.shape[0] == 0 or y_train.shape[0] == 0:
        raise ValueError("Тренировочные данные пусты")
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError(
            f"Несовпадение размеров: X_train={X_train.shape[0]}, y_train={y_train.shape[0]}"
        )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=random_state, max_iter=max_iter)),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def save_model(model: BaseEstimator, path: Union[str, Path] = DEFAULT_MODEL_PATH) -> Path:
    """
    Сохраняет обученную модель в файл через joblib.
    """
    if model is None:
        raise ValueError("Нельзя сохранить None вместо модели")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path: Union[str, Path] = DEFAULT_MODEL_PATH) -> BaseEstimator:
    """
    Загружает модель из файла.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Файл модели не найден: {path}")
    return joblib.load(path)


def predict(model: BaseEstimator, X: np.ndarray) -> np.ndarray:
    """
    Возвращает предсказания модели.
    """
    if model is None:
        raise ValueError("Модель не обучена (None)")
    if X.shape[0] == 0:
        raise ValueError("Матрица признаков пуста")
    return model.predict(X)