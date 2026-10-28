"""
Общие фикстуры pytest.

Фикстуры:
- dataset: (X, y) — загруженный датасет, session-scoped (один раз на прогон).
- split: (X_train, X_test, y_train, y_test) — session-scoped.
- trained_model: обученная Pipeline, session-scoped (обучение 1 раз).
"""

import numpy as np
import pytest

from src.data import load_dataset, split_data
from src.model import train_model


@pytest.fixture(scope="session")
def dataset():
    """Загруженный датасет (X, y)."""
    return load_dataset()


@pytest.fixture(scope="session")
def split(dataset):
    """Разделённые данные (X_train, X_test, y_train, y_test)."""
    X, y = dataset
    return split_data(X, y, test_size=0.2, random_state=42)


@pytest.fixture(scope="session")
def trained_model(split):
    """Обученная модель (обучается 1 раз на весь прогон)."""
    X_train, X_test, y_train, y_test = split
    return train_model(X_train, y_train, random_state=42)