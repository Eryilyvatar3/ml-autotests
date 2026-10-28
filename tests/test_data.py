"""
Тесты на корректность загрузки данных и разделения на train/test.
"""

import numpy as np
import pytest

from src.data import load_dataset, split_data


def test_load_dataset_shapes(dataset):
    """X и y имеют правильные размеры и количество признаков."""
    X, y = dataset
    assert X.ndim == 2, "X должен быть 2D-матрицей"
    assert y.ndim == 1, "y должен быть 1D-вектором"
    assert X.shape[0] == y.shape[0] == 569, f"Ожидалось 569 образцов, получено {X.shape[0]}"
    assert X.shape[1] == 30, f"Ожидалось 30 признаков, получено {X.shape[1]}"


def test_load_dataset_no_nan(dataset):
    """В признаках и метках не должно быть NaN."""
    X, y = dataset
    assert not np.isnan(X).any(), "В X есть NaN"
    assert not np.isnan(y).any(), "В y есть NaN"


def test_load_dataset_binary_labels(dataset):
    """Метки должны быть только 0 и 1."""
    _, y = dataset
    unique = set(np.unique(y).tolist())
    assert unique == {0, 1}, f"Ожидались метки {{0, 1}}, получено {unique}"


def test_load_dataset_has_both_classes(dataset):
    """В датасете присутствуют оба класса (иначе stratification не сработает)."""
    _, y = dataset
    counts = np.bincount(y)
    assert counts[0] > 0 and counts[1] > 0, f"Один из классов отсутствует: {counts}"


def test_split_sizes(split):
    """Train/test имеют правильные размеры (80/20 от 569)."""
    X_train, X_test, y_train, y_test = split
    total = X_train.shape[0] + X_test.shape[0]
    assert total == 569, f"Сумма train+test должна быть 569, получено {total}"
    assert X_test.shape[0] == 114, f"test должен быть ~20% (114), получено {X_test.shape[0]}"
    assert X_train.shape[0] == 455, f"train должен быть ~80% (455), получено {X_train.shape[0]}"


def test_split_preserves_features(split, dataset):
    """Количество признаков в train/test совпадает с исходным."""
    X, _ = dataset
    X_train, X_test, _, _ = split
    assert X_train.shape[1] == X.shape[1] == X_test.shape[1]


def test_split_stratified(split, dataset):
    """
    Проверка стратификации: доли классов в train и test близки к исходной.
    Допускаем отклонение ±5%.
    """
    _, y = dataset
    _, _, y_train, y_test = split

    overall_pos = y.mean()
    train_pos = y_train.mean()
    test_pos = y_test.mean()

    assert abs(train_pos - overall_pos) < 0.05, (
        f"Доля позитивного класса в train ({train_pos:.3f}) сильно отличается от исходной ({overall_pos:.3f})"
    )
    assert abs(test_pos - overall_pos) < 0.05, (
        f"Доля позитивного класса в test ({test_pos:.3f}) сильно отличается от исходной ({overall_pos:.3f})"
    )


def test_split_reproducible():
    """При одинаковом random_state split воспроизводим."""
    X, y = load_dataset()
    split1 = split_data(X, y, test_size=0.2, random_state=42)
    split2 = split_data(X, y, test_size=0.2, random_state=42)

    for arr1, arr2 in zip(split1, split2):
        assert np.array_equal(arr1, arr2), "Два вызова с одним random_state дали разные данные"


def test_split_invalid_test_size():
    """test_size вне (0, 1) → ValueError."""
    X, y = load_dataset()
    with pytest.raises(ValueError, match="test_size"):
        split_data(X, y, test_size=1.5)