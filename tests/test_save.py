"""
Тесты на сохранение и загрузку модели.
"""

import numpy as np
import pytest

from src.model import save_model, load_model, predict


def test_save_creates_file(trained_model, tmp_path):
    """save_model создаёт файл по указанному пути."""
    path = tmp_path / "test_model.joblib"
    assert not path.exists(), "Файл не должен существовать до сохранения"

    saved_path = save_model(trained_model, path)

    assert saved_path.exists(), "Файл модели не создан"
    assert saved_path.stat().st_size > 0, "Файл модели пустой"


def test_save_creates_parent_directory(trained_model, tmp_path):
    """save_model создаёт родительские папки, если их нет."""
    path = tmp_path / "nested" / "dir" / "model.joblib"
    assert not path.parent.exists()

    saved_path = save_model(trained_model, path)

    assert saved_path.exists()
    assert saved_path.parent.exists()


def test_save_none_raises(tmp_path):
    """Сохранение None → ValueError."""
    with pytest.raises(ValueError, match="Нельзя сохранить None"):
        save_model(None, tmp_path / "model.joblib")


def test_load_nonexistent_raises(tmp_path):
    """Загрузка несуществующего файла → FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="не найден"):
        load_model(tmp_path / "no_such_model.joblib")


def test_save_load_roundtrip(trained_model, split, tmp_path):
    """
    Сохранение + загрузка → предсказания идентичны исходной модели.
    Это ключевой тест: проверяет, что joblib сохраняет и восстанавливает
    модель без потерь.
    """
    _, X_test, _, _ = split
    path = tmp_path / "roundtrip_model.joblib"

    save_model(trained_model, path)
    loaded_model = load_model(path)

    preds_original = predict(trained_model, X_test)
    preds_loaded = predict(loaded_model, X_test)

    assert np.array_equal(preds_original, preds_loaded), (
        "Загруженная модель даёт другие предсказания"
    )


def test_save_load_preserves_pipeline_type(trained_model, tmp_path):
    """Загруженная модель — Pipeline с теми же шагами."""
    from sklearn.pipeline import Pipeline

    path = tmp_path / "model.joblib"
    save_model(trained_model, path)
    loaded = load_model(path)

    assert isinstance(loaded, Pipeline), f"Ожидался Pipeline, получен {type(loaded)}"
    step_names_original = [name for name, _ in trained_model.steps]
    step_names_loaded = [name for name, _ in loaded.steps]
    assert step_names_original == step_names_loaded, (
        f"Шаги Pipeline не совпадают: {step_names_original} vs {step_names_loaded}"
    )


def test_save_load_preserves_classes(trained_model, tmp_path):
    """Загруженная модель возвращает те же классы {0, 1}."""
    path = tmp_path / "model.joblib"
    save_model(trained_model, path)
    loaded = load_model(path)

    assert set(loaded.classes_.tolist()) == set(trained_model.classes_.tolist())


def test_save_default_path(trained_model):
    """
    Без явного пути save_model пишет в artifacts/model.joblib (DEFAULT_MODEL_PATH).
    Файл перезаписывается — тест не ломает существующий artifacts.
    """
    from pathlib import Path

    saved_path = save_model(trained_model)
    assert saved_path == Path("artifacts/model.joblib")
    assert saved_path.exists()