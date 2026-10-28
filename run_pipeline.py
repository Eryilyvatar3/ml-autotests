"""
End-to-end пайплайн:
    1. Загрузка датасета.
    2. Разделение на train/test.
    3. Обучение модели.
    4. Предсказания на тесте.
    5. Расчёт метрик (accuracy, precision, recall).
    6. Сохранение модели.
    7. Построение визуализаций (confusion matrix, bar chart, ROC curve).

Запуск:
    python run_pipeline.py
"""

import json
from pathlib import Path

from src.data import load_dataset, split_data
from src.model import train_model, save_model, predict
from src.metrics import calculate_metrics
from src.visualize import (
    plot_confusion_matrix,
    plot_metrics_bar,
    plot_roc_curve,
)


ARTIFACTS_DIR = Path("artifacts")


def main() -> None:
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("1. Загрузка датасета")
    print("=" * 60)
    X, y = load_dataset()
    print(f"X: {X.shape}, y: {y.shape}")
    print(f"Классы: {dict(zip(*[arr.tolist() for arr in __import__('numpy').unique(y, return_counts=True)]))}")

    print("\n" + "=" * 60)
    print("2. Разделение на train/test (80/20, stratify)")
    print("=" * 60)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
    print(f"Train: {X_train.shape[0]} образцов")
    print(f"Test:  {X_test.shape[0]} образцов")

    print("\n" + "=" * 60)
    print("3. Обучение модели (StandardScaler + LogisticRegression)")
    print("=" * 60)
    model = train_model(X_train, y_train, random_state=42)
    print(f"Модель обучена: {type(model).__name__}")
    print(f"Шаги: {[name for name, _ in model.steps]}")

    print("\n" + "=" * 60)
    print("4. Предсказания на тестовой выборке")
    print("=" * 60)
    y_pred = predict(model, X_test)
    print(f"Предсказано: {len(y_pred)} меток")
    print(f"Распределение предсказаний: {dict(zip(*[arr.tolist() for arr in __import__('numpy').unique(y_pred, return_counts=True)]))}")

    print("\n" + "=" * 60)
    print("5. Метрики качества")
    print("=" * 60)
    metrics = calculate_metrics(y_test, y_pred)
    for name, value in metrics.items():
        print(f"  {name:<10}: {value:.4f}")

    print("\n" + "=" * 60)
    print("6. Сохранение модели")
    print("=" * 60)
    model_path = save_model(model)
    print(f"Модель сохранена: {model_path}")

    print("\n" + "=" * 60)
    print("7. Визуализация")
    print("=" * 60)
    cm_path = plot_confusion_matrix(y_test, y_pred)
    print(f"Confusion matrix: {cm_path}")
    bar_path = plot_metrics_bar(metrics)
    print(f"Bar chart метрик: {bar_path}")
    roc_path = plot_roc_curve(model, X_test, y_test)
    print(f"ROC curve:        {roc_path}")

    # Сохраняем метрики в JSON — удобно для CI / отчётов
    metrics_path = ARTIFACTS_DIR / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Метрики (JSON):   {metrics_path}")

    print("\n" + "=" * 60)
    print("Готово.")
    print("=" * 60)


if __name__ == "__main__":
    main()