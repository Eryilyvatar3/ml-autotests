# ML Autotests — scikit-learn + метрики + визуализация

Система на Python 3, которая:
1. Загружает датасет.
2. Обучает модель бинарной классификации.
3. Делает предсказания на тестовой выборке.
4. Считает метрики качества (**accuracy**, **precision**, **recall**).
5. Строит визуализации (**confusion matrix**, **bar chart метрик**, **ROC-кривая**).
6. Покрыта автотестами **pytest** (37 тестов).

## Что реализовано

| Требование ДЗ | Реализация |
|---|---|
| Загрузить датасет и разделить на train/test | `src/data.py`: `load_dataset()`, `split_data()` (80/20, stratify, random_state=42) |
| Обучить модель | `src/model.py`: `train_model()` — `Pipeline(StandardScaler → LogisticRegression)` |
| Предсказания на тесте | `src/model.py`: `predict()` |
| 3 метрики | `src/metrics.py`: `calculate_metrics()` — accuracy, precision, recall |
| Автотесты (данные, модель, метрики, сохранение) | `tests/test_data.py`, `tests/test_model.py`, `tests/test_metrics.py`, `tests/test_save.py` |
| Визуализация | `src/visualize.py`: confusion matrix, bar chart метрик, ROC-кривая (matplotlib + seaborn) |

## Датасет и модель

- **Датасет:** Breast Cancer Wisconsin (Diagnostic) — встроен в `sklearn.datasets`.
  - 569 образцов, 30 признаков, бинарная классификация (`malignant=0`, `benign=1`).
  - Распределение классов: 212 / 357.
- **Модель:** `Pipeline(StandardScaler → LogisticRegression(max_iter=1000, random_state=42))`.
  - Масштабирование обязательно для корректной сходимости `lbfgs`.
- **Сплит:** 80/20 со `stratify=y`, `random_state=42` → 455 train / 114 test.

## Структура проекта
ml-autotests/
├── src/
│ ├── init.py
│ ├── data.py # загрузка + split
│ ├── model.py # train / save / load / predict
│ ├── metrics.py # accuracy / precision / recall
│ └── visualize.py # confusion matrix, bar chart, ROC curve
├── tests/
│ ├── init.py
│ ├── test_data.py # 9 тестов: корректность загрузки данных
│ ├── test_model.py # 9 тестов: корректность обучения модели
│ ├── test_metrics.py # 11 тестов: корректность вычисления метрик
│ └── test_save.py # 8 тестов: сохранение/загрузка модели
├── artifacts/ # результаты: model.joblib, *.png, metrics.json
├── conftest.py # фикстуры (dataset, split, trained_model)
├── run_pipeline.py # end-to-end скрипт
├── pytest.ini
├── requirements.txt
├── .gitignore
└── README.md

text

## Запуск

### 1. Создать окружение и установить зависимости

```bash
python -m venv .venv
.venv\Scripts\activate.bat      # Windows cmd
# source .venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
2. Запустить пайплайн (обучение + метрики + графики)
bash
python run_pipeline.py
Результаты сохраняются в artifacts/:

model.joblib — обученная модель;

confusion_matrix.png — матрица ошибок;

metrics.png — столбчатая диаграмма accuracy / precision / recall;

roc_curve.png — ROC-кривая с AUC;

metrics.json — метрики в машиночитаемом виде.

3. Запустить автотесты
bash
pytest
Или по категориям:

bash
pytest tests/test_data.py -v
pytest tests/test_model.py -v
pytest tests/test_metrics.py -v
pytest tests/test_save.py -v
Результаты прогона
Метрики на тестовой выборке (114 образцов):

Метрика	Значение
Accuracy	0.9825
Precision	0.9861
Recall	0.9861
Автотесты:

text
collected 37 items
============================ 37 passed in 0.14s ============================
Файл тестов	Кол-во	Что покрывает
tests/test_data.py	9	Размерности X (569×30) и y, отсутствие NaN, метки {0,1}, оба класса, размеры сплита 455/114, стратификация (±5%), воспроизводимость, валидация test_size
tests/test_model.py	9	Тип Pipeline с шагами scaler/clf, обученность (classes_, predict), воспроизводимость, ошибки на пустых/несовпадающих данных, predict ∈ {0,1}, predict(None) и predict(empty) → ValueError, смоук accuracy > 0.9
tests/test_metrics.py	11	Идеальные/полностью неверные предсказания, ручной расчёт (0.75/0.75/0.75), precision ≠ recall (разные метрики), zero_division=0, диапазон [0,1], ровно 3 ключа, тип float, ошибки на пустых/разной длины, смоук на реальной модели > 0.9
tests/test_save.py	8	Файл создаётся и не пустой, автосоздание папок, save(None) → ValueError, load(nonexistent) → FileNotFoundError, roundtrip (предсказания после save/load идентичны), сохранение типа Pipeline и classes_, путь по умолчанию
Визуализация
Три графика в artifacts/:

confusion_matrix.png — матрица ошибок (2×2): TP, TN, FP, FN. Показывает, где модель ошибается.

metrics.png — столбчатая диаграмма accuracy / precision / recall с подписанными значениями. Наглядное сравнение метрик.

roc_curve.png — ROC-кривая с AUC. Показывает качество разделения классов во всём диапазоне порогов.

Все графики строятся через matplotlib + seaborn в headless-режиме (Agg), сохраняются в PNG без GUI.

Как это работает
conftest.py — фикстуры session-scope: датасет загружается и модель обучается один раз на весь прогон. Тесты работают за 0.14s, а не за минуты.

src/ — модули с чистой логикой и валидацией входов. Каждая функция покрыта тестами на позитив/негатив/edge-cases.

run_pipeline.py — end-to-end скрипт для демонстрации: проходит все 7 шагов и печатает отчёт.