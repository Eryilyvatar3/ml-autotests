"""
Визуализация результатов модели:
1. Матрица ошибок (confusion matrix).
2. Столбчатая диаграмма метрик (accuracy / precision / recall).
3. ROC-кривая с AUC.

Все графики сохраняются в artifacts/.
"""

import os
from pathlib import Path
from typing import Dict, Union

# Устанавливаем MPLCONFIGDIR до импорта matplotlib, чтобы кэш шрифтов
# писался в проект, а не в C:\Users\<user>\AppData\Local\matplotlib
# (на этой машине туда нет доступа).
_MPL_CACHE_DIR = Path(__file__).resolve().parent.parent / ".mplconfig"
_MPL_CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE_DIR))

import matplotlib
matplotlib.use("Agg")  # headless-режим — не требует GUI, работает в CI/консоли

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay


ARTIFACTS_DIR = Path("artifacts")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Union[str, Path] = ARTIFACTS_DIR / "confusion_matrix.png",
) -> Path:
    """
    Строит и сохраняет матрицу ошибок.

    Returns:
        Path: путь к сохранённому PNG.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, ax=ax, cmap="Blues", colorbar=True
    )
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


def plot_metrics_bar(
    metrics: Dict[str, float],
    output_path: Union[str, Path] = ARTIFACTS_DIR / "metrics.png",
) -> Path:
    """
    Строит и сохраняет столбчатую диаграмму метрик.

    Args:
        metrics: dict вида {'accuracy': ..., 'precision': ..., 'recall': ...}.

    Returns:
        Path: путь к сохранённому PNG.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    names = list(metrics.keys())
    values = [metrics[k] for k in names]

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x=names, y=values, ax=ax, hue=names, palette="viridis", legend=False)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Metrics")
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path


def plot_roc_curve(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_path: Union[str, Path] = ARTIFACTS_DIR / "roc_curve.png",
) -> Path:
    """
    Строит и сохраняет ROC-кривую (использует predict_proba модели).

    Returns:
        Path: путь к сохранённому PNG.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
    ax.set_title("ROC Curve")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return output_path