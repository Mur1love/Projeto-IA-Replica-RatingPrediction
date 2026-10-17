"""Figuras de avaliação (Issue 7)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    path: str | Path,
    *,
    display_labels: tuple = (1, 2, 3, 4, 5),
) -> Path:
    """Salva a matriz de confusão (labels 0..4, exibidos 1..5) em PNG."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])
    _, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=list(display_labels)).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title("Matriz de confusão — Regressão Logística (TF-IDF)")
    ax.set_xlabel("Predito (rating)")
    ax.set_ylabel("Verdadeiro (rating)")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path
