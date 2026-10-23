"""Matriz de confusão (Issue 7) — RED."""

import numpy as np

from src.evaluation.plots import save_confusion_matrix


def test_save_confusion_matrix_cria_png(tmp_path):
    y_true = np.array([0, 1, 2, 3, 4, 4, 3])
    y_pred = np.array([0, 1, 2, 2, 4, 4, 3])
    out = tmp_path / "cm.png"
    assert save_confusion_matrix(y_true, y_pred, out).exists()
    assert out.stat().st_size > 0
