"""Métricas do baseline (Issue 7) — RED."""

import numpy as np
import pytest

from src.evaluation.metrics import (
    compute_metrics,
    majority_class_label,
    predict_majority,
)


def test_compute_metrics_valores_conhecidos():
    y_true = np.array([0, 1, 2, 3, 4, 4])
    y_pred = np.array([0, 1, 2, 3, 3, 4])
    m = compute_metrics(y_true, y_pred)
    assert m["accuracy"] == pytest.approx(5 / 6)
    assert 0.0 <= m["f1_macro"] <= 1.0
    assert m["mae"] == pytest.approx(1 / 6)  # erro de 1 estrela em 1 de 6
    assert m["rmse"] == pytest.approx((1 / 6) ** 0.5)


def test_majority_class_label_e_predict():
    assert majority_class_label(np.array([0, 4, 4, 1, 4])) == 4
    np.testing.assert_array_equal(
        predict_majority(np.array([0, 1, 1]), 3), np.array([1, 1, 1])
    )
