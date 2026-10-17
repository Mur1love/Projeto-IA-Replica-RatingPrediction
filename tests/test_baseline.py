"""Treino do baseline LR (Issue 7) — RED."""

import numpy as np
import scipy.sparse as sp

from src.models.baseline import load_model, save_model, train_logistic_regression


def test_train_logistic_regression_prediz_e_reproduzivel():
    X_toy = sp.csr_matrix(np.array([[2.0, 0.0], [0.0, 2.0], [1.5, 0.2], [0.1, 1.8]]))
    y_toy = np.array([0, 4, 0, 4])
    m1 = train_logistic_regression(X_toy, y_toy)
    m2 = train_logistic_regression(X_toy, y_toy)
    np.testing.assert_array_equal(m1.predict(X_toy), m2.predict(X_toy))
    assert set(m1.predict(X_toy)).issubset({0, 4})


def test_save_load_model_roundtrip(tmp_path):
    X_toy = sp.csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0]]))
    m = train_logistic_regression(X_toy, np.array([0, 4]))
    p = save_model(m, tmp_path / "lr.joblib")
    assert p.exists()
    np.testing.assert_array_equal(load_model(p).predict(X_toy), m.predict(X_toy))
