"""Baseline com Regressão Logística (Issue 7)."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression

RANDOM_STATE: int = 42
DEFAULT_LR_PARAMS: dict = {
    "C": 1.0,
    "max_iter": 1000,
    "solver": "lbfgs",
    "random_state": RANDOM_STATE,
}


def train_logistic_regression(
    X_train: sp.csr_matrix,
    y_train: np.ndarray,
    *,
    C: float = 1.0,
    max_iter: int = 1000,
    random_state: int = RANDOM_STATE,
) -> LogisticRegression:
    """Treina LR multinomial sobre TF-IDF (``lbfgs`` é determinístico dado seed)."""
    params = {**DEFAULT_LR_PARAMS, "C": C, "max_iter": max_iter, "random_state": random_state}
    model = LogisticRegression(**params)
    model.fit(X_train, np.asarray(y_train))
    return model


def predict(model: LogisticRegression, X: sp.csr_matrix) -> np.ndarray:
    """Atalho de inferência (mantém tipo do sklearn)."""
    return model.predict(X)


def save_model(model: LogisticRegression, path: str | Path) -> Path:
    """Persiste o modelo com joblib (cria diretórios pai)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path: str | Path) -> LogisticRegression:
    """Carrega modelo salvo por :func:`save_model`."""
    return joblib.load(Path(path))
