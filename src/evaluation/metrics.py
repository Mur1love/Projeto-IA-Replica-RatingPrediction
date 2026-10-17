"""Métricas do baseline (Issue 7)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error

_VALID_LABELS = (0, 1, 2, 3, 4)


def _as_label_array(valores, nome: str) -> np.ndarray:
    arr = np.asarray(valores)
    if arr.size == 0:
        raise ValueError(f"{nome}: vetor vazio")
    if arr.dtype.kind == "f":
        if not np.isfinite(arr).all():
            raise ValueError(f"{nome}: há valores ausentes (NaN/inf)")
        if not (arr == np.floor(arr)).all():
            raise ValueError(f"{nome}: há valores não inteiros")
    try:
        arr = arr.astype(np.int64)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{nome}: esperado inteiros 0..4") from exc
    invalidos = np.unique(arr[~np.isin(arr, list(_VALID_LABELS))])
    if invalidos.size:
        raise ValueError(f"{nome}: labels fora de 0..4: {invalidos[:5].tolist()}")
    return arr


def compute_metrics(y_true_labels: np.ndarray, y_pred_labels: np.ndarray) -> dict[str, float]:
    """Acurácia e F1-macro sobre labels 0..4; MAE/RMSE sobre ratings 1..5.

    MAE/RMSE usam a escala original do usuário (rating = label + 1), pois o
    artigo avalia erro em estrelas. Levanta ``ValueError`` se tamanhos
    divergirem ou labels fora de 0..4.
    """
    y_true = _as_label_array(y_true_labels, "y_true")
    y_pred = _as_label_array(y_pred_labels, "y_pred")
    if y_true.shape != y_pred.shape:
        raise ValueError(f"tamanhos diferentes: {y_true.shape} vs {y_pred.shape}")

    # Ratings na escala 1..5 para erro em estrelas.
    r_true = (y_true + 1).astype(np.float64)
    r_pred = (y_pred + 1).astype(np.float64)

    mse = float(mean_squared_error(r_true, r_pred))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "mae": float(mean_absolute_error(r_true, r_pred)),
        "rmse": float(np.sqrt(mse)),
    }


def majority_class_label(y_train_labels: np.ndarray) -> int:
    """Moda dos labels de treino; empate desempatado pelo menor label."""
    arr = _as_label_array(y_train_labels, "y_train")
    counts = np.bincount(arr, minlength=5)
    return int(np.argmax(counts))


def predict_majority(y_train_labels: np.ndarray, n: int) -> np.ndarray:
    """Vetor de tamanho ``n`` preenchido com a classe majoritária do treino."""
    if int(n) < 0:
        raise ValueError(f"n deve ser >= 0, recebi {n}")
    moda = majority_class_label(y_train_labels)
    return np.full(int(n), moda, dtype=np.int64)
