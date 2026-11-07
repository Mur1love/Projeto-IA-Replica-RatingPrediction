"""Random Forest da RQ1 e comparação complementar sobre TF-IDF."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split

from src.models.article_protocol import (
    DEFAULT_SEED, ModelSpec, compute_fold_metrics, diagnostic_scores,
)
from src.models.svm import ensure_tfidf
from src.preprocessing.embedding import (
    TEST_MATRIX_FILE, TEST_LABELS_FILE, TRAIN_MATRIX_FILE, TRAIN_LABELS_FILE,
    load_matrix,
)

# Grid informado na TAREFA. RandomizedSearchCV sorteia combinações sem reposição.
PARAM_GRID: dict[str, list] = {
    "n_estimators": [100, 200, 300, 500, 1000],
    "max_depth": [5, 10, 30, None],
    "min_samples_split": [5, 10, 20],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", 0.5, None],
    "class_weight": [None, "balanced", "balanced_subsample"],
    "max_leaf_nodes": [None, 10, 20],
    "min_impurity_decrease": [0.0, 0.01, 0.1],
}
ARTICLE_TABLE3 = {
    "mae": (0.9717, 0.0077),
    "rmse": (1.4001, 0.0087),
    "auc": (0.7367, 0.0051),
}
# A TAREFA não informa os melhores parâmetros encontrados no artigo.
ARTICLE_BEST_PARAMS: dict[str, Any] = {}


def build_random_forest(seed: int = DEFAULT_SEED) -> RandomForestClassifier:
    """Random Forest reprodutível; random_state=None no script original."""
    return RandomForestClassifier(random_state=seed, n_jobs=1)


RANDOM_FOREST_SPEC = ModelSpec(
    name="random_forest",
    build_estimator=build_random_forest,
    param_distributions=PARAM_GRID,
    article_reference=ARTICLE_TABLE3,
    article_best_params=ARTICLE_BEST_PARAMS,
)

TFIDF_RF_PARAMS: dict[str, Any] = {
    "n_estimators": 200, "max_depth": None, "min_samples_split": 2,
    "min_samples_leaf": 1, "max_features": "sqrt", "class_weight": None,
    "n_jobs": 1, "random_state": DEFAULT_SEED,
}


def _stratified_rows(y: np.ndarray, n: int, seed: int) -> np.ndarray:
    if n >= len(y):
        return np.arange(len(y))
    idx, _ = train_test_split(
        np.arange(len(y)), train_size=n, stratify=y, random_state=seed
    )
    return np.sort(idx)


def run_random_forest_tfidf(
    train_csv: str | Path,
    test_csv: str | Path,
    processed_dir: str | Path,
    *,
    sample_n: int | None = None,
    seed: int = DEFAULT_SEED,
    n_iter: int = 8,
    n_jobs: int = 1,
    baseline_json: str | Path | None = None,
) -> dict[str, Any]:
    """Busca curta em TF-IDF, treinando apenas no split oficial de treino."""
    if sample_n is not None and sample_n < 100:
        raise ValueError("sample_n deve ser >= 100")
    ensure_tfidf(train_csv, test_csv, processed_dir)
    folder = Path(processed_dir)
    x_train = load_matrix(folder / TRAIN_MATRIX_FILE)
    x_test = load_matrix(folder / TEST_MATRIX_FILE)
    y_train = np.load(folder / TRAIN_LABELS_FILE)
    y_test = np.load(folder / TEST_LABELS_FILE)
    if sample_n:
        i_train = _stratified_rows(y_train, sample_n, seed)
        i_test = _stratified_rows(y_test, max(sample_n // 4, 5), seed)
        x_train, y_train = x_train[i_train], y_train[i_train]
        x_test, y_test = x_test[i_test], y_test[i_test]

    grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 30],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", 0.5],
        "class_weight": [None, "balanced"],
    }
    model = RandomForestClassifier(random_state=seed, n_jobs=1)
    search = RandomizedSearchCV(
        model, grid, n_iter=n_iter, scoring="neg_root_mean_squared_error",
        cv=3, random_state=seed, n_jobs=n_jobs, refit=True,
    )
    started = time.perf_counter()
    search.fit(x_train, y_train)
    elapsed = time.perf_counter() - started
    pred = search.best_estimator_.predict(x_test)
    metrics = compute_fold_metrics(
        y_test, pred, diagnostic_scores(search.best_estimator_, x_test)
    )
    payload: dict[str, Any] = {
        "model": "random_forest", "features": "tfidf (Issue 6)",
        "params": search.best_params_, "params_note": "RandomizedSearchCV curta; 3 folds; RMSE",
        "protocol": "fit no train.csv e avaliação no test.csv oficial; TF-IDF ajustado só no treino",
        "seed": seed, "n_iter_requested": n_iter, "n_candidates_evaluated": len(search.cv_results_["params"]),
        "sample_n": sample_n, "n_train": int(x_train.shape[0]), "n_test": int(x_test.shape[0]),
        "n_features": int(x_train.shape[1]), "metrics": metrics,
        "fit_seconds": round(elapsed, 3), "sklearn_version": sklearn.__version__,
    }
    if sample_n is None and baseline_json and Path(baseline_json).is_file():
        baseline = json.loads(Path(baseline_json).read_text(encoding="utf-8"))["metrics_logreg"]
        common = [m for m in ("accuracy", "f1_macro", "mae", "rmse") if m in baseline]
        payload["comparison_with_baseline"] = {
            "baseline": "Regressão Logística + TF-IDF",
            "logreg": {m: baseline[m] for m in common},
            "random_forest": {m: metrics[m] for m in common},
            "delta_rf_minus_logreg": {m: metrics[m] - baseline[m] for m in common},
        }
    else:
        payload["comparison_with_baseline"] = None
    return payload
