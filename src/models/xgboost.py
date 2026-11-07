"""XGBoost da RQ1 do artigo, com as 58 features, e comparação com TF-IDF.

Parte 1 — protocolo do artigo (``XGB_SPEC``):
    Grid idêntico ao descrito na TAREFA 14: 4 × 3^8 = 8.748 combinações
    (``n_estimators``, ``learning_rate``, ``max_depth``, ``min_child_weight``,
    ``gamma``, ``subsample``, ``colsample_bytree``, ``reg_alpha`` e
    ``reg_lambda``) amostradas por ``RandomizedSearchCV`` com ``n_iter=300``.
Parte 2 — comparação complementar (``run_xgboost_tfidf``, TAREFA 14):
    XGBoost sobre a matriz TF-IDF da Issue 6, no split oficial treino -> teste,
    com busca curta, para comparar com a Regressão Logística + TF-IDF do
    baseline (``baseline_logreg.json``).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from sklearn.model_selection import RandomizedSearchCV, train_test_split
from xgboost import XGBClassifier

from src.models.article_protocol import (
    DEFAULT_SEED, ModelSpec, compute_fold_metrics, diagnostic_scores,
)
from src.models.svm import ensure_tfidf
from src.preprocessing.embedding import (
    TEST_LABELS_FILE, TEST_MATRIX_FILE, TRAIN_LABELS_FILE, TRAIN_MATRIX_FILE, load_matrix,
)

import numpy as np
import sklearn

# Grid da TAREFA 14 (RQ1): n_estimators(4) x as outras oito dimensões (3) = 8.748.
PARAM_GRID: dict[str, list] = {
    "n_estimators": [100, 200, 300, 500],
    "learning_rate": [0.05, 0.10, 0.20],
    "max_depth": [6, 8, 10],
    "min_child_weight": [1, 3, 5],
    "gamma": [0.0, 0.1, 0.3],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.4, 0.6, 0.8],
    "reg_alpha": [0, 0.1, 1],
    "reg_lambda": [1, 1.5, 2],
}

# Tabela 3 do artigo (XGBoost): (média, desvio-padrão entre os 5 folds).
ARTICLE_TABLE3 = {
    "mae": (0.9499, 0.0047),
    "rmse": (1.3721, 0.0055),
    "auc": (0.7390, 0.0046),
}
# A TAREFA não informa os melhores parâmetros encontrados no artigo.
ARTICLE_BEST_PARAMS: dict[str, Any] = {}


def build_xgboost(seed: int = DEFAULT_SEED) -> XGBClassifier:
    """``XGBClassifier`` reprodutível; ``n_jobs=1`` deixa o paralelismo externo.

    O paralelismo do protocolo fica no ``RandomizedSearchCV``/``joblib``
    (``--n-jobs -1`` do CLI), como no Random Forest. O ``tree_method`` padrão
    do xgboost 3.x é ``hist``.
    """
    return XGBClassifier(random_state=seed, n_jobs=1)


XGB_SPEC = ModelSpec(
    name="xgboost",
    build_estimator=build_xgboost,
    param_distributions=PARAM_GRID,
    article_reference=ARTICLE_TABLE3,
    article_best_params=ARTICLE_BEST_PARAMS,
)

# --- comparação com TF-IDF ---------------------------------------------------
# Como no Random Forest: busca curta e fixa (nada de grid enorme no TF-IDF),
# com o objetivo de comparar classificadores sobre a mesma matriz da Issue 6,
# treino em train.csv e avaliação no test.csv oficial.
TFIDF_XGB_GRID: dict[str, list] = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.05, 0.1, 0.2],
    "max_depth": [6, 10],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.6, 1.0],
}
MIN_TFIDF_SAMPLE = 100


def _stratified_rows(y: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Índices (ordenados) de ~``n`` linhas estratificadas pelo rótulo."""
    if n >= len(y):
        return np.arange(len(y))
    idx, _ = train_test_split(np.arange(len(y)), train_size=n, stratify=y, random_state=seed)
    return np.sort(idx)


def run_xgboost_tfidf(
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
    """Busca curta de XGBoost sobre o TF-IDF da Issue 6, avaliado no teste oficial.

    Com ``sample_n``, usa uma amostra estratificada de ``sample_n`` linhas de
    treino e ``sample_n / 4`` de teste (o vetorizador continua ajustado no
    treino completo; é só um modo rápido) e, nesse caso, NÃO compara com o
    baseline — a comparação somente vale na base completa.
    """
    if sample_n is not None and sample_n < MIN_TFIDF_SAMPLE:
        raise ValueError(f"sample_n deve ser >= {MIN_TFIDF_SAMPLE}")

    ensure_tfidf(train_csv, test_csv, processed_dir)
    pasta = Path(processed_dir)
    x_tr, x_te = load_matrix(pasta / TRAIN_MATRIX_FILE), load_matrix(pasta / TEST_MATRIX_FILE)
    y_tr, y_te = np.load(pasta / TRAIN_LABELS_FILE), np.load(pasta / TEST_LABELS_FILE)
    if sample_n:
        i_tr = _stratified_rows(y_tr, sample_n, seed)
        i_te = _stratified_rows(y_te, max(sample_n // 4, 5), seed)
        x_tr, y_tr, x_te, y_te = x_tr[i_tr], y_tr[i_tr], x_te[i_te], y_te[i_te]

    search = RandomizedSearchCV(
        XGBClassifier(random_state=seed, n_jobs=1),
        TFIDF_XGB_GRID,
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=3,
        random_state=seed,
        n_jobs=n_jobs,
        refit=True,
    )
    t0 = time.perf_counter()
    search.fit(x_tr, y_tr)
    fit_s = time.perf_counter() - t0
    model = search.best_estimator_
    pred = model.predict(x_te)
    metrics = compute_fold_metrics(y_te, pred, diagnostic_scores(model, x_te))

    payload: dict[str, Any] = {
        "model": "xgboost",
        "features": "tfidf (Issue 6)",
        "params": search.best_params_,
        "params_note": "RandomizedSearchCV curta; 3 folds; RMSE",
        "protocol": "fit no train.csv e avaliação no test.csv oficial; TF-IDF ajustado só no treino",
        "seed": seed,
        "n_iter_requested": n_iter,
        "n_candidates_evaluated": len(search.cv_results_["params"]),
        "sample_n": sample_n,
        "n_train": int(x_tr.shape[0]),
        "n_test": int(x_te.shape[0]),
        "n_features": int(x_tr.shape[1]),
        "metrics": metrics,
        "fit_seconds": round(fit_s, 3),
        "sklearn_version": sklearn.__version__,
    }
    payload["comparison_with_baseline"] = None
    if sample_n is None and baseline_json and Path(baseline_json).is_file():
        lr = json.loads(Path(baseline_json).read_text(encoding="utf-8"))["metrics_logreg"]
        comuns = [m for m in ("accuracy", "f1_macro", "mae", "rmse") if m in lr]
        payload["comparison_with_baseline"] = {
            "baseline": "Regressão Logística + TF-IDF (results/metrics/baseline_logreg.json)",
            "logreg": {m: lr[m] for m in comuns},
            "xgboost": {m: metrics[m] for m in comuns},
            "delta_xgb_minus_logreg": {m: metrics[m] - lr[m] for m in comuns},
        }
    return payload
