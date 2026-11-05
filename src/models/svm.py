"""SVM (``SVC``) da RQ1 do artigo, com as 58 features, e comparação com TF-IDF.

Parte 1 — protocolo do artigo (``SVM_SPEC``):
    Grid idêntico ao de ``code/RQ1/scripts/svm.py`` do repositório original.
    Como o grid tem só ``2 x 2 x 4 = 16`` combinações, ``n_iter=300`` no
    ``RandomizedSearchCV`` equivale a uma busca exaustiva de 16 candidatos (o
    scikit-learn avisa e limita). Os dados NÃO são padronizados, como no artigo.

Parte 2 — comparação complementar (``run_svm_tfidf``):
    SVM sobre a matriz TF-IDF da Issue 6, no split oficial treino -> teste, com
    hiperparâmetros fixos e documentados (sem busca), para comparar com a
    Regressão Logística + TF-IDF do baseline (``baseline_logreg.json``).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from src.models.article_protocol import (
    DEFAULT_SEED,
    ModelSpec,
    compute_fold_metrics,
    diagnostic_scores,
)
from src.preprocessing.embedding import (
    DEFAULT_PARAMS,
    STATS_FILE,
    TEST_LABELS_FILE,
    TEST_MATRIX_FILE,
    TRAIN_LABELS_FILE,
    TRAIN_MATRIX_FILE,
    build_embeddings,
    load_matrix,
)

# Grid do artigo (code/RQ1/scripts/svm.py).
PARAM_GRID: dict[str, list] = {
    "C": [0.01, 0.1],
    "kernel": ["linear", "rbf"],
    "gamma": ["scale", "auto", 0.01, 0.1],
}

# Tabela 3 do artigo (SVM): (média, desvio-padrão entre os 5 folds).
ARTICLE_TABLE3 = {
    "mae": (1.0431, 0.0070),
    "rmse": (1.5085, 0.0118),
    "auc": (0.7304, 0.0046),
}
# Melhores hiperparâmetros do artigo (code/RQ1/results/SVC_best_params.txt).
ARTICLE_BEST_PARAMS = {"kernel": "linear", "gamma": "scale", "C": 0.1}


def build_svm(seed: int = DEFAULT_SEED) -> SVC:
    """``SVC`` com parâmetros padrão do scikit-learn (o grid define o resto).

    ``random_state`` só afeta o SVC com ``probability=True``; é fixado por clareza.
    """
    return SVC(random_state=seed)


SVM_SPEC = ModelSpec(
    name="svm",
    build_estimator=build_svm,
    param_distributions=PARAM_GRID,
    article_reference=ARTICLE_TABLE3,
    article_best_params=ARTICLE_BEST_PARAMS,
)

# --- comparação com TF-IDF --------------------------------------------------
# Sem busca de hiperparâmetros: kernel linear (o vencedor do artigo e o usual em
# texto esparso) e C=1.0 (padrão do scikit-learn; as linhas TF-IDF já têm norma 1).
TFIDF_SVM_PARAMS: dict[str, Any] = {"kernel": "linear", "C": 1.0}
MIN_TFIDF_SAMPLE = 100


def ensure_tfidf(train_csv: str | Path, test_csv: str | Path, processed_dir: str | Path) -> None:
    """Garante os artefatos TF-IDF da Issue 6 em ``processed_dir``.

    Reaproveita os arquivos existentes se foram gerados com os parâmetros atuais
    de ``DEFAULT_PARAMS``; caso contrário (ou se faltar algum), regenera com
    ``build_embeddings`` (``fit`` só no treino).
    """
    pasta = Path(processed_dir)
    arquivos = [TRAIN_MATRIX_FILE, TEST_MATRIX_FILE, TRAIN_LABELS_FILE, TEST_LABELS_FILE, STATS_FILE]
    if all((pasta / a).is_file() for a in arquivos):
        guardados = json.loads((pasta / STATS_FILE).read_text(encoding="utf-8"))["vectorizer_params"]
        atuais = json.loads(json.dumps(DEFAULT_PARAMS))  # tuplas viram listas, como no JSON
        if guardados == atuais:
            return
    build_embeddings(train_csv, test_csv, pasta)


def _stratified_rows(y: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Índices (ordenados) de ~``n`` linhas estratificadas pelo rótulo."""
    if n >= len(y):
        return np.arange(len(y))
    idx, _ = train_test_split(np.arange(len(y)), train_size=n, stratify=y, random_state=seed)
    return np.sort(idx)


def run_svm_tfidf(
    train_csv: str | Path,
    test_csv: str | Path,
    processed_dir: str | Path,
    *,
    sample_n: int | None = None,
    seed: int = DEFAULT_SEED,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Treina o SVM no TF-IDF de treino e avalia no teste oficial.

    Com ``sample_n``, usa uma amostra estratificada de ``sample_n`` linhas de
    treino e ``sample_n / 4`` de teste, tiradas das matrizes completas (o
    vetorizador continua ajustado no treino completo; é só um modo rápido).
    """
    params = {**TFIDF_SVM_PARAMS, **(params or {})}
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

    model = SVC(random_state=seed, **params)
    t0 = time.perf_counter()
    model.fit(x_tr, y_tr)
    fit_s = time.perf_counter() - t0
    pred = model.predict(x_te)
    metrics = compute_fold_metrics(y_te, pred, diagnostic_scores(model, x_te))

    return {
        "model": "svm",
        "features": "tfidf (Issue 6)",
        "params": params,
        "params_note": "fixos, sem busca de hiperparâmetros",
        "protocol": "treino em train.csv, avaliação em test.csv (sem validação cruzada)",
        "seed": seed,
        "sample_n": sample_n,
        "n_train": int(x_tr.shape[0]),
        "n_test": int(x_te.shape[0]),
        "n_features": int(x_tr.shape[1]),
        "metrics": metrics,
        "fit_seconds": round(fit_s, 3),
        "sklearn_version": sklearn.__version__,
    }


def compare_with_baseline(svm_payload: dict[str, Any], baseline_json: str | Path) -> dict[str, Any] | None:
    """Diferença SVM − LR (mesmo TF-IDF) nas métricas em comum, ou ``None``.

    Só faz sentido na base completa: com ``sample_n`` devolve ``None``.
    """
    baseline_json = Path(baseline_json)
    if svm_payload.get("sample_n") or not baseline_json.is_file():
        return None
    lr = json.loads(baseline_json.read_text(encoding="utf-8"))["metrics_logreg"]
    comuns = [m for m in ("accuracy", "f1_macro", "mae", "rmse") if m in lr]
    return {
        "baseline": "Regressão Logística + TF-IDF (results/metrics/baseline_logreg.json)",
        "logreg": {m: lr[m] for m in comuns},
        "svm": {m: svm_payload["metrics"][m] for m in comuns},
        "delta_svm_minus_logreg": {m: svm_payload["metrics"][m] - lr[m] for m in comuns},
    }
