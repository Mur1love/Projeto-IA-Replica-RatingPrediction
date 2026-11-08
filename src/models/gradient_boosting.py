"""Gradient Boosting (``GradientBoostingClassifier``) da RQ1 e comparação com TF-IDF.

Parte 1 — protocolo do artigo (``GRADIENT_BOOSTING_SPEC``):
    Grid da TAREFA 13 (idêntico ao do artigo). São ``5 x 4 x 3 x 3 x 3 x 4 = 2160``
    combinações; o ``RandomizedSearchCV`` sorteia ``n_iter`` delas (300 no
    artigo) sem reposição. Os dados NÃO são padronizados (árvores não precisam).

    Custo: o GBT ajusta uma árvore por classe a cada iteração de boosting. Uma
    combinação com ``n_estimators=1000``, ``max_depth=10`` e ``max_features=None``
    leva dezenas de minutos em 41 mil linhas, e a busca faz ``n_iter x 5`` ajustes.
    Por isso ``n_iter`` é configurável e a execução completa com ``n_iter=300``
    não é prática em uma máquina comum (ver ``docs/diferencas.md``).

    AUC de diagnóstico: o harness prefere ``decision_function`` a ``predict_proba``.
    No GBT multiclasse a primeira devolve log-odds por classe; a TAREFA pede a
    AUC com **probabilidades** (soma de P(4 estrelas) + P(5 estrelas)). Por isso o
    estimador usado é ``GradientBoostingProbaScores``, que declara
    ``diagnostic_from_proba = True`` (gancho de ``diagnostic_scores``). As previsões
    (e, portanto, MAE, RMSE, F1, acurácia e a AUC do artigo) são idênticas às do
    ``GradientBoostingClassifier`` padrão.

Parte 2 — comparação complementar (``run_gradient_boosting_tfidf``):
    GBT sobre a matriz TF-IDF da Issue 6, no split oficial treino -> teste. Busca
    curta de hiperparâmetros feita só no treino (``search=True``) ou parâmetros
    fixos e documentados (``search=False``), para comparar com a Regressão
    Logística + TF-IDF do baseline (``baseline_logreg.json``).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV

from src.models.article_protocol import (
    DEFAULT_SEED,
    SEARCH_SCORING,
    ModelSpec,
    compute_fold_metrics,
    diagnostic_scores,
)
from src.models.svm import _stratified_rows, ensure_tfidf
from src.preprocessing.embedding import (
    TEST_LABELS_FILE,
    TEST_MATRIX_FILE,
    TRAIN_LABELS_FILE,
    TRAIN_MATRIX_FILE,
    load_matrix,
)

# Grid do artigo (TAREFA 13).
PARAM_GRID: dict[str, list] = {
    "n_estimators": [100, 200, 300, 500, 1000],
    "learning_rate": [0.001, 0.01, 0.05, 0.1],
    "max_depth": [3, 5, 10],
    "min_samples_split": [5, 10, 15],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", 0.5, None],
}

# Tabela 3 do artigo (Gradient Boosting): (média, desvio-padrão entre os 5 folds).
ARTICLE_TABLE3 = {
    "mae": (0.9612, 0.0061),
    "rmse": (1.3727, 0.0051),
    "auc": (0.7318, 0.0058),
}
# A TAREFA não informa os melhores hiperparâmetros encontrados no artigo.
ARTICLE_BEST_PARAMS: dict[str, Any] = {}


def grid_size(grid: dict[str, list] = PARAM_GRID) -> int:
    """Número de combinações do grid (2160 para o grid do artigo)."""
    return int(np.prod([len(v) for v in grid.values()]))


class GradientBoostingProbaScores(GradientBoostingClassifier):
    """``GradientBoostingClassifier`` cuja AUC de diagnóstico usa probabilidades.

    ``diagnostic_from_proba = True`` faz o harness (``diagnostic_scores``) usar
    ``predict_proba`` em vez de ``decision_function``. ``fit`` e ``predict`` não
    mudam. É atributo de classe para sobreviver ao ``clone`` da busca.
    """

    diagnostic_from_proba = True


def build_gradient_boosting(seed: int = DEFAULT_SEED) -> GradientBoostingProbaScores:
    """GBT com ``random_state`` fixo (o grid define os demais hiperparâmetros)."""
    return GradientBoostingProbaScores(random_state=seed)


GRADIENT_BOOSTING_SPEC = ModelSpec(
    name="gradient_boosting",
    build_estimator=build_gradient_boosting,
    param_distributions=PARAM_GRID,
    article_reference=ARTICLE_TABLE3,
    article_best_params=ARTICLE_BEST_PARAMS,
)

# --- comparação com TF-IDF --------------------------------------------------
# Busca curta (RandomizedSearchCV, 3 folds, RMSE) feita SOMENTE no train.csv; o teste só
# é usado na avaliação final. O grid é bem menor que o do artigo porque o TF-IDF tem
# ~28 mil colunas esparsas: ``max_features=None`` e árvores muito profundas seriam
# impraticáveis. ``max_features="sqrt"`` e 0,05 (~1,4 mil colunas por divisão).
TFIDF_GBT_GRID: dict[str, list] = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.1, 0.2],
    "max_depth": [3, 5, 8],
    "max_features": ["sqrt", 0.05],
}
# Parâmetros usados com ``search=False``: padrões do scikit-learn (100 árvores de
# profundidade 3) com ``max_features="sqrt"``. Medimos que subajustam (RMSE de
# validação ~1,41 contra ~1,18 com 300 árvores), então servem só como referência rápida.
TFIDF_GBT_FIXED_PARAMS: dict[str, Any] = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "max_depth": 3,
    "max_features": "sqrt",
}
TFIDF_N_ITER = 6
TFIDF_CV = 3
MIN_TFIDF_SAMPLE = 100


def run_gradient_boosting_tfidf(
    train_csv: str | Path,
    test_csv: str | Path,
    processed_dir: str | Path,
    *,
    sample_n: int | None = None,
    seed: int = DEFAULT_SEED,
    search: bool = True,
    n_iter: int = TFIDF_N_ITER,
    n_jobs: int | None = 1,
    params: dict[str, Any] | None = None,
    baseline_json: str | Path | None = None,
) -> dict[str, Any]:
    """Treina o GBT no TF-IDF de treino e avalia no teste oficial.

    Com ``search=True`` (padrão) faz uma busca curta só no treino e reajusta o melhor
    candidato; com ``search=False`` usa ``TFIDF_GBT_FIXED_PARAMS`` (mais ``params``).
    Com ``sample_n``, usa uma amostra estratificada de ``sample_n`` linhas de treino e
    ``sample_n / 4`` de teste, tiradas das matrizes completas (modo rápido; o
    vetorizador continua ajustado no treino completo).
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

    t0 = time.perf_counter()
    if search:
        busca = RandomizedSearchCV(
            build_gradient_boosting(seed), TFIDF_GBT_GRID, n_iter=n_iter,
            scoring=SEARCH_SCORING, cv=TFIDF_CV, random_state=seed, n_jobs=n_jobs, refit=True,
        )
        busca.fit(x_tr, y_tr)
        model = busca.best_estimator_
        used_params = dict(busca.best_params_)
        n_candidates = len(busca.cv_results_["params"])
        note = (f"RandomizedSearchCV curta (n_iter={n_iter}, {TFIDF_CV} folds, RMSE) "
                "somente no train.csv")
    else:
        used_params = {**TFIDF_GBT_FIXED_PARAMS, **(params or {})}
        model = GradientBoostingProbaScores(random_state=seed, **used_params).fit(x_tr, y_tr)
        n_candidates = 1
        note = "fixos, sem busca de hiperparâmetros"
    fit_s = time.perf_counter() - t0

    pred = model.predict(x_te)
    metrics = compute_fold_metrics(y_te, pred, diagnostic_scores(model, x_te))

    payload: dict[str, Any] = {
        "model": "gradient_boosting",
        "features": "tfidf (Issue 6)",
        "params": used_params,
        "params_note": note,
        "protocol": "treino em train.csv, avaliação em test.csv (sem validação cruzada no teste)",
        "seed": seed,
        "n_iter_requested": int(n_iter) if search else None,
        "n_candidates_evaluated": n_candidates,
        "sample_n": sample_n,
        "n_train": int(x_tr.shape[0]),
        "n_test": int(x_te.shape[0]),
        "n_features": int(x_tr.shape[1]),
        "metrics": metrics,
        "fit_seconds": round(fit_s, 3),
        "sklearn_version": sklearn.__version__,
    }
    payload["comparison_with_baseline"] = compare_with_baseline(payload, baseline_json)
    return payload


def compare_with_baseline(
    payload: dict[str, Any], baseline_json: str | Path | None
) -> dict[str, Any] | None:
    """Diferença GBT − LR (mesmo TF-IDF) nas métricas em comum, ou ``None``.

    Só faz sentido na base completa: com ``sample_n`` (ou sem o JSON do
    baseline) devolve ``None``.
    """
    if payload.get("sample_n") or baseline_json is None or not Path(baseline_json).is_file():
        return None
    lr = json.loads(Path(baseline_json).read_text(encoding="utf-8"))["metrics_logreg"]
    comuns = [m for m in ("accuracy", "f1_macro", "mae", "rmse") if m in lr]
    gbt = payload["metrics"]
    return {
        "baseline": "Regressão Logística + TF-IDF (results/metrics/baseline_logreg.json)",
        "logreg": {m: lr[m] for m in comuns},
        "gradient_boosting": {m: gbt[m] for m in comuns},
        "delta_gbt_minus_logreg": {m: gbt[m] - lr[m] for m in comuns},
    }
