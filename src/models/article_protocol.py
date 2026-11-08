"""Harness compartilhado do protocolo de avaliação do artigo (RQ1).

Reproduz, para qualquer classificador, o protocolo de ``code/train_model.py``
do repositório original, para que SVM, Random Forest, Gradient Boosting e
XGBoost (TAREFAS 11 a 14) sejam avaliados exatamente da mesma forma:

1. **Dados**: ``train.csv`` + ``test.csv`` concatenados *nessa ordem* (o teste
   participa da validação cruzada, como no artigo). Entram só as 58 features
   textuais (``str_``, ``pos_``, ``synt_``, ``lex_``, ``conc_``, ``twt_``,
   ``sbj_``); ``categoria``, ``text`` e ``rating`` nunca viram atributo.
   Linhas com ``text`` vazio **não** são removidas: as features já estão
   calculadas e o artigo usa todas as linhas.
2. **Folds**: ``StratifiedKFold`` estratificado pela chave ``categoria_rating``,
   com ``shuffle=False`` como no código original (a ordem das linhas define os
   folds). ``shuffle=True`` é aceito, mas fica registrado na saída.
3. **Busca**: ``RandomizedSearchCV`` com ``scoring="neg_root_mean_squared_error"``
   e ``random_state`` fixo, usando esses mesmos folds.
4. **Avaliação**: para cada fold, o melhor estimador é treinado de novo (refit)
   e avaliado com MAE, RMSE, F1 macro, acurácia e AUC.
5. **Agregação**: média, desvio-padrão (``ddof=1``) e IC de 95% (t de Student,
   ``n_folds - 1`` graus de liberdade) entre os folds.
6. **Holdout complementar**: com os mesmos hiperparâmetros, treina em
   ``train.csv`` e avalia em ``test.csv``.

Rótulos: o projeto usa labels 0..4 (``encode_rating``). MAE e RMSE são
calculados na escala original de 1 a 5 estrelas por ``compute_metrics``.

AUC do artigo (``auc``): classes 4 e 5 estrelas como positivas, calculada a
partir das **predições binárias** (não de scores). Isso equivale a
``(sensibilidade + especificidade) / 2``. Como diagnóstico também se calcula
``auc_scores``, a partir de ``decision_function``/``predict_proba``.

Para adicionar um modelo (TAREFAS 12 a 14), crie um ``ModelSpec`` (veja
``src/models/svm.py``) e registre-o em ``scripts/train_article_model.py``.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import scipy.stats as stats
import sklearn
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split

from src.evaluation.metrics import compute_metrics
from src.evaluation.plots import save_confusion_matrix
from src.preprocessing.encoding import encode_rating, stratify_key

CATEGORY_COL = "categoria"
TEXT_COL = "text"
RATING_COL = "rating"
NON_FEATURE_COLUMNS = (CATEGORY_COL, TEXT_COL, RATING_COL)

FEATURE_PREFIXES = ("str_", "pos_", "synt_", "lex_", "conc_", "twt_", "sbj_")
N_FEATURES = 58

# Ratings 4 e 5 estrelas = labels 3 e 4 (ver src/preprocessing/encoding.py).
POSITIVE_LABELS = (3, 4)
LABELS = (0, 1, 2, 3, 4)

SEARCH_SCORING = "neg_root_mean_squared_error"
DEFAULT_N_ITER = 300
DEFAULT_N_SPLITS = 5
DEFAULT_SEED = 42
CONFIDENCE = 0.95

# Métricas por fold, na ordem das colunas do CSV de saída.
FOLD_METRICS = ("mae", "rmse", "f1_macro", "accuracy", "auc", "auc_scores")


# --- modelo -----------------------------------------------------------------
@dataclass(frozen=True)
class ModelSpec:
    """Descreve um classificador a ser avaliado pelo protocolo do artigo.

    ``article_reference`` guarda ``{métrica: (média, desvio)}`` da Tabela 3 do
    artigo (escala original, ``ddof=1``) para comparar com a réplica.
    """

    name: str
    build_estimator: Callable[[int], BaseEstimator]
    param_distributions: Mapping[str, Sequence[Any]]
    article_reference: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    article_best_params: Mapping[str, Any] = field(default_factory=dict)


# --- dados ------------------------------------------------------------------
@dataclass
class ArticleData:
    """Dados do protocolo: só features em ``X``; o resto vive em arrays à parte."""

    X: pd.DataFrame            # as 58 features, em float
    y: np.ndarray              # labels 0..4
    groups: np.ndarray         # chave categoria_rating (só para estratificar)
    is_test: np.ndarray        # True nas linhas que vieram de test.csv
    empty_text: np.ndarray     # True onde text é vazio/ausente (linhas mantidas)
    sample_n: int | None = None

    @property
    def n_empty_text(self) -> int:
        return int(self.empty_text.sum())

    @property
    def n_rows(self) -> int:
        return len(self.y)

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    @property
    def n_train_rows(self) -> int:
        return int((~self.is_test).sum())

    @property
    def n_test_rows(self) -> int:
        return int(self.is_test.sum())


def select_feature_columns(
    columns: Sequence[str], *, expected_n: int | None = N_FEATURES
) -> list[str]:
    """Devolve as colunas de features, na ordem do CSV, ou levanta ``ValueError``.

    Barreira contra vazamento: uma coluna que não seja feature (prefixos do
    artigo) nem ``categoria``/``text``/``rating`` — por exemplo um índice
    ``Unnamed: 0`` ou um ``id`` — interrompe o processo em vez de entrar no
    modelo em silêncio.
    """
    columns = list(columns)
    features = [c for c in columns if c.startswith(FEATURE_PREFIXES)]
    inesperadas = [c for c in columns if c not in features and c not in NON_FEATURE_COLUMNS]
    if inesperadas:
        raise ValueError(
            f"colunas que não são features nem rótulos (identificação?): {inesperadas}"
        )
    if expected_n is not None and len(features) != expected_n:
        raise ValueError(f"esperava {expected_n} features, encontrei {len(features)}")
    return features


def load_article_data(
    train_csv: str | Path,
    test_csv: str | Path,
    *,
    sample_n: int | None = None,
    seed: int = DEFAULT_SEED,
    expected_n_features: int | None = N_FEATURES,
) -> ArticleData:
    """Carrega ``train.csv`` + ``test.csv`` (nessa ordem) para o protocolo.

    Com ``sample_n``, devolve uma amostra estratificada por ``categoria_rating``
    que **preserva a ordem original das linhas** (necessário porque os folds
    com ``shuffle=False`` dependem dessa ordem).
    """
    treino = pd.read_csv(train_csv)
    teste = pd.read_csv(test_csv)
    if list(treino.columns) != list(teste.columns):
        raise ValueError("train.csv e test.csv têm colunas diferentes")
    faltando = {CATEGORY_COL, RATING_COL} - set(treino.columns)
    if faltando:
        raise ValueError(f"colunas obrigatórias ausentes: {sorted(faltando)}")

    df = pd.concat([treino, teste], ignore_index=True)
    is_test = np.r_[np.zeros(len(treino), bool), np.ones(len(teste), bool)]

    features = select_feature_columns(df.columns, expected_n=expected_n_features)
    X = df[features].astype("float64")
    if X.isna().any().any():
        raise ValueError("há valores ausentes nas features; o artigo não os trata")

    if TEXT_COL in df.columns:
        vazio = df[TEXT_COL].isna() | df[TEXT_COL].astype(str).str.strip().eq("")
        empty_text = vazio.to_numpy()
    else:
        empty_text = np.zeros(len(df), bool)

    y = encode_rating(df[RATING_COL]).to_numpy()
    groups = stratify_key(df, CATEGORY_COL, RATING_COL).to_numpy()
    data = ArticleData(X=X, y=y, groups=groups, is_test=is_test, empty_text=empty_text)
    return _stratified_sample(data, sample_n, seed) if sample_n else data


def _stratified_sample(data: ArticleData, n: int, seed: int) -> ArticleData:
    if n <= 0:
        raise ValueError("sample_n deve ser positivo")
    if n >= data.n_rows:
        return data
    idx, _ = train_test_split(
        np.arange(data.n_rows), train_size=n, stratify=data.groups, random_state=seed
    )
    idx = np.sort(idx)  # mantém a ordem original (treino antes do teste)
    return ArticleData(
        X=data.X.iloc[idx].reset_index(drop=True),
        y=data.y[idx],
        groups=data.groups[idx],
        is_test=data.is_test[idx],
        empty_text=data.empty_text[idx],
        sample_n=int(n),
    )


# --- folds ------------------------------------------------------------------
def build_folds(
    groups: Sequence[str],
    n_splits: int = DEFAULT_N_SPLITS,
    *,
    shuffle: bool = False,
    seed: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Folds estratificados por ``categoria_rating`` como lista ``(treino, teste)``.

    ``shuffle=False`` (padrão) reproduz o código do artigo. ``seed`` só tem
    efeito com ``shuffle=True``.
    """
    groups = np.asarray(groups)
    _, contagens = np.unique(groups, return_counts=True)
    if contagens.min() < n_splits:
        raise ValueError(
            f"há grupo categoria_rating com {int(contagens.min())} exemplos, menos que "
            f"n_splits={n_splits}; aumente a amostra ou reduza n_splits"
        )
    kfold = StratifiedKFold(
        n_splits=n_splits, shuffle=shuffle, random_state=seed if shuffle else None
    )
    return [(tr, te) for tr, te in kfold.split(np.zeros(len(groups)), groups)]


# --- métricas ---------------------------------------------------------------
def to_binary(labels: Sequence[int]) -> np.ndarray:
    """1 para 4 e 5 estrelas (labels 3 e 4); 0 para 1 a 3 estrelas."""
    return np.isin(np.asarray(labels), POSITIVE_LABELS).astype(int)


def auc_binary_predictions(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """AUC do artigo: ``roc_auc_score`` entre as classes binárias real e predita."""
    return float(roc_auc_score(to_binary(y_true), to_binary(y_pred)))


def positive_score(scores: np.ndarray, classes: Sequence[int]) -> np.ndarray:
    """Reduz scores multiclasse a um score para "4–5 estrelas vs. 1–3".

    Margem entre a melhor classe positiva e a melhor negativa. Serve apenas
    para ranquear (AUC de diagnóstico); não é probabilidade.
    """
    scores = np.asarray(scores)
    classes = list(classes)
    if scores.ndim != 2 or scores.shape[1] != len(classes):
        raise ValueError(f"scores {scores.shape} incompatível com {len(classes)} classes")
    pos = [i for i, c in enumerate(classes) if c in POSITIVE_LABELS]
    neg = [i for i, c in enumerate(classes) if c not in POSITIVE_LABELS]
    if not pos or not neg:
        raise ValueError("é preciso ter classes positivas e negativas nos scores")
    return scores[:, pos].max(axis=1) - scores[:, neg].max(axis=1)


def diagnostic_scores(model: BaseEstimator, X: np.ndarray) -> np.ndarray | None:
    """Score "4–5 vs. 1–3" a partir de ``decision_function`` ou ``predict_proba``.

    Devolve ``None`` se o modelo não expõe nenhum dos dois.
    """
    classes = getattr(model, "classes_", None)
    if classes is None:
        return None
    # Estimadores com ``diagnostic_from_proba = True`` (ex.: Gradient Boosting) pedem
    # probabilidades mesmo expondo ``decision_function`` (log-odds).
    prefer_proba = getattr(model, "diagnostic_from_proba", False) and hasattr(model, "predict_proba")
    if hasattr(model, "decision_function") and not prefer_proba:
        return positive_score(model.decision_function(X), classes)
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(X))
        pos = [i for i, c in enumerate(classes) if c in POSITIVE_LABELS]
        return proba[:, pos].sum(axis=1)
    return None


def compute_fold_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    scores: np.ndarray | None = None,
) -> dict[str, float]:
    """MAE, RMSE (em estrelas), F1 macro, acurácia, AUC do artigo e AUC por scores.

    ``auc_scores`` é ``NaN`` quando ``scores`` não é informado.
    """
    out = compute_metrics(np.asarray(y_true), np.asarray(y_pred))
    out["auc"] = auc_binary_predictions(y_true, y_pred)
    out["auc_scores"] = (
        float(roc_auc_score(to_binary(y_true), scores)) if scores is not None else float("nan")
    )
    return {k: float(out[k]) for k in FOLD_METRICS}


def confidence_interval(values: Sequence[float]) -> tuple[float, float]:
    """IC de 95% da média pela distribuição t, com ``len(values) - 1`` graus de liberdade.

    Atenção: o código original do artigo calcula o IC depois de anexar a linha
    da média ao DataFrame (6 valores, 5 graus de liberdade), o que o torna cerca
    de 24% mais estreito. Aqui o IC usa apenas os valores por fold; veja
    ``docs/diferencas.md``.
    """
    x = np.asarray(values, dtype=float)
    if x.size < 2:
        raise ValueError("o IC precisa de pelo menos 2 folds")
    mean, sem = float(x.mean()), float(stats.sem(x))
    if sem == 0.0:
        return mean, mean
    low, high = stats.t.interval(CONFIDENCE, x.size - 1, loc=mean, scale=sem)
    return float(low), float(high)


def aggregate_folds(
    fold_metrics: pd.DataFrame, metrics: Sequence[str] = FOLD_METRICS
) -> dict[str, dict[str, Any]]:
    """Média, desvio-padrão (``ddof=1``, como o pandas), erro-padrão e IC de 95%."""
    out: dict[str, dict[str, Any]] = {}
    for m in metrics:
        v = fold_metrics[m].to_numpy(dtype=float)
        if np.isnan(v).any():
            continue  # métrica indisponível (ex.: auc_scores sem scores)
        low, high = confidence_interval(v)
        out[m] = {
            "mean": float(v.mean()),
            "std": float(v.std(ddof=1)),
            "sem": float(stats.sem(v)),
            "ci95": [low, high],
        }
    return out


# --- execução ---------------------------------------------------------------
@dataclass
class ArticleResult:
    spec: ModelSpec
    config: dict[str, Any]
    best_params: dict[str, Any]
    n_candidates: int
    search_best_rmse: float
    fold_metrics: pd.DataFrame
    aggregate: dict[str, dict[str, Any]]
    y_true: np.ndarray
    oof_pred: np.ndarray        # predição fora-da-dobra de cada linha
    confusion: np.ndarray       # matriz 5x5 agregada dos 5 folds
    holdout: dict[str, Any] | None
    timing: dict[str, float]


def _fit_and_score(estimator, params, X, y, train_idx, test_idx):
    """Treina com ``params`` em ``train_idx`` e avalia em ``test_idx``."""
    model = clone(estimator).set_params(**params)
    model.fit(X[train_idx], y[train_idx])
    pred = model.predict(X[test_idx])
    scores = diagnostic_scores(model, X[test_idx])
    return pred, compute_fold_metrics(y[test_idx], pred, scores)


def run_article_protocol(
    spec: ModelSpec,
    data: ArticleData,
    *,
    n_iter: int = DEFAULT_N_ITER,
    n_splits: int = DEFAULT_N_SPLITS,
    seed: int = DEFAULT_SEED,
    shuffle: bool = False,
    n_jobs: int | None = 1,
    verbose: int = 0,
) -> ArticleResult:
    """Executa o protocolo do artigo para ``spec`` sobre ``data``."""
    X = data.X.to_numpy(dtype=float)
    y = data.y
    folds = build_folds(data.groups, n_splits, shuffle=shuffle, seed=seed)

    t0 = time.perf_counter()
    # refit=False: o artigo reajusta no fim sobre todos os dados, mas esse modelo
    # nunca é usado (cada fold é retreinado abaixo); evitá-lo poupa um treino cheio.
    search = RandomizedSearchCV(
        spec.build_estimator(seed),
        param_distributions=dict(spec.param_distributions),
        n_iter=n_iter,
        scoring=SEARCH_SCORING,
        cv=folds,
        random_state=seed,
        refit=False,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    search.fit(X, y)
    t_search = time.perf_counter() - t0
    best_params = dict(search.best_params_)
    n_candidates = len(search.cv_results_["params"])

    # Um job por fold + o holdout (treino em train.csv, avaliação em test.csv).
    jobs = [(tr, te) for tr, te in folds]
    holdout_idx = None
    if data.n_train_rows and data.n_test_rows:
        holdout_idx = (np.flatnonzero(~data.is_test), np.flatnonzero(data.is_test))
        jobs.append(holdout_idx)

    estimator = spec.build_estimator(seed)
    t1 = time.perf_counter()
    outputs = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(_fit_and_score)(estimator, best_params, X, y, tr, te) for tr, te in jobs
    )
    t_eval = time.perf_counter() - t1

    fold_outputs = outputs[: len(folds)]
    oof = np.full(len(y), -1, dtype=np.int64)
    rows = []
    for i, ((_, te), (pred, met)) in enumerate(zip(folds, fold_outputs), start=1):
        oof[te] = pred
        rows.append({"fold": i, "n_train": len(jobs[i - 1][0]), "n_test": len(te), **met})
    fold_df = pd.DataFrame(rows)

    holdout = None
    if holdout_idx is not None:
        _, h_metrics = outputs[-1]
        holdout = {
            "n_train": int(len(holdout_idx[0])),
            "n_test": int(len(holdout_idx[1])),
            "metrics": h_metrics,
        }

    config = {
        "n_iter_requested": int(n_iter),
        "n_splits": int(n_splits),
        "seed": int(seed),
        "shuffle": bool(shuffle),
        "scoring": SEARCH_SCORING,
        "fold_stratification": "categoria_rating",
        "refit_per_fold": True,
        "n_jobs": n_jobs,
    }
    return ArticleResult(
        spec=spec,
        config=config,
        best_params=best_params,
        n_candidates=n_candidates,
        search_best_rmse=float(-search.best_score_),
        fold_metrics=fold_df,
        aggregate=aggregate_folds(fold_df),
        y_true=y,
        oof_pred=oof,
        confusion=confusion_matrix(y, oof, labels=list(LABELS)),
        holdout=holdout,
        timing={"search": t_search, "evaluation": t_eval},
    )


# --- saída ------------------------------------------------------------------
def _article_comparison(spec: ModelSpec, aggregate: Mapping[str, Mapping[str, Any]]) -> dict:
    """Lado a lado com a Tabela 3 do artigo (média e desvio) e diferença das médias."""
    out: dict[str, Any] = {}
    for metric, (ref_mean, ref_std) in spec.article_reference.items():
        if metric not in aggregate:
            continue
        mean = aggregate[metric]["mean"]
        out[metric] = {
            "article_mean": ref_mean,
            "article_std": ref_std,
            "replica_mean": mean,
            "replica_std": aggregate[metric]["std"],
            "delta_mean": mean - ref_mean,
        }
    return out


def summarize(result: ArticleResult, data: ArticleData) -> dict[str, Any]:
    """Dicionário serializável com tudo o que é registrado em ``<modelo>_rq1.json``."""
    spec = result.spec
    return {
        "model": spec.name,
        "protocol": {
            **result.config,
            "n_candidates_evaluated": result.n_candidates,
            "data_source": "train.csv + test.csv concatenados (o teste participa da validação cruzada)",
        },
        "data": {
            "n_rows": data.n_rows,
            "n_train_csv_rows": data.n_train_rows,
            "n_test_csv_rows": data.n_test_rows,
            "n_features": data.n_features,
            "feature_prefixes": list(FEATURE_PREFIXES),
            "excluded_columns": list(NON_FEATURE_COLUMNS) + ["categoria_rating"],
            "sample_n": data.sample_n,
            "empty_text_rows_kept": data.n_empty_text,
        },
        "param_grid": {k: list(v) for k, v in spec.param_distributions.items()},
        "best_params": result.best_params,
        "search_best_rmse": result.search_best_rmse,
        "metrics": result.aggregate,
        "article": {
            "best_params": dict(spec.article_best_params),
            "comparison": _article_comparison(spec, result.aggregate),
        },
        "sklearn_version": sklearn.__version__,
        "timing_seconds": {k: round(v, 3) for k, v in result.timing.items()},
    }


def save_article_results(
    result: ArticleResult, data: ArticleData, output_dir: str | Path
) -> dict[str, Path]:
    """Grava JSON agregado, CSV por fold, JSON do holdout e matriz de confusão.

    Arquivos (``<m>`` = nome do modelo): ``metrics/<m>_rq1.json``,
    ``metrics/<m>_rq1_folds.csv``, ``metrics/<m>_rq1_holdout.json`` e
    ``figures/confusion_matrix_<m>.png``.
    """
    name = result.spec.name
    out = Path(output_dir)
    metrics_dir, figures_dir = out / "metrics", out / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "summary": metrics_dir / f"{name}_rq1.json",
        "folds": metrics_dir / f"{name}_rq1_folds.csv",
        "holdout": metrics_dir / f"{name}_rq1_holdout.json",
        "confusion_matrix": figures_dir / f"confusion_matrix_{name}.png",
    }
    paths["summary"].write_text(
        json.dumps(summarize(result, data), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    result.fold_metrics.to_csv(paths["folds"], index=False)

    holdout_payload = {
        "model": name,
        "protocol": "treino em train.csv, avaliação em test.csv, com os hiperparâmetros da busca",
        "best_params": result.best_params,
        "sample_n": data.sample_n,
        **(result.holdout or {"note": "sem linhas de treino ou de teste para o holdout"}),
        "sklearn_version": sklearn.__version__,
    }
    paths["holdout"].write_text(
        json.dumps(holdout_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    save_confusion_matrix(
        result.y_true,
        result.oof_pred,
        paths["confusion_matrix"],
        title=f"Matriz de confusão — {name.upper()} (predições dos 5 folds)",
    )
    return paths
