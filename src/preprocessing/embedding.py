"""Vetorização TF-IDF da coluna ``text`` (versão "embedada" da base).

Pipeline::

    train.csv / test.csv
        -> normalize_dataframe (coluna ``text``; linhas vazias removidas)
        -> TfidfVectorizer.fit_transform(treino)      # ajuste SÓ no treino
        -> TfidfVectorizer.transform(teste)           # reaproveita o vocabulário e o IDF
        -> data/processed/tfidf_*.npz / tfidf_vectorizer.joblib / tfidf_stats.json

Evitando vazamento de dados (data leakage):
    O vocabulário e os pesos IDF são estatísticas da coleção. Se fossem
    calculados com o teste junto, o modelo "enxergaria" termos e frequências
    do teste. Por isso o ``fit`` acontece apenas em ``train.csv``; o teste (e,
    depois, qualquer texto novo na inferência) só passa por ``transform``.
    Termos que existem só no teste não entram no vocabulário e são ignorados.

As 58 features textuais do CSV NÃO entram nesta etapa: o CSV é lido apenas
com as colunas ``text`` e ``rating`` (``usecols``).

Linhas descartadas:
    Textos vazios após a normalização são removidos (``empty="drop"``). Como
    isso muda o número de linhas, o rótulo ``y`` é gerado a partir do mesmo
    DataFrame filtrado e salvo junto da matriz, mantendo as linhas alinhadas.

Uso pela linha de comando (a partir da raiz do repositório)::

    python -m src.preprocessing.embedding
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.validation import check_is_fitted

from src.preprocessing.encoding import encode_rating
from src.preprocessing.normalization import normalize_dataframe

logger = logging.getLogger(__name__)

DEFAULT_PARAMS: dict[str, Any] = {
    "lowercase": True,
    "strip_accents": "unicode",
    "ngram_range": (1, 2),
    "min_df": 5,
    "max_df": 0.95,
    "sublinear_tf": True,
}

# Nomes dos artefatos gravados em data/processed/.
VECTORIZER_FILE = "tfidf_vectorizer.joblib"
TRAIN_MATRIX_FILE = "tfidf_train.npz"
TEST_MATRIX_FILE = "tfidf_test.npz"
TRAIN_LABELS_FILE = "tfidf_train_y.npy"
TEST_LABELS_FILE = "tfidf_test_y.npy"
STATS_FILE = "tfidf_stats.json"

TEXT_COL = "text"
RATING_COL = "rating"

def build_vectorizer(**overrides: Any) -> TfidfVectorizer:
    """Cria um ``TfidfVectorizer`` ainda NÃO ajustado.

    Parte de ``DEFAULT_PARAMS``; qualquer argumento nomeado o sobrescreve
    (inclusive parâmetros extras do scikit-learn, como ``token_pattern``).
    Cada chamada devolve uma instância nova, sem estado compartilhado.
    """
    return TfidfVectorizer(**{**DEFAULT_PARAMS, **overrides})


def _as_texts(textos: Iterable[Any]) -> list[str]:
    """Lista de strings; None/NaN/não-string viram "" (documento sem termos)."""
    return [t if isinstance(t, str) else "" for t in textos]


def fit_vectorizer(
    textos_treino: Iterable[Any], vectorizer: TfidfVectorizer | None = None
) -> tuple[TfidfVectorizer, sp.csr_matrix]:
    """Ajusta o vetorizador nos textos de TREINO e devolve ``(vetorizador, matriz)``.

    Este é o único ponto do módulo que chama ``fit``. Se ``vectorizer`` não for
    informado, usa ``build_vectorizer()`` com os parâmetros padrão.
    """
    vec = vectorizer if vectorizer is not None else build_vectorizer()
    matriz = vec.fit_transform(_as_texts(textos_treino))
    return vec, matriz.tocsr()


def transform_texts(vectorizer: TfidfVectorizer, textos: Iterable[Any]) -> sp.csr_matrix:
    """Transforma textos com um vetorizador JÁ ajustado (teste ou inferência).

    Nunca reajusta: vocabulário e IDF permanecem os do treino. Levanta
    ``sklearn.exceptions.NotFittedError`` se o vetorizador não foi ajustado,
    em vez de ajustá-lo em silêncio com dados do teste.
    """
    check_is_fitted(vectorizer, "vocabulary_")
    return vectorizer.transform(_as_texts(textos)).tocsr()

def matrix_stats(matriz: sp.spmatrix) -> dict[str, Any]:
    """Formato, nº de valores não nulos e esparsidade de uma matriz esparsa.

    esparsidade = fração de células iguais a zero = 1 - nnz / (linhas * colunas).
    """
    n_linhas, n_colunas = matriz.shape
    if n_linhas == 0 or n_colunas == 0:
        raise ValueError(f"matriz vazia, formato {matriz.shape}")
    nnz = int(matriz.nnz)
    return {
        "n_docs": int(n_linhas),
        "n_features": int(n_colunas),
        "nnz": nnz,
        "sparsity": 1.0 - nnz / (n_linhas * n_colunas),
        "avg_nonzero_per_doc": nnz / n_linhas,
    }

def vocabulary_size(vectorizer: TfidfVectorizer) -> int:
    """Dimensão do vocabulário (número de colunas da matriz TF-IDF)."""
    check_is_fitted(vectorizer, "vocabulary_")
    return len(vectorizer.vocabulary_)


def save_vectorizer(vectorizer: TfidfVectorizer, path: str | Path) -> Path:
    """Persiste o vetorizador ajustado (joblib). Recusa vetorizador não ajustado."""
    check_is_fitted(vectorizer, "vocabulary_")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, path)
    return path


def load_vectorizer(path: str | Path) -> TfidfVectorizer:
    """Recarrega o vetorizador salvo por ``save_vectorizer``.

    Atenção: o joblib/pickle depende da versão do scikit-learn. A versão usada
    é registrada em ``tfidf_stats.json``. Carregue apenas arquivos gerados
    pelo próprio projeto.
    """
    vec = joblib.load(Path(path))
    check_is_fitted(vec, "vocabulary_")
    return vec


def save_matrix(matriz: sp.spmatrix, path: str | Path) -> Path:
    """Salva a matriz esparsa em formato ``.npz`` comprimido."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sp.save_npz(path, matriz.tocsr(), compressed=True)
    return path


def load_matrix(path: str | Path) -> sp.csr_matrix:
    """Carrega uma matriz salva por ``save_matrix``."""
    return sp.load_npz(Path(path)).tocsr()

def _load_split(path: str | Path, normalize_kwargs: dict[str, Any]) -> tuple[pd.DataFrame, int]:
    """Lê só ``text`` e ``rating`` (sem as 58 features) e normaliza o texto.

    Retorna ``(df_normalizado, n_linhas_removidas)``.
    """
    bruto = pd.read_csv(path, usecols=[TEXT_COL, RATING_COL])
    df = normalize_dataframe(bruto, TEXT_COL, empty="drop", **normalize_kwargs)
    return df, len(bruto) - len(df)


def build_embeddings(
    train_csv: str | Path,
    test_csv: str | Path,
    out_dir: str | Path,
    *,
    normalize_kwargs: dict[str, Any] | None = None,
    **vectorizer_params: Any,
) -> dict[str, Any]:
    """Gera e salva a versão TF-IDF da base. Retorna o dicionário de estatísticas.

    O vetorizador é ajustado apenas em ``train_csv``; ``test_csv`` só é
    transformado. Grava em ``out_dir``: matrizes de treino e teste, rótulos
    alinhados (``encode_rating``: 0..4), vetorizador e ``tfidf_stats.json``.
    """
    normalize_kwargs = dict(normalize_kwargs or {})
    out_dir = Path(out_dir)
    params = {**DEFAULT_PARAMS, **vectorizer_params}

    treino, descartadas_treino = _load_split(train_csv, normalize_kwargs)
    teste, descartadas_teste = _load_split(test_csv, normalize_kwargs)

    vec, x_treino = fit_vectorizer(treino[TEXT_COL], build_vectorizer(**params))
    x_teste = transform_texts(vec, teste[TEXT_COL])

    y_treino = encode_rating(treino[RATING_COL]).to_numpy()
    y_teste = encode_rating(teste[RATING_COL]).to_numpy()

    stats = {
        "sklearn_version": sklearn.__version__,
        "vectorizer_params": params,
        "normalization": normalize_kwargs,
        "vocabulary_size": vocabulary_size(vec),
        "train": {**matrix_stats(x_treino), "n_dropped_empty": descartadas_treino},
        "test": {**matrix_stats(x_teste), "n_dropped_empty": descartadas_teste},
    }

    save_vectorizer(vec, out_dir / VECTORIZER_FILE)
    save_matrix(x_treino, out_dir / TRAIN_MATRIX_FILE)
    save_matrix(x_teste, out_dir / TEST_MATRIX_FILE)
    np.save(out_dir / TRAIN_LABELS_FILE, y_treino)
    np.save(out_dir / TEST_LABELS_FILE, y_teste)
    (out_dir / STATS_FILE).write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    logger.info("Vocabulário: %d termos", stats["vocabulary_size"])
    for nome in ("train", "test"):
        s = stats[nome]
        logger.info(
            "%s: formato=(%d, %d) | esparsidade=%.4f%% | ~%.1f termos/doc | removidas=%d",
            nome, s["n_docs"], s["n_features"], 100 * s["sparsity"],
            s["avg_nonzero_per_doc"], s["n_dropped_empty"],
        )
    return stats


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Gera a versão TF-IDF da base.")
    parser.add_argument("--train", default="data/raw/train.csv")
    parser.add_argument("--test", default="data/raw/test.csv")
    parser.add_argument("--out", default="data/processed")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    build_embeddings(args.train, args.test, args.out)


if __name__ == "__main__":
    main()