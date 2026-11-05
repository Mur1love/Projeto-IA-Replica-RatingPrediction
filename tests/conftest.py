"""Fixtures compartilhadas pelos testes da Issue 10.

Os CSVs (data/raw/train.csv) não são versionados. Os testes que precisam
deles usam a fixture ``amostra_real`` e são PULADOS (skip) se o arquivo não
existir na máquina; os demais testes continuam rodando normalmente.
"""
from pathlib import Path

import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
TRAIN_CSV = RAIZ / "data" / "raw" / "train.csv"

# Comentários reais citados no artigo (Figuras 1 e 3): (texto, rating).
COMENTARIOS_DO_ARTIGO = [
    (
        "Eu amei a embalagem, textura, quantidade, qualidade, preço... tudo! "
        "Mas não consigo usar por conta do perfume que é muito, muito forte",
        2,
    ),
    ("Não gostei. Câmera pequena, não tem visão do bebê todo, só uma parte.", 1),
]


@pytest.fixture(scope="session")
def amostra_real() -> pd.DataFrame:
    """2000 linhas (text, rating) sorteadas do train.csv real, seed fixa."""
    if not TRAIN_CSV.exists():
        pytest.skip("data/raw/train.csv não encontrado (dados não são versionados).")
    df = pd.read_csv(TRAIN_CSV, usecols=["text", "rating"]).dropna()
    return df.sample(n=min(2000, len(df)), random_state=0).reset_index(drop=True)


# --- Protocolo do artigo (RQ1): CSVs sintéticos com as 58 features ----------
ARTICLE_PREFIX_COUNTS = {"str": 7, "pos": 19, "synt": 5, "lex": 10, "conc": 10, "twt": 5, "sbj": 2}
ARTICLE_FEATURES = [
    f"{prefixo}_f{i}" for prefixo, n in ARTICLE_PREFIX_COUNTS.items() for i in range(n)
]
_PALAVRAS = {1: "pessimo horrivel", 2: "ruim fraco", 3: "normal razoavel", 4: "bom legal", 5: "otimo excelente"}


def make_article_frame(n_por_grupo: int, seed: int, categorias=("auto", "pets")) -> pd.DataFrame:
    """Linhas sintéticas no formato do CSV do artigo (categoria, text, rating + 58 features).

    As features têm um deslocamento ligado ao rating, para o SVM conseguir aprender algo.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    linhas = []
    for categoria in categorias:
        for rating in range(1, 6):
            for _ in range(n_por_grupo):
                feats = rng.normal(size=len(ARTICLE_FEATURES)) + 0.4 * rating
                texto = f"{_PALAVRAS[rating]} produto {categoria} chegou"
                linhas.append([categoria, texto, rating, *feats])
    df = pd.DataFrame(linhas, columns=["categoria", "text", "rating", *ARTICLE_FEATURES])
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


@pytest.fixture
def article_csvs(tmp_path):
    """(train.csv, test.csv) sintéticos: 10 grupos categoria_rating, 24+8 linhas cada."""
    treino = make_article_frame(24, seed=1)
    teste = make_article_frame(8, seed=2)
    treino.loc[0, "text"] = None  # texto ausente: o protocolo NÃO remove linhas
    treino.to_csv(tmp_path / "train.csv", index=False)
    teste.to_csv(tmp_path / "test.csv", index=False)
    return tmp_path / "train.csv", tmp_path / "test.csv"
