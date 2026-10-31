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
