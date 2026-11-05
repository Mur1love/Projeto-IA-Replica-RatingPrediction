"""Protocolo do artigo sobre os CSVs reais (pulado se data/raw/ não existir)."""
import numpy as np
import pandas as pd
import pytest

from src.models.article_protocol import build_folds, load_article_data
from tests.conftest import RAIZ

TRAIN_CSV = RAIZ / "data" / "raw" / "train.csv"
TEST_CSV = RAIZ / "data" / "raw" / "test.csv"

pytestmark = pytest.mark.skipif(
    not (TRAIN_CSV.exists() and TEST_CSV.exists()),
    reason="data/raw/train.csv e test.csv não encontrados (dados não são versionados).",
)


@pytest.fixture(scope="module")
def real():
    return load_article_data(TRAIN_CSV, TEST_CSV)


def test_base_real_tem_58_features_e_todas_as_linhas(real):
    assert real.n_features == 58
    assert real.n_rows == 41172 + 10293           # o artigo usa treino + teste
    assert (real.n_train_rows, real.n_test_rows) == (41172, 10293)
    assert real.n_empty_text == 2                  # 1 por split; mantidas, não removidas
    assert not real.X.isna().any().any()


def test_features_reais_nao_incluem_texto_nem_identificacao(real):
    assert not {"text", "categoria", "rating", "categoria_rating"} & set(real.X.columns)
    assert real.X.columns.str.match(r"^(str|pos|synt|lex|conc|twt|sbj)_").all()


def test_folds_reais_sao_estratificados_e_particionam(real):
    folds = build_folds(real.groups, 5)
    assert sum(len(te) for _, te in folds) == real.n_rows
    totais = pd.Series(real.groups).value_counts(normalize=True)
    for _, te in folds:
        prop = pd.Series(real.groups[te]).value_counts(normalize=True).reindex(totais.index)
        assert (prop - totais).abs().max() < 0.002


def test_amostra_real_de_2000_linhas(real):
    amostra = load_article_data(TRAIN_CSV, TEST_CSV, sample_n=2000)
    assert amostra.n_rows == 2000 and len(set(amostra.groups)) == 50
    assert np.isin(amostra.y, [0, 1, 2, 3, 4]).all()
    build_folds(amostra.groups, 5)                 # todos os grupos têm >= 5 exemplos
