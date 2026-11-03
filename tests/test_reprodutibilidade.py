"""Issue 10 — reprodutibilidade com seed fixa (split, treino e pipeline completo)."""
import random

import numpy as np
import pandas as pd
import pytest

from src.data.splitting import split_train_validation
from src.evaluation.metrics import compute_metrics
from src.models.baseline import RANDOM_STATE, predict, train_logistic_regression
from src.preprocessing.embedding import (
    build_embeddings,
    fit_vectorizer,
    load_matrix,
    transform_texts,
    build_vectorizer,
    TRAIN_MATRIX_FILE,
    TEST_MATRIX_FILE,
)
from src.preprocessing.encoding import encode_rating
from src.preprocessing.normalization import normalize_dataframe

POSITIVAS = ["otimo", "adorei", "excelente", "perfeito", "recomendo", "bom"]
NEGATIVAS = ["pessimo", "horrivel", "quebrou", "decepcao", "ruim", "devolvi"]
NEUTRAS = ["produto", "chegou", "entrega", "caixa", "preco", "compra", "dias"]


def _base_sintetica(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """Reviews artificiais: o rating influencia as palavras (aprendível)."""
    rng = random.Random(seed)
    linhas = []
    for i in range(n):
        rating = 1 + i % 5
        pos = (rating - 1) / 4
        palavras = [rng.choice(NEUTRAS) for _ in range(4)]
        palavras += [
            rng.choice(POSITIVAS if rng.random() < pos else NEGATIVAS) for _ in range(3)
        ]
        rng.shuffle(palavras)
        linhas.append({"text": " ".join(palavras), "rating": rating})
    return pd.DataFrame(linhas)


def _pipeline(df: pd.DataFrame, seed: int) -> tuple[dict, np.ndarray]:
    limpo = normalize_dataframe(df, "text")
    treino, validacao = split_train_validation(limpo, random_state=seed)
    vec, x_tr = fit_vectorizer(treino["text"], build_vectorizer(min_df=2))
    x_val = transform_texts(vec, validacao["text"])
    y_tr = encode_rating(treino["rating"]).to_numpy()
    y_val = encode_rating(validacao["rating"]).to_numpy()
    modelo = train_logistic_regression(x_tr, y_tr, random_state=seed)
    pred = predict(modelo, x_val)
    return compute_metrics(y_val, pred), pred


def test_seed_padrao_do_projeto_e_42():
    assert RANDOM_STATE == 42


def test_split_com_mesma_seed_e_identico():
    df = _base_sintetica()
    a_tr, a_val = split_train_validation(df, random_state=42)
    b_tr, b_val = split_train_validation(df, random_state=42)
    pd.testing.assert_frame_equal(a_tr, b_tr)
    pd.testing.assert_frame_equal(a_val, b_val)


def test_split_com_seeds_diferentes_muda_a_particao():
    df = _base_sintetica()
    a_tr, _ = split_train_validation(df, random_state=1)
    b_tr, _ = split_train_validation(df, random_state=2)
    assert not a_tr["text"].equals(b_tr["text"])


def test_treino_com_mesma_seed_gera_modelo_identico():
    df = normalize_dataframe(_base_sintetica(), "text")
    vec, x = fit_vectorizer(df["text"], build_vectorizer(min_df=2))
    y = encode_rating(df["rating"]).to_numpy()
    m1 = train_logistic_regression(x, y, random_state=42)
    m2 = train_logistic_regression(x, y, random_state=42)
    np.testing.assert_array_equal(m1.coef_, m2.coef_)
    np.testing.assert_array_equal(predict(m1, x), predict(m2, x))


def test_pipeline_completo_repetido_da_mesmas_metricas_e_previsoes():
    df = _base_sintetica()
    metricas_1, pred_1 = _pipeline(df, seed=42)
    metricas_2, pred_2 = _pipeline(df, seed=42)
    assert metricas_1 == metricas_2
    np.testing.assert_array_equal(pred_1, pred_2)


def test_pipeline_aprende_algo_melhor_que_o_acaso():
    """Sanidade: base artificial aprendível deve superar o acaso (20% com 5 classes)."""
    metricas, _ = _pipeline(_base_sintetica(), seed=42)
    assert metricas["accuracy"] > 0.30


def test_pipeline_em_amostra_real_e_reprodutivel(amostra_real):
    m1, p1 = _pipeline(amostra_real, seed=42)
    m2, p2 = _pipeline(amostra_real, seed=42)
    assert m1 == m2
    np.testing.assert_array_equal(p1, p2)


def test_build_embeddings_duas_vezes_gera_matrizes_identicas(tmp_path):
    df = _base_sintetica(300)
    treino, teste = df.iloc[:240], df.iloc[240:]
    treino.to_csv(tmp_path / "train.csv", index=False)
    teste.to_csv(tmp_path / "test.csv", index=False)

    for saida in ("a", "b"):
        build_embeddings(
            tmp_path / "train.csv", tmp_path / "test.csv", tmp_path / saida, min_df=2
        )
    for arquivo in (TRAIN_MATRIX_FILE, TEST_MATRIX_FILE):
        a = load_matrix(tmp_path / "a" / arquivo)
        b = load_matrix(tmp_path / "b" / arquivo)
        assert abs(a - b).max() == 0
