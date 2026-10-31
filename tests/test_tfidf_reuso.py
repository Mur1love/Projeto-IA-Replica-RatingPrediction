"""Issue 10 — reutilização do vetorizador TF-IDF (fit só no treino)."""
import numpy as np
import pytest

from src.data.splitting import split_train_validation
from src.preprocessing.embedding import (
    build_vectorizer,
    fit_vectorizer,
    load_vectorizer,
    save_vectorizer,
    transform_texts,
    vocabulary_size,
)
from src.preprocessing.normalization import normalize_dataframe


@pytest.fixture(scope="module")
def treino_e_validacao(amostra_real):
    df = normalize_dataframe(amostra_real, "text")
    return split_train_validation(df)


def test_transform_reaproveita_vocabulario_e_idf(treino_e_validacao):
    treino, validacao = treino_e_validacao
    vec, _ = fit_vectorizer(treino["text"])
    vocab_antes = dict(vec.vocabulary_)
    idf_antes = vec.idf_.copy()

    transform_texts(vec, validacao["text"])

    assert dict(vec.vocabulary_) == vocab_antes
    np.testing.assert_array_equal(vec.idf_, idf_antes)


def test_treino_e_validacao_tem_as_mesmas_colunas(treino_e_validacao):
    treino, validacao = treino_e_validacao
    vec, x_treino = fit_vectorizer(treino["text"])
    x_val = transform_texts(vec, validacao["text"])
    assert x_treino.shape[1] == x_val.shape[1] == vocabulary_size(vec)
    assert x_val.shape[0] == len(validacao)


def test_transform_do_treino_igual_ao_fit_transform(treino_e_validacao):
    treino, _ = treino_e_validacao
    vec, x_fit = fit_vectorizer(treino["text"])
    x_transform = transform_texts(vec, treino["text"])
    assert abs(x_fit - x_transform).max() < 1e-12


def test_termo_so_da_validacao_nao_entra_no_vocabulario(treino_e_validacao):
    treino, validacao = treino_e_validacao
    vec, _ = fit_vectorizer(treino["text"])
    so_validacao = {
        p for t in validacao["text"] for p in t.split()
    } - {p for t in treino["text"] for p in t.split()}
    if not so_validacao:
        pytest.skip("a amostra não tem palavra exclusiva da validação")
    assert not (so_validacao & set(vec.vocabulary_))


def test_vetorizador_salvo_e_recarregado_gera_a_mesma_matriz(treino_e_validacao, tmp_path):
    treino, validacao = treino_e_validacao
    vec, _ = fit_vectorizer(treino["text"])
    caminho = save_vectorizer(vec, tmp_path / "tfidf.joblib")
    recarregado = load_vectorizer(caminho)
    a = transform_texts(vec, validacao["text"])
    b = transform_texts(recarregado, validacao["text"])
    assert abs(a - b).max() == 0


def test_reuso_com_textos_pequenos_sem_dados_reais():
    """Sempre roda (não depende do CSV): min_df=1 para um vocabulário minúsculo."""
    vec, _ = fit_vectorizer(
        ["produto otimo", "produto ruim quebrado", "otimo recomendo"],
        build_vectorizer(min_df=1, max_df=1.0),
    )
    x = transform_texts(vec, ["produto desconhecidoxyz"])
    assert x.shape == (1, vocabulary_size(vec))
    assert "desconhecidoxyz" not in vec.vocabulary_
    assert x.nnz > 0  # 'produto' é conhecido
