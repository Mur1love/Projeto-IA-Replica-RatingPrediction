import json

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import TfidfVectorizer

from src.preprocessing.embedding import (
    DEFAULT_PARAMS,
    STATS_FILE,
    TEST_LABELS_FILE,
    TEST_MATRIX_FILE,
    TRAIN_LABELS_FILE,
    TRAIN_MATRIX_FILE,
    VECTORIZER_FILE,
    build_embeddings,
    build_vectorizer,
    fit_vectorizer,
    load_matrix,
    load_vectorizer,
    matrix_stats,
    save_matrix,
    save_vectorizer,
    transform_texts,
    vocabulary_size,
)

PEQUENO = {"min_df": 1, "max_df": 1.0}

TREINO = [
    "produto muito bom chegou rapido",
    "nao gostei produto muito ruim",
    "chegou rapido e bem embalado",
    "qualidade ruim nao recomendo",
    "muito bom recomendo produto",
    "entrega rapida produto bom",
]
TESTE = [
    "produto bom e chegou rapido",
    "nao recomendo qualidade pessima", 
]

def test_build_vectorizer_usa_parametros_iniciais():
    vec = build_vectorizer()
    assert isinstance(vec, TfidfVectorizer)
    for nome, valor in DEFAULT_PARAMS.items():
        assert getattr(vec, nome) == valor
    assert set(DEFAULT_PARAMS) == {
        "lowercase", "strip_accents", "ngram_range", "min_df", "max_df", "sublinear_tf",
    }


def test_build_vectorizer_aceita_overrides_e_nao_altera_o_padrao():
    vec = build_vectorizer(min_df=2, ngram_range=(1, 1))
    assert vec.min_df == 2 and vec.ngram_range == (1, 1)
    assert DEFAULT_PARAMS["min_df"] == 5 and DEFAULT_PARAMS["ngram_range"] == (1, 2)


def test_build_vectorizer_devolve_instancias_independentes():
    assert build_vectorizer() is not build_vectorizer()

def test_formato_da_matriz_de_treino():
    vec, x = fit_vectorizer(TREINO, build_vectorizer(**PEQUENO))
    assert sp.issparse(x) and x.format == "csr"
    assert x.shape == (len(TREINO), vocabulary_size(vec))
    assert x.dtype == np.float64
    assert (x.data > 0).all()  


def test_linhas_normalizadas_em_norma_l2():
    _, x = fit_vectorizer(TREINO, build_vectorizer(**PEQUENO))
    normas = np.sqrt(x.multiply(x).sum(axis=1)).A1
    np.testing.assert_allclose(normas, 1.0)


def test_ngram_range_gera_unigramas_e_bigramas():
    vec, _ = fit_vectorizer(TREINO, build_vectorizer(**PEQUENO))
    termos = set(vec.vocabulary_)
    assert "produto" in termos            
    assert "nao recomendo" in termos      

def test_transform_do_teste_tem_as_mesmas_colunas_do_treino():
    vec, x_treino = fit_vectorizer(TREINO, build_vectorizer(**PEQUENO))
    x_teste = transform_texts(vec, TESTE)
    assert x_teste.shape == (len(TESTE), x_treino.shape[1])
    assert x_teste.format == "csr"


def test_vetorizador_reutilizado_em_novos_textos():
    vec, x_treino = fit_vectorizer(TREINO, build_vectorizer(**PEQUENO))
    novos = ["produto bom", "nao gostei nada", "chegou rapido demais"]
    x_novos = transform_texts(vec, novos)
    assert x_novos.shape == (3, x_treino.shape[1])
    
    assert (transform_texts(vec, novos) != x_novos).nnz == 0

    x_repetido = transform_texts(vec, [TREINO[0]])
    np.testing.assert_allclose(x_repetido.toarray(), x_treino[0].toarray(), atol=1e-12)


def test_texto_sem_termos_do_vocabulario_vira_vetor_zerado():
    vec, _ = fit_vectorizer(TREINO, build_vectorizer(**PEQUENO))
    x = transform_texts(vec, ["zzz qqq", "", None, float("nan")])
    assert x.shape == (4, vocabulary_size(vec))
    assert x.nnz == 0


def test_transform_antes_do_fit_levanta_erro():
    with pytest.raises(NotFittedError):
        transform_texts(build_vectorizer(**PEQUENO), TESTE)

def test_vocabulario_vem_apenas_do_treino():
    vec, _ = fit_vectorizer(TREINO, build_vectorizer(**PEQUENO))
    transform_texts(vec, TESTE)
    assert "pessima" not in vec.vocabulary_
    referencia = build_vectorizer(**PEQUENO).fit(TREINO)
    assert vec.vocabulary_ == referencia.vocabulary_


def test_idf_nao_muda_ao_transformar_o_teste():
    vec, _ = fit_vectorizer(TREINO, build_vectorizer(**PEQUENO))
    idf_antes = vec.idf_.copy()
    transform_texts(vec, TESTE)
    np.testing.assert_array_equal(vec.idf_, idf_antes)

    junto = build_vectorizer(**PEQUENO).fit(TREINO + TESTE)
    assert len(junto.vocabulary_) > len(vec.vocabulary_)


def test_termo_so_do_teste_e_ignorado_na_transformacao():
    vec, _ = fit_vectorizer(TREINO, build_vectorizer(**PEQUENO))
    x = transform_texts(vec, ["pessima"])
    assert x.nnz == 0

def test_strip_accents_unifica_formas_nao_normalizadas():
    textos = ["não gostei", "nao gostei", "NÃO gostei"]
    vec, _ = fit_vectorizer(textos, build_vectorizer(**PEQUENO, ngram_range=(1, 1)))
    assert set(vec.vocabulary_) == {"nao", "gostei"}


def test_max_df_remove_termos_presentes_em_quase_todos_os_documentos():
    textos = ["produto bom", "produto ruim", "produto otimo", "produto caro"]
    vec, _ = fit_vectorizer(
        textos, build_vectorizer(min_df=1, max_df=0.9, ngram_range=(1, 1))
    )
    assert "produto" not in vec.vocabulary_


def test_min_df_remove_termos_raros():
    textos = ["produto bom", "produto bom", "produto xyzraro"]
    vec, _ = fit_vectorizer(
        textos, build_vectorizer(min_df=2, max_df=1.0, ngram_range=(1, 1))
    )
    assert "xyzraro" not in vec.vocabulary_
    assert {"produto", "bom"} <= set(vec.vocabulary_)


def test_sublinear_tf_reduz_peso_de_palavra_repetida():
    textos = ["bom bom bom bom ruim", "ruim"]
    comum = dict(min_df=1, max_df=1.0, ngram_range=(1, 1))
    _, x_lin = fit_vectorizer(textos, build_vectorizer(sublinear_tf=False, **comum))
    vec_sub, x_sub = fit_vectorizer(textos, build_vectorizer(sublinear_tf=True, **comum))
    col = vec_sub.vocabulary_["bom"]
    assert x_sub[0, col] < x_lin[0, col]

def test_matrix_stats_valores_conhecidos():
    m = sp.csr_matrix(np.array([[1.0, 0, 0, 0], [0, 2.0, 3.0, 0]]))
    s = matrix_stats(m)
    assert s["n_docs"] == 2 and s["n_features"] == 4
    assert s["nnz"] == 3
    assert s["sparsity"] == pytest.approx(1 - 3 / 8)
    assert s["avg_nonzero_per_doc"] == pytest.approx(1.5)


def test_matrix_stats_recusa_matriz_vazia():
    with pytest.raises(ValueError):
        matrix_stats(sp.csr_matrix((0, 5)))


def test_vocabulary_size_exige_vetorizador_ajustado():
    with pytest.raises(NotFittedError):
        vocabulary_size(build_vectorizer())
    vec, x = fit_vectorizer(TREINO, build_vectorizer(**PEQUENO))
    assert vocabulary_size(vec) == x.shape[1]

def test_vetorizador_persistido_e_recarregavel(tmp_path):
    vec, _ = fit_vectorizer(TREINO, build_vectorizer(**PEQUENO))
    caminho = save_vectorizer(vec, tmp_path / "sub" / VECTORIZER_FILE)
    assert caminho.exists()

    recarregado = load_vectorizer(caminho)
    assert recarregado.vocabulary_ == vec.vocabulary_
    np.testing.assert_array_equal(recarregado.idf_, vec.idf_)

    novos = TESTE + ["produto bom demais"]
    assert (transform_texts(recarregado, novos) != transform_texts(vec, novos)).nnz == 0


def test_save_vectorizer_recusa_vetorizador_nao_ajustado(tmp_path):
    with pytest.raises(NotFittedError):
        save_vectorizer(build_vectorizer(), tmp_path / VECTORIZER_FILE)


def test_matriz_salva_e_recarregada_identica(tmp_path):
    _, x = fit_vectorizer(TREINO, build_vectorizer(**PEQUENO))
    caminho = save_matrix(x, tmp_path / TRAIN_MATRIX_FILE)
    x2 = load_matrix(caminho)
    assert x2.shape == x.shape and x2.format == "csr"
    assert (x2 != x).nnz == 0

@pytest.fixture
def csvs(tmp_path):
    """train/test sintéticos com colunas extras (simulam as 58 features)."""
    treino = pd.DataFrame(
        {
            "categoria": ["a"] * 8,
            "text": [
                "Produto MUITO bom, chegou rápido!!!",
                "Não gostei, produto ruim",
                "Chegou rápido e bem embalado",
                None,                              
                "Qualidade ruim, não recomendo",
                "Muito bom, recomendo o produto",
                "   ",                             
                "Entrega rápida, produto bom",
            ],
            "rating": [5, 1, 4, 3, 1, 5, 2, 4],
            "str_qtcharacters": range(8),          
        }
    )
    teste = pd.DataFrame(
        {
            "categoria": ["a"] * 3,
            "text": ["Produto bom e rápido", "Não recomendo, péssima qualidade", None],
            "rating": [5, 1, 3],
            "str_qtcharacters": range(3),
        }
    )
    treino.to_csv(tmp_path / "train.csv", index=False)
    teste.to_csv(tmp_path / "test.csv", index=False)
    return tmp_path / "train.csv", tmp_path / "test.csv"


def test_pipeline_gera_e_salva_todos_os_artefatos(csvs, tmp_path):
    train_csv, test_csv = csvs
    out = tmp_path / "processed"
    stats = build_embeddings(train_csv, test_csv, out, **PEQUENO)

    for nome in (
        VECTORIZER_FILE, TRAIN_MATRIX_FILE, TEST_MATRIX_FILE,
        TRAIN_LABELS_FILE, TEST_LABELS_FILE, STATS_FILE,
    ):
        assert (out / nome).exists(), nome

    x_treino = load_matrix(out / TRAIN_MATRIX_FILE)
    x_teste = load_matrix(out / TEST_MATRIX_FILE)
    vec = load_vectorizer(out / VECTORIZER_FILE)
    assert x_treino.shape == (6, vocabulary_size(vec))   
    assert x_teste.shape == (2, x_treino.shape[1])       
    assert stats["train"]["n_dropped_empty"] == 2
    assert stats["test"]["n_dropped_empty"] == 1


def test_pipeline_mantem_linhas_alinhadas_com_os_rotulos(csvs, tmp_path):
    train_csv, test_csv = csvs
    out = tmp_path / "processed"
    build_embeddings(train_csv, test_csv, out, **PEQUENO)

    y_treino = np.load(out / TRAIN_LABELS_FILE)
    y_teste = np.load(out / TEST_LABELS_FILE)

    assert y_treino.tolist() == [4, 0, 3, 0, 4, 3]
    assert y_teste.tolist() == [4, 0]
    assert len(y_treino) == load_matrix(out / TRAIN_MATRIX_FILE).shape[0]
    assert len(y_teste) == load_matrix(out / TEST_MATRIX_FILE).shape[0]


def test_pipeline_ajusta_so_no_treino(csvs, tmp_path):
    train_csv, test_csv = csvs
    out = tmp_path / "processed"
    build_embeddings(train_csv, test_csv, out, **PEQUENO)
    vec = load_vectorizer(out / VECTORIZER_FILE)
    assert "pessima" not in vec.vocabulary_   


def test_pipeline_registra_dimensao_e_esparsidade(csvs, tmp_path):
    train_csv, test_csv = csvs
    out = tmp_path / "processed"
    build_embeddings(train_csv, test_csv, out, **PEQUENO)

    stats = json.loads((out / STATS_FILE).read_text(encoding="utf-8"))
    x_treino = load_matrix(out / TRAIN_MATRIX_FILE)
    assert stats["vocabulary_size"] == x_treino.shape[1]
    assert stats["train"]["n_features"] == stats["test"]["n_features"] == stats["vocabulary_size"]
    assert stats["train"]["sparsity"] == pytest.approx(1 - x_treino.nnz / np.prod(x_treino.shape))
    assert 0 <= stats["test"]["sparsity"] <= 1
    assert stats["vectorizer_params"]["ngram_range"] == [1, 2]
    assert stats["sklearn_version"]


def test_pipeline_ignora_as_features_textuais_do_csv(csvs, tmp_path):
    train_csv, test_csv = csvs
    out = tmp_path / "processed"
    build_embeddings(train_csv, test_csv, out, **PEQUENO)
    vec = load_vectorizer(out / VECTORIZER_FILE)
    assert not any("str_qtcharacters" in termo for termo in vec.vocabulary_)

    assert load_matrix(out / TRAIN_MATRIX_FILE).shape[1] == len(vec.vocabulary_)