import numpy as np
import pandas as pd
import pytest

from src.preprocessing.encoding import (
    LABEL_TO_RATING,
    RATING_TO_LABEL,
    decode_rating,
    encode_dataframe,
    encode_rating,
    stratify_key,
    verify_distribution,
)


# --- mapeamento -----------------------------------------------------------
def test_mapeamento_explicito():
    assert RATING_TO_LABEL == {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}


def test_mapeamento_inverso_consistente():
    assert LABEL_TO_RATING == {0: 1, 1: 2, 2: 3, 3: 4, 4: 5}
    for rating, label in RATING_TO_LABEL.items():
        assert LABEL_TO_RATING[label] == rating


def test_encode_preserva_ordem():
    labels = encode_rating([1, 2, 3, 4, 5])
    assert labels.tolist() == [0, 1, 2, 3, 4]
    assert (np.diff(labels) > 0).all()


# --- ida e volta ----------------------------------------------------------
@pytest.mark.parametrize("rating", [1, 2, 3, 4, 5])
def test_ida_e_volta_escalar(rating):
    assert decode_rating(encode_rating(rating)) == rating


def test_ida_e_volta_array_embaralhado():
    ratings = np.array([5, 1, 3, 3, 2, 5, 4, 1])
    assert decode_rating(encode_rating(ratings)).tolist() == ratings.tolist()


def test_ida_e_volta_no_sentido_inverso():
    for label in range(5):
        assert encode_rating(decode_rating(label)) == label


def test_escalar_devolve_int_python():
    assert isinstance(encode_rating(3), int)
    assert encode_rating(3) == 2


# --- formatos de entrada --------------------------------------------------
def test_series_preserva_indice_e_nome():
    s = pd.Series([5, 1, 3], index=[10, 20, 30], name="rating")
    out = encode_rating(s)
    assert isinstance(out, pd.Series)
    assert out.index.tolist() == [10, 20, 30]
    assert out.name == "rating"
    assert out.tolist() == [4, 0, 2]
    assert out.dtype == np.int64


def test_float_com_valor_inteiro_e_aceito():
    # pandas lê a coluna como float quando há NaN em algum lugar
    assert encode_rating(pd.Series([1.0, 5.0])).tolist() == [0, 4]


def test_lista_vazia():
    assert encode_rating([]).tolist() == []


# --- validação ------------------------------------------------------------
@pytest.mark.parametrize("ruim", [0, 6, -1, [1, 2, 9]])
def test_rating_fora_do_intervalo(ruim):
    with pytest.raises(ValueError, match="fora do esperado"):
        encode_rating(ruim)


@pytest.mark.parametrize("ruim", [-1, 5, 10])
def test_label_fora_do_intervalo(ruim):
    with pytest.raises(ValueError, match="fora do esperado"):
        decode_rating(ruim)


def test_rating_nan_e_rejeitado():
    with pytest.raises(ValueError, match="ausentes"):
        encode_rating(pd.Series([1.0, np.nan]))


def test_rating_nao_inteiro_e_rejeitado():
    with pytest.raises(ValueError, match="não inteiros"):
        encode_rating([2.5])


@pytest.mark.parametrize("ruim", [["3", "4"], pd.Series(["3", "4"]), [True, False]])
def test_tipos_nao_numericos_sao_rejeitados(ruim):
    with pytest.raises(ValueError, match="numérico"):
        encode_rating(ruim)


# --- DataFrame ------------------------------------------------------------
@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "categoria": ["livros", "pets", "livros", "moda", "pets"],
            "text": ["a", "b", "c", "d", "e"],
            "rating": [5, 1, 3, 2, 4],
        },
        index=[7, 8, 9, 10, 11],
    )


def test_encode_dataframe_nao_altera_original(df):
    original = df.copy()
    out = encode_dataframe(df)
    pd.testing.assert_frame_equal(df, original)
    assert "label" not in df.columns
    assert out["label"].tolist() == [4, 0, 2, 1, 3]
    assert out["rating"].tolist() == df["rating"].tolist()  # rating mantido


def test_encode_dataframe_colunas_customizadas(df):
    out = encode_dataframe(df, rating_col="rating", label_col="y")
    assert "y" in out.columns


def test_decode_da_coluna_label_recupera_rating(df):
    out = encode_dataframe(df)
    assert decode_rating(out["label"]).tolist() == df["rating"].tolist()


# --- distribuição ---------------------------------------------------------
def test_verify_distribution_ok():
    ratings = [1, 1, 2, 3, 3, 3, 5]
    tabela = verify_distribution(ratings, encode_rating(ratings))
    assert tabela["n_original"].tolist() == [2, 1, 3, 0, 1]
    assert tabela["n_original"].tolist() == tabela["n_codificado"].tolist()
    assert tabela["label"].tolist() == [0, 1, 2, 3, 4]


def test_verify_distribution_detecta_divergencia():
    ratings = [1, 2, 3]
    labels_errados = [0, 1, 1]  # o 3 virou 2 estrelas
    with pytest.raises(ValueError, match="divergente"):
        verify_distribution(ratings, labels_errados)


def test_verify_distribution_tamanhos_diferentes():
    with pytest.raises(ValueError, match="tamanhos"):
        verify_distribution([1, 2, 3], [0, 1])


# --- categoria ------------------------------------------------------------
def test_stratify_key(df):
    assert stratify_key(df).tolist() == ["livros_5", "pets_1", "livros_3", "moda_2", "pets_4"]


# --- integração com os modelos (por que 0..4) ------------------------------
def test_xgboost_aceita_labels_codificados():
    xgb = pytest.importorskip("xgboost")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 4))
    ratings = np.tile([1, 2, 3, 4, 5], 12)
    modelo = xgb.XGBClassifier(n_estimators=2)
    modelo.fit(X, encode_rating(ratings))
    preds = modelo.predict(X)
    assert set(decode_rating(preds)) <= {1, 2, 3, 4, 5}
