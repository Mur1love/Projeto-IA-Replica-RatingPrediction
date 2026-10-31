"""Issue 10 — ida e volta do label encoding, também com os ratings reais."""
import numpy as np
import pandas as pd
import pytest

from src.preprocessing.encoding import (
    decode_rating,
    encode_dataframe,
    encode_rating,
    verify_distribution,
)


def test_ida_e_volta_dos_cinco_ratings():
    ratings = np.array([1, 2, 3, 4, 5])
    labels = encode_rating(ratings)
    np.testing.assert_array_equal(labels, [0, 1, 2, 3, 4])
    np.testing.assert_array_equal(decode_rating(labels), ratings)


def test_ida_e_volta_em_amostra_real(amostra_real):
    ratings = amostra_real["rating"].astype(int)
    labels = encode_rating(ratings)
    assert labels.between(0, 4).all()
    pd.testing.assert_series_equal(
        decode_rating(labels), ratings.astype("int64"), check_names=False
    )


def test_encoding_real_e_bijetor_e_preserva_ordem(amostra_real):
    ratings = amostra_real["rating"].astype(int).to_numpy()
    labels = encode_rating(ratings)  # ndarray entra, ndarray sai
    assert len(set(ratings)) == len(set(labels))
    ordem = np.argsort(ratings, kind="stable")
    assert (np.diff(labels[ordem]) >= 0).all()


def test_distribuicao_igual_antes_e_depois_em_amostra_real(amostra_real):
    ratings = amostra_real["rating"].astype(int)
    labels = encode_rating(ratings)
    tabela = verify_distribution(ratings, labels)  # levanta ValueError se divergir
    assert tabela["n_original"].sum() == len(ratings)
    assert (tabela["n_original"] == tabela["n_codificado"]).all()


def test_encode_dataframe_real_mantem_linhas_e_rating(amostra_real):
    out = encode_dataframe(amostra_real)
    assert len(out) == len(amostra_real)
    pd.testing.assert_series_equal(out["rating"], amostra_real["rating"])
    assert (decode_rating(out["label"]) == out["rating"].astype("int64")).all()


@pytest.mark.parametrize("ruim", [0, 6, -1])
def test_rating_invalido_e_rejeitado(ruim):
    with pytest.raises(ValueError):
        encode_rating([3, ruim])
