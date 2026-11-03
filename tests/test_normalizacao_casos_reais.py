"""Issue 10 — normalização testada com casos reais da base."""
import unicodedata

import pytest

from src.preprocessing.normalization import normalize
from tests.conftest import COMENTARIOS_DO_ARTIGO


def test_comentario_da_figura_1_do_artigo():
    texto = COMENTARIOS_DO_ARTIGO[0][0]
    assert normalize(texto) == (
        "eu amei a embalagem, textura, quantidade, qualidade, preco.. tudo! "
        "mas nao consigo usar por conta do perfume que e muito, muito forte"
    )


def test_comentario_da_figura_3_do_artigo():
    texto = COMENTARIOS_DO_ARTIGO[1][0]
    assert normalize(texto) == (
        "nao gostei. camera pequena, nao tem visao do bebe todo, so uma parte."
    )


def test_review_informal_com_abreviacoes_e_repeticoes():
    texto = "Pq vc não recomendou?!!! Ameiiii o produto, tá 10/10"
    assert normalize(texto) == "porque voce nao recomendou?!! ameii o produto, esta 10/10"


def test_negacao_nao_e_perdida():
    """'não' é decisiva para o rating (artigo, feature F54): vira 'nao', nunca some."""
    for texto, _ in COMENTARIOS_DO_ARTIGO:
        antes = texto.lower().replace(".", " ").replace(",", " ").split().count("não")
        depois = normalize(texto).replace(".", " ").replace(",", " ").split().count("nao")
        assert antes >= 1
        assert depois == antes


@pytest.mark.parametrize("texto,_rating", COMENTARIOS_DO_ARTIGO)
def test_normalizacao_do_artigo_e_idempotente(texto, _rating):
    uma_vez = normalize(texto)
    assert normalize(uma_vez) == uma_vez


def test_invariantes_em_amostra_real(amostra_real):
    """Em 2000 reviews reais: sempre str, minúsculas, sem acento e sem espaço sobrando."""
    for texto in amostra_real["text"]:
        out = normalize(texto)
        assert isinstance(out, str)
        assert out == out.lower()
        assert out == out.strip()
        assert "  " not in out
        sem_acento = [
            c for c in unicodedata.normalize("NFD", out)
            if unicodedata.category(c) == "Mn" and c not in {"\ufe0e", "\ufe0f"}
        ]
        assert not sem_acento, f"acento sobrou em: {out!r}"


def test_normalizacao_idempotente_em_amostra_real(amostra_real):
    for texto in amostra_real["text"]:
        uma_vez = normalize(texto)
        assert normalize(uma_vez) == uma_vez, f"não idempotente: {texto!r}"


def test_amostra_real_quase_nenhum_texto_fica_vazio(amostra_real):
    vazios = sum(normalize(t) == "" for t in amostra_real["text"])
    assert vazios / len(amostra_real) < 0.05
