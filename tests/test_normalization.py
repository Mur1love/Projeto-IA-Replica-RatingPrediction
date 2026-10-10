import numpy as np
import pandas as pd
import pytest

from src.preprocessing.normalization import normalize, normalize_dataframe


def test_minusculas_e_espacos():
    assert normalize("  Produto   MUITO \t\n bom ") == "produto muito bom"


def test_strip_accent_unifica_nao():
    assert normalize("não") == normalize("nao") == "nao"
    assert normalize("Não recomendo, decepção") == "nao recomendo, decepcao"


def test_strip_accent_false_preserva_acentos():
    assert normalize("Não gostei", strip_accent=False) == "não gostei"


def test_emoji_com_seletor_de_variacao_preservado():
    assert normalize("amei ❤️") == "amei ❤️"


def test_repeticoes_viram_dois():
    assert normalize("Bom!!!! Ameiii ooo") == "bom!! ameii oo"
    assert normalize("loop ee") == "loop ee"
    assert normalize("ameiii", collapse_repeats=False) == "ameiii"


def test_repeticao_nao_altera_numeros():
    assert normalize("custou 1000 reais, nota 10") == "custou 1000 reais, nota 10"


def test_abreviacoes():
    assert normalize("Nao gostei, vc precisa melhorar pq eh caro") == (
        "nao gostei, voce precisa melhorar porque e caro"
    )
    assert normalize("o produto q comprei tá ruim", strip_accent=False) == (
        "o produto que comprei está ruim"
    )
    assert normalize("tb gostei, vcs entregam") == "tambem gostei, voces entregam"


def test_abreviacao_so_em_palavra_inteira():
    assert normalize("qualidade boa, vcard") == "qualidade boa, vcard"


def test_abreviacoes_desligadas_ou_customizadas():
    assert normalize("vc pq", expand_abbreviations=False) == "vc pq"
    assert normalize("blz vc", abbreviations={"blz": "beleza"}) == "beleza vc"


def test_pontuacao_emoji_e_emoticon_mantidos_por_padrao():
    assert normalize("Adorei 😍 :) Bom, barato!") == "adorei 😍 :) bom, barato!"


def test_strip_punctuation_mantem_emoji():
    assert normalize("Bom, barato! 😍", strip_punctuation=True) == "bom barato 😍"


def test_numeros_mantidos_ou_trocados_por_token():
    assert normalize("Nota 10") == "nota 10"
    assert normalize("Nota 10, pagou 29,90", number_token="<num>") == "nota <num>, pagou <num>"


def test_html_falso_positivo_nao_e_removido():
    assert normalize("Aperte <W> e <Shift>") == "aperte <w> e <shift>"


@pytest.mark.parametrize("vazio", [None, np.nan, "", "   ", "\n\t", 123])
def test_entradas_vazias_viram_string_vazia(vazio):
    assert normalize(vazio) == ""


def test_reviews_curtas_preservadas():
    assert normalize("Ok") == "ok"
    assert normalize("10") == "10"


def test_nao_altera_a_entrada():
    abrev = {"vc": "você"}
    normalize("Vc  ESTÁ certo!!!", abbreviations=abrev)
    assert abrev == {"vc": "você"}


@pytest.mark.parametrize("strip_accent", [True, False])
@pytest.mark.parametrize(
    "texto",
    [
        "Produto  MUITO   bom!!! Ameiii ooo produto  :)",
        "Nao gostei, vc precisa melhorar pq eh caro",
        "Chegou q nem tá funcionando, não recomendo 😡😡😡",
    ],
)
def test_idempotente(texto, strip_accent):
    uma_vez = normalize(texto, strip_accent=strip_accent)
    assert normalize(uma_vez, strip_accent=strip_accent) == uma_vez


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "categoria": ["livros", "pets", "games", "moda"],
            "text": ["Amei!!! Vc  é demais", np.nan, "   ", "Não gostei"],
            "rating": [5, 3, 1, 2],
        }
    )


def test_dataframe_remove_vazios_e_nao_altera_original(df):
    original = df.copy()
    out = normalize_dataframe(df)
    assert out["text"].tolist() == ["amei!! voce e demais", "nao gostei"]
    assert out["rating"].tolist() == [5, 2]
    assert out.index.tolist() == [0, 1]
    pd.testing.assert_frame_equal(df, original)


def test_dataframe_fill_mantem_linhas(df):
    out = normalize_dataframe(df, empty="fill", fill_value="<vazio>")
    assert out["text"].tolist() == ["amei!! voce e demais", "<vazio>", "<vazio>", "nao gostei"]


def test_dataframe_repassa_parametros(df):
    assert normalize_dataframe(df, strip_accent=False)["text"].iloc[-1] == "não gostei"


def test_dataframe_empty_invalido(df):
    with pytest.raises(ValueError):
        normalize_dataframe(df, empty="ignorar")