"""Normalização leve de textos de reviews (Issue 4)."""

import re
import unicodedata

import pandas as pd

ABREVIACOES = {
    "vc": "você",
    "vcs": "vocês",
    "pq": "porque",
    "eh": "é",
    "tb": "também",
    "tbm": "também",
    "q": "que",
    "tá": "está",
}

# Repetição de 3+ (dígitos ficam de fora para não alterar "1000").
_REPETIDO = re.compile(r"([^\d\s])\1{2,}")
_ESPACOS = re.compile(r"\s+")
_NUMERO = re.compile(r"\d+(?:[.,]\d+)*")
# Seletores de variação dos emojis (ex.: ❤️) não são acentos.
_SELETORES_EMOJI = {"\ufe0e", "\ufe0f"}


def _remover_acentos(texto):
    decomposto = unicodedata.normalize("NFD", texto)
    limpo = "".join(
        c for c in decomposto
        if unicodedata.category(c) != "Mn" or c in _SELETORES_EMOJI
    )
    return unicodedata.normalize("NFC", limpo)


def _expandir_abreviacoes(texto, abreviacoes):
    if not abreviacoes:
        return texto
    chaves = sorted(abreviacoes, key=len, reverse=True)
    padrao = re.compile(r"(?<!\w)(" + "|".join(map(re.escape, chaves)) + r")(?!\w)")
    return padrao.sub(lambda m: abreviacoes[m.group(1)], texto)


def normalize(
    texto,
    *,
    strip_accent=True,
    collapse_repeats=True,
    expand_abbreviations=True,
    abbreviations=None,
    strip_punctuation=False,
    number_token=None,
):
    """Normaliza um texto de review (função pura, não altera a entrada).

    Aplica, nesta ordem: minúsculas, repetições, abreviações, acentos,
    pontuação, números e espaços. Entradas vazias, None ou NaN viram "".

    strip_accent: remove acentos (não == nao).
    collapse_repeats: 3+ caracteres repetidos viram 2 (ameiii -> ameii).
    expand_abbreviations / abbreviations: expande vc, pq etc.; aceita dicionário próprio.
    strip_punctuation: troca pontuação por espaço (também desfaz emoticons como :)).
    number_token: se informado, substitui cada número por esse token.
    """
    if not isinstance(texto, str):
        return ""

    t = unicodedata.normalize("NFC", texto).lower()

    if collapse_repeats:
        t = _REPETIDO.sub(r"\1\1", t)
    if expand_abbreviations:
        t = _expandir_abreviacoes(t, ABREVIACOES if abbreviations is None else abbreviations)
    if strip_accent:
        t = _remover_acentos(t)
    if strip_punctuation:
        t = "".join(" " if unicodedata.category(c).startswith("P") else c for c in t)
    if number_token is not None:
        t = _NUMERO.sub(lambda _: number_token, t)

    return _ESPACOS.sub(" ", t).strip()


def normalize_dataframe(df, text_col="text", *, empty="drop", fill_value="", **kwargs):
    """Normaliza a coluna de texto e trata os vazios.

    empty="drop" remove as linhas vazias; empty="fill" preenche com fill_value.
    Retorna uma cópia com índice reiniciado. kwargs vão para normalize.
    """
    if empty not in {"drop", "fill"}:
        raise ValueError(f"empty deve ser 'drop' ou 'fill', recebi {empty!r}")

    out = df.copy()
    out[text_col] = [normalize(t, **kwargs) for t in out[text_col]]
    vazio = out[text_col] == ""

    if empty == "drop":
        out = out.loc[~vazio]
    else:
        out.loc[vazio, text_col] = fill_value
    return out.reset_index(drop=True)