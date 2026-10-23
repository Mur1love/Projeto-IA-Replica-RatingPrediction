"""Divisões reproduzíveis de treino e validação (Issue 7)."""

from __future__ import annotations

import math

import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocessing.normalization import normalize


def split_train_validation(
    df: pd.DataFrame,
    *,
    validation_size: float = 0.2,
    random_state: int = 42,
    text_col: str = "text",
    rating_col: str = "rating",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide avaliações brutas em treino e validação, estratificando por nota.

    Textos que ficam vazios após a normalização do projeto são excluídos antes
    do split, assim como na pipeline baseline. As linhas devolvidas mantêm os
    valores e todas as colunas originais; a entrada não é modificada.
    """
    ausentes = {text_col, rating_col} - set(df.columns)
    if ausentes:
        raise ValueError(f"colunas obrigatórias ausentes: {sorted(ausentes)}")

    try:
        frac = float(validation_size)
    except (TypeError, ValueError) as exc:
        raise ValueError("validation_size deve estar entre 0 e 1") from exc
    if not math.isfinite(frac) or not 0 < frac < 1:
        raise ValueError("validation_size deve estar entre 0 e 1")

    keep = df[text_col].map(normalize).ne("")
    eligible = df.loc[keep].reset_index(drop=True)
    if eligible.empty:
        raise ValueError("nenhuma linha com texto não vazio disponível para split")

    train, validation = train_test_split(
        eligible,
        test_size=frac,
        stratify=eligible[rating_col],
        random_state=random_state,
    )
    return train.reset_index(drop=True), validation.reset_index(drop=True)
