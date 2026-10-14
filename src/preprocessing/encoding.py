"""Label encoding do rating (Issue #5).

Mapeamento adotado (explícito e reversível)::

    rating (nota do usuário)  ->  label (classe do modelo)
            1 estrela                    0
            2 estrelas                   1
            3 estrelas                   2
            4 estrelas                   3
            5 estrelas                   4

Por que 0..4 e não 1..5?
    O XGBoost (melhor modelo do artigo) exige classes contíguas começando
    em 0 e levanta ``ValueError: Invalid classes inferred from unique values
    of y`` com rótulos 1..5. Scikit-Learn (LR, RF, SVM, GBT) aceita qualquer
    rótulo, então usar 0..4 em todos os modelos mantém a comparação justa.

Ordem preservada:
    O mapeamento mantém a ordem das notas (label maior = nota maior). Isso
    importa porque o artigo também avalia MAE e RMSE sobre a nota (Seção 3.4):
    ``decode_rating`` devolve o valor na escala 1..5 original.

A coluna ``categoria`` NÃO recebe label encoding:
    O artigo usa apenas features textuais, então a categoria não entra no
    modelo como atributo (usá-la seria vazamento de informação para a RQ4).
    Ela só serve para agrupar resultados e estratificar as partições, e o
    scikit-learn aceita strings para isso. Use ``stratify_key`` para montar
    a chave "categoria + rating" da divisão estratificada do artigo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Fonte única da verdade: encode e decode derivam deste dicionário.
RATING_TO_LABEL: dict[int, int] = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
LABEL_TO_RATING: dict[int, int] = {l: r for r, l in RATING_TO_LABEL.items()}


def _lookup_table(mapping: dict[int, int]) -> np.ndarray:
    """Tabela de consulta: posição = valor de entrada, conteúdo = saída."""
    tabela = np.full(max(mapping) + 1, -1, dtype=np.int64)
    for origem, destino in mapping.items():
        tabela[origem] = destino
    return tabela


_ENCODE_LUT = _lookup_table(RATING_TO_LABEL)
_DECODE_LUT = _lookup_table(LABEL_TO_RATING)


def _validar(valores, validos, nome: str) -> np.ndarray:
    """Converte para array int64 ou levanta ValueError com mensagem clara.

    Aceita inteiros e floats com valor inteiro (5.0 vira 5: o pandas lê a
    coluna como float quando há NaN). Rejeita NaN, 2.5, strings, bool e
    valores fora do conjunto válido, em vez de codificar errado em silêncio.
    """
    if isinstance(valores, pd.Series):
        if not pd.api.types.is_numeric_dtype(valores) or pd.api.types.is_bool_dtype(valores):
            raise ValueError(f"{nome}: esperado tipo numérico, recebi {valores.dtype}")
        arr = valores.to_numpy(dtype="float64", na_value=np.nan)
    else:
        arr = np.atleast_1d(np.asarray(valores))
        if arr.dtype.kind not in "iuf":
            raise ValueError(f"{nome}: esperado tipo numérico, recebi {arr.dtype}")

    if arr.dtype.kind == "f":
        if not np.isfinite(arr).all():
            raise ValueError(f"{nome}: há valores ausentes (NaN/inf)")
        if not (arr == np.floor(arr)).all():
            raise ValueError(f"{nome}: há valores não inteiros")
    arr = arr.astype(np.int64)

    invalidos = np.unique(arr[~np.isin(arr, list(validos))])
    if invalidos.size:
        raise ValueError(
            f"{nome}: valores fora do esperado {sorted(validos)}: {invalidos[:5].tolist()}"
        )
    return arr


def _devolver(resultado: np.ndarray, original):
    """Devolve no mesmo formato da entrada: Series, escalar ou ndarray."""
    if isinstance(original, pd.Series):
        return pd.Series(resultado, index=original.index, name=original.name, dtype="int64")
    if np.ndim(original) == 0:
        return int(resultado[0])
    return resultado


def encode_rating(ratings):
    """Converte rating 1..5 em label 0..4.

    Aceita escalar, lista, ndarray ou ``pd.Series`` (o índice é preservado).
    Levanta ``ValueError`` para NaN, não inteiros ou notas fora de 1..5.
    """
    arr = _validar(ratings, RATING_TO_LABEL, "rating")
    return _devolver(_ENCODE_LUT[arr], ratings)


def decode_rating(labels):
    """Converte label 0..4 de volta para rating 1..5 (inverso de encode)."""
    arr = _validar(labels, LABEL_TO_RATING, "label")
    return _devolver(_DECODE_LUT[arr], labels)


def encode_dataframe(
    df: pd.DataFrame, rating_col: str = "rating", label_col: str = "label"
) -> pd.DataFrame:
    """Devolve uma cópia de ``df`` com a coluna ``label_col`` adicionada.

    A coluna ``rating`` original é mantida (necessária para MAE/RMSE e para
    conferir a reversibilidade).
    """
    out = df.copy()
    out[label_col] = encode_rating(out[rating_col])
    return out


def verify_distribution(ratings, labels) -> pd.DataFrame:
    """Confere se a contagem por classe é a mesma antes e depois do encoding.

    Retorna uma tabela ``rating | label | n_original | n_codificado`` e levanta
    ``ValueError`` se qualquer contagem divergir (ou se o tamanho mudou).
    """
    ratings_arr = _validar(ratings, RATING_TO_LABEL, "rating")
    labels_arr = _validar(labels, LABEL_TO_RATING, "label")
    if ratings_arr.size != labels_arr.size:
        raise ValueError(
            f"tamanhos diferentes: {ratings_arr.size} ratings e {labels_arr.size} labels"
        )

    linhas = [
        {
            "rating": r,
            "label": l,
            "n_original": int((ratings_arr == r).sum()),
            "n_codificado": int((labels_arr == l).sum()),
        }
        for r, l in RATING_TO_LABEL.items()
    ]
    tabela = pd.DataFrame(linhas)
    if (tabela["n_original"] != tabela["n_codificado"]).any():
        raise ValueError(f"distribuição divergente após o encoding:\n{tabela}")
    return tabela


def stratify_key(
    df: pd.DataFrame, categoria_col: str = "categoria", rating_col: str = "rating"
) -> pd.Series:
    """Chave 'categoria_rating' para a divisão estratificada do artigo.

    Exemplo: ``livros_5``. Passe-a em ``stratify=`` (train_test_split) ou como
    ``y`` do ``StratifiedKFold``; não precisa de encoding numérico.
    """
    return df[categoria_col].astype(str) + "_" + df[rating_col].astype(int).astype(str)
