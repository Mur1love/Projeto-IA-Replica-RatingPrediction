"""Testes do split de validação (Issue 7)."""

import pandas as pd
import pytest

from src.data.splitting import split_train_validation


def _reviews_with_one_empty_text():
    rows = [
        {
            "text": f"review nota {rating} exemplo {i}",
            "rating": rating,
            "categoria": f"categoria-{i % 2}",
            "marker": f"linha-{rating}-{i}",
        }
        for rating in range(1, 6)
        for i in range(10)
    ]
    rows.append(
        {"text": "   ", "rating": 1, "categoria": "vazia", "marker": "vazia"}
    )
    return pd.DataFrame(rows)


def test_split_train_validation_eh_estratificado_reprodutivel_e_preserva_linhas():
    original = _reviews_with_one_empty_text()
    original_copy = original.copy(deep=True)

    train_a, validation_a = split_train_validation(original)
    train_b, validation_b = split_train_validation(original)

    assert len(train_a) == 40
    assert len(validation_a) == 10
    assert train_a["rating"].value_counts().sort_index().tolist() == [8] * 5
    assert validation_a["rating"].value_counts().sort_index().tolist() == [2] * 5
    assert "vazia" not in set(train_a["marker"]) | set(validation_a["marker"])
    assert set(train_a["marker"]).isdisjoint(validation_a["marker"])
    assert set(train_a["marker"]) | set(validation_a["marker"]) == set(
        original.loc[original["marker"] != "vazia", "marker"]
    )
    assert train_a.columns.tolist() == original.columns.tolist()
    assert validation_a.columns.tolist() == original.columns.tolist()
    assert train_a["marker"].tolist() == train_b["marker"].tolist()
    assert validation_a["marker"].tolist() == validation_b["marker"].tolist()
    pd.testing.assert_frame_equal(original, original_copy)
    assert validation_a.loc[0, "text"].startswith("review nota")


@pytest.mark.parametrize("validation_size", [0, 1, -0.1, 1.1])
def test_split_train_validation_rejeita_fracao_fora_de_zero_a_um(validation_size):
    with pytest.raises(ValueError, match="validation_size"):
        split_train_validation(
            _reviews_with_one_empty_text(), validation_size=validation_size
        )
