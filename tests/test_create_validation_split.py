"""Teste do gerador local de validation.csv (Issue 7)."""

import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLIT_SCRIPT = REPO_ROOT / "scripts" / "create_validation_split.py"


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


def test_create_validation_split_salva_csv_bruto_sem_alterar_origem(tmp_path):
    train_path = tmp_path / "train.csv"
    validation_path = tmp_path / "validation.csv"
    _reviews_with_one_empty_text().to_csv(train_path, index=False)
    source_bytes = train_path.read_bytes()

    result = subprocess.run(
        [
            sys.executable,
            str(SPLIT_SCRIPT),
            "--train",
            str(train_path),
            "--out",
            str(validation_path),
            "--validation-size",
            "0.2",
            "--seed",
            "42",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert validation_path.is_file()
    assert train_path.read_bytes() == source_bytes
    validation = pd.read_csv(validation_path)
    assert len(validation) == 10
    assert validation.columns.tolist() == ["text", "rating", "categoria", "marker"]
    assert validation["rating"].value_counts().sort_index().tolist() == [2] * 5
    assert "vazia" not in validation["marker"].tolist()
    assert validation.loc[0, "text"].startswith("review nota")
