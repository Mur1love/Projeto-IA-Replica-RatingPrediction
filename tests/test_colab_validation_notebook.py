"""Testes do notebook Colab de validação (Issue 7)."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "notebooks" / "baseline_logreg_validation_colab.ipynb"


def _ratings_frame(count_per_rating: int, *, include_empty: bool = False):
    words = {1: "terrivel", 2: "ruim", 3: "neutro", 4: "bom", 5: "excelente"}
    rows = [
        {
            "text": f"produto {words[rating]} experiencia {i}",
            "rating": rating,
            "categoria": f"categoria-{i % 2}",
            "marker": f"linha-{rating}-{i}",
        }
        for rating in range(1, 6)
        for i in range(count_per_rating)
    ]
    if include_empty:
        rows.append(
            {"text": "   ", "rating": 1, "categoria": "vazia", "marker": "vazia"}
        )
    return pd.DataFrame(rows)


def test_notebook_colab_validation_tem_split_e_saidas():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell["source"])
        for cell in nb["cells"]
        if cell["cell_type"] == "code"
    )
    for token in [
        "files.upload",
        "SAMPLE_N",
        "validation_size",
        "VALIDATION_OUT",
        "baseline_logreg_validation.json",
        "baseline_logreg_comparison.csv",
        "confusion_matrix_logreg_validation.png",
        "files.download",
    ]:
        assert token in code, f"faltando no notebook: {token}"


def test_notebook_colab_validation_executa_com_dados_locais(tmp_path):
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    output_dir = tmp_path / "results"
    validation_path = tmp_path / "data" / "raw" / "validation.csv"
    _ratings_frame(100, include_empty=True).to_csv(train_path, index=False)
    _ratings_frame(20).to_csv(test_path, index=False)

    env = os.environ.copy()
    env.update(
        {
            "SAMPLE_N": "400",
            "OUTPUT_DIR": str(output_dir),
            "VALIDATION_OUT": str(validation_path),
            "TRAIN_CSV": str(train_path),
            "TEST_CSV": str(test_path),
        }
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            str(NOTEBOOK),
            "--output",
            "colab_validation_smoke.ipynb",
            "--output-dir",
            str(tmp_path),
            "--ExecutePreprocessor.timeout=600",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "colab_validation_smoke.ipynb").is_file()
    assert validation_path.is_file()
    assert len(pd.read_csv(validation_path)) == 80

    full_metrics = json.loads(
        (output_dir / "metrics" / "baseline_logreg.json").read_text(
            encoding="utf-8"
        )
    )
    validation_metrics = json.loads(
        (output_dir / "metrics" / "baseline_logreg_validation.json").read_text(
            encoding="utf-8"
        )
    )
    assert set(full_metrics["metrics_logreg"]) == {
        "accuracy",
        "f1_macro",
        "mae",
        "rmse",
    }
    assert set(validation_metrics["metrics_validation"]) == {
        "accuracy",
        "f1_macro",
        "mae",
        "rmse",
    }
    assert (output_dir / "figures" / "confusion_matrix_logreg.png").stat().st_size > 0
    assert (
        output_dir / "figures" / "confusion_matrix_logreg_validation.png"
    ).stat().st_size > 0
    assert len(
        pd.read_csv(output_dir / "metrics" / "baseline_logreg_comparison.csv")
    ) == 4
