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


def test_notebook_validation_fallback_funciona_sem_modulos_issue7(tmp_path):
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cell3 = "".join(nb["cells"][3]["source"])
    verify_fallbacks = r"""
import importlib.abc
import pathlib
import sys

blocked = {
    "src.data.splitting",
    "src.models.baseline",
    "src.evaluation.metrics",
    "src.evaluation.plots",
}

class MissingIssue7Modules(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in blocked:
            raise ModuleNotFoundError(
                f"No module named {fullname!r} (simulated origin/main)", name=fullname
            )

sys.meta_path.insert(0, MissingIssue7Modules())
namespace = {
    "__name__": "__main__",
    "Path": pathlib.Path,
    "DATA_DIR": pathlib.Path(sys.argv[2]) / "data" / "raw",
}
exec(compile(sys.argv[1], "<validation-cell-3>", "exec"), namespace)

import numpy as np
import pandas as pd
import scipy.sparse as sp

rows = [
    {"text": f"review rating {rating} sample {i}", "rating": rating, "marker": f"{rating}-{i}"}
    for rating in range(1, 6)
    for i in range(10)
]
rows.append({"text": "   ", "rating": 1, "marker": "empty"})
train, validation = namespace["split_train_validation"](pd.DataFrame(rows))
assert (len(train), len(validation)) == (40, 10)
assert "empty" not in set(train.marker) | set(validation.marker)

X = sp.csr_matrix([[2.0, 0.0], [0.0, 2.0], [1.5, 0.2], [0.1, 1.8]])
y = np.array([0, 4, 0, 4])
model = namespace["train_logistic_regression"](X, y)
assert model.predict(X).shape == (4,)
assert namespace["compute_metrics"](y, y)["accuracy"] == 1.0
assert namespace["majority_class_label"](np.array([0, 1, 1, 4])) == 1
assert namespace["predict_majority"](np.array([0, 1, 1, 4]), 2).tolist() == [1, 1]

figure = pathlib.Path(sys.argv[2]) / "fallback.png"
namespace["save_confusion_matrix"](y, model.predict(X), figure)
assert figure.is_file() and figure.stat().st_size > 0
"""
    env = os.environ.copy()
    env["OUTPUT_DIR"] = str(tmp_path / "results")
    result = subprocess.run(
        [sys.executable, "-c", verify_fallbacks, cell3, str(tmp_path)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


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
    assert full_metrics["n_train"] == 400
    assert full_metrics["n_test"] == 100
    assert full_metrics["n_dropped_train"] == 0
    assert set(full_metrics["metrics_majority"]) == {
        "accuracy",
        "f1_macro",
        "mae",
        "rmse",
    }
    assert full_metrics["majority_label"] in range(5)
    assert set(validation_metrics["metrics_validation"]) == {
        "accuracy",
        "f1_macro",
        "mae",
        "rmse",
    }
    assert validation_metrics["n_source_rows"] == 400
    assert validation_metrics["n_dropped_empty"] == 0
    assert validation_metrics["n_train"] == 320
    assert validation_metrics["n_validation"] == 80
    assert (output_dir / "figures" / "confusion_matrix_logreg.png").stat().st_size > 0
    assert (
        output_dir / "figures" / "confusion_matrix_logreg_validation.png"
    ).stat().st_size > 0
    comparison = pd.read_csv(output_dir / "metrics" / "baseline_logreg_comparison.csv")
    assert len(comparison) == 4
    assert comparison.columns.tolist() == [
        "metric",
        "sample_train_test",
        "sample_80_validation",
        "validation_minus_test",
        "better_direction",
    ]
