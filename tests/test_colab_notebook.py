"""Notebook Colab (Issue 7) — RED."""

import json
import ast
import subprocess
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def test_notebook_colab_tem_celulas_obrigatorias():
    nb = json.loads(
        Path("notebooks/baseline_logreg_colab.ipynb").read_text(encoding="utf-8")
    )
    src = "\n".join(
        "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"
    )
    for token in [
        "files.upload",
        "train_logistic_regression",
        "compute_metrics",
        "save_confusion_matrix",
        "SEED",
        "SAMPLE_N",
    ]:
        assert token in src, f"faltando {token}"


def test_notebook_nao_forca_pandas_pinado_no_colab():
    nb = json.loads(
        Path("notebooks/baseline_logreg_colab.ipynb").read_text(encoding="utf-8")
    )
    cell1 = "".join(nb["cells"][1]["source"])
    assert "google.colab" in cell1 and "IN_COLAB" in cell1
    assert "pip install -q scikit-learn matplotlib nltk joblib" in cell1


def test_notebook_amostra_tamanho_solicitado_em_treino_e_teste():
    nb = json.loads(
        Path("notebooks/baseline_logreg_colab.ipynb").read_text(encoding="utf-8")
    )
    cell4 = "".join(nb["cells"][4]["source"])
    module = ast.parse(cell4)
    assignments = [
        node
        for node in module.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Tuple)
            and any(
                isinstance(name, ast.Name)
                and name.id in {"tr_sample", "te_sample"}
                for name in target.elts
            )
            for target in node.targets
        )
    ]
    assert len(assignments) == 2

    train_df = pd.DataFrame(
        {"text": [f"train {r}" for r in range(1, 6) for _ in range(50)],
         "rating": [r for r in range(1, 6) for _ in range(50)]}
    )
    test_df = pd.DataFrame(
        {"text": [f"test {r}" for r in range(1, 6) for _ in range(100)],
         "rating": [r for r in range(1, 6) for _ in range(100)]}
    )
    namespace = {
        "train_test_split": train_test_split,
        "train_df": train_df,
        "test_df": test_df,
        "SAMPLE_N": 100,
        "SEED": 42,
        "min": min,
        "max": max,
    }
    exec(compile(ast.Module(body=assignments, type_ignores=[]), "<sample-cell>", "exec"), namespace)

    assert len(namespace["tr_sample"]) == 100
    assert len(namespace["te_sample"]) == 200


def test_cell3_funciona_quando_main_ainda_nao_tem_modulos_issue7(tmp_path):
    nb = json.loads(
        Path("notebooks/baseline_logreg_colab.ipynb").read_text(encoding="utf-8")
    )
    cell3 = "".join(nb["cells"][3]["source"])
    verify_fallbacks = r"""
import importlib.abc
import pathlib
import sys

blocked = {
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
namespace = {"__name__": "__main__"}
exec(compile(sys.argv[1], "<colab-cell-3>", "exec"), namespace)

import numpy as np
import scipy.sparse as sp

X = sp.csr_matrix([[2.0, 0.0], [0.0, 2.0], [1.5, 0.2], [0.1, 1.8]])
y = np.array([0, 4, 0, 4])
model = namespace["train_logistic_regression"](X, y)
assert model.predict(X).shape == (4,)

metrics = namespace["compute_metrics"](
    np.array([0, 1, 2, 3, 4]), np.array([0, 1, 2, 3, 3])
)
assert metrics["accuracy"] == 0.8
assert metrics["mae"] == 0.2
assert abs(metrics["rmse"] - (1 / 5) ** 0.5) < 1e-12
assert namespace["majority_class_label"](np.array([0, 1, 1, 4])) == 1
assert namespace["predict_majority"](np.array([0, 1, 1, 4]), 3).tolist() == [1, 1, 1]

figure = pathlib.Path(sys.argv[2])
namespace["save_confusion_matrix"](
    np.array([0, 1, 2, 3, 4]), np.array([0, 1, 2, 3, 3]), figure
)
assert figure.is_file() and figure.stat().st_size > 0
"""
    result = subprocess.run(
        [sys.executable, "-c", verify_fallbacks, cell3, str(tmp_path / "cm.png")],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
