"""Notebook Colab (Issue 7) — RED."""

import json
from pathlib import Path


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
