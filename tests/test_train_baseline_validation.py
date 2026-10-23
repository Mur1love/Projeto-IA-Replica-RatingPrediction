"""Testes da pipeline de treino/validação da Issue 7."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train_baseline_validation.py"


def _training_reviews():
    words = {1: "terrivel", 2: "ruim", 3: "neutro", 4: "bom", 5: "excelente"}
    rows = [
        {
            "text": f"produto {words[rating]} experiencia {i}",
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


def test_cli_treina_sem_ler_test_salva_split_metricas_figura_e_comparacao(tmp_path):
    train_path = tmp_path / "train.csv"
    validation_path = tmp_path / "validation.csv"
    baseline_path = tmp_path / "baseline.json"
    metrics_path = tmp_path / "validation_metrics.json"
    figure_path = tmp_path / "validation_confusion.png"
    comparison_path = tmp_path / "comparison.csv"

    _training_reviews().to_csv(train_path, index=False)
    baseline_path.write_text(
        json.dumps(
            {
                "metrics_logreg": {
                    "accuracy": 0.5,
                    "f1_macro": 0.48,
                    "mae": 0.7,
                    "rmse": 1.1,
                }
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--train",
            str(train_path),
            "--validation-out",
            str(validation_path),
            "--baseline-metrics",
            str(baseline_path),
            "--out-metrics",
            str(metrics_path),
            "--out-figure",
            str(figure_path),
            "--comparison-out",
            str(comparison_path),
            "--validation-size",
            "0.2",
            "--seed",
            "42",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    assert validation_path.is_file()
    assert metrics_path.is_file()
    assert figure_path.is_file() and figure_path.stat().st_size > 0
    assert comparison_path.is_file()

    validation = pd.read_csv(validation_path)
    assert len(validation) == 10
    assert validation.columns.tolist() == ["text", "rating", "categoria", "marker"]
    assert validation["rating"].value_counts().sort_index().tolist() == [2] * 5
    assert "vazia" not in validation["marker"].tolist()
    assert validation.loc[0, "text"].startswith("produto ")

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert payload["n_source_rows"] == 51
    assert payload["n_dropped_empty"] == 1
    assert payload["n_train"] == 40
    assert payload["n_validation"] == 10
    assert payload["seed"] == 42
    assert payload["validation_size"] == 0.2
    assert payload["validation_csv_sha256"] == hashlib.sha256(
        validation_path.read_bytes()
    ).hexdigest()
    assert set(payload["metrics_validation"]) == {
        "accuracy",
        "f1_macro",
        "mae",
        "rmse",
    }

    comparison = pd.read_csv(comparison_path)
    assert comparison["metric"].tolist() == ["accuracy", "f1_macro", "mae", "rmse"]
    assert comparison["better_direction"].tolist() == [
        "higher",
        "higher",
        "lower",
        "lower",
    ]
