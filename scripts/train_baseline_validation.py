"""Treina a LR em 80% do treino e avalia nos 20% reservados (Issue 7)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import sklearn

from src.data.splitting import split_train_validation
from src.evaluation.metrics import compute_metrics
from src.evaluation.plots import save_confusion_matrix
from src.models.baseline import train_logistic_regression
from src.preprocessing.embedding import (
    DEFAULT_PARAMS,
    build_vectorizer,
    fit_vectorizer,
    transform_texts,
)
from src.preprocessing.encoding import encode_rating
from src.preprocessing.normalization import normalize_dataframe

TEXT_COL = "text"
RATING_COL = "rating"
METRIC_NAMES = ("accuracy", "f1_macro", "mae", "rmse")


def _load_baseline_metrics(path: str | Path) -> dict[str, float]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    metrics = payload.get("metrics_logreg")
    if not isinstance(metrics, dict) or set(METRIC_NAMES) - set(metrics):
        raise ValueError(
            f"{path}: esperado JSON baseline com metrics_logreg e {METRIC_NAMES}"
        )
    return {name: float(metrics[name]) for name in METRIC_NAMES}


def _write_comparison(
    baseline: dict[str, float],
    validation: dict[str, float],
    path: str | Path,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "metric",
                "full_train_official_test",
                "train_80_validation",
                "validation_minus_test",
                "better_direction",
            ],
        )
        writer.writeheader()
        for name in METRIC_NAMES:
            writer.writerow(
                {
                    "metric": name,
                    "full_train_official_test": baseline[name],
                    "train_80_validation": validation[name],
                    "validation_minus_test": validation[name] - baseline[name],
                    "better_direction": "higher" if name in {"accuracy", "f1_macro"} else "lower",
                }
            )
    return path


def run_validation_experiment(
    train_csv: str | Path,
    validation_out: str | Path,
    baseline_metrics: str | Path,
    out_metrics: str | Path,
    out_figure: str | Path,
    comparison_out: str | Path,
    *,
    validation_size: float = 0.2,
    seed: int = 42,
    C: float = 1.0,
    max_iter: int = 1000,
) -> dict[str, Any]:
    """Executa treino/validação sem carregar nem avaliar o teste oficial."""
    baseline = _load_baseline_metrics(baseline_metrics)
    raw = pd.read_csv(train_csv)
    train_raw, validation_raw = split_train_validation(
        raw, validation_size=validation_size, random_state=seed
    )

    validation_path = Path(validation_out)
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    validation_raw.to_csv(validation_path, index=False)

    train_df = normalize_dataframe(
        train_raw[[TEXT_COL, RATING_COL]], TEXT_COL, empty="drop"
    )
    validation_df = normalize_dataframe(
        validation_raw[[TEXT_COL, RATING_COL]], TEXT_COL, empty="drop"
    )
    vectorizer, X_train = fit_vectorizer(
        train_df[TEXT_COL], build_vectorizer(**DEFAULT_PARAMS)
    )
    X_validation = transform_texts(vectorizer, validation_df[TEXT_COL])
    y_train = encode_rating(train_df[RATING_COL]).to_numpy()
    y_validation = encode_rating(validation_df[RATING_COL]).to_numpy()

    model = train_logistic_regression(
        X_train,
        y_train,
        C=C,
        max_iter=max_iter,
        random_state=seed,
    )
    predictions = model.predict(X_validation)
    metrics_validation = compute_metrics(y_validation, predictions)
    save_confusion_matrix(y_validation, predictions, out_figure)
    comparison_path = _write_comparison(
        baseline, metrics_validation, comparison_out
    )

    dropped_empty = len(raw) - len(train_raw) - len(validation_raw)
    payload: dict[str, Any] = {
        "seed": seed,
        "validation_size": validation_size,
        "n_source_rows": int(len(raw)),
        "n_dropped_empty": int(dropped_empty),
        "n_train": int(len(train_df)),
        "n_validation": int(len(validation_df)),
        "validation_csv": str(validation_path),
        "validation_csv_sha256": hashlib.sha256(
            validation_path.read_bytes()
        ).hexdigest(),
        "sklearn_version": sklearn.__version__,
        "lr_params": {
            "C": C,
            "max_iter": max_iter,
            "solver": "lbfgs",
            "random_state": seed,
        },
        "vectorizer_params": dict(DEFAULT_PARAMS),
        "metrics_validation": metrics_validation,
        "baseline_metrics_path": str(baseline_metrics),
        "comparison_csv": str(comparison_path),
    }
    metrics_path = Path(out_metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Treina LR em 80% do treino e avalia nos 20% de validação."
    )
    parser.add_argument("--train", default="data/raw/train.csv")
    parser.add_argument("--validation-out", default="data/raw/validation.csv")
    parser.add_argument(
        "--baseline-metrics", default="results/metrics/baseline_logreg.json"
    )
    parser.add_argument(
        "--out-metrics", default="results/metrics/baseline_logreg_validation.json"
    )
    parser.add_argument(
        "--out-figure",
        default="results/figures/confusion_matrix_logreg_validation.png",
    )
    parser.add_argument(
        "--comparison-out", default="results/metrics/baseline_logreg_comparison.csv"
    )
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    args = parser.parse_args(argv)

    payload = run_validation_experiment(
        args.train,
        args.validation_out,
        args.baseline_metrics,
        args.out_metrics,
        args.out_figure,
        args.comparison_out,
        validation_size=args.validation_size,
        seed=args.seed,
        C=args.C,
        max_iter=args.max_iter,
    )
    print(
        "Treino/validação: "
        f"n_train={payload['n_train']} n_validation={payload['n_validation']} "
        f"(excluídas={payload['n_dropped_empty']}, seed={args.seed})"
    )
    print(f"Métricas validação: {payload['metrics_validation']}")
    print(f"CSV local: {payload['validation_csv']}")
    print(f"Comparação: {payload['comparison_csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
