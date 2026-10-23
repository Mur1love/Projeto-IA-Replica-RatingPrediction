"""Treino fim a fim do baseline TF-IDF + Regressão Logística (Issue 7)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

from src.evaluation.metrics import (
    compute_metrics,
    majority_class_label,
    predict_majority,
)
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

logger = logging.getLogger(__name__)

TEXT_COL = "text"
RATING_COL = "rating"


def _load_normalized(csv_path: str | Path) -> tuple[pd.DataFrame, int]:
    bruto = pd.read_csv(csv_path, usecols=[TEXT_COL, RATING_COL])
    df = normalize_dataframe(bruto, TEXT_COL, empty="drop")
    return df, len(bruto) - len(df)


def _stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n <= 0 or n >= len(df):
        return df
    sub, _ = train_test_split(
        df, train_size=n, stratify=df[RATING_COL], random_state=seed
    )
    return sub.reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Treina o baseline TF-IDF + LR (Issue 7).")
    parser.add_argument("--train", default="data/raw/train.csv")
    parser.add_argument("--test", default="data/raw/test.csv")
    parser.add_argument("--out-metrics", default="results/metrics/baseline_logreg.json")
    parser.add_argument("--out-figure", default="results/figures/confusion_matrix_logreg.png")
    parser.add_argument("--sample-n", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--val-size", type=float, default=0.0)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    np.random.seed(args.seed)

    train_df, dropped_train = _load_normalized(args.train)
    test_df, dropped_test = _load_normalized(args.test)

    if args.sample_n and args.sample_n > 0:
        train_df = _stratified_sample(train_df, args.sample_n, args.seed)
        n_test = min(len(test_df), max(200, args.sample_n // 4))
        test_df = _stratified_sample(test_df, n_test, args.seed)

    # Validação opcional sem tocar no teste: checa C e retreina no treino cheio.
    val_metrics = None
    fit_df = train_df
    if args.val_size and args.val_size > 0.0:
        tr_part, val_part = train_test_split(
            train_df, test_size=args.val_size, stratify=train_df[RATING_COL],
            random_state=args.seed,
        )
        vec_val, x_tr = fit_vectorizer(tr_part[TEXT_COL], build_vectorizer(**DEFAULT_PARAMS))
        x_val = transform_texts(vec_val, val_part[TEXT_COL])
        y_tr = encode_rating(tr_part[RATING_COL]).to_numpy()
        y_val = encode_rating(val_part[RATING_COL]).to_numpy()
        probe = train_logistic_regression(x_tr, y_tr, C=args.C, max_iter=args.max_iter, random_state=args.seed)
        val_metrics = compute_metrics(y_val, probe.predict(x_val))
        logger.info("Validação (C=%s): %s", args.C, val_metrics)
        fit_df = train_df

    vec, x_train = fit_vectorizer(fit_df[TEXT_COL], build_vectorizer(**DEFAULT_PARAMS))
    x_test = transform_texts(vec, test_df[TEXT_COL])
    y_train = encode_rating(fit_df[RATING_COL]).to_numpy()
    y_test = encode_rating(test_df[RATING_COL]).to_numpy()

    model = train_logistic_regression(
        x_train, y_train, C=args.C, max_iter=args.max_iter, random_state=args.seed
    )
    y_pred = model.predict(x_test)
    metrics_lr = compute_metrics(y_test, y_pred)

    maj_label = majority_class_label(y_train)
    metrics_maj = compute_metrics(y_test, predict_majority(y_train, len(y_test)))

    save_confusion_matrix(y_test, y_pred, args.out_figure)

    payload = {
        "seed": args.seed,
        "lr_params": {"C": args.C, "max_iter": args.max_iter, "solver": "lbfgs"},
        "vectorizer_params": dict(DEFAULT_PARAMS),
        "normalization": {},
        "n_train": int(len(fit_df)),
        "n_test": int(len(test_df)),
        "n_dropped_train": int(dropped_train),
        "n_dropped_test": int(dropped_test),
        "sklearn_version": sklearn.__version__,
        "majority_label": int(maj_label),
        "metrics_logreg": metrics_lr,
        "metrics_majority": metrics_maj,
    }
    if val_metrics is not None:
        payload["metrics_val"] = val_metrics

    out_m = Path(args.out_metrics)
    out_m.parent.mkdir(parents=True, exist_ok=True)
    out_m.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    logger.info("Treino: n=%d teste: n=%d (removidas %d/%d)", len(fit_df), len(test_df), dropped_train, dropped_test)
    logger.info("LR: %s", metrics_lr)
    logger.info("Majoritária (label=%d): %s", maj_label, metrics_maj)
    logger.info("Métricas → %s | Figura → %s", out_m, args.out_figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
