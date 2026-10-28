"""Compara o baseline TF-IDF + Regressão Logística com e sem normalização."""


from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

from src.data.splitting import split_train_validation
from src.evaluation.metrics import compute_metrics
from src.models.baseline import train_logistic_regression
from src.preprocessing.embedding import (
    DEFAULT_PARAMS,
    build_vectorizer,
    fit_vectorizer,
    matrix_stats,
    transform_texts,
)
from src.preprocessing.encoding import encode_rating
from src.preprocessing.normalization import normalize

logger = logging.getLogger("compare_normalization")

TEXT_COL = "text"
RATING_COL = "rating"
REFERENCE = "sem_normalizacao"
METRICS = ("accuracy", "f1_macro", "mae", "rmse")
HIGHER_IS_BETTER = {"accuracy": True, "f1_macro": True, "mae": False, "rmse": False}

_OFF = {"strip_accent": False, "collapse_repeats": False, "expand_abbreviations": False}

# nome -> kwargs de normalize(); None = texto bruto, sem chamar normalize().
VARIANTS: dict[str, dict[str, Any] | None] = {
    "sem_normalizacao": None,
    "normalizacao_completa": {},
    "apenas_acentos": {**_OFF, "strip_accent": True},
    "apenas_repeticoes": {**_OFF, "collapse_repeats": True},
    "apenas_abreviacoes": {**_OFF, "expand_abbreviations": True},
    "completa_sem_pontuacao": {"strip_punctuation": True},
    "completa_numero_token": {"number_token": "NUM"},
}


# --------------------------------------------------------------------------- #
# Dados
# --------------------------------------------------------------------------- #
def load_raw(path: str | Path) -> pd.DataFrame:
    """Lê só ``text`` e ``rating``; texto ausente vira string vazia."""
    df = pd.read_csv(path, usecols=[TEXT_COL, RATING_COL])
    df[TEXT_COL] = df[TEXT_COL].where(df[TEXT_COL].map(lambda t: isinstance(t, str)), "")
    return df


def common_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Mantém as linhas não vazias sob a normalização completa (regra do baseline)."""
    keep = df[TEXT_COL].map(normalize).ne("")
    return df.loc[keep].reset_index(drop=True)


def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n >= len(df):
        return df
    sub, _ = train_test_split(df, train_size=n, stratify=df[RATING_COL], random_state=seed)
    return sub.reset_index(drop=True)


def apply_variant(texts: pd.Series, kwargs: dict[str, Any] | None) -> list[str]:
    """Aplica uma variante ao texto. ``None`` devolve o texto bruto."""
    if kwargs is None:
        return texts.tolist()
    return [normalize(t, **kwargs) for t in texts]


# --------------------------------------------------------------------------- #
# Experimento
# --------------------------------------------------------------------------- #
def run_variant(
    name: str,
    kwargs: dict[str, Any] | None,
    train: pd.DataFrame,
    test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    *,
    seed: int,
    C: float,
    max_iter: int,
) -> tuple[dict[str, Any], np.ndarray]:
    """Treina e avalia uma variante. Devolve (linha da tabela, previsões no teste)."""
    t0 = time.perf_counter()
    x_text_train = apply_variant(train[TEXT_COL], kwargs)
    x_text_test = apply_variant(test[TEXT_COL], kwargs)

    vec, x_train = fit_vectorizer(x_text_train, build_vectorizer(**DEFAULT_PARAMS))
    x_test = transform_texts(vec, x_text_test)
    model = train_logistic_regression(x_train, y_train, C=C, max_iter=max_iter, random_state=seed)
    y_pred = model.predict(x_test)

    row = {
        "variante": name,
        **compute_metrics(y_test, y_pred),
        "vocabulario": len(vec.vocabulary_),
        "termos_por_doc": round(matrix_stats(x_train)["avg_nonzero_per_doc"], 2),
        "tempo_s": round(time.perf_counter() - t0, 1),
    }
    logger.info(
        "%-24s acc=%.4f f1=%.4f mae=%.4f rmse=%.4f vocab=%d (%.0fs)",
        name, row["accuracy"], row["f1_macro"], row["mae"], row["rmse"],
        row["vocabulario"], row["tempo_s"],
    )
    return row, y_pred


def fast_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Mesmas métricas de ``compute_metrics``, vetorizadas (para o bootstrap).

    Usa a matriz de confusão via ``bincount`` em vez de ``f1_score``, que é
    lento demais para ser chamado milhares de vezes. Os testes conferem que o
    resultado é idêntico ao de ``compute_metrics``.
    """
    cm = np.bincount(5 * y_true + y_pred, minlength=25).reshape(5, 5)
    tp = np.diag(cm).astype(float)
    denom = 2 * tp + (cm.sum(axis=0) - tp) + (cm.sum(axis=1) - tp)
    f1 = np.divide(2 * tp, denom, out=np.zeros(5), where=denom > 0)
    # Como o sklearn: a média macro só considera classes presentes em y_true ou y_pred.
    presentes = (cm.sum(axis=0) + cm.sum(axis=1)) > 0
    f1 = f1[presentes]
    err = (y_pred - y_true).astype(float)  # a escala 1..5 só desloca; o erro é igual
    return {
        "accuracy": float(tp.sum() / len(y_true)),
        "f1_macro": float(f1.mean()),
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
    }


def paired_bootstrap(
    y_true: np.ndarray,
    pred_ref: np.ndarray,
    preds_alt: dict[str, np.ndarray],
    *,
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict[str, dict[str, dict[str, float]]]:
    """IC 95% da diferença (alternativa − referência) por bootstrap pareado.

    Cada reamostragem de linhas do teste é aplicada à referência e a todas as
    alternativas, o que cancela a variação entre documentos e isola a diferença
    entre os modelos. ``significativo`` = o IC 95% não contém zero.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = {v: {m: [] for m in METRICS} for v in preds_alt}
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        base = fast_metrics(yt, pred_ref[idx])
        for v, pred in preds_alt.items():
            alt = fast_metrics(yt, pred[idx])
            for m in METRICS:
                diffs[v][m].append(alt[m] - base[m])

    out: dict[str, dict[str, dict[str, float]]] = {}
    for v, por_metrica in diffs.items():
        out[v] = {}
        for m, vals in por_metrica.items():
            lo, hi = np.percentile(vals, [2.5, 97.5])
            out[v][m] = {
                "ic95_low": float(lo),
                "ic95_high": float(hi),
                "significativo": bool(lo > 0 or hi < 0),
            }
    return out


def build_comparison(rows: list[dict[str, Any]], boot: dict[str, dict]) -> pd.DataFrame:
    """Tabela única: métricas, delta vs. referência e significância."""
    df = pd.DataFrame(rows).set_index("variante")
    ref = df.loc[REFERENCE]
    for m in METRICS:
        df[f"delta_{m}"] = df[m] - ref[m]
    for m in ("accuracy", "mae"):
        df[f"sig_{m}"] = [
            "ref" if v == REFERENCE else ("sim" if boot[v][m]["significativo"] else "não")
            for v in df.index
        ]
    return df.reset_index()


def _ptbr(texto: str) -> str:
    """1,234.5 -> 1.234,5 (padrão numérico das tabelas do README)."""
    return texto.translate(str.maketrans({",": ".", ".": ","}))


def to_markdown(df: pd.DataFrame) -> str:
    """Tabela em markdown (números em pt-BR) para colar no documento de diferenças."""
    cab = ["Variante", "Acurácia", "F1 macro", "MAE", "RMSE", "Vocab.",
           "Δ acurácia", "Δ MAE", "Sig. acc", "Sig. MAE"]
    linhas = ["| " + " | ".join(cab) + " |",
              "| " + " | ".join(["---"] + ["---:"] * 7 + [":---:"] * 2) + " |"]
    for _, r in df.iterrows():
        n = lambda x, f: _ptbr(format(x, f))  # noqa: E731
        linhas.append(
            f"| `{r.variante}` | {n(r.accuracy, '.4f')} | {n(r.f1_macro, '.4f')} | "
            f"{n(r.mae, '.4f')} | {n(r.rmse, '.4f')} | {n(int(r.vocabulario), ',')} | "
            f"{n(r.delta_accuracy, '+.4f')} | {n(r.delta_mae, '+.4f')} | "
            f"{r.sig_accuracy} | {r.sig_mae} |"
        )
    return "\n".join(linhas)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Baseline com e sem normalização.")
    p.add_argument("--train", default="data/raw/train.csv")
    p.add_argument("--test", default="data/raw/test.csv")
    p.add_argument("--out-dir", default="results/metrics")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--max-iter", type=int, default=1000)
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--sample-size", type=int, default=0,
                   help="Amostra estratificada do treino para execução rápida (0 = base completa)")
    p.add_argument("--protocol", choices=("test", "validation"), default="test",
                   help="test: treino completo -> teste oficial. validation: 80%% do treino -> "
                        "20%% de validação (mesmo split da Issue 7; não toca no teste)")
    p.add_argument("--validation-size", type=float, default=0.2)
    p.add_argument("--prefix", default=None,
                   help="Prefixo dos arquivos de saída (padrão: normalization_comparison)")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.protocol == "validation":
        train, test = split_train_validation(
            load_raw(args.train), validation_size=args.validation_size, random_state=args.seed
        )
        prefix = args.prefix or "normalization_comparison_validation"
    else:
        train = common_rows(load_raw(args.train))
        test = common_rows(load_raw(args.test))
        prefix = args.prefix or "normalization_comparison"
    if args.sample_size and args.sample_size > 0:
        train = stratified_sample(train, args.sample_size, args.seed)
        test = stratified_sample(test, max(args.sample_size // 4, 100), args.seed)
        prefix = args.prefix or f"{prefix}_sample"

    y_train = encode_rating(train[RATING_COL]).to_numpy()
    y_test = encode_rating(test[RATING_COL]).to_numpy()
    logger.info("Treino: %d | Teste: %d (mesmas linhas em todas as variantes)", len(train), len(test))

    rows, preds = [], {}
    for name, kwargs in VARIANTS.items():
        row, preds[name] = run_variant(
            name, kwargs, train, test, y_train, y_test,
            seed=args.seed, C=args.C, max_iter=args.max_iter,
        )
        rows.append(row)

    logger.info("Bootstrap pareado (%d reamostragens)...", args.n_bootstrap)
    boot = paired_bootstrap(
        y_test, preds[REFERENCE],
        {n: p_ for n, p_ in preds.items() if n != REFERENCE},
        n_resamples=args.n_bootstrap, seed=args.seed,
    )

    table = build_comparison(rows, boot)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{prefix}.csv"
    json_path = out_dir / f"{prefix}.json"
    md_path = out_dir / f"{prefix}.md"

    table.to_csv(csv_path, index=False, float_format="%.6f")
    md_path.write_text(to_markdown(table) + "\n", encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {
                "seed": args.seed,
                "lr_params": {"C": args.C, "max_iter": args.max_iter, "solver": "lbfgs"},
                "vectorizer_params": {k: list(v) if isinstance(v, tuple) else v
                                      for k, v in DEFAULT_PARAMS.items()},
                "variants": VARIANTS,
                "reference": REFERENCE,
                "protocol": args.protocol,
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "n_bootstrap": args.n_bootstrap,
                "sklearn_version": sklearn.__version__,
                "bootstrap_vs_reference": boot,
            },
            indent=2, ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )
    logger.info("Tabela → %s\nMarkdown → %s\nDetalhes → %s", csv_path, md_path, json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
