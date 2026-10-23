"""Pipeline reproduzível fim a fim do baseline (TF-IDF + Regressão Logística).

Um único comando executa as quatro etapas sobre os CSVs de ``data/raw/``::

    1. normalização   (src.preprocessing.normalization)
    2. encoding       (src.preprocessing.encoding: rating 1..5 -> label 0..4)
    3. embedding      (src.preprocessing.embedding: TF-IDF, fit só no treino)
    4. modelagem      (src.models.baseline + src.evaluation: LR, métricas, figura)

As etapas 1 a 3 são executadas por ``build_embeddings``; a etapa 4 treina a
Regressão Logística, avalia no teste e compara com o baseline trivial
(prever sempre a classe majoritária).

Uso (a partir da raiz do repositório)::

    python run_baseline.py                          # base completa
    python run_baseline.py --sample-size 2000       # execução rápida (testes)
    python run_baseline.py --seed 7 --C 0.5

Execuções com ``--sample-size`` gravam em subpastas/arquivos com o prefixo
``baseline_sample`` e NÃO sobrescrevem os artefatos da base completa.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent
# Permite "from src..." mesmo se o script for chamado por caminho absoluto.
sys.path.insert(0, str(RAIZ))

from src.evaluation.metrics import compute_metrics, predict_majority  # noqa: E402
from src.evaluation.plots import save_confusion_matrix  # noqa: E402
from src.models.baseline import (  # noqa: E402
    predict,
    save_model,
    train_logistic_regression,
)
from src.preprocessing.embedding import (  # noqa: E402
    TEST_LABELS_FILE,
    TEST_MATRIX_FILE,
    TRAIN_LABELS_FILE,
    TRAIN_MATRIX_FILE,
    build_embeddings,
    load_matrix,
)

logger = logging.getLogger("run_baseline")

TEXT_COL = "text"
RATING_COL = "rating"
MIN_SAMPLE_SIZE = 100  # abaixo disso o TF-IDF (min_df=5) fica sem vocabulário útil
TEST_FRACTION = 0.25  # teste da amostra = 25% do tamanho do treino (split 80/20)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Executa o baseline TF-IDF + Regressão Logística de ponta a ponta."
    )
    p.add_argument("--train", type=Path, default=RAIZ / "data" / "raw" / "train.csv",
                   help="CSV de treino (padrão: data/raw/train.csv)")
    p.add_argument("--test", type=Path, default=RAIZ / "data" / "raw" / "test.csv",
                   help="CSV de teste (padrão: data/raw/test.csv)")
    p.add_argument("--processed-dir", type=Path, default=RAIZ / "data" / "processed",
                   help="Onde gravar TF-IDF, rótulos, vetorizador e modelo")
    p.add_argument("--metrics-dir", type=Path, default=RAIZ / "results" / "metrics",
                   help="Onde gravar o JSON de métricas")
    p.add_argument("--figures-dir", type=Path, default=RAIZ / "results" / "figures",
                   help="Onde gravar a matriz de confusão")
    p.add_argument("--seed", type=int, default=42,
                   help="Semente da amostragem e da Regressão Logística (padrão: 42)")
    p.add_argument("--sample-size", type=int, default=None,
                   help=f"Nº de linhas de treino para execução rápida (mín. {MIN_SAMPLE_SIZE}); "
                        "o teste usa 25%% desse valor. Omitido = base completa")
    p.add_argument("--C", type=float, default=1.0,
                   help="Regularização da Regressão Logística (padrão: 1.0)")
    p.add_argument("--max-iter", type=int, default=1000,
                   help="Máximo de iterações da Regressão Logística (padrão: 1000)")
    args = p.parse_args(argv)

    if args.sample_size is not None and args.sample_size < MIN_SAMPLE_SIZE:
        p.error(f"--sample-size deve ser >= {MIN_SAMPLE_SIZE}")
    for nome in ("train", "test"):
        caminho = getattr(args, nome)
        if not caminho.is_file():
            p.error(f"arquivo não encontrado: {caminho} (veja data/README.md para obter os dados)")
    return args


def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Amostra ~``n`` linhas mantendo a proporção de cada nota (reprodutível)."""
    if n >= len(df):
        return df
    return (
        df.groupby(RATING_COL, group_keys=False)
        .sample(frac=n / len(df), random_state=seed)
        .reset_index(drop=True)
    )


def prepare_inputs(args: argparse.Namespace, processed_dir: Path) -> tuple[Path, Path, dict]:
    """Devolve os CSVs a usar. Com --sample-size, grava amostras estratificadas."""
    if args.sample_size is None:
        return args.train, args.test, {"mode": "full"}

    treino = pd.read_csv(args.train, usecols=[TEXT_COL, RATING_COL])
    teste = pd.read_csv(args.test, usecols=[TEXT_COL, RATING_COL])
    n_teste = max(int(args.sample_size * TEST_FRACTION), 5)

    amostra_treino = stratified_sample(treino, args.sample_size, args.seed)
    amostra_teste = stratified_sample(teste, n_teste, args.seed)

    pasta = processed_dir / "input"
    pasta.mkdir(parents=True, exist_ok=True)
    caminho_treino = pasta / "train_sample.csv"
    caminho_teste = pasta / "test_sample.csv"
    amostra_treino.to_csv(caminho_treino, index=False)
    amostra_teste.to_csv(caminho_teste, index=False)
    info = {
        "mode": "sample",
        "sample_size_requested": args.sample_size,
        "n_train_sampled": len(amostra_treino),
        "n_test_sampled": len(amostra_teste),
    }
    return caminho_treino, caminho_teste, info


def run(args: argparse.Namespace) -> dict:
    sample = args.sample_size is not None
    processed_dir = args.processed_dir / "sample" if sample else args.processed_dir
    prefixo = "baseline_sample" if sample else "baseline"
    tempos: dict[str, float] = {}

    # --- Etapas 1 a 3: normalização, encoding e embedding (TF-IDF) ---------------
    t0 = time.perf_counter()
    train_csv, test_csv, info_entrada = prepare_inputs(args, processed_dir)
    tempos["amostragem"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    stats = build_embeddings(train_csv, test_csv, processed_dir)
    tempos["normalizacao_encoding_embedding"] = time.perf_counter() - t0

    x_treino = load_matrix(processed_dir / TRAIN_MATRIX_FILE)
    x_teste = load_matrix(processed_dir / TEST_MATRIX_FILE)
    y_treino = np.load(processed_dir / TRAIN_LABELS_FILE)
    y_teste = np.load(processed_dir / TEST_LABELS_FILE)

    # --- Etapa 4: modelagem e avaliação ------------------------------------------
    t0 = time.perf_counter()
    modelo = train_logistic_regression(
        x_treino, y_treino, C=args.C, max_iter=args.max_iter, random_state=args.seed
    )
    tempos["treino"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = predict(modelo, x_teste)
    metricas_modelo = compute_metrics(y_teste, y_pred)
    metricas_majoritaria = compute_metrics(
        y_teste, predict_majority(y_treino, len(y_teste))
    )
    tempos["avaliacao"] = time.perf_counter() - t0

    # --- Artefatos de saída (nomes padronizados) ---------------------------------
    args.metrics_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    modelo_path = save_model(modelo, processed_dir / f"{prefixo}_lr.joblib")
    figura_path = save_confusion_matrix(
        y_teste, y_pred, args.figures_dir / f"{prefixo}_confusion_matrix.png"
    )
    np.save(processed_dir / f"{prefixo}_test_pred.npy", y_pred)

    resultado = {
        "config": {
            "seed": args.seed,
            "C": args.C,
            "max_iter": args.max_iter,
            "train_csv": str(train_csv),
            "test_csv": str(test_csv),
            **info_entrada,
        },
        "data": {
            "n_train": int(x_treino.shape[0]),
            "n_test": int(x_teste.shape[0]),
            "vocabulary_size": stats["vocabulary_size"],
            "n_dropped_empty_train": stats["train"]["n_dropped_empty"],
            "n_dropped_empty_test": stats["test"]["n_dropped_empty"],
        },
        "metrics": {
            "logistic_regression": metricas_modelo,
            "majority_class": metricas_majoritaria,
        },
        "seconds": {k: round(v, 3) for k, v in tempos.items()},
        "artifacts": {
            "model": str(modelo_path),
            "confusion_matrix": str(figura_path),
            "tfidf_dir": str(processed_dir),
        },
    }
    metricas_path = args.metrics_dir / f"{prefixo}_metrics.json"
    metricas_path.write_text(
        json.dumps(resultado, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    resultado["artifacts"]["metrics"] = str(metricas_path)
    return resultado


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    inicio = time.perf_counter()

    resultado = run(args)

    m = resultado["metrics"]
    logger.info("")
    logger.info("=== Resultado no teste (n=%d) ===", resultado["data"]["n_test"])
    for nome, rotulo in (("logistic_regression", "Regressão Logística"),
                         ("majority_class", "Classe majoritária ")):
        v = m[nome]
        logger.info("%s | acc=%.4f  f1_macro=%.4f  MAE=%.4f  RMSE=%.4f",
                    rotulo, v["accuracy"], v["f1_macro"], v["mae"], v["rmse"])
    logger.info("")
    logger.info("Artefatos:")
    for nome, caminho in resultado["artifacts"].items():
        logger.info("  %-17s %s", nome, caminho)
    logger.info("Tempo total: %.1fs", time.perf_counter() - inicio)


if __name__ == "__main__":
    main()
