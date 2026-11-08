"""Gradient Boosting sobre a matriz TF-IDF (comparação complementar da RQ1).

Treina o GBT no TF-IDF de ``train.csv`` (busca curta de hiperparâmetros, só no
treino; ``--no-search`` usa parâmetros fixos) e avalia em ``test.csv``, o mesmo
split oficial do baseline. Na base completa, registra a
diferença para a Regressão Logística + TF-IDF (``results/metrics/baseline_logreg.json``).

Exemplos (a partir da raiz do repositório)::

    python scripts/train_gradient_boosting_tfidf.py --sample-n 2000   # modo rápido
    python scripts/train_gradient_boosting_tfidf.py                   # base completa

Sem ``--output-dir``, a base completa grava em ``results/`` e o modo rápido em
``results/sample/``, para nunca sobrescrever os resultados completos. O modo rápido
amostra as matrizes TF-IDF completas de ``--processed-dir`` (não grava nada à parte).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.article_protocol import DEFAULT_SEED  # noqa: E402
from src.models.gradient_boosting import (  # noqa: E402
    MIN_TFIDF_SAMPLE,
    TFIDF_N_ITER,
    run_gradient_boosting_tfidf,
)

logger = logging.getLogger("train_gradient_boosting_tfidf")

RAIZ = Path(__file__).resolve().parents[1]
OUTPUT_NAME = "gradient_boosting_tfidf.json"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Gradient Boosting sobre TF-IDF vs. Regressão Logística.")
    p.add_argument("--train", type=Path, default=RAIZ / "data" / "raw" / "train.csv")
    p.add_argument("--test", type=Path, default=RAIZ / "data" / "raw" / "test.csv")
    p.add_argument("--processed-dir", type=Path, default=RAIZ / "data" / "processed",
                   help="onde estão (ou serão gerados) os artefatos TF-IDF")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="pasta de saída; padrão: results/ ou results/sample/")
    p.add_argument("--baseline-json", type=Path,
                   default=RAIZ / "results" / "metrics" / "baseline_logreg.json",
                   help="métricas da Regressão Logística para a comparação")
    p.add_argument("--sample-n", type=int, default=None,
                   help=f"execução rápida com ~N linhas de treino (e N/4 de teste); mínimo {MIN_TFIDF_SAMPLE}")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--n-iter", type=int, default=TFIDF_N_ITER,
                   help=f"candidatos da busca curta (padrão: {TFIDF_N_ITER})")
    p.add_argument("--no-search", action="store_true",
                   help="não busca: usa parâmetros fixos (100 árvores, profundidade 3)")
    p.add_argument("--n-jobs", type=int, default=-1,
                   help="processos paralelos da busca (padrão: -1 = todos)")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    for nome in ("train", "test"):
        if not getattr(args, nome).is_file():
            parser.error(f"arquivo não encontrado: {getattr(args, nome)} (veja data/README.md)")
    if args.sample_n is not None and args.sample_n < MIN_TFIDF_SAMPLE:
        parser.error(f"--sample-n deve ser >= {MIN_TFIDF_SAMPLE}")

    out_dir = args.output_dir or (RAIZ / "results" / ("sample" if args.sample_n else ""))

    try:
        payload = run_gradient_boosting_tfidf(
            args.train, args.test, args.processed_dir,
            sample_n=args.sample_n, seed=args.seed, search=not args.no_search,
            n_iter=args.n_iter, n_jobs=args.n_jobs,
            baseline_json=args.baseline_json,
        )
    except ValueError as exc:
        parser.error(str(exc))

    destino = out_dir / "metrics" / OUTPUT_NAME
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    m = payload["metrics"]
    logger.info("GBT + TF-IDF (%d treino, %d teste, %d colunas) em %.1fs | %d candidato(s): %s",
                payload["n_train"], payload["n_test"], payload["n_features"],
                payload["fit_seconds"], payload["n_candidates_evaluated"], payload["params"])
    logger.info("acc=%.4f  f1_macro=%.4f  MAE=%.4f  RMSE=%.4f  AUC=%.4f  AUC(prob)=%.4f",
                m["accuracy"], m["f1_macro"], m["mae"], m["rmse"], m["auc"], m["auc_scores"])
    comp = payload["comparison_with_baseline"]
    if comp:
        d = comp["delta_gbt_minus_logreg"]
        logger.info("Δ (GBT − LR): " + "  ".join(f"{k}={v:+.4f}" for k, v in d.items()))
    else:
        logger.info("Sem comparação com o baseline (modo rápido ou %s ausente).", args.baseline_json.name)
    logger.info("-> %s", destino)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
