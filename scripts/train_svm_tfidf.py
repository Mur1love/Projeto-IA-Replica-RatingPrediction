"""SVM sobre TF-IDF (comparação complementar da RQ1) contra a LR do baseline.

Exemplos (a partir da raiz do repositório)::

    python scripts/train_svm_tfidf.py --sample-n 2000   # modo rápido
    python scripts/train_svm_tfidf.py                   # base completa

Grava ``results/metrics/svm_tfidf.json`` (base completa) ou
``results/sample/metrics/svm_tfidf.json`` (modo rápido). A comparação com
``baseline_logreg.json`` só é registrada na base completa.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.article_protocol import DEFAULT_SEED  # noqa: E402
from src.models.svm import TFIDF_SVM_PARAMS, compare_with_baseline, run_svm_tfidf  # noqa: E402

logger = logging.getLogger("train_svm_tfidf")
RAIZ = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="SVM + TF-IDF vs. LR + TF-IDF (baseline).")
    p.add_argument("--train", type=Path, default=RAIZ / "data" / "raw" / "train.csv")
    p.add_argument("--test", type=Path, default=RAIZ / "data" / "raw" / "test.csv")
    p.add_argument("--processed-dir", type=Path, default=RAIZ / "data" / "processed",
                   help="onde estão (ou serão geradas) as matrizes TF-IDF da Issue 6")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="pasta de saída; padrão: results/ ou results/sample/")
    p.add_argument("--baseline-json", type=Path, default=RAIZ / "results" / "metrics" / "baseline_logreg.json")
    p.add_argument("--sample-n", type=int, default=None, help="execução rápida com ~N linhas de treino")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--C", type=float, default=TFIDF_SVM_PARAMS["C"], help="regularização do SVC")
    p.add_argument("--kernel", default=TFIDF_SVM_PARAMS["kernel"], choices=["linear", "rbf"])
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    for nome in ("train", "test"):
        if not getattr(args, nome).is_file():
            p.error(f"arquivo não encontrado: {getattr(args, nome)} (veja data/README.md)")

    # O modo rápido amostra as matrizes completas de data/processed/ (não grava nada lá).
    out_dir = args.output_dir or (RAIZ / "results" / ("sample" if args.sample_n else ""))

    try:
        payload = run_svm_tfidf(
            args.train, args.test, args.processed_dir, sample_n=args.sample_n, seed=args.seed,
            params={"C": args.C, "kernel": args.kernel},
        )
    except ValueError as exc:
        p.error(str(exc))
    payload["comparison_with_baseline"] = compare_with_baseline(payload, args.baseline_json)

    out = out_dir / "metrics" / "svm_tfidf.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    logger.info("SVM + TF-IDF (n_treino=%d, n_teste=%d): %s", payload["n_train"], payload["n_test"], payload["metrics"])
    if payload["comparison_with_baseline"]:
        logger.info("SVM − LR: %s", payload["comparison_with_baseline"]["delta_svm_minus_logreg"])
    logger.info("-> %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
