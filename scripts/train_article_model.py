"""Avalia um modelo com o protocolo do artigo (RQ1) sobre as 58 features.

Exemplos (a partir da raiz do repositório)::

    python scripts/train_article_model.py --model svm --sample-n 2000   # modo rápido
    python scripts/train_article_model.py --model svm                   # protocolo completo

Sem ``--output-dir``, a base completa grava em ``results/`` e o modo rápido
(``--sample-n``) em ``results/sample/``, para nunca sobrescrever os resultados
completos. Veja ``src/models/article_protocol.py`` para o protocolo.

Para acompanhar progresso em log (tempo e % da busca/avaliação), use
``--verbose 10`` e rode o monitor::

    python scripts/train_article_model.py --model logistic_regression \
        --verbose 10 > results/<log>.log 2>&1
    python scripts/monitor_regressao_logistica.py --log results/<log>.log
"""
from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.article_protocol import (  # noqa: E402
    DEFAULT_N_ITER,
    DEFAULT_N_SPLITS,
    DEFAULT_SEED,
    load_article_data,
    run_article_protocol,
    save_article_results,
)
from src.models.svm import SVM_SPEC  # noqa: E402
from src.models.random_forest import RANDOM_FOREST_SPEC  # noqa: E402
from src.models.xgboost import XGB_SPEC  # noqa: E402
from src.models.gradient_boosting import GRADIENT_BOOSTING_SPEC  # noqa: E402
from src.models.logistic_regression import LOGISTIC_REGRESSION_SPEC  # noqa: E402

logger = logging.getLogger("train_article_model")

RAIZ = Path(__file__).resolve().parents[1]

# TAREFAS 12 a 14 (e próximas): importe o ModelSpec do novo modelo e acrescente aqui.
MODELS = {
    spec.name: spec
    for spec in (SVM_SPEC, RANDOM_FOREST_SPEC, XGB_SPEC, GRADIENT_BOOSTING_SPEC,
                 LOGISTIC_REGRESSION_SPEC)
}

# Candidatos da busca no modo rápido (--sample-n). Modelos fora daqui usam o padrão (300):
# o grid do SVM só tem 16 combinações. RF e GBT têm grids grandes e treinos caros.
QUICK_N_ITER = {"random_forest": 10, "gradient_boosting": 5}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Protocolo do artigo (RQ1) com as 58 features.")
    p.add_argument("--model", required=True, choices=sorted(MODELS), help="modelo a avaliar")
    p.add_argument("--n-iter", type=int, default=None,
                   help=f"candidatos do RandomizedSearchCV (padrão: {DEFAULT_N_ITER})")
    p.add_argument("--n-splits", type=int, default=DEFAULT_N_SPLITS,
                   help=f"folds estratificados (padrão: {DEFAULT_N_SPLITS})")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help=f"semente da busca e da amostragem (padrão: {DEFAULT_SEED})")
    p.add_argument("--sample-n", type=int, default=None,
                   help="usa só ~N linhas (amostra estratificada) para uma execução rápida")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="pasta de saída (metrics/ e figures/); padrão: results/ ou results/sample/")
    p.add_argument("--train", type=Path, default=RAIZ / "data" / "raw" / "train.csv")
    p.add_argument("--test", type=Path, default=RAIZ / "data" / "raw" / "test.csv")
    p.add_argument("--shuffle", action="store_true",
                   help="embaralha os folds (o artigo NÃO embaralha; fica registrado na saída)")
    p.add_argument("--n-jobs", type=int, default=-1,
                   help="processos paralelos (padrão: -1 = todos, como no artigo)")
    p.add_argument("--verbose", type=int, default=0, metavar="N",
                   help="verbosidade da busca/avaliação (use 10 para acompanhar por log)")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    for nome in ("train", "test"):
        if not getattr(args, nome).is_file():
            parser.error(f"arquivo não encontrado: {getattr(args, nome)} (veja data/README.md)")
    if args.sample_n is not None and args.sample_n <= 0:
        parser.error("--sample-n deve ser positivo")

    spec = MODELS[args.model]
    n_iter = args.n_iter
    if n_iter is None:
        n_iter = QUICK_N_ITER.get(args.model, DEFAULT_N_ITER) if args.sample_n else DEFAULT_N_ITER
    out_dir = args.output_dir or (RAIZ / "results" / ("sample" if args.sample_n else ""))

    try:
        data = load_article_data(args.train, args.test, sample_n=args.sample_n, seed=args.seed)
        logger.info("Dados: %d linhas (%d de train.csv + %d de test.csv), %d features",
                    data.n_rows, data.n_train_rows, data.n_test_rows, data.n_features)
        with warnings.catch_warnings():
            # O grid do SVM tem 16 combinações < n_iter: o aviso do sklearn é esperado
            # e a contagem real fica registrada em n_candidates_evaluated.
            warnings.filterwarnings("ignore", message="The total space of parameters")
            result = run_article_protocol(
                spec, data, n_iter=n_iter, n_splits=args.n_splits, seed=args.seed,
                shuffle=args.shuffle, n_jobs=args.n_jobs, verbose=args.verbose,
            )
    except ValueError as exc:  # amostra pequena demais, colunas inesperadas etc.
        parser.error(str(exc))

    paths = save_article_results(result, data, out_dir)

    logger.info("Candidatos avaliados: %d (n_iter pedido: %d) | melhores parâmetros: %s",
                result.n_candidates, n_iter, result.best_params)
    for metric in ("mae", "rmse", "auc"):
        agg = result.aggregate[metric]
        ref = spec.article_reference.get(metric)
        ref_txt = f" | artigo: {ref[0]:.4f} ± {ref[1]:.4f}" if ref else ""
        logger.info("%-5s %.4f ± %.4f  IC95%% [%.4f, %.4f]%s", metric.upper(),
                    agg["mean"], agg["std"], *agg["ci95"], ref_txt)
    if result.holdout:
        h = result.holdout["metrics"]
        logger.info("Holdout (train.csv -> test.csv): MAE %.4f RMSE %.4f AUC %.4f",
                    h["mae"], h["rmse"], h["auc"])
    for nome, caminho in paths.items():
        logger.info("%s -> %s", nome, caminho)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())