"""Treina XGBoost sobre TF-IDF e compara com o baseline LR."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.xgboost import run_xgboost_tfidf

ROOT = Path(__file__).resolve().parents[1]

def main(argv=None):
    parser = argparse.ArgumentParser(description="XGBoost + TF-IDF (comparação complementar).")
    parser.add_argument("--train", type=Path, default=ROOT/"data/raw/train.csv")
    parser.add_argument("--test", type=Path, default=ROOT/"data/raw/test.csv")
    parser.add_argument("--processed-dir", type=Path, default=ROOT/"data/processed")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"results")
    parser.add_argument("--baseline-json", type=Path, default=ROOT/"results/metrics/baseline_logreg.json")
    parser.add_argument("--sample-n", type=int, default=None)
    parser.add_argument("--n-iter", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=1)
    args=parser.parse_args(argv)
    for f in (args.train,args.test):
        if not f.is_file(): parser.error(f"arquivo não encontrado: {f} (veja data/README.md)")
    result=run_xgboost_tfidf(args.train,args.test,args.processed_dir,
        sample_n=args.sample_n,seed=args.seed,n_iter=args.n_iter,n_jobs=args.n_jobs,
        baseline_json=args.baseline_json)
    out=args.output_dir/ "metrics"
    out.mkdir(parents=True,exist_ok=True)
    path=out/"xgboost_tfidf.json"
    path.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(f"MAE {result['metrics']['mae']:.4f} | RMSE {result['metrics']['rmse']:.4f} | AUC {result['metrics']['auc']:.4f}")
    print(f"Artefato: {path}")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
