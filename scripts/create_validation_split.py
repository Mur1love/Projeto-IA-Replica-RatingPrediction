"""Gera localmente o CSV bruto de validação (Issue 7)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data.splitting import split_train_validation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Gera um split estratificado local de validação a partir do train.csv."
    )
    parser.add_argument("--train", default="data/raw/train.csv")
    parser.add_argument("--out", default="data/raw/validation.csv")
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    source_path = Path(args.train)
    raw = pd.read_csv(source_path)
    train, validation = split_train_validation(
        raw, validation_size=args.validation_size, random_state=args.seed
    )

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    validation.to_csv(output_path, index=False)

    n_excluded = len(raw) - len(train) - len(validation)
    print(
        f"Split salvo em {output_path}: treino={len(train)}, "
        f"validação={len(validation)}, excluídas={n_excluded}, seed={args.seed}"
    )
    print("Distribuição rating na validação:")
    print(validation["rating"].value_counts().sort_index().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
