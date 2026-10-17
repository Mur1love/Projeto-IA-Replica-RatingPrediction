# Métricas do baseline (Issue 7)

Comando: `.venv/bin/python scripts/train_baseline.py --seed 42 --C 1.0 --max-iter 1000`
Seed fixa: 42. TF-IDF `fit` só no treino; teste só `transform`.
Vetorizador: `(1,2)`, `min_df=5`, `max_df=0.95`, `sublinear_tf=True`.
Artefatos: `baseline_logreg.json` (métricas LR + majoritária) e
`../figures/confusion_matrix_logreg.png`.
Treino 41171 / teste 10292 (1 vazio removido por split).
