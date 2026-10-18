# Métricas do baseline (Issue 7)

Comando: `.venv/bin/python scripts/train_baseline.py --seed 42 --C 1.0 --max-iter 1000`
Seed fixa: 42. TF-IDF `fit` só no treino; teste só `transform`.
Vetorizador: `(1,2)`, `min_df=5`, `max_df=0.95`, `sublinear_tf=True`.
Artefatos: `baseline_logreg.json` (métricas LR + majoritária) e
`../figures/confusion_matrix_logreg.png`.
Treino 41171 / teste 10292 (1 vazio removido por split).

## Experimento de validação 80/20

Comando: `python scripts/train_baseline_validation.py --seed 42 --validation-size 0.2`.
O treino de validação usa 32.936 registros para ajustar o TF-IDF e a LR e
avalia em 8.235 registros separados do `train.csv`; não lê nem usa `test.csv`.
O CSV bruto de validação é recriável em `data/raw/validation.csv`, mas não é
versionado por causa da licença não especificada dos reviews.

Artefatos versionados: `baseline_logreg_validation.json`,
`baseline_logreg_comparison.csv` e
`../figures/confusion_matrix_logreg_validation.png`. A comparação coloca lado a
lado o baseline treinado em 100% do treino e avaliado no teste oficial e o
modelo treinado em 80% do treino e avaliado na validação; portanto, os tamanhos
de treino e os conjuntos avaliados diferem.

| Métrica | Treino completo → teste oficial | Treino 80% → validação | Validação − teste |
| --- | ---: | ---: | ---: |
| Acurácia | 0,4898 | 0,4860 | −0,0038 |
| F1 macro | 0,4862 | 0,4827 | −0,0035 |
| MAE (estrelas) | 0,6819 | 0,7004 | +0,0185 |
| RMSE (estrelas) | 1,0619 | 1,0879 | +0,0260 |

As métricas são próximas. O modelo de validação foi treinado com menos dados,
então a pequena diferença não é uma comparação perfeitamente equivalente.
