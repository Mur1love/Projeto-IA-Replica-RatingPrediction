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


## Comparação com e sem normalização

Comando: `python scripts/compare_normalization.py --protocol validation` (decisão, não usa o teste)
e `python scripts/compare_normalization.py` (confirmação no teste oficial). Seed fixa: 42.
Sete variantes de pré-processamento, mesmas linhas e mesmo modelo em todas; bootstrap pareado
(1.000 reamostragens) para o IC 95% da diferença contra `sem_normalizacao`.

Artefatos versionados (para cada protocolo, `.csv` = tabela única, `.md` = a mesma tabela
formatada, `.json` = configuração e ICs do bootstrap):
`normalization_comparison_validation.*` e `normalization_comparison.*`.
A coluna `tempo_s` dos CSVs varia a cada execução; as demais são determinísticas no mesmo ambiente.

Resultado: nenhuma variante se distingue da referência em acurácia ou F1 (|Δ| ≤ 0,0026). Análise e
decisão em [`docs/diferencas.md`](../../docs/diferencas.md).


## SVM com as 58 features do artigo (RQ1) e comparação com TF-IDF

Comando: `python scripts/train_article_model.py --model svm` (protocolo completo; o modo
rápido `--sample-n 2000` grava em `results/sample/`, que não é versionado).
Protocolo: `train.csv` + `test.csv` concatenados, 5 folds estratificados por `categoria_rating`
sem embaralhar, `RandomizedSearchCV` (`n_iter=300`, que avalia 16 candidatos), RMSE como
critério, seed 42. Detalhes e diferenças em [`docs/diferencas.md`](../../docs/diferencas.md), seção 3.

Artefatos versionados: `svm_rq1.json` (média, desvio-padrão e IC 95% entre os folds, melhores
parâmetros e comparação com a Tabela 3 do artigo), `svm_rq1_folds.csv` (métricas por fold),
`svm_rq1_holdout.json` (treino em `train.csv`, avaliação em `test.csv`),
`../figures/confusion_matrix_svm.png` (predições fora-da-dobra dos 5 folds) e `svm_tfidf.json`
(`python scripts/train_svm_tfidf.py`: SVM sobre TF-IDF e diferença para `baseline_logreg.json`).

Leitura das métricas: `auc` é a AUC do artigo (classes 4–5 estrelas positivas, a partir de predições
binárias); `auc_scores` é um diagnóstico a partir de `decision_function`. MAE e RMSE estão na escala
de 1 a 5 estrelas. O IC de 95% usa `t` com 4 graus de liberdade e é mais largo que o publicado no
artigo (ver seção 3.3 de `docs/diferencas.md`).

## XGBoost com as 58 features do artigo (RQ1) e comparação com TF-IDF

Comando do protocolo: `python scripts/train_article_model.py --model xgboost` (base completa, 5 folds,
`n_iter=300` sobre o grid de 26.244 combinações da TAREFA 14). **Nesta etapa apenas o modo rápido foi
executado** (restrição de hardware: `--sample-n 2000 --n-iter 20`, 33 s; e TF-IDF com `--sample-n 2000`,
6:22) — os artefatos ficam em `results/sample/`, que **não é versionado**; por isso não há artefatos
XGBoost versionados em `results/metrics/`. Números, limitações e justificativa em
[`docs/diferencas.md`](../../docs/diferencas.md), seção 4; a comparação com a Tabela 3 do artigo e os
artefatos oficiais ficam versionados apenas quando a execução completa for concluída
(`results/metrics/xgboost_rq1{,_folds.csv,_holdout}.json`, `xgboost_tfidf.json` e
`../figures/confusion_matrix_xgboost.png`).

## Gradient Boosting com as 58 features do artigo (RQ1) e comparação com TF-IDF

Comando do protocolo: `python scripts/train_article_model.py --model gradient_boosting` (base completa,
5 folds, grid de 2.160 combinações da TAREFA 13, com `--n-iter` reduzido em relação aos 300 do artigo).
**Nesta etapa apenas o modo rápido foi executado** (`--sample-n 2000`, `n_iter=5`; e TF-IDF com
`--sample-n 2000`, 36 s) — os artefatos ficam em `results/sample/`, que **não é versionado**; por isso
não há artefatos do Gradient Boosting versionados em `results/metrics/`. Números, limitações e
justificativa em [`docs/diferencas.md`](../../docs/diferencas.md), seção 5; os artefatos oficiais ficam
versionados apenas quando a execução completa for concluída (`results/metrics/gradient_boosting_rq1{,_folds.csv,_holdout}.json`,
`gradient_boosting_tfidf.json` e `../figures/confusion_matrix_gradient_boosting.png`).

Leitura das métricas: `auc` é a AUC do artigo (classes 4–5 estrelas positivas, a partir de predições
binárias); `auc_scores` é um diagnóstico a partir das **probabilidades** (`predict_proba`). MAE e RMSE
estão na escala de 1 a 5 estrelas.
