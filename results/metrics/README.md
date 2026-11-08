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

Artefatos versionados: `normalization_comparison_validation.json` e `normalization_comparison.json`
(configuração, variantes e ICs do bootstrap). O script também gera tabelas `.csv` e `.md` em
`results/metrics/`, mas esses arquivos são locais e não versionados; as tabelas completas
estão reproduzidas em [`docs/diferencas.md`](../../docs/diferencas.md), seção 2.3.
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

## Random Forest com as 58 features do artigo (RQ1) e comparação com TF-IDF

Comando do protocolo: `python scripts/train_article_model.py --model random_forest` (base completa,
5 folds, grid de 19.440 combinações, `n_iter=300`). **A execução completa não foi realizada nesta
etapa.** O artefato versionado `random_forest_rq1.json` registra `sample_n=5000`, `n_iter_requested=1`,
`n_candidates_evaluated=1` e um `param_grid` com um único valor por hiperparâmetro, gerado com
scikit-learn 1.8.0; portanto, é uma execução de amostra/smoke test, **não o protocolo do artigo**.
O CSV `random_forest_rq1_folds.csv` confirma folds de 4.000/1.000 linhas sobre a amostra de 5.000.
Números, limitações e comandos em [`docs/diferencas.md`](../../docs/diferencas.md), seção 6.

Artefatos versionados: `random_forest_rq1.json`, `random_forest_rq1_folds.csv`,
`random_forest_rq1_holdout.json`, `random_forest_tfidf.json` (amostra de 5.000, `n_iter=1`) e
`../figures/confusion_matrix_random_forest.png` (OOF da amostra). A execução completa deve gerar os
mesmos nomes em `results/metrics/`, com `sample_n=null` e os 300 candidatos.

Leitura das métricas: `auc` é a AUC do artigo (classes 4–5 estrelas positivas, a partir de predições
binárias); `auc_scores` é um diagnóstico por scores. MAE e RMSE estão na escala de 1 a 5 estrelas.

## XGBoost com as 58 features do artigo (RQ1) e comparação com TF-IDF

Comando do protocolo: `python scripts/train_article_model.py --model xgboost` (base completa, 5 folds,
`n_iter=300` sobre o grid de 26.244 combinações da TAREFA 14). **Nesta etapa apenas o modo rápido foi
executado** (restrição de hardware: `--sample-n 2000 --n-iter 20`, 33 s; e TF-IDF com `--sample-n 2000`,
6:22) — os artefatos ficam em `results/sample/`, que **não é versionado**; por isso não há artefatos
XGBoost versionados em `results/metrics/`. Números, limitações e justificativa em
[`docs/diferencas.md`](../../docs/diferencas.md), seção 4; a comparação com a Tabela 3 do artigo e os
artefatos oficiais ficam versionados apenas quando a execução completa for concluída
(`results/metrics/xgboost_rq1.json`, `xgboost_rq1_folds.csv`, `xgboost_rq1_holdout.json`,
`xgboost_tfidf.json` e `../figures/confusion_matrix_xgboost.png`).

## Gradient Boosting com as 58 features do artigo (RQ1) e comparação com TF-IDF

Comando do protocolo: `python scripts/train_article_model.py --model gradient_boosting` (base completa,
5 folds, grid de 2.160 combinações da TAREFA 13, com `--n-iter` reduzido em relação aos 300 do artigo).
**Nesta etapa apenas o modo rápido foi executado** (`--sample-n 2000`, `n_iter=5`; e TF-IDF com
`--sample-n 2000`, 36 s) — os artefatos ficam em `results/sample/`, que **não é versionado**; por isso
não há artefatos do Gradient Boosting versionados em `results/metrics/`. Números, limitações e
justificativa em [`docs/diferencas.md`](../../docs/diferencas.md), seção 5; os artefatos oficiais ficam
versionados apenas quando a execução completa for concluída (`results/metrics/gradient_boosting_rq1.json`,
`gradient_boosting_rq1_folds.csv`, `gradient_boosting_rq1_holdout.json`,
`gradient_boosting_tfidf.json` e `../figures/confusion_matrix_gradient_boosting.png`).

Leitura das métricas: `auc` é a AUC do artigo (classes 4–5 estrelas positivas, a partir de predições
binárias); `auc_scores` é um diagnóstico a partir das **probabilidades** (`predict_proba`). MAE e RMSE
estão na escala de 1 a 5 estrelas.

## Consolidação da RQ1 (estado atual)

Apenas o **SVM** foi executado no protocolo completo do artigo. As demais linhas estão implementadas,
mas foram executadas em amostra ou modo rápido e **não são comparáveis à Tabela 3**.

| Modelo | Execução | MAE | RMSE | AUC | Status |
| --- | --- | ---: | ---: | ---: | --- |
| SVM | Completa (51.465 linhas) | 1,0430 ± 0,0062 | 1,5085 ± 0,0112 | 0,7303 ± 0,0047 | Comparável ao artigo |
| Random Forest | Amostra 5.000, 1 candidato | 1,0784 ± 0,0200 | 1,5165 ± 0,0296 | 0,7168 ± 0,0110 | Parcial |
| XGBoost | Modo rápido 2.000, 20 candidatos | 1,0215 ± 0,0327 | 1,4337 ± 0,0490 | 0,7204 ± 0,0306 | Parcial |
| Gradient Boosting | Modo rápido 2.000, 5 candidatos | 1,0400 ± 0,0549 | 1,4527 ± 0,0584 | 0,7072 ± 0,0133 | Parcial |
| Regressão Logística | LR + TF-IDF (não é a LR do artigo) | 0,6819 | 1,0619 | n/d | Adicional; 58 features pendente |

Referência do artigo (Tabela 3): GBT 0,9612 / 1,3727 / 0,7318; LR 1,0149 / 1,4691 / 0,7328;
RF 0,9717 / 1,4001 / 0,7367; XGB 0,9499 / 1,3721 / 0,7390; SVM 1,0431 / 1,5085 / 0,7304.
A consolidação completa, o holdout e a comparação TF-IDF × 58 features estão em
[`docs/diferencas.md`](../../docs/diferencas.md), seção 7, e em
[`docs/documentacao.md`](../../docs/documentacao.md), seção 10.
