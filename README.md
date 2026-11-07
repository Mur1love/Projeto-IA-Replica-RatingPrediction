# Replica de Rating Prediction com Features Textuais

Projeto da disciplina de Inteligência Artificial dedicado à réplica do artigo *Rating Prediction in Brazilian Portuguese Reviews: An Approach Based on Textual Features*. O trabalho investiga a previsão de avaliações de produtos da Amazon Brasil, escritas em português brasileiro, utilizando características textuais e modelos tradicionais de aprendizado de máquina.

## Equipe

| Integrante | Conta no GitHub | Responsabilidade inicial |
| --- | --- | --- |
| Murilo | [`@Mur1love`](https://github.com/Mur1love) | Liderança e organização do projeto |
| David Carvalho | [`@DaviidCarvallho`](https://github.com/DaviidCarvallho) | A definir |
| Eduardo Araujo | [`@Duuduaraujo`](https://github.com/Duuduaraujo) | A definir |
| Enriko Martins | [`@EnrikoMartins`](https://github.com/EnrikoMartins) | A definir |
| João Pazzin | [`@joaoppazzin1`](https://github.com/joaoppazzin1) | A definir |

## Descrição do projeto

O objetivo deste projeto é reproduzir a metodologia e os experimentos apresentados no artigo de referência. A tarefa consiste em prever a nota, de 1 a 5 estrelas, atribuída por usuários da Amazon a partir apenas do texto das avaliações.

A réplica investigará a extração de características textuais e a aplicação de modelos de classificação, com atenção especial aos seguintes pontos:

- comparação entre SVM, Random Forest, Logistic Regression, Gradient Boosting e XGBoost;
- avaliação de diferentes grupos de características textuais, como características léxicas, sintáticas, estruturais e de classe gramatical;
- estudo de ablação para analisar a contribuição de cada grupo de características;
- seleção de características por meio de Recursive Feature Elimination (RFE);
- comparação do desempenho entre diferentes categorias de produtos.

As métricas principais serão MAE, RMSE e AUC, conforme o protocolo descrito no artigo. As decisões metodológicas, resultados obtidos e diferenças em relação ao trabalho original serão documentados neste repositório ao longo do desenvolvimento.

## Artigo replicado

O artigo utilizado como base está disponível localmente na raiz deste repositório:

- [`RatingPredictionInBrazilianPortugueseReviews.pdf`](RatingPredictionInBrazilianPortugueseReviews.pdf)

## Estrutura do repositório

```text
.
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   └── task.md
│   └── pull_request_template.md
├── data/
│   ├── raw/
│   │   └── .gitkeep
│   ├── processed/
│   │   └── .gitkeep
│   └── README.md
├── notebooks/
│   └── .gitkeep
├── scripts/
│   └── check_environment.py
├── src/
│   ├── data/
│   │   └── __init__.py
│   ├── preprocessing/
│   │   └── __init__.py
│   ├── models/
│   │   └── __init__.py
│   ├── evaluation/
│   │   └── __init__.py
│   ├── utils/
│   │   └── __init__.py
│   └── __init__.py
├── tests/
│   ├── .gitkeep
│   ├── __init__.py
│   └── test_environment.py
├── results/
│   ├── figures/
│   │   └── .gitkeep
│   └── metrics/
│       └── .gitkeep
├── RatingPredictionInBrazilianPortugueseReviews.pdf
├── .gitignore
├── README.md
├── requirements.txt
└── run_baseline.py
```

### Finalidade dos diretórios

- `data/raw/`: datasets originais, sem alterações. Os arquivos de dados não devem ser versionados por padrão.
- `data/processed/`: dados gerados por limpeza, transformação ou preparação.
- `notebooks/`: análises exploratórias, experimentos e registros de investigação em Jupyter.
- `scripts/`: utilitários executados diretamente, como a verificação do ambiente (`scripts/check_environment.py`).
- `src/data/`: código reutilizável de carregamento e manipulação inicial dos dados.
- `src/preprocessing/`: código reutilizável de pré-processamento e extração de características.
- `src/models/`: implementação e treinamento dos modelos.
- `src/evaluation/`: avaliação dos modelos e cálculo das métricas.
- `src/utils/`: funções auxiliares compartilhadas pelo projeto.
- `tests/`: testes automatizados do código produzido pela equipe.
- `results/figures/`: figuras e gráficos gerados pelos experimentos.
- `results/metrics/`: resultados das métricas gerados pelos experimentos.
- `.github/`: modelos para Issues e Pull Requests.

Os notebooks devem apoiar a exploração e a comunicação dos experimentos. O código reutilizável deve ser mantido em `src/` para evitar concentrar toda a implementação nos notebooks.

## Configuração do ambiente

O projeto requer **Python 3.12 ou mais recente** — piso exigido pelas versões travadas em `requirements.txt` (`numpy` e `scipy` exigem 3.12+). A instalação foi validada em **Linux com Python 3.14**; os comandos para Windows estão documentados abaixo, mas ainda não foram validados em uma máquina Windows.

As dependências têm as versões fixadas em `requirements.txt` para garantir um ambiente reproduzível entre os integrantes.

1. Clone o repositório e acesse sua pasta:

   ```bash
   git clone <URL_DO_REPOSITORIO>
   cd <NOME_DO_REPOSITORIO>
   ```

2. Crie um ambiente virtual:

   ```bash
   python -m venv .venv
   ```

3. Ative o ambiente virtual:

   No Linux ou macOS:

   ```bash
   source .venv/bin/activate
   ```

   No Windows (PowerShell):

   ```powershell
   .venv\Scripts\Activate.ps1
   ```

   Se o PowerShell bloquear a execução do script de ativação (política padrão em algumas máquinas), execute `Set-ExecutionPolicy -Scope Process RemoteSigned` antes de ativar, ou use o prompt do cmd: `.venv\Scripts\activate.bat`.

4. Instale as dependências:

   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

5. Verifique a instalação:

   ```bash
   python scripts/check_environment.py
   ```

   O comando importa todas as dependências principais e imprime as versões instaladas. Se algo estiver faltando, ele termina com código de saída diferente de zero.

### Decisão sobre dependências de NLP

A normalização de textos (Issue 4) utilizará o **nltk**, incluído em `requirements.txt`: ele é instalável apenas via pip e oferece *stopwords* e o stemmer RSLP em português. Os dados linguísticos necessários são leves e baixados sob demanda com `nltk.download(...)` — na Issue 4 serão usados `stopwords`, `rslp` e `punkt_tab`. O **spacy** foi descartado por depender de um modelo de pipeline completo (`pt_core_news_sm`) baixado separadamente — mais pesado e mais um artefato a manter. Observação: os dados do nltk não são travados pelo `requirements.txt`; a versão utilizada deve ser registrada na documentação de decisões (Issue 11). Caso a lematização se mostre necessária, a decisão poderá ser revista na Issue 4.

## Organização dos dados

Os datasets originais devem ser armazenados em `data/raw/` e preservados sem modificações. Em `data/processed/` devem ser salvos somente os dados resultantes de limpeza, transformação ou preparação.

Datasets e artefatos gerados podem ser grandes ou conter informações que não devem ser publicadas. Por isso, o `.gitignore` impede o versionamento desses arquivos por padrão. A forma de obtenção dos dados e eventuais decisões de preparação devem ser documentadas em `data/README.md`.

## Executando o baseline

O script `run_baseline.py` (na raiz) reproduz o baseline **TF-IDF + Regressão Logística** de ponta a ponta, com um único comando, a partir de `data/raw/train.csv` e `data/raw/test.csv` (ver `data/README.md` para obtê-los). Com o ambiente virtual ativado, na raiz do repositório:

```bash
python run_baseline.py
```

O comando executa, em sequência:

1. **Normalização** do texto (`src/preprocessing/normalization.py`);
2. **Encoding** do rating, de 1..5 para 0..4 (`src/preprocessing/encoding.py`);
3. **Embedding** TF-IDF, com `fit` apenas no treino e `transform` no teste (`src/preprocessing/embedding.py`);
4. **Modelagem**: treino da Regressão Logística, avaliação no teste e comparação com o baseline trivial de classe majoritária (`src/models/baseline.py`, `src/evaluation/`).

Apenas as colunas `text` e `rating` são usadas; as 58 features dos CSVs não fazem parte do baseline.

### Execução rápida (para testes)

```bash
python run_baseline.py --sample-size 2000
```

Usa uma amostra estratificada por nota (2000 linhas de treino e 25% disso no teste) e termina em poucos segundos. As saídas levam o prefixo `baseline_sample` e ficam em `data/processed/sample/`, então **não sobrescrevem** os resultados da base completa. O valor mínimo aceito é 100.

### Argumentos

| Argumento | Padrão | Descrição |
| --- | --- | --- |
| `--train` | `data/raw/train.csv` | CSV de treino |
| `--test` | `data/raw/test.csv` | CSV de teste |
| `--processed-dir` | `data/processed` | Saída do TF-IDF, rótulos, vetorizador e modelo |
| `--metrics-dir` | `results/metrics` | Saída do JSON de métricas |
| `--figures-dir` | `results/figures` | Saída da matriz de confusão |
| `--seed` | `42` | Semente da amostragem e da Regressão Logística |
| `--sample-size` | *(base completa)* | Nº de linhas de treino na execução rápida |
| `--C` | `1.0` | Regularização da Regressão Logística |
| `--max-iter` | `1000` | Máximo de iterações da Regressão Logística |

Use `python run_baseline.py --help` para ver a lista completa.

### Saídas

Os nomes abaixo valem para a base completa; na execução rápida, `baseline` vira `baseline_sample`.

| Arquivo | Conteúdo |
| --- | --- |
| `results/metrics/baseline_metrics.json` | Configuração, tamanhos dos dados, métricas (acurácia, F1-macro, MAE, RMSE) da Regressão Logística e da classe majoritária, e tempo de cada etapa |
| `results/figures/baseline_confusion_matrix.png` | Matriz de confusão no teste (notas 1 a 5) |
| `data/processed/tfidf_train.npz`, `tfidf_test.npz` | Matrizes TF-IDF esparsas |
| `data/processed/tfidf_train_y.npy`, `tfidf_test_y.npy` | Rótulos (0..4) alinhados às matrizes |
| `data/processed/tfidf_vectorizer.joblib` | Vetorizador ajustado no treino |
| `data/processed/tfidf_stats.json` | Estatísticas do TF-IDF, parâmetros e versão do scikit-learn |
| `data/processed/baseline_lr.joblib` | Modelo de Regressão Logística treinado |
| `data/processed/baseline_test_pred.npy` | Previsões (labels 0..4) no teste |

MAE e RMSE são calculados na escala original de 1 a 5 estrelas. Todos esses arquivos são ignorados pelo Git (ver `.gitignore`).

### Reprodutibilidade

Com a mesma `--seed`, os mesmos CSVs e as mesmas versões das dependências (`requirements.txt`), duas execuções geram métricas idênticas. Trocar a semente altera a amostra da execução rápida e a inicialização da Regressão Logística.

## Protocolo do artigo: SVM com as 58 features (RQ1)

`scripts/train_article_model.py` replica o protocolo de avaliação do artigo (`code/train_model.py` do repositório original) para um classificador treinado com as **58 features textuais** dos CSVs (`str_*`, `pos_*`, `synt_*`, `lex_*`, `conc_*`, `twt_*`, `sbj_*`). `categoria`, `text` e `rating` nunca entram na matriz `X`; uma coluna inesperada (por exemplo, um índice ou `id`) interrompe a execução. Com o ambiente virtual ativado, na raiz do repositório:

```bash
python scripts/train_article_model.py --model svm --sample-n 2000   
python scripts/train_article_model.py --model svm                   
python scripts/train_svm_tfidf.py                                   
```

O protocolo, implementado em `src/models/article_protocol.py` e compartilhado pelos próximos modelos:

1. `train.csv` + `test.csv` concatenados, nessa ordem (**o teste participa da validação cruzada**, como no artigo); nenhuma linha é removida, nem as de `text` vazio;
2. 5 folds com `StratifiedKFold(shuffle=False)` estratificados por `categoria_rating`;
3. `RandomizedSearchCV` com `scoring="neg_root_mean_squared_error"` e `random_state=42` sobre esses folds;
4. refit do melhor estimador em cada fold e cálculo de MAE, RMSE (em estrelas), F1 macro, acurácia e AUC;
5. média, desvio-padrão e IC de 95% (t de Student) entre os folds;
6. complemento: treino em `train.csv` e avaliação em `test.csv` com os mesmos hiperparâmetros (holdout).

A AUC segue o artigo: classes 4 e 5 estrelas como positivas, calculada a partir das **predições binárias**. Como diagnóstico, `auc_scores` usa `decision_function`.

**Modo rápido**: `--sample-n 2000` usa uma amostra estratificada por `categoria_rating` que preserva a ordem das linhas. Levou cerca de 2 minutos em 1 CPU. As saídas vão para `results/sample/` (ignorado pelo Git), então **não sobrescrevem** os resultados completos.

**Execução completa**: é cara. O `SVC` sem padronização dos dados (como no artigo) cresce de forma superlinear com o número de linhas: ~8,5 s para treinar com 3.200 linhas em kernel linear, num núcleo. Na execução de referência (`--n-jobs -1`), a busca levou ≈ 7,6 h e a avaliação dos 5 folds e do holdout ≈ 16 min; o tempo depende do número de núcleos (o padrão, `-1`, usa todos).

| Argumento | Padrão | Descrição |
| --- | --- | --- |
| `--model` | *(obrigatório)* | Modelo a avaliar (`svm`) |
| `--n-iter` | `300` | Candidatos do `RandomizedSearchCV` |
| `--n-splits` | `5` | Folds estratificados |
| `--seed` | `42` | Semente da busca e da amostragem |
| `--sample-n` | *(base completa)* | Nº de linhas da execução rápida |
| `--output-dir` | `results/` (`results/sample/` com `--sample-n`) | Pasta de saída (`metrics/` e `figures/`) |
| `--shuffle` | desligado | Embaralha os folds (o artigo não embaralha; fica registrado no JSON) |
| `--n-jobs` | `-1` | Processos paralelos |
| `--train`, `--test` | `data/raw/*.csv` | CSVs de entrada |

**Atenção ao `n_iter`**: o grid do SVM tem só 2 × 2 × 4 = 16 combinações, então `n_iter=300` equivale a uma busca exaustiva de **16 candidatos**, também no artigo. O JSON registra `n_iter_requested` e `n_candidates_evaluated`.

Saídas (`<m>` = nome do modelo, `svm`):

| Arquivo | Conteúdo |
| --- | --- |
| `results/metrics/<m>_rq1.json` | Configuração, melhores parâmetros, média, desvio-padrão e IC 95% de cada métrica, e comparação com a Tabela 3 do artigo |
| `results/metrics/<m>_rq1_folds.csv` | Métricas de cada fold |
| `results/metrics/<m>_rq1_holdout.json` | Treino em `train.csv`, avaliação em `test.csv` |
| `results/figures/confusion_matrix_<m>.png` | Matriz de confusão agregada (predições fora-da-dobra dos 5 folds) |
| `results/metrics/svm_tfidf.json` | SVM sobre a matriz TF-IDF (Issue 6) e diferença para a Regressão Logística do baseline |

`scripts/train_svm_tfidf.py` treina o SVM (kernel linear, `C=1.0`, sem busca) sobre o TF-IDF no split oficial treino → teste e registra a diferença contra `baseline_logreg.json`. A comparação só é gravada na base completa. As decisões e diferenças em relação ao artigo estão em [`docs/diferencas.md`](docs/diferencas.md), seção 3.

**Adicionar outro modelo (TAREFAS 12 a 14)**: crie um `ModelSpec` (como `SVM_SPEC` em `src/models/svm.py`) com o estimador, o grid e a linha da Tabela 3, e registre-o em `MODELS` no script.

## Protocolo do artigo: XGBoost com as 58 features (RQ1)

O XGBoost foi o melhor modelo do artigo nas três métricas da Tabela 3 (MAE, RMSE e AUC) e o escolhido para as RQs seguintes. O mesmo protocolo da seção anterior se aplica; muda o estimador (`XGBClassifier` de `src/models/xgboost.py`) e o grid, extraído da Tarefa 14:

```bash
python scripts/train_article_model.py --model xgboost --sample-n 2000   # modo rápido
python scripts/train_article_model.py --model xgboost                   # protocolo completo
python scripts/train_xgboost_tfidf.py                                   # comparação com TF-IDF
```

| Argumento | Padrão | Descrição |
| --- | --- | --- |
| `--model` | *(obrigatório)* | `xgboost` |
| `--n-iter` | `300` | Candidatos do `RandomizedSearchCV` (no modo rápido use um valor menor, ex.: `--n-iter 20`) |

Os demais argumentos (`--n-splits`, `--seed`, `--sample-n`, `--output-dir`, `--shuffle`, `--n-jobs`, `--train`, `--test`) são iguais aos da seção SVM.

**Grid**: 9 dimensões — `n_estimators` (4 valores) × `learning_rate`, `max_depth`, `min_child_weight`, `gamma`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda` (3 valores cada) = **26.244 combinações**, das quais `n_iter=300` sorteia 300 candidatos por fold. Cada fit de árvore `hist` é mais barato que um `SVC`, mas o corte total é maior que o do SVM (∼3 h estimadas de busca). **A entrega da TAREFA 14 usou o modo rápido** (limitação de hardware do notebook responsável); números, limitações e comando da execução completa em [`docs/diferencas.md`](docs/diferencas.md), seção 4.

Saídas (`<m>` = `xgboost`):

| Arquivo | Conteúdo |
| --- | --- |
| `results/metrics/<m>_rq1.json` | Configuração, melhores parâmetros, média, desvio-padrão e IC 95% de cada métrica, e comparação com a Tabela 3 |
| `results/metrics/<m>_rq1_folds.csv` | Métricas de cada fold |
| `results/metrics/<m>_rq1_holdout.json` | Treino em `train.csv`, avaliação em `test.csv` |
| `results/figures/confusion_matrix_<m>.png` | Matriz de confusão agregada (predições fora-da-dobra dos 5 folds) |
| `results/metrics/<m>_tfidf.json` | XGBoost sobre a matriz TF-IDF (Issue 6) e diferença para a Regressão Logística do baseline |

`scripts/train_xgboost_tfidf.py` reutiliza o TF-IDF da Issue 6 (`ensure_tfidf`), faz uma busca curta (`n_iter=8`, 3 folds, RMSE) só no treino e avalia no teste oficial; a comparação com `baseline_logreg.json` só se grava na base completa. As decisões e diferenças em relação ao artigo estão em [`docs/diferencas.md`](docs/diferencas.md), seção 4.

## Testes automatizados

Os testes ficam em `tests/` e usam [pytest](https://docs.pytest.org/). Com o ambiente virtual ativado, na raiz do repositório:

```bash
python -m pytest
```

Para ver também a cobertura de `src/` (as linhas não cobertas aparecem na coluna `Missing`):

```bash
python -m pytest --cov --cov-fail-under=80
```

A configuração está em `pytest.ini` (pasta de testes) e `.coveragerc` (cobertura medida apenas sobre `src/`).

Cobertura por etapa do pipeline:

| Etapa | Arquivos de teste |
|---|---|
| Normalização | `test_normalization.py`, `test_normalizacao_casos_reais.py` |
| Label encoding | `test_encoding.py`, `test_encoding_dados_reais.py` |
| Embedding TF-IDF | `test_embedding.py`, `test_tfidf_reuso.py` |
| Métricas e baseline | `test_metrics.py`, `test_metricas_valores_conhecidos.py`, `test_baseline.py` |
| Reprodutibilidade (seed fixa) | `test_reprodutibilidade.py`, `test_splitting.py` |
| Protocolo do artigo e SVM (RQ1) | `test_article_protocol.py`, `test_svm.py`, `test_train_article_model.py`, `test_article_protocol_dados_reais.py` |

Observações:

- Os testes `*_casos_reais`, `*_dados_reais`, `test_tfidf_reuso.py` e parte de `test_reprodutibilidade.py` leem uma amostra de `data/raw/train.csv`. Se o arquivo não existir na máquina, esses testes são **pulados** (`SKIPPED`), e os demais continuam rodando. Veja `data/README.md` para obter os CSVs.
- `test_train_baseline.py` executa o script do baseline e exige `data/raw/train.csv` e `data/raw/test.csv`.
- `test_article_protocol_dados_reais.py` é **pulado** sem `data/raw/train.csv` e `test.csv`; os demais testes do protocolo usam CSVs sintéticos e rodam em qualquer máquina.
- Antes de abrir um Pull Request, rode `python -m pytest`: todos os testes devem passar.

## Fluxo de desenvolvimento

O fluxo esperado é:

```text
Issue → Branch → Commits → Pull Request → Code Review → Approval → Merge
```

Regras do projeto:

1. Toda alteração de código deve estar associada a uma Issue.
2. Cada atividade deve ser desenvolvida em uma branch própria.
3. Não devem ser realizados commits diretamente na `main`.
4. Toda alteração deve entrar na `main` por meio de Pull Request.
5. O Pull Request deve estar associado à Issue correspondente. Quando o merge concluir a atividade, use `Closes #numero_da_issue` na descrição do PR.
6. O autor do Pull Request não pode aprovar o próprio PR.
7. Todo PR precisa de pelo menos uma aprovação de outro integrante da equipe antes do merge.
8. Cada integrante deve utilizar sua própria conta GitHub e realizar seus próprios commits.
9. O histórico do GitHub será utilizado para acompanhar a participação individual dos integrantes.

### Exemplo de início de uma atividade

Depois de criar ou escolher uma Issue, atualize a `main` local e crie uma branch com um nome descritivo:

```bash
git switch main
git pull
git switch -c issue-12-descricao-curta
```

Faça commits pequenos e claros. Ao concluir a atividade, envie a branch e abra um Pull Request utilizando o modelo do repositório.

## Referências

### Artigo principal

Marreira, E.; Oliveira, M.; Melo, T. *Rating Prediction in Brazilian Portuguese Reviews: An Approach Based on Textual Features*. Artigo disponibilizado neste repositório em [`RatingPredictionInBrazilianPortugueseReviews.pdf`](RatingPredictionInBrazilianPortugueseReviews.pdf).

### Repositório original

Marreira, E. *Rating Prediction with Textual Features*. Repositório utilizado como referência para a implementação e os experimentos: [github.com/emanuellemarreira/rating-prediction-with-textual-features](https://github.com/emanuellemarreira/rating-prediction-with-textual-features).
