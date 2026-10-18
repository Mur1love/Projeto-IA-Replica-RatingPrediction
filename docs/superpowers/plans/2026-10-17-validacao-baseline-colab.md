# Split de Validação e Comparação no Colab — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Gerar localmente o split estratificado de validação (20% do treino), treinar e comparar o baseline de validação com o baseline oficial, e entregar um novo notebook Colab reproduzível.

**Architecture:** `src/data/splitting.py` será a fonte única da divisão train/validation e preservará as linhas/colunas originais no CSV local. Um CLI de validação usará apenas `train.csv`, ajustará TF-IDF somente nos 80% de treino, salvará métricas/figura/tabela de comparação, e uma nova cópia de notebook permitirá repetir o fluxo no Colab.

**Tech Stack:** Python >=3.12, pandas, numpy, scikit-learn 1.9.1, scipy, matplotlib, pytest, Jupyter/nbformat.

**Spec:** Issue 7 em `ISSUES.md`; implementação aprovada em `docs/superpowers/plans/2026-10-17-baseline-regressao-logistica.md`; decisão aprovada nesta conversa: split de 20%, CSV derivado local em `data/raw/`, código/notebook/resultados na branch, CSV fora do Git por licença não especificada.

## Global Constraints

- Usar `validation_size=0.20`, estratificar por `rating` e `random_state=42`.
- Dividir após remover o texto vazio conforme a normalização existente: dos 41.172 registros brutos, 41.171 são utilizáveis; 32.936 vão para treino e 8.235 para validação.
- `data/raw/validation.csv` guarda os registros originais, todas as colunas, sem normalização; não alterar `train.csv` nem `test.csv`.
- O CSV de validação é gerado localmente e continua ignorado pelo Git; versionar o gerador, instruções, notebook, métricas e figuras — não os reviews.
- O treinamento de validação usa apenas o `train.csv`; `fit` do TF-IDF ocorre só nos 32.936 registros de treino. O teste oficial não participa desse experimento.
- Comparar `baseline_logreg.json` (modelo treinado em 100% do train e avaliado no test oficial) com `baseline_logreg_validation.json` (modelo treinado em 80% do train e avaliado nos 20% reservados); documentar que os tamanhos de treino e conjuntos avaliados são diferentes.
- Manter `C=1.0`, `max_iter=1000`, `solver="lbfgs"`, TF-IDF `DEFAULT_PARAMS` e seed 42 nos dois experimentos.
- Artefatos versionados da validação: `results/metrics/baseline_logreg_validation.json`, `results/metrics/baseline_logreg_comparison.csv` e `results/figures/confusion_matrix_logreg_validation.png`.
- Adicionar ao `.gitignore` exceções somente para esses três artefatos; manter outros arquivos em `results/` e todos os dados em `data/raw/` ignorados.

## Review Focus

- Texto vazio/NaN: excluir antes de dividir, sem desalinhamento de features/rótulos; teste de split inclui vazio e verifica contagem/linhas preservadas (Task 1).
- Estratificação e reprodutibilidade: cada rating mantém proporção próxima e a seed 42 repete exatamente os mesmos registros (Task 1).
- Vazamento: o vetorizador de validação não pode ajustar em validation ou test; teste do CLI recebe apenas `train.csv` e comprova os tamanhos/artefatos sem abrir `test.csv` (Task 2).
- CSV derivado: salvar todas as colunas e valores brutos em `data/raw/validation.csv`, mas não adicioná-lo ao índice; checar com `git check-ignore` (Tasks 1 e 4).
- Comparação correta: acurácia/F1 são maiores quando melhor; MAE/RMSE menores; o relatório identifica explicitamente “treino completo → teste oficial” versus “80% treino → 20% validação” (Task 2).

---

### Task 1: Split reutilizável e CSV local de validação

**Files:**
- Create: `src/data/splitting.py`
- Create: `scripts/create_validation_split.py`
- Create: `tests/test_splitting.py`
- Create: `tests/test_create_validation_split.py`

**Interfaces:**
- Produces: `split_train_validation(df: pd.DataFrame, *, validation_size: float = 0.2, random_state: int = 42, text_col: str = "text", rating_col: str = "rating") -> tuple[pd.DataFrame, pd.DataFrame]`, retornando `(train, validation)` em linhas brutas e mantendo todas as colunas.
- Produces CLI: `python scripts/create_validation_split.py --train data/raw/train.csv --out data/raw/validation.csv --validation-size 0.2 --seed 42`.
- O splitter remove linhas cujo texto vira vazio com `normalize`, valida que `0 < validation_size < 1`, estratifica por rating e reseta índices sem mutar a entrada.

- [ ] **Step 1: Write failing tests**

`test_split_train_validation_eh_estratificado_reprodutivel_e_preserva_linhas` usa um DataFrame sintético com cinco ratings, uma linha vazia e coluna extra `marker`; verifica split 40/10 após remover o vazio, distribuições 8/2 por classe, ausência de interseção, preservação das colunas/valores e mesma seleção em duas chamadas.

`test_create_validation_split_salva_csv_bruto_sem_alterar_origem` passa CSV sintético ao CLI e verifica saída com colunas originais e 20% dos registros válidos, além de confirmar que o CSV de origem permanece byte a byte igual.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_splitting.py tests/test_create_validation_split.py -v`
Expected: FAIL porque `src.data.splitting`/CLI ainda não existem.

- [ ] **Step 3: Implement splitter and CLI**

Usar `normalize` apenas para identificar linhas vazias e `sklearn.model_selection.train_test_split(..., stratify=rating, random_state=seed)` para obter a divisão; salvar somente validation com `index=False`, preservando valores e colunas originais.

- [ ] **Step 4: Run focused tests**

Run: `.venv/bin/python -m pytest tests/test_splitting.py tests/test_create_validation_split.py -v`
Expected: PASS; divisão repetível e CSV sintético válido.

- [ ] **Step 5: Commit**

```bash
git add src/data/splitting.py scripts/create_validation_split.py tests/test_splitting.py tests/test_create_validation_split.py
git commit -m "feat: gera split local de validacao estratificado (issue 7)"
```

### Task 2: Treino de validação, métricas e comparação versionável

**Files:**
- Create: `scripts/train_baseline_validation.py`
- Create: `tests/test_train_baseline_validation.py`
- Modify: `.gitignore` (permitir somente os novos JSON/CSV/PNG da validação)
- Modify: `data/README.md` (documentar criação local e não versionamento do CSV)
- Modify: `results/metrics/README.md` (comando, tamanhos e interpretação da comparação)
- Create: `results/metrics/baseline_logreg_validation.json` (gerado)
- Create: `results/metrics/baseline_logreg_comparison.csv` (gerado)
- Create: `results/figures/confusion_matrix_logreg_validation.png` (gerado)

**Interfaces:**
- Consumes: `split_train_validation` (Task 1), normalização/encoding/embedding existentes, `train_logistic_regression`, `compute_metrics`, `save_confusion_matrix`, e o baseline `results/metrics/baseline_logreg.json`.
- Produces CLI: `python scripts/train_baseline_validation.py --train data/raw/train.csv --validation-out data/raw/validation.csv --baseline-metrics results/metrics/baseline_logreg.json --out-metrics results/metrics/baseline_logreg_validation.json --out-figure results/figures/confusion_matrix_logreg_validation.png --comparison-out results/metrics/baseline_logreg_comparison.csv --validation-size 0.2 --seed 42 --C 1.0 --max-iter 1000`.
- O JSON registra `seed`, `validation_size`, `n_source_rows`, `n_dropped_empty`, `n_train`, `n_validation`, `validation_csv`, `validation_csv_sha256`, `sklearn_version`, `lr_params`, `vectorizer_params` e `metrics_validation` (acurácia/F1 macro/MAE/RMSE).
- O CSV de comparação tem colunas `metric`, `full_train_official_test`, `train_80_validation`, `validation_minus_test` e `better_direction`.

- [ ] **Step 1: Write failing tests**

`test_cli_treina_sem_ler_test_salva_split_metricas_figura_e_comparacao` usa um train CSV sintético e um JSON baseline sintético; verifica saída validation CSV com todas as colunas, tamanhos coerentes, quatro métricas, figura PNG não vazia e quatro linhas de comparação. O CLI não recebe argumento `--test`.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_train_baseline_validation.py -v`
Expected: FAIL porque o CLI ainda não existe.

- [ ] **Step 3: Implement `scripts/train_baseline_validation.py`**

Splitar e gravar CSV bruto via Task 1; normalizar separadamente train/validation; ajustar TF-IDF só em train; transformar validation; treinar LR com parâmetros fixados; salvar métrica, figura e CSV de comparação. Não carregar nem avaliar o teste oficial.

- [ ] **Step 4: Run focused tests**

Run: `.venv/bin/python -m pytest tests/test_train_baseline_validation.py tests/test_splitting.py tests/test_create_validation_split.py -v`
Expected: PASS; o caso sintético comprova a pipeline sem acessar test.csv.

- [ ] **Step 5: Commit**

```bash
git add scripts/train_baseline_validation.py tests/test_train_baseline_validation.py .gitignore data/README.md results/metrics/README.md
git commit -m "feat: treina baseline com validacao e compara metricas (issue 7)"
```

### Task 3: Novo notebook Colab com os dois experimentos

**Files:**
- Create: `notebooks/baseline_logreg_validation_colab.ipynb`
- Create: `tests/test_colab_validation_notebook.py`

**Interfaces:**
- O notebook faz upload de `train.csv` e `test.csv`, cria `data/raw/validation.csv` local com 20% estratificado, executa baseline completo train→test e baseline train-80%→validation, mostra a tabela comparativa e disponibiliza downloads do CSV de validação e resultados.
- Inclui fallback equivalente de split/modelo/métricas/plot quando o Colab clonou `main` antes do merge da Issue 7; se os módulos do projeto existirem, importa-os. No Colab, instalar dependências sem sobrescrever `numpy`/`pandas` do runtime.
- Saídas exatas: baseline completo em `baseline_logreg.json` e `confusion_matrix_logreg.png`; validação em `baseline_logreg_validation.json` e `confusion_matrix_logreg_validation.png`; comparação em `baseline_logreg_comparison.csv`.
- `SAMPLE_N=0` roda todos os dados por padrão; um valor positivo amostra estratificadamente o treino e o teste para smoke run. `OUTPUT_DIR` e `VALIDATION_OUT` permitem smoke tests sem sobrescrever artefatos/dados reais.

- [ ] **Step 1: Write failing tests**

`test_notebook_colab_validation_tem_split_e_saidas` verifica que o notebook contém upload, split 20%/seed 42, gravação `data/raw/validation.csv`, métricas dos dois experimentos, comparação e downloads. `test_notebook_colab_validation_executa_com_dados_locais` executa notebook em modo smoke (`SAMPLE_N=400`, `OUTPUT_DIR` e `VALIDATION_OUT` em `tmp_path`) e verifica JSON/PNG/CSV válidos sem sobrescrever os resultados oficiais.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_colab_validation_notebook.py -v`
Expected: FAIL porque o notebook e teste ainda não existem.

- [ ] **Step 3: Implement notebook**

Manter células pedagógicas e reproduzíveis: setup/upload; configuração/imports; geração e download local do split; baseline completo; validação 80/20 e comparação; downloads. Reutilizar configurações da pipeline e indicar que o CSV derivado não é versionado.

- [ ] **Step 4: Run notebook verification**

Run: `.venv/bin/python -m pytest tests/test_colab_validation_notebook.py -v` e `SAMPLE_N=400 .venv/bin/python -m jupyter nbconvert --to notebook --execute notebooks/baseline_logreg_validation_colab.ipynb --output /tmp/colab_validation_smoke.ipynb --ExecutePreprocessor.timeout=600` com os paths de saída temporários.
Expected: PASS e notebook termina sem exceções.

- [ ] **Step 5: Commit**

```bash
git add notebooks/baseline_logreg_validation_colab.ipynb tests/test_colab_validation_notebook.py
git commit -m "feat: adiciona notebook colab para validacao (issue 7)"
```

### Task 4: Execução com os dados oficiais e artefatos de revisão

**Files:**
- Generate locally, ignored: `data/raw/validation.csv`
- Generate and version: the three validation artifacts named in Global Constraints.

- [ ] **Step 1: Generate the local validation split and run the validation CLI**

Run: `.venv/bin/python scripts/train_baseline_validation.py --seed 42 --validation-size 0.2`
Expected: CSV local com 8.235 linhas e todas as colunas; métricas/figura/comparação gravadas em `results/`.

- [ ] **Step 2: Verify split, comparison, original artifacts and ignore rules**

Run: `.venv/bin/python -m pytest -q`, `git check-ignore data/raw/validation.csv`, and verify both experiment JSONs/figures plus comparison CSV are non-empty and tracked/trackable.
Expected: suite verde; `validation.csv` ignorado; arquivos de resultados versionáveis; baseline full original não foi alterado.

- [ ] **Step 3: Commit generated results and documentation**

```bash
git add results/metrics/baseline_logreg_validation.json results/metrics/baseline_logreg_comparison.csv results/figures/confusion_matrix_logreg_validation.png data/README.md results/metrics/README.md .gitignore
git commit -m "docs: registra comparacao do baseline com validacao (issue 7)"
```

---

## Self-Review

- Cobertura: Task 1 cria split determinístico e CSV bruto local; Task 2 treina sem acessar test e salva métricas/figura/comparação; Task 3 cobre uso no Colab; Task 4 executa os dados reais e põe artefatos no branch.
- Interfaces: `split_train_validation` é consumida tanto pelo CLI quanto pelo notebook; o JSON de baseline existente fornece os campos `metrics_logreg` usados pela tabela comparativa.
- Review Focus: vazios/estratificação são testados na Task 1; isolamento do teste na Task 2; ignorar CSV e preservar baseline verificados na Task 4; direções da comparação verificadas na Task 2.
- Proporção: plano limitado ao split, um trainer de validação, notebook e três artefatos necessários; sem ajustar hiperparâmetros nem alterar a pipeline full baseline.
