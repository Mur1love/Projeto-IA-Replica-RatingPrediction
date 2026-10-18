# Baseline com Regressão Logística (Issue 7) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Treinar o primeiro modelo da réplica (TF-IDF + Regressão Logística) sobre os splits oficiais e salvar métricas + matriz de confusão em `results/`.

**Architecture:** Reusar `src/preprocessing/` já mergeado na `main` (normalize → encode → TF-IDF fit só no treino); adicionar `src/evaluation/metrics.py` (métricas puras), `src/models/baseline.py` (treino LR + baseline majoritária), `src/evaluation/plots.py` (matriz de confusão), `scripts/train_baseline.py` (CLI fim a fim com seed fixa) e `notebooks/baseline_logreg_colab.ipynb` (mesma pipeline para treino no Google Colab).

**Tech Stack:** Python 3.12+, scikit-learn 1.9.1 (LogisticRegression, DummyClassifier/majority, confusion_matrix), scipy sparse, matplotlib 3.11.2, joblib, pytest 9.1.1, pandas, numpy.

**Spec:** Issue 7 remota `[TAREFA 7] - BASELINE COM REGRESSÃO LOGÍSTICA` + `ISSUES.md` seção "Issue 7" (splits oficiais `train.csv` 41.172 / `test.csv` 10.293, métricas acurácia/F1-macro/MAE/RMSE, matriz em `results/figures/`, métricas+parâmetros em `results/metrics/`, baseline trivial classe majoritária, seed fixa). Pré-requisitos: **já mergeados na `main` em `a0fcc9f`** (`src/preprocessing/normalization.py`, `encoding.py`, `embedding.py` com `DEFAULT_PARAMS`, `build_embeddings`/`fit_vectorizer`/`transform_texts`). Verificado em 2026-10-17: `pytest -q` 96 passed.

## Global Constraints

- Python >= 3.12 (requirements travadas: numpy 2.5.3, pandas 3.0.6, scikit-learn 1.9.1, scipy 1.18.1, matplotlib 3.11.2, pytest 9.1.1).
- Splits oficiais: `fit` do TF-IDF e treino da LR SOMENTE em `data/raw/train.csv`; `data/raw/test.csv` só `transform`/avaliação — nunca `fit` no teste.
- Labels: `encode_rating` 1..5 → 0..4; acurácia/F1 sobre labels 0..4; MAE/RMSE sobre ratings 1..5 decodificados (`label+1` / `decode_rating`).
- Seed fixa `RANDOM_STATE = 42` em todo sorteio (`train_test_split`, `LogisticRegression`, `numpy`); registrar `sklearn_version` no JSON.
- Não versionar CSVs nem matrizes grandes (respeitar `.gitignore`: `data/raw/`, `data/processed/*.npz/*.joblib`); versionar apenas código, testes, JSON de métricas e PNG da figura.
- Branch: `issue-7-baseline` criada a partir da `main` atual (`a0fcc9f`, já contém issues 4+5+6); PR com `Closes #7`; sem commit direto na `main`.
- Notebook Colab: `notebooks/baseline_logreg_colab.ipynb` deve rodar a **mesma** pipeline do CLI (mesmos `DEFAULT_PARAMS`, `seed=42`, `fit` só no treino); dados via upload (`files.upload`) pois `data/raw/*.csv` não é versionado; `pip install -r requirements.txt` no Colab pode divergir de versão — registrar `sklearn_version` no output e comparar com o JSON local.

## Review Focus

- Texto vazio/NaN após `normalize` vira `""` e é descartado (`empty="drop"`); contagem `n_dropped_empty` deve ir ao JSON — espera-se 1 por split; ignorar silenciosamente quebraria alinhamento X/y.
- Termo só do teste fora do vocabulário é ignorado no `transform` (vetor zerado); linha zerada não deve quebrar `predict` nem inflar métrica.
- Desbalanceamento acidental no split de validação (se `--val-size` usado): usar `stratify=stratify_key` ou `stratify=y`; sem isso F1-macro da validação mente.
- `LogisticRegression` sem `max_iter` suficiente não converge em TF-IDF esparso (~40k docs): esperar `ConvergenceWarning`; fixar `max_iter=1000` e documentar.
- Rótulo fora de 0..4 ou NaN em `y` deve levantar `ValueError` (via `encode_rating`), nunca codificar silenciosamente e contaminar MAE/RMSE.

---

### Task 1: Métricas + baseline majoritária (`src/evaluation/`)

**Files:**
- Create: `src/evaluation/metrics.py`
- Create: `tests/test_metrics.py`

**Interfaces:**
- Consumes: `src.preprocessing.encoding.decode_rating` (ou `label+1`) para MAE/RMSE na escala 1..5.
- Produces:
  - `def compute_metrics(y_true_labels: np.ndarray, y_pred_labels: np.ndarray) -> dict[str, float]` — chaves exatas `accuracy`, `f1_macro`, `mae`, `rmse`.
  - `def majority_class_label(y_train_labels: np.ndarray) -> int` — moda; empate desempatado pelo menor label.
  - `def predict_majority(y_train_labels: np.ndarray, n: int) -> np.ndarray` — vetor tamanho `n` com a moda.

- [ ] **Step 1: Write the failing test**

```python
def test_compute_metrics_valores_conhecidos():
    y_true = np.array([0, 1, 2, 3, 4, 4])
    y_pred = np.array([0, 1, 2, 3, 3, 4])
    m = compute_metrics(y_true, y_pred)
    assert m["accuracy"] == pytest.approx(5/6)
    assert 0.0 <= m["f1_macro"] <= 1.0
    assert m["mae"] == pytest.approx(1/6)  # erro de 1 estrela em 1 de 6
    assert m["rmse"] == pytest.approx((1/6) ** 0.5)

def test_majority_class_label_e_predict():
    assert majority_class_label(np.array([0, 4, 4, 1, 4])) == 4
    np.testing.assert_array_equal(predict_majority(np.array([0, 1, 1]), 3), np.array([1, 1, 1]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.evaluation.metrics'` ou `ImportError`.

- [ ] **Step 3: Implement `compute_metrics`, `majority_class_label`, `predict_majority` in `src/evaluation/metrics.py`**

Usar `sklearn.metrics.accuracy_score`, `f1_score(average="macro", zero_division=0)`, `mean_absolute_error` e `mean_squared_error(squared=False)` — ou `sqrt` manual se API exigir — sempre após decodificar `label+1` para MAE/RMSE. Validar tamanhos iguais e labels em 0..4, levantar `ValueError` caso contrário. Sem I/O neste módulo.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_metrics.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/metrics.py tests/test_metrics.py
git commit -m "feat: adiciona metricas e baseline majoritaria (issue 7)"
```

### Task 2: Treino da Regressão Logística (`src/models/`)

**Files:**
- Create: `src/models/baseline.py`
- Create: `tests/test_baseline.py`
- Modify: `src/models/__init__.py` (exportar nomes, 1 linha)

**Interfaces:**
- Consumes: matrizes `scipy.sparse.csr_matrix` de `src.preprocessing.embedding.fit_vectorizer`/`transform_texts`; `compute_metrics` da Task 1 (só no script, não aqui).
- Produces:
  - `RANDOM_STATE: int = 42`, `DEFAULT_LR_PARAMS: dict = {"C": 1.0, "max_iter": 1000, "solver": "lbfgs", "random_state": 42}`
  - `def train_logistic_regression(X_train: sp.csr_matrix, y_train: np.ndarray, *, C: float = 1.0, max_iter: int = 1000, random_state: int = 42) -> LogisticRegression`
  - `def save_model(model: LogisticRegression, path: str | Path) -> Path`
  - `def load_model(path: str | Path) -> LogisticRegression`

- [ ] **Step 1: Write the failing test**

```python
def test_train_logistic_regression_prediz_e_reproduzivel():
    X_toy = sp.csr_matrix(np.array([[2.0, 0.0], [0.0, 2.0], [1.5, 0.2], [0.1, 1.8]]))
    y_toy = np.array([0, 4, 0, 4])
    m1 = train_logistic_regression(X_toy, y_toy)
    m2 = train_logistic_regression(X_toy, y_toy)
    np.testing.assert_array_equal(m1.predict(X_toy), m2.predict(X_toy))
    assert set(m1.predict(X_toy)).issubset({0, 4})

def test_save_load_model_roundtrip(tmp_path):
    X_toy = sp.csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0]]))
    m = train_logistic_regression(X_toy, np.array([0, 4]))
    p = save_model(m, tmp_path / "lr.joblib")
    assert p.exists()
    np.testing.assert_array_equal(load_model(p).predict(X_toy), m.predict(X_toy))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_baseline.py -v`
Expected: FAIL with `ModuleNotFoundError` / `ImportError`.

- [ ] **Step 3: Implement em `src/models/baseline.py`**

Envolver `sklearn.linear_model.LogisticRegression(**{**DEFAULT_LR_PARAMS, "C": C, "max_iter": max_iter, "random_state": random_state})` + `fit`. `save/load` via `joblib.dump/load` criando parent dirs. Não normalizar/escalonar (TF-IDF já L2). Documentar que `lbfgs` multinomial é determinístico dado `random_state`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_baseline.py tests/test_metrics.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/models/baseline.py src/models/__init__.py tests/test_baseline.py
git commit -m "feat: adiciona treino da regressao logistica (issue 7)"
```

### Task 3: Matriz de confusão (`src/evaluation/plots.py`)

**Files:**
- Create: `src/evaluation/plots.py`
- Create: `tests/test_plots.py`

**Interfaces:**
- Consumes: `y_true_labels`, `y_pred_labels` (0..4) da Task 2.
- Produces:
  - `def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, path: str | Path, *, display_labels: tuple = (1,2,3,4,5)) -> Path` — salva PNG 6x5, `ConfusionMatrixDisplay`, `cmap="Blues"`, `tight_layout`.

- [ ] **Step 1: Write the failing test**

```python
def test_save_confusion_matrix_cria_png(tmp_path):
    y_true = np.array([0, 1, 2, 3, 4, 4, 3])
    y_pred = np.array([0, 1, 2, 2, 4, 4, 3])
    out = tmp_path / "cm.png"
    assert save_confusion_matrix(y_true, y_pred, out).exists()
    assert out.stat().st_size > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_plots.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `save_confusion_matrix` em `src/evaluation/plots.py`**

Usar `sklearn.metrics.confusion_matrix(labels=[0,1,2,3,4])` + `ConfusionMatrixDisplay(...).plot(...)` com `matplotlib.use("Agg")`; `plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()`. Criar parent dirs. Sem `plt.show()`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_plots.py -v`
Expected: PASS e PNG gerado.

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/plots.py tests/test_plots.py
git commit -m "feat: adiciona matriz de confusao (issue 7)"
```

### Task 4: CLI fim a fim (`scripts/train_baseline.py`)

**Files:**
- Create: `scripts/train_baseline.py`
- Create: `tests/test_train_baseline.py` (teste de fumaça com amostra)

**Interfaces:**
- Consumes: `normalize_dataframe`, `encode_rating`, `build_vectorizer`/`fit_vectorizer`/`transform_texts`, `train_logistic_regression`, `compute_metrics`/`majority_class_label`, `save_confusion_matrix`.
- Produces (CLI):
  - `python scripts/train_baseline.py --train data/raw/train.csv --test data/raw/test.csv --out-metrics results/metrics/baseline_logreg.json --out-figure results/figures/confusion_matrix_logreg.png --sample-n 2000 --seed 42 --C 1.0 --max-iter 1000 --val-size 0.0`
  - JSON exato: `{seed, lr_params, vectorizer_params, normalization:{}, n_train, n_test, n_dropped_train, n_dropped_test, sklearn_version, metrics_logreg:{accuracy,f1_macro,mae,rmse}, metrics_majority:{...}, majority_label}`.

- [ ] **Step 1: Write the failing test**

```python
def test_cli_sample_gera_json_e_png(tmp_path):
    metrics_p = tmp_path / "m.json"
    fig_p = tmp_path / "cm.png"
    rc = subprocess.run([sys.executable, "scripts/train_baseline.py", "--sample-n", "400", "--seed", "42", "--out-metrics", str(metrics_p), "--out-figure", str(fig_p)], capture_output=True, text=True, timeout=300)
    assert rc.returncode == 0, rc.stderr
    assert metrics_p.exists() and fig_p.exists()
    payload = json.loads(metrics_p.read_text(encoding="utf-8"))
    assert set(payload["metrics_logreg"]) == {"accuracy", "f1_macro", "mae", "rmse"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_train_baseline.py -v`
Expected: FAIL (script não existe → `returncode != 0`).

- [ ] **Step 3: Implement `scripts/train_baseline.py`**

Ordem: `argparse` → `np.random.seed(seed)` → ler só `usecols=["text","rating"]` → `normalize_dataframe(empty="drop")` → opcional `sample-n` estratificado (`train_test_split(stratify=y, random_state=seed)`) → opcional validação `train_test_split(test_size=val-size, stratify=...)` para checar `C` sem tocar no teste → `build_vectorizer(**DEFAULT_PARAMS)` + `fit` só no treino → `train_logistic_regression` → `compute_metrics` no teste para LR e para majoritária → `save_confusion_matrix` → gravar JSON + `print` resumo. Reusar exatamente `DEFAULT_PARAMS` do embedding.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_train_baseline.py -v`
Expected: PASS em < 5 min (amostra 400).

- [ ] **Step 5: Commit**

```bash
git add scripts/train_baseline.py tests/test_train_baseline.py
git commit -m "feat: adiciona CLI fim a fim do baseline (issue 7)"
```

### Task 5: Treino completo + artefatos + doc

**Files:**
- Create: `results/metrics/baseline_logreg.json` (gerado, versionar)
- Create: `results/figures/confusion_matrix_logreg.png` (gerado, versionar)
- Modify: `README.md` ou `data/README.md`? Não — criar seção curta em PR description + anotar em `results/metrics/README.md` (criar, 10 linhas: comando, seed, config).

**Interfaces:**
- Consumes: CLI da Task 4.
- Produces: artefatos versionados + linha de comando documentada.

- [ ] **Step 1: Write the failing test (contrato dos artefatos)**

```python
def test_artefatos_finais_existem_e_sao_validos():
    payload = json.loads(Path("results/metrics/baseline_logreg.json").read_text(encoding="utf-8"))
    assert payload["seed"] == 42 and payload["n_train"] + payload["n_dropped_train"] == 41172
    assert Path("results/figures/confusion_matrix_logreg.png").exists()
```

(Pode viver em `tests/test_train_baseline.py` como segundo teste; escrevê-lo antes da execução full faz falhar até o full rodar.)

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_train_baseline.py::test_artefatos_finais_existem_e_sao_validos -v`
Expected: FAIL (`FileNotFoundError`).

- [ ] **Step 3: Executar treino completo**

```bash
.venv/bin/python scripts/train_baseline.py --seed 42 --C 1.0 --max-iter 1000
```

Tempo esperado: 1–5 min em CPU (41k treino, vocabulário ~ dezenas de milhares). Se `ConvergenceWarning`, registrar e manter `max_iter=1000` (não silenciar sem documentar).

- [ ] **Step 4: Run tests to verify all pass**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS (todos os testes 4–7 + antigos).

- [ ] **Step 5: Commit**

```bash
git add results/metrics/baseline_logreg.json results/figures/confusion_matrix_logreg.png results/metrics/README.md
git commit -m "feat: baseline regressao logistica com metricas e matriz (issue 7)"
```

### Task 6: Notebook Colab para (re)treino (`notebooks/baseline_logreg_colab.ipynb`)

**Files:**
- Create: `notebooks/baseline_logreg_colab.ipynb`
- Create: `tests/test_colab_notebook.py` (valida estrutura/nbconvert, não treina full)

**Interfaces:**
- Consumes: mesmos módulos das Tasks 1-4 (`embedding`, `baseline`, `metrics`, `plots`); CLI da Task 4 como referência.
- Produces: notebook com 6 células numeradas:
  1. `!git clone <URL> && %cd` + `!pip install -q -r requirements.txt` (ou `%pip install scikit-learn==1.9.1 pandas==3.0.6 matplotlib==3.11.2` se clone falhar).
  2. Upload dos CSVs (`from google.colab import files; files.upload()` → salvar em `data/raw/train.csv`, `data/raw/test.csv`) + `!ls -lh data/raw/`.
  3. Imports + `SEED=42`, `SAMPLE_N=2000` parametrizável no topo.
  4. Treino amostra via `fit_vectorizer`/`train_logistic_regression`/`compute_metrics` (mesmos defaults do CLI) + `print` métricas.
  5. Full (mesma chamada sem `sample-n`) + `save_confusion_matrix` com `plt.show()` inline + exibição do JSON.
  6. Download dos artefatos (`files.download("results/metrics/baseline_logreg.json")`).

- [ ] **Step 1: Write the failing test**

```python
def test_notebook_colab_tem_celulas_obrigatorias():
    nb = json.loads(Path("notebooks/baseline_logreg_colab.ipynb").read_text(encoding="utf-8"))
    src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    for token in ["files.upload", "train_logistic_regression", "compute_metrics", "save_confusion_matrix", "SEED", "SAMPLE_N"]:
        assert token in src, f"faltando {token}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_colab_notebook.py -v`
Expected: FAIL (`FileNotFoundError`).

- [ ] **Step 3: Implement `notebooks/baseline_logreg_colab.ipynb`**

Criar via Jupytext/nbformat ou Jupyter local com kernel python3; fixar `SEED=42`, `C=1.0`, `MAX_ITER=1000`, `DEFAULT_PARAMS` importado (não copiado); célula amostra default `SAMPLE_N=2000` para rodar em <2 min no Colab free; célula full comentada com aviso de tempo (~1–5 min). Incluir markdown inicial explicando: repositório não versiona CSVs → upload obrigatório; `fit` só no treino.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_colab_notebook.py -v && .venv/bin/python -m jupyter nbconvert --to notebook --execute notebooks/baseline_logreg_colab.ipynb --output /tmp/colab_smoke.ipynb --ExecutePreprocessor.timeout=600 2>&1 | tail -n 5`
Expected: PASS no pytest; nbconvert executa sem erro (pode usar amostra pequena via env `SAMPLE_N=200`).

- [ ] **Step 5: Commit**

```bash
git add notebooks/baseline_logreg_colab.ipynb tests/test_colab_notebook.py
git commit -m "feat: adiciona notebook colab do baseline (issue 7)"
```
