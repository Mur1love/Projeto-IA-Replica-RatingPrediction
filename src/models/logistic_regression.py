"""Regressão Logística (``LogisticRegression``) da RQ1 do artigo, com as 58 features.

Parte única — protocolo do artigo (``LOGISTIC_REGRESSION_SPEC``):
    Grid idêntico ao de ``code/RQ1/scripts/logistic_regression.py`` do repositório
    original: 5 × 5 × 2 × 1 × 1 = 50 combinações. Como ``n_iter=300`` é maior que
    o grid, a busca é exaustiva (o scikit-learn avisa e limita), como no SVM.

    A contraparte TF-IDF desta LR **já existe**: é o baseline da issue #7
    (``results/metrics/baseline_logreg.json``), cuja acurácia, F1 macro, MAE e
    RMSE no teste oficial são a referência de TF-IDF. Esta execução cobre a
    pendência 4 da seção 7.6 de ``docs/diferencas.md`` e deixa a réplica com
    dois modelos completos no protocolo do artigo (SVM e LR).

Convergência e escala: as features não são padronizadas (fidelidade ao artigo,
mesma ressalva da seção 3.2 de ``docs/diferencas.md``) e ``max_iter=10000`` vem
do grid. Avisos de convergência de ``sag``/``saga`` em ``C`` pequenos são
esperados no scikit-learn atual e registram o mesmo comportamento do original.
"""
from __future__ import annotations

from sklearn.linear_model import LogisticRegression

from src.models.article_protocol import DEFAULT_SEED, ModelSpec

# Grid do artigo (code/RQ1/scripts/logistic_regression.py), na ordem do original.
PARAM_GRID: dict[str, list] = {
    "C": [0.01, 0.1, 1, 10, 100],
    "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
    "class_weight": [None, "balanced"],
    "random_state": [None],
    "max_iter": [10000],
}

# Tabela 3 do artigo (Logistic Regression): (média, desvio-padrão entre os 5 folds).
ARTICLE_TABLE3 = {
    "mae": (1.0149, 0.0043),
    "rmse": (1.4691, 0.0042),
    "auc": (0.7328, 0.0064),
}
# Melhores hiperparâmetros do artigo
# (code/RQ1/results/LogisticRegression_best_params.txt).
ARTICLE_BEST_PARAMS = {
    "solver": "lbfgs",
    "max_iter": 10000,
    "class_weight": None,
    "C": 100,
}

N_GRID_COMBINATIONS = 50  # 5 (C) x 5 (solver) x 2 (class_weight) x 1 x 1


def build_logistic_regression(seed: int = DEFAULT_SEED) -> LogisticRegression:
    """``LogisticRegression`` sem parâmetros extras (o grid define o resto).

    O ``seed`` é aceito apenas para manter a assinatura comum aos ``ModelSpec``:
    o grid do artigo fixa ``random_state=None``, e o valor não afeta
    ``lbfgs``/``newton-cg`` nem o resultado dos demais solvers neste protocolo.
    """
    return LogisticRegression()


LOGISTIC_REGRESSION_SPEC = ModelSpec(
    name="logistic_regression",
    build_estimator=build_logistic_regression,
    param_distributions=PARAM_GRID,
    article_reference=ARTICLE_TABLE3,
    article_best_params=ARTICLE_BEST_PARAMS,
)
