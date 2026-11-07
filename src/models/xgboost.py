"""XGBoost da RQ1 do artigo, com as 58 features, e comparação com TF-IDF.

Parte 1 — protocolo do artigo (``XGB_SPEC``):
    Grid idêntico ao descrito na TAREFA 14: 4 × 3^8 = 8.748 combinações
    (``n_estimators``, ``learning_rate``, ``max_depth``, ``min_child_weight``,
    ``gamma``, ``subsample``, ``colsample_bytree``, ``reg_alpha`` e
    ``reg_lambda``) amostradas por ``RandomizedSearchCV`` com ``n_iter=300``.
Parte 2 — comparação complementar (``run_xgboost_tfidf``, TAREFA 14):
    XGBoost sobre a matriz TF-IDF da Issue 6, no split oficial treino -> teste,
    com busca curta, para comparar com a Regressão Logística + TF-IDF do
    baseline (``baseline_logreg.json``).
"""
from __future__ import annotations

from typing import Any

from xgboost import XGBClassifier

from src.models.article_protocol import DEFAULT_SEED, ModelSpec

# Grid da TAREFA 14 (RQ1): n_estimators(4) x as outras oito dimensões (3) = 8.748.
PARAM_GRID: dict[str, list] = {
    "n_estimators": [100, 200, 300, 500],
    "learning_rate": [0.05, 0.10, 0.20],
    "max_depth": [6, 8, 10],
    "min_child_weight": [1, 3, 5],
    "gamma": [0.0, 0.1, 0.3],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.4, 0.6, 0.8],
    "reg_alpha": [0, 0.1, 1],
    "reg_lambda": [1, 1.5, 2],
}

# Tabela 3 do artigo (XGBoost): (média, desvio-padrão entre os 5 folds).
ARTICLE_TABLE3 = {
    "mae": (0.9499, 0.0047),
    "rmse": (1.3721, 0.0055),
    "auc": (0.7390, 0.0046),
}
# A TAREFA não informa os melhores parâmetros encontrados no artigo.
ARTICLE_BEST_PARAMS: dict[str, Any] = {}


def build_xgboost(seed: int = DEFAULT_SEED) -> XGBClassifier:
    """``XGBClassifier`` reprodutível; ``n_jobs=1`` deixa o paralelismo externo.

    O paralelismo do protocolo fica no ``RandomizedSearchCV``/``joblib``
    (``--n-jobs -1`` do CLI), como no Random Forest. O ``tree_method`` padrão
    do xgboost 3.x é ``hist``.
    """
    return XGBClassifier(random_state=seed, n_jobs=1)


XGB_SPEC = ModelSpec(
    name="xgboost",
    build_estimator=build_xgboost,
    param_distributions=PARAM_GRID,
    article_reference=ARTICLE_TABLE3,
    article_best_params=ARTICLE_BEST_PARAMS,
)
