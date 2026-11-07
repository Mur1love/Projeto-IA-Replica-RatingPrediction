"""Testes da configuração e integração do Random Forest na RQ1."""
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.models.article_protocol import (
    ModelSpec, build_folds, compute_fold_metrics, run_article_protocol,
)
from src.models.random_forest import (
    ARTICLE_TABLE3, PARAM_GRID, RANDOM_FOREST_SPEC, build_random_forest,
)


def test_grid_random_forest_contem_parametros_do_enunciado():
    assert PARAM_GRID == {
        "n_estimators": [100, 200, 300, 500, 1000],
        "max_depth": [5, 10, 30, None],
        "min_samples_split": [5, 10, 20],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", 0.5, None],
        "class_weight": [None, "balanced", "balanced_subsample"],
        "max_leaf_nodes": [None, 10, 20],
        "min_impurity_decrease": [0.0, 0.01, 0.1],
    }


def test_estimador_random_forest_e_reprodutivel():
    model = build_random_forest(42)
    assert isinstance(model, RandomForestClassifier)
    assert model.random_state == 42
    assert model.n_jobs == 1


def test_referencia_tabela_3():
    assert ARTICLE_TABLE3 == {
        "mae": (0.9717, 0.0077),
        "rmse": (1.4001, 0.0087),
        "auc": (0.7367, 0.0051),
    }


def test_folds_estratificados_sem_perda_de_linhas():
    groups = np.repeat(["a", "b", "c"], 10)
    folds = build_folds(groups, n_splits=5)
    assert len(folds) == 5
    for train, valid in folds:
        assert len(set(train) & set(valid)) == 0
        assert len(train) + len(valid) == len(groups)


def test_metricas_incluem_auc_binaria_e_auc_scores():
    y = np.array([0, 1, 2, 3, 4] * 4)
    pred = np.array([0, 1, 1, 3, 4] * 4)
    scores = np.linspace(0, 1, len(y))
    result = compute_fold_metrics(y, pred, scores)
    assert set(result) == {"mae", "rmse", "f1_macro", "accuracy", "auc", "auc_scores"}
    assert 0 <= result["auc"] <= 1
    assert 0 <= result["auc_scores"] <= 1


def test_random_forest_roda_com_grid_reduzido_e_probabilidades():
    from tests.conftest import make_article_frame
    from src.models.article_protocol import load_article_data
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        make_article_frame(24, seed=11).to_csv(d/"train.csv", index=False)
        make_article_frame(8, seed=12).to_csv(d/"test.csv", index=False)
        data = load_article_data(d/"train.csv", d/"test.csv")
        spec = ModelSpec(
            "random_forest_test", build_random_forest,
            {"n_estimators": [10], "max_depth": [5], "min_samples_split": [2],
             "min_samples_leaf": [1], "max_features": ["sqrt"]},
        )
        result = run_article_protocol(spec, data, n_iter=1, n_splits=5, n_jobs=1)
        assert len(result.fold_metrics) == 5
        assert result.fold_metrics["auc_scores"].notna().all()
        assert result.confusion.sum() == data.n_rows
        assert result.holdout["n_train"] == 240
        assert result.holdout["n_test"] == 80
