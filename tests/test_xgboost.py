"""Testes da configuração e integração do XGBoost na RQ1."""
from dataclasses import replace
import json

import pytest
from xgboost import XGBClassifier

from src.models.article_protocol import load_article_data, run_article_protocol
from src.models.xgboost import (
    ARTICLE_TABLE3, PARAM_GRID, XGB_SPEC, build_xgboost, run_xgboost_tfidf,
)


def test_grid_xgboost_contem_parametros_do_enunciado():
    assert PARAM_GRID == {
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


def test_estimador_xgboost_e_reprodutivel():
    model = build_xgboost(42)
    assert isinstance(model, XGBClassifier)
    assert model.random_state == 42
    assert model.n_jobs == 1


def test_referencia_tabela_3():
    assert ARTICLE_TABLE3 == {
        "mae": (0.9499, 0.0047),
        "rmse": (1.3721, 0.0055),
        "auc": (0.7390, 0.0046),
    }


def test_protocolo_xgboost_com_grid_reduzido(article_csvs):
    train_csv, test_csv = article_csvs
    data = load_article_data(train_csv, test_csv)
    spec = replace(
        XGB_SPEC,
        name="xgboost_test",
        param_distributions={"n_estimators": [10], "max_depth": [4], "learning_rate": [0.1]},
        article_reference={},
    )
    result = run_article_protocol(spec, data, n_iter=1, n_splits=5, n_jobs=1)
    assert len(result.fold_metrics) == 5
    assert result.fold_metrics["auc_scores"].notna().all()  # predict_proba existe
    assert result.confusion.sum() == data.n_rows
    assert result.holdout["n_train"] == 240
    assert result.holdout["n_test"] == 80


def _baseline_json(tmp_path):
    """Baseline LR de mentira, como nos testes de CLI."""
    base = tmp_path / "baseline.json"
    base.write_text(json.dumps({"metrics_logreg": {
        "accuracy": 0.5, "f1_macro": 0.5, "mae": 0.7, "rmse": 1.1}}), encoding="utf-8")
    return base


def test_xgboost_tfidf_com_comparacao_na_base_completa(article_csvs, tmp_path):
    train_csv, test_csv = article_csvs
    payload = run_xgboost_tfidf(
        train_csv, test_csv, tmp_path / "proc", n_iter=4, baseline_json=_baseline_json(tmp_path)
    )
    assert payload["model"] == "xgboost"
    assert payload["features"] == "tfidf (Issue 6)"
    assert payload["params_note"]
    assert set(payload["comparison_with_baseline"]["delta_xgb_minus_logreg"]) == {
        "accuracy", "f1_macro", "mae", "rmse"}


def test_xgboost_tfidf_modo_rapido_nao_compara(article_csvs, tmp_path):
    train_csv, test_csv = article_csvs
    payload = run_xgboost_tfidf(
        train_csv, test_csv, tmp_path / "proc", sample_n=100, baseline_json=_baseline_json(tmp_path)
    )
    assert payload["sample_n"] == 100
    assert payload["comparison_with_baseline"] is None
