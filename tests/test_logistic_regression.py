"""Regressão Logística da RQ1: grid do artigo, referências e execução no protocolo."""
import json
import warnings

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.models.article_protocol import (
    ModelSpec,
    load_article_data,
    run_article_protocol,
    save_article_results,
)
from src.models.logistic_regression import (
    ARTICLE_BEST_PARAMS,
    ARTICLE_TABLE3,
    LOGISTIC_REGRESSION_SPEC,
    PARAM_GRID,
    build_logistic_regression,
)

# Grid pequeno para os testes ficarem rápidos; o grid real é testado à parte.
GRID_PEQUENO = {"C": [0.1], "solver": ["lbfgs", "liblinear"], "class_weight": [None]}
SPEC_PEQUENO = ModelSpec(
    "logistic_regression", build_logistic_regression, GRID_PEQUENO,
    ARTICLE_TABLE3, ARTICLE_BEST_PARAMS,
)


@pytest.fixture(scope="module")
def dados(tmp_path_factory):
    from tests.conftest import make_article_frame

    pasta = tmp_path_factory.mktemp("logistic_regression")
    make_article_frame(24, seed=1).to_csv(pasta / "train.csv", index=False)
    make_article_frame(8, seed=2).to_csv(pasta / "test.csv", index=False)
    return load_article_data(pasta / "train.csv", pasta / "test.csv")


@pytest.fixture(scope="module")
def resultado(dados):
    return run_article_protocol(SPEC_PEQUENO, dados, n_iter=2, n_splits=5, seed=42)


# --- configuração do artigo --------------------------------------------------
def test_grid_e_identico_ao_do_artigo():
    assert PARAM_GRID == {
        "C": [0.01, 0.1, 1, 10, 100],
        "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
        "class_weight": [None, "balanced"],
        "random_state": [None],
        "max_iter": [10000],
    }
    assert LOGISTIC_REGRESSION_SPEC.param_distributions == PARAM_GRID


def test_grid_tem_50_combinacoes():
    assert len(PARAM_GRID["C"]) * len(PARAM_GRID["solver"]) * len(PARAM_GRID["class_weight"]) \
        * len(PARAM_GRID["random_state"]) * len(PARAM_GRID["max_iter"]) == 50


def test_referencia_da_tabela_3_do_artigo():
    assert LOGISTIC_REGRESSION_SPEC.article_reference == {
        "mae": (1.0149, 0.0043), "rmse": (1.4691, 0.0042), "auc": (0.7328, 0.0064),
    }
    assert LOGISTIC_REGRESSION_SPEC.article_best_params == {
        "solver": "lbfgs", "max_iter": 10000, "class_weight": None, "C": 100,
    }


def test_estimador_e_logisticregression_sem_padronizacao():
    modelo = LOGISTIC_REGRESSION_SPEC.build_estimator(42)
    assert isinstance(modelo, LogisticRegression)   # sem Pipeline/StandardScaler, como no artigo
    assert modelo.get_params()["random_state"] is None   # o grid do artigo fixa None


def test_n_iter_300_satura_o_grid(dados):
    """n_iter=300 >= grid -> busca exaustiva; no spec de teste, os 2 candidatos."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = run_article_protocol(SPEC_PEQUENO, dados, n_iter=300, n_splits=5, seed=42)
    assert r.config["n_iter_requested"] == 300
    assert r.n_candidates == 2   # no grid real do artigo, seriam os 50 (test_grid_tem_50_combinacoes)


# --- runner ----------------------------------------------------------------
def test_um_resultado_por_fold_com_todas_as_metricas(resultado):
    df = resultado.fold_metrics
    assert len(df) == 5 and df["fold"].tolist() == [1, 2, 3, 4, 5]
    for m in ("mae", "rmse", "f1_macro", "accuracy", "auc", "auc_scores"):
        assert df[m].notna().all()
    assert (df["n_train"] + df["n_test"] == 320).all()


def test_melhores_parametros_vem_do_grid(resultado):
    """Os melhores parâmetros vêm só das chaves do grid de teste (C, solver, class_weight)."""
    assert set(resultado.best_params) >= {"C", "solver", "class_weight"}
    assert resultado.best_params["solver"] in GRID_PEQUENO["solver"]
    assert resultado.best_params["C"] in GRID_PEQUENO["C"]
    assert resultado.n_candidates == 2


def test_liblinear_multiclasse_falha_no_sklearn_atual(dados):
    """`liblinear` só faz binário desde a remoção do OvR (sklearn >= 1.8): 10 dos 50
    candidatos do grid real falham (score NaN) e o ranking os ignora. Diferença
    registrada em docs/diferencas.md."""
    modelo = LogisticRegression(C=0.1, solver="liblinear", max_iter=10000)
    with pytest.raises(ValueError, match="liblinear"):
        modelo.fit(dados.X.to_numpy(), dados.y)


def test_predicoes_fora_da_dobra_cobrem_todas_as_linhas(resultado, dados):
    assert (resultado.oof_pred >= 0).all() and len(resultado.oof_pred) == dados.n_rows
    assert resultado.confusion.shape == (5, 5) and resultado.confusion.sum() == dados.n_rows


def test_agregado_bate_com_a_media_dos_folds(resultado):
    mae = resultado.fold_metrics["mae"]
    assert resultado.aggregate["mae"]["mean"] == pytest.approx(mae.mean())
    assert resultado.aggregate["mae"]["std"] == pytest.approx(mae.std(ddof=1))
    low, high = resultado.aggregate["mae"]["ci95"]
    assert low < mae.mean() < high


def test_holdout_treina_no_treino_e_avalia_no_teste(resultado, dados):
    h = resultado.holdout
    assert h["n_train"] == dados.n_train_rows == 240
    assert h["n_test"] == dados.n_test_rows == 80
    assert set(h["metrics"]) >= {"mae", "rmse", "auc"}


def test_execucao_e_reprodutivel(dados, resultado):
    outra = run_article_protocol(SPEC_PEQUENO, dados, n_iter=2, n_splits=5, seed=42)
    assert outra.best_params == resultado.best_params
    pd.testing.assert_frame_equal(outra.fold_metrics, resultado.fold_metrics)


# --- saída ------------------------------------------------------------------
def test_save_article_results_grava_os_quatro_artefatos(resultado, dados, tmp_path):
    paths = save_article_results(resultado, dados, tmp_path)
    assert paths["summary"].name == "logistic_regression_rq1.json"
    assert paths["folds"].name == "logistic_regression_rq1_folds.csv"
    assert paths["holdout"].name == "logistic_regression_rq1_holdout.json"
    assert paths["confusion_matrix"].name == "confusion_matrix_logistic_regression.png"
    assert all(p.stat().st_size > 0 for p in paths.values())

    resumo = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert resumo["protocol"]["n_iter_requested"] == 2
    assert resumo["data"]["n_features"] == 58 and "text" in resumo["data"]["excluded_columns"]
    assert resumo["data"]["empty_text_rows_kept"] == 0
    comp = resumo["article"]["comparison"]["mae"]
    assert comp["article_mean"] == 1.0149
    assert comp["delta_mean"] == pytest.approx(comp["replica_mean"] - 1.0149)

    folds = pd.read_csv(paths["folds"])
    assert list(folds.columns[:3]) == ["fold", "n_train", "n_test"] and len(folds) == 5
    holdout = json.loads(paths["holdout"].read_text(encoding="utf-8"))
    assert holdout["n_train"] == 240 and holdout["n_test"] == 80
