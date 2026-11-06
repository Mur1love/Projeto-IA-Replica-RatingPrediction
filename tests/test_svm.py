"""SVM da RQ1: grid do artigo, runner do protocolo, saída e comparação TF-IDF."""
import json
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC

from src.models.article_protocol import (
    ModelSpec,
    load_article_data,
    run_article_protocol,
    save_article_results,
)
from src.models.svm import (
    ARTICLE_BEST_PARAMS,
    ARTICLE_TABLE3,
    PARAM_GRID,
    SVM_SPEC,
    TFIDF_SVM_PARAMS,
    build_svm,
    compare_with_baseline,
    ensure_tfidf,
    run_svm_tfidf,
)

# Grid pequeno para os testes ficarem rápidos; o grid real é testado à parte.
GRID_PEQUENO = {"C": [0.1], "kernel": ["linear", "rbf"], "gamma": ["scale"]}
SPEC_PEQUENO = ModelSpec("svm", build_svm, GRID_PEQUENO, ARTICLE_TABLE3, ARTICLE_BEST_PARAMS)


@pytest.fixture(scope="module")
def dados(tmp_path_factory):
    from tests.conftest import make_article_frame

    pasta = tmp_path_factory.mktemp("svm")
    make_article_frame(24, seed=1).to_csv(pasta / "train.csv", index=False)
    make_article_frame(8, seed=2).to_csv(pasta / "test.csv", index=False)
    return load_article_data(pasta / "train.csv", pasta / "test.csv")


@pytest.fixture(scope="module")
def resultado(dados):
    return run_article_protocol(SPEC_PEQUENO, dados, n_iter=2, n_splits=5, seed=42)


# --- configuração do artigo --------------------------------------------------
def test_grid_e_identico_ao_do_artigo():
    assert PARAM_GRID == {
        "C": [0.01, 0.1],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto", 0.01, 0.1],
    }
    assert SVM_SPEC.param_distributions == PARAM_GRID


def test_referencia_da_tabela_3_do_artigo():
    assert SVM_SPEC.article_reference == {
        "mae": (1.0431, 0.0070), "rmse": (1.5085, 0.0118), "auc": (0.7304, 0.0046),
    }
    assert SVM_SPEC.article_best_params == {"kernel": "linear", "gamma": "scale", "C": 0.1}


def test_estimador_e_svc_padrao_sem_padronizacao():
    modelo = SVM_SPEC.build_estimator(42)
    assert isinstance(modelo, SVC)             # sem Pipeline/StandardScaler, como no artigo
    assert modelo.get_params()["C"] == 1.0 and modelo.get_params()["kernel"] == "rbf"


def test_n_iter_300_equivale_a_16_candidatos(dados):
    """O grid tem 2x2x4=16 combinações: n_iter=300 vira busca exaustiva de 16."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = run_article_protocol(SVM_SPEC, dados, n_iter=300, n_splits=5, seed=42)
    assert r.config["n_iter_requested"] == 300
    assert r.n_candidates == 16


# --- runner ----------------------------------------------------------------
def test_um_resultado_por_fold_com_todas_as_metricas(resultado):
    df = resultado.fold_metrics
    assert len(df) == 5 and df["fold"].tolist() == [1, 2, 3, 4, 5]
    for m in ("mae", "rmse", "f1_macro", "accuracy", "auc", "auc_scores"):
        assert df[m].notna().all()
    assert (df["n_train"] + df["n_test"] == 320).all()


def test_melhores_parametros_vem_do_grid(resultado):
    assert resultado.best_params["kernel"] in GRID_PEQUENO["kernel"]
    assert resultado.n_candidates == 2


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


def test_estimador_so_enxerga_as_58_features(dados):
    """Garante que text/categoria/rating não chegam ao modelo."""
    vistos = {}

    class Espiao(DummyClassifier):
        def fit(self, X, y, sample_weight=None):
            vistos["n_features"] = np.asarray(X).shape[1]
            return super().fit(X, y, sample_weight)

    spec = ModelSpec("dummy", lambda seed: Espiao(), {"strategy": ["prior", "most_frequent"]})
    run_article_protocol(spec, dados, n_iter=2, n_splits=5)
    assert vistos["n_features"] == 58


def test_shuffle_fica_registrado(dados):
    r = run_article_protocol(SPEC_PEQUENO, dados, n_iter=1, n_splits=5, shuffle=True)
    assert r.config["shuffle"] is True
    assert run_article_protocol(SPEC_PEQUENO, dados, n_iter=1).config["shuffle"] is False


# --- saída ------------------------------------------------------------------
def test_save_article_results_grava_os_quatro_artefatos(resultado, dados, tmp_path):
    paths = save_article_results(resultado, dados, tmp_path)
    assert paths["summary"].name == "svm_rq1.json"
    assert paths["folds"].name == "svm_rq1_folds.csv"
    assert paths["holdout"].name == "svm_rq1_holdout.json"
    assert paths["confusion_matrix"].name == "confusion_matrix_svm.png"
    assert all(p.stat().st_size > 0 for p in paths.values())

    resumo = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert resumo["protocol"]["n_splits"] == 5 and resumo["protocol"]["shuffle"] is False
    assert resumo["protocol"]["n_candidates_evaluated"] == 2
    assert resumo["data"]["n_features"] == 58 and "text" in resumo["data"]["excluded_columns"]
    assert resumo["data"]["empty_text_rows_kept"] == 0
    assert set(resumo["metrics"]["mae"]) == {"mean", "std", "sem", "ci95"}
    comp = resumo["article"]["comparison"]["mae"]
    assert comp["article_mean"] == 1.0431
    assert comp["delta_mean"] == pytest.approx(comp["replica_mean"] - 1.0431)

    folds = pd.read_csv(paths["folds"])
    assert list(folds.columns[:3]) == ["fold", "n_train", "n_test"] and len(folds) == 5
    holdout = json.loads(paths["holdout"].read_text(encoding="utf-8"))
    assert holdout["n_train"] == 240 and holdout["n_test"] == 80


# --- comparação com TF-IDF --------------------------------------------------
@pytest.fixture
def tfidf_csvs(tmp_path):
    from tests.conftest import make_article_frame

    make_article_frame(24, seed=1).to_csv(tmp_path / "train.csv", index=False)
    make_article_frame(8, seed=2).to_csv(tmp_path / "test.csv", index=False)
    return tmp_path / "train.csv", tmp_path / "test.csv", tmp_path / "processed"


def test_run_svm_tfidf_usa_split_oficial_e_params_fixos(tfidf_csvs):
    train_csv, test_csv, processed = tfidf_csvs
    r = run_svm_tfidf(train_csv, test_csv, processed)
    assert r["params"] == TFIDF_SVM_PARAMS == {"kernel": "linear", "C": 1.0}
    assert r["n_train"] == 240 and r["n_test"] == 80
    assert r["sample_n"] is None
    assert set(r["metrics"]) >= {"accuracy", "f1_macro", "mae", "rmse", "auc", "auc_scores"}
    assert r["metrics"]["mae"] < 1.0   # o texto sintético é quase determinístico por nota


def test_ensure_tfidf_reaproveita_e_regenera(tfidf_csvs):
    train_csv, test_csv, processed = tfidf_csvs
    ensure_tfidf(train_csv, test_csv, processed)
    arquivo = processed / "tfidf_train.npz"
    marca = arquivo.stat().st_mtime_ns
    ensure_tfidf(train_csv, test_csv, processed)           # parâmetros iguais: reaproveita
    assert arquivo.stat().st_mtime_ns == marca
    stats_path = processed / "tfidf_stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    stats["vectorizer_params"]["min_df"] = 99               # artefato de outra configuração
    stats_path.write_text(json.dumps(stats), encoding="utf-8")
    ensure_tfidf(train_csv, test_csv, processed)           # diferente: regenera
    assert json.loads(stats_path.read_text(encoding="utf-8"))["vectorizer_params"]["min_df"] == 5


def test_run_svm_tfidf_modo_amostra_e_validacoes(tfidf_csvs):
    train_csv, test_csv, processed = tfidf_csvs
    r = run_svm_tfidf(train_csv, test_csv, processed, sample_n=100)
    assert r["n_train"] == 100 and r["n_test"] == 25
    with pytest.raises(ValueError, match="sample_n"):
        run_svm_tfidf(train_csv, test_csv, processed, sample_n=10)


def test_compare_with_baseline_diferenca_svm_menos_lr(tmp_path):
    base = tmp_path / "baseline.json"
    base.write_text(json.dumps({"metrics_logreg": {
        "accuracy": 0.49, "f1_macro": 0.486, "mae": 0.68, "rmse": 1.06}}), encoding="utf-8")
    svm = {"sample_n": None, "metrics": {"accuracy": 0.50, "f1_macro": 0.49, "mae": 0.66,
                                          "rmse": 1.04, "auc": 0.8, "auc_scores": 0.9}}
    comp = compare_with_baseline(svm, base)
    assert comp["delta_svm_minus_logreg"]["mae"] == pytest.approx(-0.02)
    assert comp["delta_svm_minus_logreg"]["accuracy"] == pytest.approx(0.01)
    assert set(comp["logreg"]) == {"accuracy", "f1_macro", "mae", "rmse"}


def test_compare_with_baseline_omite_em_amostra_ou_sem_arquivo(tmp_path):
    svm = {"sample_n": 2000, "metrics": {}}
    assert compare_with_baseline(svm, tmp_path / "nao_existe.json") is None
    base = tmp_path / "b.json"
    base.write_text(json.dumps({"metrics_logreg": {"mae": 1.0}}), encoding="utf-8")
    assert compare_with_baseline(svm, base) is None        # amostra não se compara à base completa
