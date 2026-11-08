"""Testes do Gradient Boosting (RQ1): grid, estimador, protocolo e comparação com TF-IDF.

Usam CSVs sintéticos pequenos (sem depender de ``data/raw``) e grids reduzidos,
então rodam em poucos segundos em qualquer máquina.
"""
from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC

import src.models.gradient_boosting as gb
from scripts import train_article_model, train_gradient_boosting_tfidf
from src.models.article_protocol import (
    N_FEATURES,
    diagnostic_scores,
    load_article_data,
    positive_score,
    run_article_protocol,
    save_article_results,
)
from src.models.gradient_boosting import (
    ARTICLE_TABLE3,
    GRADIENT_BOOSTING_SPEC,
    PARAM_GRID,
    GradientBoostingProbaScores,
    build_gradient_boosting,
    grid_size,
    run_gradient_boosting_tfidf,
)

RAIZ = Path(__file__).resolve().parents[1]
PREFIXOS = ("str_", "pos_", "synt_", "lex_", "conc_", "twt_", "sbj_")

_POSITIVAS = ["otimo", "adorei", "recomendo", "perfeito", "excelente"]
_NEUTRAS = ["normal", "razoavel", "mediano", "regular", "comum"]
_NEGATIVAS = ["pessimo", "horrivel", "ruim", "decepcao", "lixo"]


# --- dados sintéticos ---------------------------------------------------------
def _csv_sintetico(n_por_grupo: int, seed: int) -> pd.DataFrame:
    """2 categorias x 5 notas x ``n_por_grupo`` linhas, com 58 features e texto."""
    rng = np.random.RandomState(seed)
    linhas = []
    for categoria in ("auto", "pets"):
        for rating in (1, 2, 3, 4, 5):
            for _ in range(n_por_grupo):
                palavras = _NEGATIVAS if rating <= 2 else _NEUTRAS if rating == 3 else _POSITIVAS
                linha = {
                    "categoria": categoria,
                    "text": " ".join(rng.choice(palavras, 3)),
                    "rating": rating,
                }
                for i in range(N_FEATURES):
                    linha[f"{PREFIXOS[i % len(PREFIXOS)]}f{i}"] = rating + rng.randn()
                linhas.append(linha)
    return pd.DataFrame(linhas)


@pytest.fixture
def csvs(tmp_path):
    train, test = tmp_path / "train.csv", tmp_path / "test.csv"
    _csv_sintetico(10, seed=0).to_csv(train, index=False)
    _csv_sintetico(4, seed=1).to_csv(test, index=False)
    return train, test


@pytest.fixture
def spec_pequeno():
    """Mesmo estimador do GBT, mas com um grid minúsculo para os testes."""
    grid = {"n_estimators": [5, 10], "max_depth": [2, 3], "max_features": ["sqrt", None]}
    return dataclasses.replace(GRADIENT_BOOSTING_SPEC, param_distributions=grid)


# --- grid e referência do artigo ---------------------------------------------
def test_grid_igual_ao_do_artigo():
    assert PARAM_GRID == {
        "n_estimators": [100, 200, 300, 500, 1000],
        "learning_rate": [0.001, 0.01, 0.05, 0.1],
        "max_depth": [3, 5, 10],
        "min_samples_split": [5, 10, 15],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", 0.5, None],
    }


def test_grid_tem_2160_combinacoes():
    assert grid_size() == 5 * 4 * 3 * 3 * 3 * 4 == 2160


def test_referencia_da_tabela_3():
    assert ARTICLE_TABLE3 == {
        "mae": (0.9612, 0.0061),
        "rmse": (1.3727, 0.0051),
        "auc": (0.7318, 0.0058),
    }


def test_spec_registrado_com_nome_e_referencia():
    assert GRADIENT_BOOSTING_SPEC.name == "gradient_boosting"
    assert GRADIENT_BOOSTING_SPEC.article_reference == ARTICLE_TABLE3
    assert dict(GRADIENT_BOOSTING_SPEC.param_distributions) == PARAM_GRID


# --- estimador ----------------------------------------------------------------
def test_estimador_e_gradient_boosting_com_seed_42():
    modelo = build_gradient_boosting()
    assert isinstance(modelo, GradientBoostingClassifier)
    assert modelo.random_state == 42
    assert build_gradient_boosting(7).random_state == 7


def test_estimadores_sao_independentes():
    assert build_gradient_boosting() is not build_gradient_boosting()


def _xy(n=300, seed=0):
    rng = np.random.RandomState(seed)
    return rng.rand(n, 6), rng.randint(0, 5, n)


def test_previsoes_iguais_as_do_gbt_padrao():
    X, y = _xy()
    nosso = build_gradient_boosting().set_params(n_estimators=15, max_depth=3).fit(X, y)
    padrao = GradientBoostingClassifier(random_state=42, n_estimators=15, max_depth=3).fit(X, y)
    assert (nosso.predict(X) == padrao.predict(X)).all()


def test_diagnostico_usa_probabilidades_4_e_5_estrelas():
    X, y = _xy()
    modelo = clone(build_gradient_boosting()).set_params(n_estimators=15).fit(X, y)
    assert modelo.diagnostic_from_proba is True  # sobrevive ao clone
    esperado = modelo.predict_proba(X)[:, [3, 4]].sum(axis=1)
    assert np.allclose(diagnostic_scores(modelo, X), esperado)


def test_gbt_padrao_continua_usando_decision_function():
    X, y = _xy()
    padrao = GradientBoostingClassifier(random_state=0, n_estimators=10).fit(X, y)
    esperado = positive_score(padrao.decision_function(X), padrao.classes_)
    assert np.allclose(diagnostic_scores(padrao, X), esperado)


def test_gancho_nao_muda_svm_nem_random_forest():
    X, y = _xy()
    svc = SVC(kernel="linear").fit(X, y)
    assert np.allclose(
        diagnostic_scores(svc, X), positive_score(svc.decision_function(X), svc.classes_)
    )
    rf = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
    assert np.allclose(diagnostic_scores(rf, X), rf.predict_proba(X)[:, [3, 4]].sum(axis=1))


def test_gancho_so_vale_se_o_modelo_tem_predict_proba():
    class SoDecisao:
        classes_ = np.arange(5)
        diagnostic_from_proba = True

        def decision_function(self, X):
            return np.tile(np.arange(5.0), (len(X), 1))

    scores = diagnostic_scores(SoDecisao(), np.zeros((3, 2)))
    assert np.allclose(scores, 4.0 - 2.0)  # max(positivas) - max(negativas)


# --- protocolo do artigo com o GBT -------------------------------------------
def test_protocolo_completo_com_gbt(csvs, spec_pequeno, tmp_path):
    data = load_article_data(*csvs)
    assert data.n_features == N_FEATURES
    resultado = run_article_protocol(spec_pequeno, data, n_iter=3, n_splits=2, n_jobs=1)

    assert resultado.n_candidates == 3
    assert set(resultado.best_params) <= set(spec_pequeno.param_distributions)
    assert len(resultado.fold_metrics) == 2
    for metrica in ("mae", "rmse", "f1_macro", "accuracy", "auc", "auc_scores"):
        assert metrica in resultado.aggregate
        assert np.isfinite(resultado.aggregate[metrica]["mean"])
    assert resultado.holdout is not None
    assert resultado.holdout["n_train"] == data.n_train_rows
    assert resultado.holdout["n_test"] == data.n_test_rows
    assert (resultado.oof_pred >= 0).all()  # toda linha foi prevista fora-da-dobra

    caminhos = save_article_results(resultado, data, tmp_path / "saida")
    assert caminhos["summary"].name == "gradient_boosting_rq1.json"
    assert caminhos["folds"].name == "gradient_boosting_rq1_folds.csv"
    assert caminhos["confusion_matrix"].name == "confusion_matrix_gradient_boosting.png"
    for caminho in caminhos.values():
        assert caminho.is_file()
    resumo = json.loads(caminhos["summary"].read_text(encoding="utf-8"))
    assert set(resumo["article"]["comparison"]) == {"mae", "rmse", "auc"}
    assert resumo["article"]["comparison"]["mae"]["article_mean"] == 0.9612


def test_protocolo_e_reprodutivel(csvs, spec_pequeno):
    data = load_article_data(*csvs)
    a = run_article_protocol(spec_pequeno, data, n_iter=3, n_splits=2, n_jobs=1)
    b = run_article_protocol(spec_pequeno, data, n_iter=3, n_splits=2, n_jobs=1)
    assert a.best_params == b.best_params
    pd.testing.assert_frame_equal(a.fold_metrics, b.fold_metrics)


def test_protocolo_nao_remove_linhas_de_texto_vazio(csvs, spec_pequeno):
    train, test = csvs
    df = pd.read_csv(train)
    df.loc[0, "text"] = np.nan
    df.to_csv(train, index=False)
    data = load_article_data(train, test)
    assert data.n_empty_text == 1
    assert data.n_rows == len(df) + len(pd.read_csv(test))


# --- comparação com TF-IDF ----------------------------------------------------
def test_tfidf_sem_busca(csvs, tmp_path):
    train, test = csvs
    r = run_gradient_boosting_tfidf(
        train, test, tmp_path / "tfidf", search=False, params={"n_estimators": 5}
    )
    assert r["model"] == "gradient_boosting"
    assert r["params"]["n_estimators"] == 5
    assert r["params"]["max_depth"] == gb.TFIDF_GBT_FIXED_PARAMS["max_depth"]
    assert r["n_candidates_evaluated"] == 1 and r["n_iter_requested"] is None
    assert r["n_train"] == 100 and r["n_test"] == 40
    for metrica in ("mae", "rmse", "accuracy", "f1_macro", "auc", "auc_scores"):
        assert np.isfinite(r["metrics"][metrica])
    assert r["comparison_with_baseline"] is None


def test_tfidf_com_busca_curta(csvs, tmp_path, monkeypatch):
    monkeypatch.setattr(gb, "TFIDF_GBT_GRID", {"n_estimators": [5, 10], "max_depth": [2, 3]})
    train, test = csvs
    r = run_gradient_boosting_tfidf(train, test, tmp_path / "tfidf", n_iter=2, n_jobs=1)
    assert r["n_candidates_evaluated"] == 2 and r["n_iter_requested"] == 2
    assert set(r["params"]) == {"n_estimators", "max_depth"}
    assert "somente no train.csv" in r["params_note"]


def test_tfidf_modo_rapido_usa_amostra(csvs, tmp_path):
    train, test = csvs
    r = run_gradient_boosting_tfidf(
        train, test, tmp_path / "tfidf", sample_n=100, search=False,
        params={"n_estimators": 5},
    )
    assert r["sample_n"] == 100
    assert r["n_train"] == 100  # só há 100 linhas; a amostra não passa disso
    assert r["n_test"] == 25


def test_tfidf_rejeita_amostra_pequena_demais(csvs, tmp_path):
    train, test = csvs
    with pytest.raises(ValueError, match="sample_n"):
        run_gradient_boosting_tfidf(train, test, tmp_path / "tfidf", sample_n=50)


def test_comparacao_com_baseline(tmp_path):
    baseline = tmp_path / "baseline_logreg.json"
    baseline.write_text(json.dumps({"metrics_logreg": {
        "accuracy": 0.50, "f1_macro": 0.48, "mae": 0.70, "rmse": 1.00,
    }}), encoding="utf-8")
    payload = {"sample_n": None, "metrics": {
        "accuracy": 0.45, "f1_macro": 0.40, "mae": 0.90, "rmse": 1.30,
    }}
    comp = gb.compare_with_baseline(payload, baseline)
    assert comp["delta_gbt_minus_logreg"]["accuracy"] == pytest.approx(-0.05)
    assert comp["delta_gbt_minus_logreg"]["mae"] == pytest.approx(0.20)
    assert comp["logreg"]["rmse"] == 1.00 and comp["gradient_boosting"]["rmse"] == 1.30


def test_comparacao_ausente_no_modo_rapido_ou_sem_baseline(tmp_path):
    baseline = tmp_path / "baseline_logreg.json"
    baseline.write_text(json.dumps({"metrics_logreg": {"accuracy": 0.5}}), encoding="utf-8")
    metricas = {"accuracy": 0.4}
    assert gb.compare_with_baseline({"sample_n": 2000, "metrics": metricas}, baseline) is None
    assert gb.compare_with_baseline({"sample_n": None, "metrics": metricas}, tmp_path / "x.json") is None
    assert gb.compare_with_baseline({"sample_n": None, "metrics": metricas}, None) is None


# --- scripts ------------------------------------------------------------------
def test_script_do_protocolo_registra_o_gbt():
    assert train_article_model.MODELS["gradient_boosting"] is GRADIENT_BOOSTING_SPEC
    args = train_article_model.build_parser().parse_args(["--model", "gradient_boosting"])
    assert args.model == "gradient_boosting"
    quick = train_article_model.QUICK_N_ITER
    assert quick["gradient_boosting"] < quick["random_forest"]


def test_script_do_protocolo_usa_n_iter_pequeno_no_modo_rapido(csvs, monkeypatch):
    capturado = {}

    def falso_protocolo(spec, data, **kwargs):
        capturado.update(kwargs)
        raise RuntimeError("parar aqui")

    monkeypatch.setattr(train_article_model, "run_article_protocol", falso_protocolo)
    train, test = csvs
    base = ["--model", "gradient_boosting", "--train", str(train), "--test", str(test)]

    with pytest.raises(RuntimeError):
        train_article_model.main(base + ["--sample-n", "60"])
    assert capturado["n_iter"] == train_article_model.QUICK_N_ITER["gradient_boosting"]

    with pytest.raises(RuntimeError):
        train_article_model.main(base + ["--sample-n", "60", "--n-iter", "2"])
    assert capturado["n_iter"] == 2  # o argumento explícito vence

    with pytest.raises(RuntimeError):
        train_article_model.main(base)
    assert capturado["n_iter"] == 300  # base completa: padrão do artigo


def test_cli_gbt_gera_os_artefatos(csvs, spec_pequeno, tmp_path, monkeypatch):
    # Grid reduzido: o do artigo (até 1000 árvores com profundidade 10) é lento demais.
    monkeypatch.setitem(train_article_model.MODELS, "gradient_boosting", spec_pequeno)
    train, test = csvs
    out = tmp_path / "saida"
    rc = train_article_model.main([
        "--model", "gradient_boosting", "--train", str(train), "--test", str(test),
        "--n-iter", "2", "--n-splits", "2", "--n-jobs", "1", "--output-dir", str(out),
    ])
    assert rc == 0
    resumo = json.loads((out / "metrics" / "gradient_boosting_rq1.json").read_text("utf-8"))
    assert resumo["model"] == "gradient_boosting"
    assert resumo["protocol"]["n_iter_requested"] == 2
    assert resumo["data"]["n_rows"] == 140 and resumo["data"]["n_features"] == 58
    assert "auc_scores" in resumo["metrics"]
    for nome in ("metrics/gradient_boosting_rq1_folds.csv",
                 "metrics/gradient_boosting_rq1_holdout.json",
                 "figures/confusion_matrix_gradient_boosting.png"):
        assert (out / nome).stat().st_size > 0


def test_cli_gbt_modo_rapido_com_sample_n(csvs, spec_pequeno, tmp_path, monkeypatch):
    monkeypatch.setitem(train_article_model.MODELS, "gradient_boosting", spec_pequeno)
    train, test = csvs
    out = tmp_path / "rapido"
    rc = train_article_model.main([
        "--model", "gradient_boosting", "--train", str(train), "--test", str(test),
        "--sample-n", "100", "--n-splits", "2", "--n-jobs", "1", "--output-dir", str(out),
    ])
    assert rc == 0
    resumo = json.loads((out / "metrics" / "gradient_boosting_rq1.json").read_text("utf-8"))
    assert resumo["data"]["sample_n"] == 100 and resumo["data"]["n_rows"] == 100
    assert resumo["protocol"]["n_iter_requested"] == train_article_model.QUICK_N_ITER["gradient_boosting"]


def test_cli_tfidf_valida_argumentos(csvs, capsys):
    train, test = csvs
    with pytest.raises(SystemExit):
        train_gradient_boosting_tfidf.main(
            ["--train", str(train), "--test", str(test), "--sample-n", "10"])
    assert "--sample-n" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        train_gradient_boosting_tfidf.main(["--train", "nao_existe.csv", "--test", str(test)])
    assert "não encontrado" in capsys.readouterr().err


def test_cli_tfidf_modo_rapido_nao_compara(csvs, tmp_path):
    train, test = csvs
    saida = tmp_path / "saida"
    codigo = train_gradient_boosting_tfidf.main([
        "--train", str(train), "--test", str(test), "--processed-dir", str(tmp_path / "proc"),
        "--output-dir", str(saida), "--sample-n", "100", "--no-search",
    ])
    assert codigo == 0
    dados = json.loads((saida / "metrics" / "gradient_boosting_tfidf.json").read_text("utf-8"))
    assert dados["sample_n"] == 100 and dados["comparison_with_baseline"] is None
    assert (tmp_path / "proc" / "tfidf_train.npz").is_file()  # amostra as matrizes completas


def test_cli_tfidf_com_comparacao_na_base_completa(csvs, tmp_path):
    train, test = csvs
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"metrics_logreg": {
        "accuracy": 0.5, "f1_macro": 0.5, "mae": 0.7, "rmse": 1.1}}), encoding="utf-8")
    saida = tmp_path / "saida"
    codigo = train_gradient_boosting_tfidf.main([
        "--train", str(train), "--test", str(test), "--processed-dir", str(tmp_path / "proc"),
        "--output-dir", str(saida), "--baseline-json", str(baseline), "--no-search",
    ])
    assert codigo == 0
    dados = json.loads((saida / "metrics" / "gradient_boosting_tfidf.json").read_text("utf-8"))
    assert dados["n_candidates_evaluated"] == 1
    assert set(dados["comparison_with_baseline"]["delta_gbt_minus_logreg"]) == {
        "accuracy", "f1_macro", "mae", "rmse"}


def test_cli_tfidf_roda_como_script_pela_raiz(csvs, tmp_path):
    train, test = csvs
    rc = subprocess.run(
        [sys.executable, "scripts/train_gradient_boosting_tfidf.py",
         "--train", str(train), "--test", str(test), "--processed-dir", str(tmp_path / "proc"),
         "--output-dir", str(tmp_path / "saida"), "--sample-n", "100", "--no-search"],
        capture_output=True, text=True, timeout=300, cwd=RAIZ,
    )
    assert rc.returncode == 0, rc.stderr
    assert (tmp_path / "saida" / "metrics" / "gradient_boosting_tfidf.json").exists()
