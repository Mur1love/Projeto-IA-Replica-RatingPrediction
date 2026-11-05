"""Harness do protocolo do artigo (RQ1): features, folds, métricas e agregação."""
import numpy as np
import pandas as pd
import pytest
import scipy.stats as stats
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from src.models.article_protocol import (
    N_FEATURES,
    FOLD_METRICS,
    aggregate_folds,
    auc_binary_predictions,
    build_folds,
    compute_fold_metrics,
    confidence_interval,
    diagnostic_scores,
    load_article_data,
    positive_score,
    select_feature_columns,
    to_binary,
)
from tests.conftest import ARTICLE_FEATURES


# --- features ---------------------------------------------------------------
def test_ha_58_features_com_os_prefixos_do_artigo():
    assert len(ARTICLE_FEATURES) == N_FEATURES == 58


def test_select_feature_columns_exclui_rotulos_e_mantem_a_ordem():
    cols = ["categoria", "text", "rating", *ARTICLE_FEATURES]
    assert select_feature_columns(cols) == ARTICLE_FEATURES


@pytest.mark.parametrize("extra", ["Unnamed: 0", "id", "review_id", "categoria_rating", "text_len"])
def test_coluna_de_identificacao_interrompe_o_processo(extra):
    cols = ["categoria", "text", "rating", extra, *ARTICLE_FEATURES]
    with pytest.raises(ValueError, match="identificação"):
        select_feature_columns(cols)


def test_numero_de_features_diferente_de_58_interrompe():
    with pytest.raises(ValueError, match="58"):
        select_feature_columns(["categoria", "text", "rating", *ARTICLE_FEATURES[:-1]])


def test_load_article_data_so_tem_as_58_features(article_csvs):
    data = load_article_data(*article_csvs)
    assert list(data.X.columns) == ARTICLE_FEATURES
    for proibida in ("text", "categoria", "rating", "categoria_rating"):
        assert proibida not in data.X.columns
    assert data.X.shape == (240 + 80, 58)
    assert data.X.dtypes.eq("float64").all()


def test_load_article_data_concatena_treino_depois_teste(article_csvs):
    train_csv, test_csv = article_csvs
    data = load_article_data(train_csv, test_csv)
    assert data.n_train_rows == 240 and data.n_test_rows == 80
    assert not data.is_test[:240].any() and data.is_test[240:].all()
    treino = pd.read_csv(train_csv)
    np.testing.assert_allclose(data.X.iloc[:240].to_numpy(), treino[ARTICLE_FEATURES].to_numpy())


def test_rotulos_0_a_4_e_chave_categoria_rating(article_csvs):
    data = load_article_data(*article_csvs)
    assert sorted(set(data.y)) == [0, 1, 2, 3, 4]
    assert len(set(data.groups)) == 10
    assert all(g.rsplit("_", 1)[1] in "12345" for g in data.groups)


def test_linhas_com_texto_vazio_sao_mantidas(article_csvs):
    data = load_article_data(*article_csvs)
    assert data.n_rows == 320          # nenhuma linha removida
    assert data.n_empty_text == 1      # a do text ausente da fixture


def test_features_ausentes_interrompem(article_csvs, tmp_path):
    train_csv, test_csv = article_csvs
    df = pd.read_csv(train_csv)
    df.loc[3, ARTICLE_FEATURES[0]] = np.nan
    df.to_csv(train_csv, index=False)
    with pytest.raises(ValueError, match="ausentes"):
        load_article_data(train_csv, test_csv)


def test_colunas_diferentes_entre_treino_e_teste_interrompem(article_csvs):
    train_csv, test_csv = article_csvs
    pd.read_csv(test_csv).drop(columns=[ARTICLE_FEATURES[0]]).to_csv(test_csv, index=False)
    with pytest.raises(ValueError, match="diferentes"):
        load_article_data(train_csv, test_csv)


# --- amostra (--sample-n) ---------------------------------------------------
def test_sample_n_estratifica_e_preserva_a_ordem(article_csvs):
    completo = load_article_data(*article_csvs)
    amostra = load_article_data(*article_csvs, sample_n=100, seed=7)
    assert amostra.n_rows == 100 and amostra.sample_n == 100
    assert np.all(np.diff(amostra.is_test.astype(int)) >= 0)  # treino antes do teste
    prop = pd.Series(amostra.groups).value_counts(normalize=True)
    esperado = pd.Series(completo.groups).value_counts(normalize=True)
    assert (prop - esperado.reindex(prop.index)).abs().max() < 0.03
    assert amostra.n_train_rows > 0 and amostra.n_test_rows > 0


def test_sample_n_e_reprodutivel_e_depende_da_seed(article_csvs):
    a = load_article_data(*article_csvs, sample_n=100, seed=7)
    b = load_article_data(*article_csvs, sample_n=100, seed=7)
    c = load_article_data(*article_csvs, sample_n=100, seed=8)
    pd.testing.assert_frame_equal(a.X, b.X)
    assert not a.X.equals(c.X)


def test_sample_n_maior_que_a_base_devolve_tudo(article_csvs):
    assert load_article_data(*article_csvs, sample_n=10_000).n_rows == 320


# --- folds ------------------------------------------------------------------
def test_folds_particionam_as_linhas(article_csvs):
    data = load_article_data(*article_csvs)
    folds = build_folds(data.groups, 5)
    assert len(folds) == 5
    testes = np.concatenate([te for _, te in folds])
    assert sorted(testes) == list(range(data.n_rows))          # cada linha testa uma vez
    for tr, te in folds:
        assert not set(tr) & set(te)


def test_folds_sao_estratificados_por_categoria_rating(article_csvs):
    data = load_article_data(*article_csvs)
    for _, te in build_folds(data.groups, 5):
        contagens = pd.Series(data.groups[te]).value_counts()
        assert len(contagens) == 10 and contagens.min() >= 6 and contagens.max() <= 7  # 32/5


def test_folds_sem_shuffle_reproduzem_o_codigo_do_artigo(article_csvs):
    data = load_article_data(*article_csvs)
    esperado = list(StratifiedKFold(5, shuffle=False, random_state=None).split(data.X, data.groups))
    obtido = build_folds(data.groups, 5, shuffle=False)
    for (tr_a, te_a), (tr_b, te_b) in zip(esperado, obtido):
        np.testing.assert_array_equal(tr_a, tr_b)
        np.testing.assert_array_equal(te_a, te_b)


def test_folds_com_shuffle_dependem_da_seed(article_csvs):
    g = load_article_data(*article_csvs).groups
    a1, a2 = build_folds(g, 5, shuffle=True, seed=1), build_folds(g, 5, shuffle=True, seed=1)
    b = build_folds(g, 5, shuffle=True, seed=2)
    assert all((x[1] == y[1]).all() for x, y in zip(a1, a2))
    assert not all((x[1] == y[1]).all() for x, y in zip(a1, b))


def test_folds_recusam_grupo_menor_que_n_splits():
    with pytest.raises(ValueError, match="n_splits"):
        build_folds(np.array(["a"] * 10 + ["b"] * 3), 5)


# --- AUC --------------------------------------------------------------------
def test_to_binary_usa_4_e_5_estrelas_como_positivas():
    assert to_binary([0, 1, 2, 3, 4]).tolist() == [0, 0, 0, 1, 1]


def test_auc_do_artigo_predicao_perfeita_e_chute_constante():
    y = np.array([0, 1, 2, 3, 4, 4])
    assert auc_binary_predictions(y, y) == 1.0
    assert auc_binary_predictions(y, np.full(6, 4)) == 0.5   # tudo "positivo"
    assert auc_binary_predictions(y, np.zeros(6, int)) == 0.5  # tudo "negativo"


def test_auc_do_artigo_igual_a_media_de_sensibilidade_e_especificidade():
    # 3 negativos (labels 0,1,2) e 3 positivos (3,4,4); predições binárias:
    y_true = np.array([0, 1, 2, 3, 4, 4])
    y_pred = np.array([0, 3, 2, 3, 3, 1])   # vira: 0,1,0 | 1,1,0
    sens, espec = 2 / 3, 2 / 3
    assert auc_binary_predictions(y_true, y_pred) == pytest.approx((sens + espec) / 2)


def test_erro_entre_classes_do_mesmo_lado_nao_afeta_a_auc():
    y_true = np.array([0, 0, 4, 4])
    assert auc_binary_predictions(y_true, np.array([2, 1, 3, 4])) == 1.0  # 1–3 e 4–5 continuam certos


def test_positive_score_e_auc_por_scores():
    scores = np.array([[0.9, 0.0, 0.0, 0.0, 0.1], [0.0, 0.0, 0.1, 0.8, 0.0]])
    margem = positive_score(scores, [0, 1, 2, 3, 4])
    assert margem[0] < 0 < margem[1]


def test_positive_score_valida_o_formato():
    with pytest.raises(ValueError):
        positive_score(np.zeros((3, 4)), [0, 1, 2, 3, 4])
    with pytest.raises(ValueError, match="positivas e negativas"):
        positive_score(np.zeros((3, 2)), [3, 4])


def test_diagnostic_scores_com_svc_e_sem_scores():
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1, 2, 3, 4], 20)
    X = rng.normal(size=(100, 4)) + y[:, None]
    s = diagnostic_scores(SVC(kernel="linear").fit(X, y), X)
    assert s.shape == (100,)
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(to_binary(y), s) > 0.8
    assert diagnostic_scores(object(), X) is None


# --- métricas por fold ------------------------------------------------------
def test_compute_fold_metrics_em_estrelas_e_com_todas_as_chaves():
    m = compute_fold_metrics([0, 4], [4, 0])      # erra por 4 estrelas nos dois
    assert set(m) == set(FOLD_METRICS)
    assert m["mae"] == 4.0 and m["rmse"] == 4.0 and m["accuracy"] == 0.0
    assert np.isnan(m["auc_scores"])


def test_compute_fold_metrics_valores_conhecidos():
    y_true, y_pred = np.array([0, 1, 2, 3, 4]), np.array([0, 1, 2, 3, 3])
    m = compute_fold_metrics(y_true, y_pred)
    assert m["mae"] == pytest.approx(0.2) and m["rmse"] == pytest.approx(np.sqrt(0.2))
    assert m["accuracy"] == pytest.approx(0.8)


# --- agregação e IC ---------------------------------------------------------
def test_ic_usa_t_com_n_menos_1_graus_de_liberdade():
    v = [1, 2, 3, 4, 5]
    low, high = confidence_interval(v)
    meia = stats.t.ppf(0.975, 4) * np.std(v, ddof=1) / np.sqrt(5)
    assert (low, high) == pytest.approx((3 - meia, 3 + meia))


def test_ic_reproduz_valores_do_artigo_por_fold():
    """MAE por fold publicado em code/RQ1/results/SVC_results.csv."""
    mae = [1.0431, 1.0429, 1.0487, 1.0318, 1.049]
    assert np.mean(mae) == pytest.approx(1.0431, abs=5e-5)
    assert np.std(mae, ddof=1) == pytest.approx(0.0070, abs=5e-5)
    low, high = confidence_interval(mae)
    # IC correto entre 5 folds; o do artigo, [1.0366, 1.0496], é mais estreito
    # porque o código original inclui a linha da média no cálculo.
    assert (round(low, 4), round(high, 4)) == (1.0345, 1.0517)


def test_ic_recusa_menos_de_dois_folds_e_aceita_variancia_zero():
    with pytest.raises(ValueError):
        confidence_interval([1.0])
    assert confidence_interval([2.0, 2.0, 2.0]) == (2.0, 2.0)


def test_aggregate_folds_mean_std_ic():
    df = pd.DataFrame({m: [1.0, 2.0, 3.0, 4.0, 5.0] for m in FOLD_METRICS})
    df["auc_scores"] = np.nan                    # métrica indisponível é omitida
    agg = aggregate_folds(df)
    assert "auc_scores" not in agg
    assert agg["mae"]["mean"] == 3.0
    assert agg["mae"]["std"] == pytest.approx(np.std([1, 2, 3, 4, 5], ddof=1))
    assert agg["mae"]["ci95"][0] < 3.0 < agg["mae"]["ci95"][1]
