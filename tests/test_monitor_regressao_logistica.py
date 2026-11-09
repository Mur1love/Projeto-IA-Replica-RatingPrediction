"""monitor_regressao_logistica.py: fases, %, tempo e ETA a partir de um log simulado."""
import pytest

from scripts.monitor_regressao_logistica import (
    Estado,
    _bar,
    _hhmmss,
    _num,
    _pct,
    interpretar_log,
    render,
)

TOTAL_BUSCA = 250       # 50 candidatos x 5 folds
TOTAL_AVALIACAO = 6     # 5 folds + holdout
DADOS = "Dados: 51465 linhas (41172 de train.csv + 10293 de test.csv), 58 features"


def linha_end(cand, fold, tempo="5.3s"):
    """Marca de fit concluído no formato verbose do scikit-learn ("5.3s" ou "1.2min")."""
    return f"[CV {fold}/5; {cand}/50] END C=0.01, solver=lbfgs;, score=-1.495 total time=  {tempo}"


def linha_done(n, *, elapsed="0:01:00"):
    """Marca do joblib puro (fase de avaliação)."""
    return f"Parallel(n_jobs=-1): Done {n} tasks | elapsed: {elapsed} remaining: 5.0s"


def test_sem_log_aguarda():
    e = interpretar_log([], TOTAL_BUSCA, TOTAL_AVALIACAO)
    assert e.fase == "aguardando"
    t = render(e, TOTAL_BUSCA, TOTAL_AVALIACAO, "x.log", monitor_s=7.0)
    assert "aguardando" in t and "0:00:07" in t


def test_fase_dados_sem_busca():
    e = interpretar_log([DADOS], TOTAL_BUSCA, TOTAL_AVALIACAO)
    assert e.fase == "dados" and not e.busca_iniciada
    t = render(e, TOTAL_BUSCA, TOTAL_AVALIACAO, "x.log")
    assert "protocolo 0,0%" in t


def test_primeiro_lote_sem_end():
    linhas = [DADOS, "Fitting 5 folds for each of 50 candidates, totalling 250 fits"]
    e = interpretar_log(linhas, TOTAL_BUSCA, TOTAL_AVALIACAO)
    assert e.fase == "busca" and e.busca_iniciada and e.feitos_busca == 0
    t = render(e, TOTAL_BUSCA, TOTAL_AVALIACAO, "x.log", busca_ancora_s=90.0)
    assert "0/250" in t and "nenhum fit concluído" in t and "0:01:30" in t


def test_progresso_da_busca_com_media_e_eta():
    linhas = [
        DADOS,
        "Fitting 5 folds for each of 50 candidates, totalling 250 fits",
        linha_end(1, 1),          # 5.3 s
        linha_end(1, 2),          # 5.3 s
        linha_end(2, 1, tempo="1.2min"),   # 72.0 s
    ]
    e = interpretar_log(linhas, TOTAL_BUSCA, TOTAL_AVALIACAO)
    assert e.fase == "busca"
    assert (e.dados or "").startswith("Dados:")
    assert e.feitos_busca == 3 and e.feitos_avaliacao == 0
    assert e.tempo_medio_fit_s == pytest.approx((5.3 + 5.3 + 72.0) / 3)
    t = render(e, TOTAL_BUSCA, TOTAL_AVALIACAO, "x.log", busca_ancora_s=120.0, workers=10)
    assert "1.2%" in t                       # busca 3/250 e protocolo 3/256 arredondam iguais
    assert "fit médio" in t and "ETA protocolo" in t
    assert "ritmo" in t and "1,5 fits/min" in t   # 3 fits / 2 min


def test_avaliacao_apos_fim_do_contador():
    linhas = [
        DADOS,
        "Fitting 5 folds for each of 50 candidates, totalling 250 fits",
        *(linha_end(i, i) for i in range(1, TOTAL_BUSCA + 1)),   # 250 ENDs
        linha_done(1, elapsed="0:00:02"),
        linha_done(4, elapsed="0:00:07"),
    ]
    e = interpretar_log(linhas, TOTAL_BUSCA, TOTAL_AVALIACAO)
    assert e.fase == "avaliacao"
    assert e.feitos_busca == TOTAL_BUSCA and e.feitos_avaliacao == 4
    assert e.tempo_avaliacao_s == 7.0
    t = render(e, TOTAL_BUSCA, TOTAL_AVALIACAO, "x.log", workers=10)
    assert "busca 100%" in t and "protocolo 99" in t


def test_done_marca_avaliacao_mesmo_sem_busca_completa_no_monitor():
    """Monitor iniciado no meio: a linha Done basta para classificar a avaliação."""
    e = interpretar_log([linha_done(3, elapsed="0:00:05")], TOTAL_BUSCA, TOTAL_AVALIACAO)
    assert e.fase == "avaliacao" and e.feitos_avaliacao == 3


def test_conclusao_com_resumo_e_tempos():
    linhas = [
        DADOS,
        "Fitting 5 folds for each of 50 candidates, totalling 250 fits",
        "Candidatos avaliados: 50 (n_iter pedido: 300) | melhores parâmetros: {'C': 100}",
        "MAE 1.0149 ± 0.0043  IC95% [1.0109, 1.0190]",
        "summary -> results/metrics/logistic_regression_rq1.json",
    ]
    e = interpretar_log(linhas, TOTAL_BUSCA, TOTAL_AVALIACAO)
    assert e.fase == "concluido"
    assert len(e.resultados) == 3
    t = render(e, TOTAL_BUSCA, TOTAL_AVALIACAO, "x.log", busca_ancora_s=1800.0)
    assert "concluído" in t and "100%" in t and "0:30:00" in t


def test_bar_e_helpers():
    assert _bar(100.0, 4) == "████" and _bar(0.0, 4) == "░░░░"
    assert _bar(50.0, 4).endswith("░░")
    assert _pct(127, 256) == pytest.approx(49.6, abs=0.1)
    assert _pct(9, 6) == 100.0 and _pct(0, 0) == 0.0
    assert _hhmmss(1809.0) == "0:30:09"
    assert _num(12.546) == "12,5"


def test_estado_default():
    e = Estado()
    assert e.fase == "aguardando" and e.feitos_busca == 0 and not e.busca_iniciada
    assert e.tempo_medio_fit_s == 0.0
