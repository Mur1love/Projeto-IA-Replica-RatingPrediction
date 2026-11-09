"""monitor_regressao_logistica.py: fases, %, tempo e estimativas a partir de um log simulado."""
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


def linha_done(n, *, elapsed="0:01:00"):
    """Uma marca de progresso joblib (contadores reais não recuam na fase)."""
    return f"Parallel(n_jobs=-1): Done {n} tasks | elapsed: {elapsed} remaining: 5.0s"


def test_sem_log_aguarda():
    e = interpretar_log([], TOTAL_BUSCA, TOTAL_AVALIACAO)
    assert e.fase == "aguardando"
    t = render(e, TOTAL_BUSCA, TOTAL_AVALIACAO, "x.log", monitor_s=7.0)
    assert "aguardando" in t and "0:00:07" in t


def test_fase_dados_sem_busca():
    linhas = ["Dados: 51465 linhas (41172 de train.csv + 10293 de test.csv), 58 features"]
    e = interpretar_log(linhas, TOTAL_BUSCA, TOTAL_AVALIACAO)
    assert e.fase == "dados"
    assert "protocolo 0,0%" in render(e, TOTAL_BUSCA, TOTAL_AVALIACAO, "x.log")


def test_primeiro_lote_sem_done():
    linhas = [
        "Dados: 51465 linhas (41172 de train.csv + 10293 de test.csv), 58 features",
        "Fitting 5 folds for each of 50 candidates, totalling 250 fits",
    ]
    e = interpretar_log(linhas, TOTAL_BUSCA, TOTAL_AVALIACAO)
    assert e.fase == "busca" and e.busca_iniciada and e.feitos_busca == 0
    t = render(e, TOTAL_BUSCA, TOTAL_AVALIACAO, "x.log")
    assert "0/250" in t and "primeiro lote" in t


def test_progresso_da_busca_com_eta():
    linhas = [
        "Dados: 51465 linhas (41172 de train.csv + 10293 de test.csv), 58 features",
        "Fitting 5 folds for each of 50 candidates, totalling 250 fits",
        linha_done(40, elapsed="0:00:15.2"),
        linha_done(153, elapsed="0:05:12"),
    ]
    e = interpretar_log(linhas, TOTAL_BUSCA, TOTAL_AVALIACAO)
    assert e.fase == "busca"
    assert (e.dados or "").startswith("Dados:")
    assert e.feitos_busca == 153 and e.feitos_avaliacao == 0
    assert e.tempo_busca_s == 312.0
    t = render(e, TOTAL_BUSCA, TOTAL_AVALIACAO, "x.log")
    assert "61.2%" in t.replace(",", ".") or "61,2%" in t
    assert "ETA protocolo" in t and "ritmo" in t


def test_decorrido_projeta_ate_agora():
    linhas = [linha_done(50, elapsed="0:05:00")]
    e = interpretar_log(linhas, TOTAL_BUSCA, TOTAL_AVALIACAO)
    t = render(e, TOTAL_BUSCA, TOTAL_AVALIACAO, "x.log", agora=1000.0, log_mtime=660.0)
    # 300 s elapsed + 340 s desde a última escrita
    assert "0:10:40" in t


def test_contador_reiniciando_marca_avaliacao():
    linhas = [
        linha_done(TOTAL_BUSCA),
        linha_done(1, elapsed="0:00:02"),      # avaliação reinicia
        linha_done(4, elapsed="0:00:07"),
    ]
    e = interpretar_log(linhas, TOTAL_BUSCA, TOTAL_AVALIACAO)
    assert e.fase == "avaliacao"
    assert e.feitos_busca == TOTAL_BUSCA and e.feitos_avaliacao == 4
    assert e.tempo_avaliacao_s == 7.0
    t = render(e, TOTAL_BUSCA, TOTAL_AVALIACAO, "x.log")
    assert "protocolo 99" in t or "protocolo 98" in t
    assert "busca 100%" in t


def test_conclusao_com_resumo():
    linhas = [
        linha_done(TOTAL_BUSCA, elapsed="0:30:00"),
        linha_done(6, elapsed="0:00:09"),
        "Candidatos avaliados: 50 (n_iter pedido: 300) | melhores parâmetros: {'C': 100}",
        "MAE 1.0149 ± 0.0043  IC95% [1.0109, 1.0190]",
        "summary -> results/metrics/logistic_regression_rq1.json",
    ]
    e = interpretar_log(linhas, TOTAL_BUSCA, TOTAL_AVALIACAO)
    assert e.fase == "concluido"
    assert len(e.resultados) == 3
    t = render(e, TOTAL_BUSCA, TOTAL_AVALIACAO, "x.log")
    assert "concluído" in t and "100%" in t and "0:30:09" in t


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
