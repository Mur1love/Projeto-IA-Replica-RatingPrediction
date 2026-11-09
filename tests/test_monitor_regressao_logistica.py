"""monitor_regressao_logistica.py: fases, % e finalização a partir de um log simulado."""
import pytest

from scripts.monitor_regressao_logistica import (
    Estado,
    _hhmmss,
    _pct,
    interpretar_log,
    render,
)


TOTAL_BUSCA = 250       # 50 candidatos x 5 folds
TOTAL_AVALIACAO = 6     # 5 folds + holdout


def linhas_busca(n, *, elapsed="0:01:00"):
    """Uma linha de progresso joblib (barras reais do joblib nunca recuam dentro da fase)."""
    return [
        f"Parallel(n_jobs=-1): Done {n} tasks | elapsed: {elapsed} remaining: 5.0s"
    ]


def test_sem_log_aguarda():
    e = interpretar_log([], TOTAL_BUSCA, TOTAL_AVALIACAO)
    assert e.fase == "aguardando"
    assert "aguardando" in render(e, TOTAL_BUSCA, TOTAL_AVALIACAO, 7.0, "x.log")


def test_fase_dados_e_progresso_da_busca():
    linhas = [
        "Dados: 51465 linhas (41172 de train.csv + 10293 de test.csv), 58 features",
        *linhas_busca(40, elapsed="0:00:15.2"),
        *linhas_busca(153, elapsed="0:05:12"),
    ]
    e = interpretar_log(linhas, TOTAL_BUSCA, TOTAL_AVALIACAO)
    assert e.fase == "busca"
    assert (e.dados or "").startswith("Dados:")
    assert e.feitos_busca == 153 and e.feitos_avaliacao == 0
    assert e.tempo_busca_s == 312.0
    t = render(e, TOTAL_BUSCA, TOTAL_AVALIACAO, 315.0, "x.log")
    assert "busca há 0:05:12" in t and "61" in t and "ETA" in t


def test_contador_reiniciando_marca_avaliacao():
    linhas = [
        *linhas_busca(TOTAL_BUSCA),
        "Parallel(n_jobs=-1): Done 1 tasks | elapsed: 0:00:02",   # avaliação reinicia
        "Parallel(n_jobs=-1): Done 4 tasks | elapsed: 0:00:07",
    ]
    e = interpretar_log(linhas, TOTAL_BUSCA, TOTAL_AVALIACAO)
    assert e.fase == "avaliacao"
    assert e.feitos_busca == TOTAL_BUSCA and e.feitos_avaliacao == 4


def test_conclusao_com_resumo():
    linhas = [
        *linhas_busca(TOTAL_BUSCA, elapsed="0:30:00"),
        "Parallel(n_jobs=-1): Done 6 tasks | elapsed: 0:00:09",
        "Candidatos avaliados: 50 (n_iter pedido: 300) | melhores parâmetros: {'C': 100, 'class_weight': None}",
        "MAE 1.0149 ± 0.0043  IC95% [1.0109, 1.0190]",
        "summary -> results/metrics/logistic_regression_rq1.json",
    ]
    e = interpretar_log(linhas, TOTAL_BUSCA, TOTAL_AVALIACAO)
    assert e.fase == "concluido"
    assert len(e.resultados) == 3
    t = render(e, TOTAL_BUSCA, TOTAL_AVALIACAO, 20.0, "x.log")
    assert "concluído" in t and "100%" in t and "0:30:09" in t


def test_pct_e_hhmmss_cap_naos_estouram():
    assert _pct(127, 256) == pytest.approx(49.6, abs=0.1)
    assert _pct(9, 6) == 100.0 and _pct(0, 0) == 0.0
    assert _hhmmss(312.0) == "0:05:12"


def test_estado_default():
    e = Estado()
    assert e.fase == "aguardando" and e.feitos_busca == 0
