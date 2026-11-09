"""Monitor de execução do protocolo (RQ1): fase, %, tempo decorrido e restante.

Acompanha o log do ``train_article_model.py`` rodado com ``--verbose 11`` e
``PYTHONUNBUFFERED=1`` (sem isso, os workers loky bufferiam as marcas ``[CV]``::

    # lancar (do jeito usado nesta etapa; independente do terminal):
    PYTHONUNBUFFERED=1 setsid nohup .venv/bin/python -u scripts/train_article_model.py \
        --model logistic_regression --verbose 11 --n-jobs -1 \
        > results/logistic_regression_run.log 2>&1 &
    echo $! > results/logistic_regression_run.pid

    # acompanhar (Ctrl-C encerra o monitor, nao a execucao):
    python scripts/monitor_regressao_logistica.py --log results/logistic_regression_run.log

Usos extras:

    python scripts/monitor_regressao_logistica.py --uma-vez      # foto unica, sem loop
    python scripts/monitor_regressao_logistica.py --intervalo 5  # atualiza a cada 5 s

Por fase:

1. ``dados``      — carregamento do CSV (linha ``Dados: ...``);
2. ``busca``      — cada fit concluido imprime ``[CV k/5; i/50] END ... total
   time=Xs``; o monitor conta os ENDs, mostra %, ritmo e ETA (tempo medio de
   fit medido x workers); antes do primeiro END, avisa que nenhum lote acabou;
3. ``avaliacao``  — 5 folds + holdout com os melhores hiperparametros; esse
   ``Parallel`` e do joblib puro e imprime ``Done N tasks | elapsed: ...``;
4. ``concluido``  — metricas e caminhos dos artefatos (``summary -> ...``).

Tempo: o sklearn verbose nao imprime o ``elapsed`` do joblib durante a busca;
o decorrido e ancorado na ``Fitting N folds...`` vista pelo monitor (acurado
quando o monitor acompanha desde o inicio ``≈`` se entrar no meio). O ``total
time=`` de cada END alimenta o ETA. MAE/RMSE/AUC e holdout vm só ao final.

Denominadores da LR: ``min(n_iter, 50) ≈ 50`` candidatos × 5 folds = 250 fits
na busca e ``5 folds + 1 holdout = 6`` na avaliacao (protocolo: 256 fits).
``--candidatos``, ``--n-iter``, ``--n-splits``, ``--workers`` e ``--sem-holdout``
ajustam outras execucoes.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

N_CANDIDATOS_LR = 50  # grid do artigo: 5 C × 5 solver × 2 class_weight

# "[CV 3/5; 1/50] END .... score=... total time=   5.3s" (sklearn, busca)
# (>1 min aparece como "total time=  1.2min")
_CV_END = re.compile(r"\[CV\s+\d+/\d+;\s*\d+/\d+\]\s*END.*?total time=\s*([\d.]+)(s|min)")
# "Parallel(n_jobs=-1): Done 3 tasks | elapsed: 0:00:07" (joblib puro, avaliação)
_DONE = re.compile(r"Done\s+(\d+)\s+tasks|Done\s+(\d+)\s+(?:of|out of)\s+(\d+)")
_ELAPSED = re.compile(r"elapsed:\s*([\d:]+\.?\d*)s?")
_FITTING = re.compile(r"Fitting\s+\d+\s+folds for each of")
_FIM_RESUMO = re.compile(r"summary\s+->\s+\S+")
_METRICA = re.compile(r"\s*(MAE|RMSE|AUC)\s+\d")


@dataclass
class Estado:
    """O que o log informa em um instante, já traduzido para progresso."""

    fase: str = "aguardando"                # aguardando|dados|busca|avaliacao|concluido
    busca_iniciada: bool = False            # True após a linha "Fitting N folds..."
    feitos_busca: int = 0
    feitos_avaliacao: int = 0
    tempos_fit_s: list[float] = field(default_factory=list)  # "total time=" por fit
    tempo_avaliacao_s: float | None = None  # elapsed do joblib na avaliação
    dados: str | None = None
    resultados: list[str] = field(default_factory=list)      # métricas/arte­fatos finais

    @property
    def tempo_medio_fit_s(self) -> float:
        return sum(self.tempos_fit_s) / len(self.tempos_fit_s) if self.tempos_fit_s else 0.0


def _segundos(texto: str) -> float:
    """``0.7s``/``0.7``/``0:00:15`` -> segundos."""
    texto = texto.strip()
    if texto.endswith("s"):
        texto = texto[:-1]
    if ":" in texto:
        h, m, s = texto.split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)
    return float(texto)


def interpretar_log(linhas: list[str], total_busca: int, total_avaliacao: int) -> Estado:
    """Converte as linhas do log em :class:`Estado`.

    Na busca, cada fit concluído gera uma marca ``[CV ...] END`` — elas são
    contadas em ``feitos_busca`` e o ``total time=`` de cada uma entra em
    ``tempos_fit_s``. A fase de avaliação começa quando há ``total_busca`` ENDs
    (ou quando o contador do joblib reinicia); o ``Done N tasks | elapsed:``
    da avaliação acumula em ``feitos_avaliacao`` e ``tempo_avaliacao_s``.
    Métricas e caminhos de artefatos só aparecem no fim, em ``resultados``.
    """
    estado = Estado()
    in_busca = True  # enquanto True, contadores ativos são da busca
    for bruta in linhas:
        linha = bruta.rstrip()
        if not linha:
            continue

        if "Dados:" in linha and "features" in linha:
            estado.dados = linha.strip()
            if estado.fase == "aguardando":
                estado.fase = "dados"

        if _FITTING.search(linha):
            estado.busca_iniciada = True
            if not estado.resultados:
                estado.fase = "busca"  # a linha Fitting vem depois de Dados

        fim_cv = _CV_END.search(linha)
        if fim_cv and in_busca:
            valor = float(fim_cv.group(1)) * (60.0 if fim_cv.group(2) == "min" else 1.0)
            estado.tempos_fit_s.append(valor)
            estado.feitos_busca = min(estado.feitos_busca + 1, total_busca)
            estado.fase = "busca"
            continue

        bate = _DONE.search(linha)
        if bate:
            feitos = int(bate.group(2) or bate.group(1))
            in_busca = False  # a busca (verbose sklearn) não emite Done: essas linhas são da avaliação
            estado.feitos_avaliacao = max(estado.feitos_avaliacao, min(feitos, total_avaliacao))
            estado.fase = "avaliacao"
            tempo = _ELAPSED.search(linha)
            if tempo:
                estado.tempo_avaliacao_s = max(
                    estado.tempo_avaliacao_s or 0.0, _segundos(tempo.group(1))
                )
            continue

        if "Candidatos avaliados:" in linha or "Holdout (" in linha or _FIM_RESUMO.search(linha):
            estado.fase = "concluido"
            estado.resultados.append(linha.strip())
        elif _METRICA.match(linha):
            estado.resultados.append(linha.strip())

    if estado.resultados:
        estado.fase = "concluido"
    if estado.fase == "busca" and 0 < total_busca <= estado.feitos_busca:
        estado.fase = "avaliacao"
    if estado.fase == "avaliacao" and 0 < total_avaliacao <= estado.feitos_avaliacao:
        estado.fase = "concluido"
    return estado


# --- apresentação -----------------------------------------------------------
def _pct(parte: int, total: int) -> float:
    return max(0.0, min(100.0, 100.0 * parte / float(total))) if total else 0.0


def _hhmmss(segundos: float) -> str:
    segundos = max(0, int(round(segundos)))
    return f"{segundos // 3600}:{segundos % 3600 // 60:02d}:{segundos % 60:02d}"


def _num(valor: float) -> str:
    """Decimal com vírgula (pt-BR), 1 casa."""
    return f"{valor:.1f}".replace(".", ",")


def _bar(pct: float, largura: int = 24) -> str:
    cheio = int(round(pct / 100.0 * largura))
    return "█" * cheio + "░" * max(0, largura - cheio)


def render(estado: Estado, total_busca: int, total_avaliacao: int, log: str,
           *, monitor_s: float = 0.0, busca_ancora_s: float | None = None,
           workers: int = 1) -> str:
    """Bloco de 1–2 linhas com fase, %, tempo decorrido e tempo restante."""
    total_protocolo = total_busca + total_avaliacao
    pct_protocolo = _pct(estado.feitos_busca + estado.feitos_avaliacao, total_protocolo)
    medio = estado.tempo_medio_fit_s

    if estado.fase == "aguardando":
        return f"aguardando o log em '{log}'... (monitor há {_hhmmss(monitor_s)})"
    if estado.fase == "dados":
        return f"fase: dados | {estado.dados or 'carregado'} | protocolo 0,0% (busca não iniciou)"

    if estado.fase == "busca":
        pct_busca = _pct(estado.feitos_busca, total_busca)
        cabeca = (f"fase de busca de hiperparâmetros | {pct_busca:5.1f}% {_bar(pct_busca)} "
                  f"({estado.feitos_busca}/{total_busca} fits, protocolo {pct_protocolo:.1f}%)")
        decorrido = f"busca há ≈{_hhmmss(busca_ancora_s or 0)}" if busca_ancora_s else ""
        if not estado.feitos_busca:
            return (f"{cabeca}\n  {decorrido + ' | ' if decorrido else ''}"
                    f"nenhum fit concluído ainda — os 250 fits estão distribuídos "
                    f"nos workers; a primeira marca END abrira o ETA")
        resta_busca = (total_busca - estado.feitos_busca) * medio / workers
        eta_total = resta_busca + total_avaliacao * medio / workers
        ritmo = (estado.feitos_busca / (busca_ancora_s or 0.0) * 60) if busca_ancora_s else 0.0
        detalhe = (
            f"  {decorrido + ' | ' if decorrido else ''}"
            f"ritmo {_num(ritmo)} fits/min | fit médio {_num(medio)} s | "
            f"resta: {_hhmmss(resta_busca)} (busca) | ETA protocolo ≈ {_hhmmss(eta_total)}"
        )
        return f"{cabeca}\n{detalhe}"

    if estado.fase == "avaliacao":
        pct_aval = _pct(estado.feitos_avaliacao, total_avaliacao)
        resta_aval = (total_avaliacao - estado.feitos_avaliacao) * medio / workers
        decorrido = (f"avaliação há {_hhmmss(estado.tempo_avaliacao_s)}"
                     if estado.tempo_avaliacao_s else "")
        return (f"fase de avaliação (folds + holdout) | {pct_aval:5.1f}% {_bar(pct_aval)} "
                f"({estado.feitos_avaliacao}/{total_avaliacao} fits)\n  "
                f"{decorrido + ' | ' if decorrido else ''}busca 100% | "
                f"protocolo {pct_protocolo:.1f}% | resta ≈ {_hhmmss(resta_aval)}")

    fim = (busca_ancora_s or 0) + (estado.tempo_avaliacao_s or 0)
    return (f"concluído — 100% do protocolo | total ≈ {_hhmmss(fim)} "
            f"(busca {_hhmmss(busca_ancora_s or 0)} + avaliação "
            f"{_hhmmss(estado.tempo_avaliacao_s or 0)})")


def principal(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Fase, %, tempo decorrido e restante da execução RQ1 (a partir do log --verbose 11)."
    )
    p.add_argument("--log", type=Path, default=Path("results/logistic_regression_run.log"),
                   help="arquivo de log do train_article_model.py")
    p.add_argument("--n-iter", type=int, default=300, help="n_iter pedido na execução (padrão 300)")
    p.add_argument("--candidatos", type=int, default=None,
                   help=f"candidatos realmente avaliados (padrão {N_CANDIDATOS_LR} na LR)")
    p.add_argument("--n-splits", type=int, default=5, help="folds usados na execução (padrão 5)")
    p.add_argument("--workers", type=int, default=0,
                   help="núcleos paralelos para o ETA (0 = os.cpu_count())")
    p.add_argument("--sem-holdout", action="store_true",
                   help="a execução não tem holdout (ex.: dados de um único arquivo)")
    p.add_argument("--intervalo", type=float, default=2.0, help="segundos entre leituras (loop)")
    p.add_argument("--uma-vez", action="store_true",
                   help="imprime o estado atual e encerra (código 0 = concluído; 1 = em execução)")
    args = p.parse_args(argv)

    candidatos = args.candidatos or min(args.n_iter, N_CANDIDATOS_LR)
    total_busca = candidatos * args.n_splits
    total_avaliacao = args.n_splits + (0 if args.sem_holdout else 1)
    workers = args.workers or (os.cpu_count() or 1)
    inicio_monitor = time.monotonic()
    busca_ancora: float | None = None  # tempo decorrido da busca segundo o monitor
    ansi = sys.stdout.isatty()

    try:
        while True:
            agora = time.monotonic()
            monitor_s = agora - inicio_monitor
            if args.log.is_file():
                linhas = args.log.read_text(encoding="utf-8", errors="replace").splitlines()
                estado = interpretar_log(linhas, total_busca, total_avaliacao)
                if estado.busca_iniciada and busca_ancora is None:
                    busca_ancora = monitor_s  # âncora da busca (desde o monitor)
            else:
                estado = Estado()  # --log ainda não existe

            linha = render(
                estado, total_busca, total_avaliacao, str(args.log),
                monitor_s=monitor_s,
                busca_ancora_s=(agora - inicio_monitor) if (estado.busca_iniciada and busca_ancora is not None) else None,
                workers=workers,
            )
            if ansi:
                print("\033[2J\033[H" + linha, flush=True)
            else:
                print(linha, flush=True)

            if estado.fase == "concluido":
                print("\n— resultado (log completo em %s) —" % args.log, flush=True)
                for r in estado.resultados:
                    print(r, flush=True)
                return 0
            if args.uma_vez:
                return 1  # em execução
            time.sleep(args.intervalo)
    except KeyboardInterrupt:
        print("\nmonitor encerrado; a execução continua (pare-a no terminal 1).", flush=True)
        return 130


if __name__ == "__main__":
    raise SystemExit(principal())
