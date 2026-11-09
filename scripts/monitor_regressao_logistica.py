"""Monitor de execução do protocolo (RQ1): tempo decorrido, fase e % concluído.

Acompanha o log do ``train_article_model.py`` rodado com ``--verbose 10``::

    # terminal 1 (a execução; pode ficar em background)
    python scripts/train_article_model.py --model logistic_regression \
        --verbose 10 > results/logistic_regression_run.log 2>&1

    # terminal 2 (o monitor; Ctrl-C encerra o monitor, não a execução)
    python scripts/monitor_regressao_logistica.py --log results/logistic_regression_run.log

Fases lidas do log:

1. ``dados``      — carregamento do CSV (linha ``Dados: ...``);
2. ``busca``      — ``RandomizedSearchCV`` (contadores ``Done N tasks`` do joblib);
3. ``avaliacao``  — 5 folds + holdout com os melhores hiperparâmetros (o contador
   do joblib reinicia a partir de 1);
4. ``concluido``  — quando aparecem ``Candidatos avaliados:`` e os caminhos dos
   artefatos (``summary -> ...``); o monitor então repete as métricas e encerra.

A busca da LR avalia ``min(n_iter, 50) ≈ 50`` candidatos × 5 folds = 250 ajustes
(grid do artigo tem 50 combinações). Para monitorar outro modelo, passe
``--candidatos`` explicitamente.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

N_CANDIDATOS_LR = 50  # grid do artigo: 5 C x 5 solver x 2 class_weight
FOLD_METRICAS = ("MAE", "RMSE", "AUC")

_TAREFAS = re.compile(r"Done\s+(\d+)\s+tasks")
_DE = re.compile(r"Done\s+(\d+)\s+(?:of|out of)\s+(\d+)")
_DECURRIDO = re.compile(r"elapsed:\s*([\d:]+\.?\d*)s?")

FIM_RESUMO = re.compile(r"summary\s+->\s+\S+")


@dataclass
class Estado:
    """O que o log informa em um instante, já traduzido para progresso."""

    fase: str = "aguardando"
    feitos_busca: int = 0
    feitos_avaliacao: int = 0
    tempo_busca_s: float | None = None       # elapsed do joblib na busca
    tempo_avaliacao_s: float | None = None   # elapsed do joblib na avaliação
    dados: str | None = None
    resultados: list[str] = field(default_factory=list)   # linhas finais de métricas/artefatos
    ultimas_linhas: list[str] = field(default_factory=list)


def _segundos(elapsed: str) -> float:
    """``0.7s``/``0.7``/``0:00:15`` -> segundos."""
    text = elapsed.strip().removesuffix("s")
    if ":" in text:
        h, m, s = text.split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)
    return float(text)


def interpretar_log(linhas: list[str], total_busca: int, total_avaliacao: int) -> Estado:
    """Converte as linhas do log em :class:`Estado`.

    O contador do joblib cresce na busca (até ``total_busca``) e reinicia na
    avaliação; cada queda marca o início de uma fase. Métricas e caminhos de
    artefatos aparecem só no fim e são devolvidos em ``resultados``.
    """
    estado = Estado()
    fase_busca = True           # True = contadores ativos são da busca
    for linha in linhas:
        linha = linha.rstrip()
        if not linha:
            continue

        if "Dados:" in linha and "features" in linha:
            estado.dados = linha.strip()
            estado.fase = "dados"

        bate = _TAREFAS.search(linha) or _DE.search(linha)
        if bate:
            feitos = int(bate.group(1))
            if fase_busca and feitos < estado.feitos_busca:
                fase_busca = False          # contador reiniciou: avaliação
            if fase_busca:
                estado.feitos_busca = min(max(estado.feitos_busca, feitos), total_busca)
                estado.fase = "busca"
                tempo = _DECURRIDO.search(linha)
                if tempo:
                    estado.tempo_busca_s = max(estado.tempo_busca_s or 0.0, _segundos(tempo.group(1)))
            else:
                estado.feitos_avaliacao = max(estado.feitos_avaliacao, feitos)
                estado.fase = "avaliacao"
                tempo = _DECURRIDO.search(linha)
                if tempo:
                    estado.tempo_avaliacao_s = max(
                        estado.tempo_avaliacao_s or 0.0, _segundos(tempo.group(1))
                    )
            continue

        if "Candidatos avaliados:" in linha:
            estado.fase = "concluido"
            estado.resultados.append(linha.strip())
        elif "Holdout (" in linha or FIM_RESUMO.search(linha):
            estado.fase = "concluido"
            estado.resultados.append(linha.strip())
        elif re.match(r"\s*(MAE|RMSE|AUC)\s+\d", linha):
            estado.resultados.append(linha.strip())

    if estado.resultados:
        estado.fase = "concluido"
    estado.ultimas_linhas = [l.strip() for l in linhas if l.strip()][-1:]
    total = total_busca + total_avaliacao
    if estado.fase == "avaliacao" and estado.feitos_avaliacao >= total_avaliacao:
        estado.fase = "concluido"
    if estado.fase == "busca" and estado.feitos_busca >= total_busca:
        estado.fase = "avaliacao"
    return estado


def _pct(parte: int, total: int) -> float:
    return max(0.0, min(100.0, 100.0 * parte / float(total))) if total else 0.0


def _hhmmss(segundos: float) -> str:
    segundos = max(0, int(round(segundos)))
    return f"{segundos // 3600}:{segundos % 3600 // 60:02d}:{segundos % 60:02d}"


def render(estado: Estado, total_busca: int, total_avaliacao: int, total_s: float,
           log: str) -> str:
    """Linha única com fase, % e tempo, pronta para redesenhar no terminal.

    O tempo preferido é o ``elapsed`` impresso pelo próprio joblib (exato por
    fase); o relógio do monitor só aparece quando não há nada melhor.
    """
    total = total_busca + total_avaliacao
    if estado.fase == "aguardando":
        return f"aguardando o log em '{log}'..."
    if estado.fase == "dados":
        # Sem relógio aqui: a leitura de dados leva segundos; o tempo conta a partir
        # do `elapsed` que o joblib imprime na busca/avaliação.
        return f"fase: dados | {estado.dados or 'carregado'}"

    if estado.fase == "busca":
        decorrido = (
            f"busca há {_hhmmss(estado.tempo_busca_s)}" if estado.tempo_busca_s else monitor_h
        )
        taxa = (
            estado.feitos_busca / estado.tempo_busca_s
            if estado.feitos_busca and estado.tempo_busca_s
            else None
        )
        eta = f" | ETA {_hhmmss((total_busca - estado.feitos_busca) / taxa)}" if taxa else ""
        return (
            f"[{decorrido}] fase: busca | {estado.feitos_busca}/{total_busca} fits "
            f"({_pct(estado.feitos_busca, total_busca):5.1f}%)"
            f"{eta} | protocolo {_pct(estado.feitos_busca, total):.1f}%"
        )
    if estado.fase == "avaliacao":
        decorrido = (
            f"avaliação há {_hhmmss(estado.tempo_avaliacao_s)}"
            if estado.tempo_avaliacao_s
            else monitor_h
        )
        return (
            f"[{decorrido}] fase: avaliação (folds + holdout) | "
            f"{estado.feitos_avaliacao}/{total_avaliacao} fits "
            f"({_pct(estado.feitos_avaliacao, total_avaliacao):.0f}%) | busca 100%"
        )
    fim = (estado.tempo_busca_s or 0) + (estado.tempo_avaliacao_s or 0)
    return (
        f"[total ≈ {_hhmmss(fim)}] concluído — 100% do protocolo "
        f"(busca {_hhmmss(estado.tempo_busca_s or 0)} + avaliação "
        f"{_hhmmss(estado.tempo_avaliacao_s or 0)})"
    )


def principal(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Tempos e %% da busca/avaliação na RQ1, a partir de um log com --verbose 10."
    )
    p.add_argument("--log", type=Path, default=Path("results/logistic_regression_run.log"),
                   help="arquivo de log do train_article_model.py")
    p.add_argument("--n-iter", type=int, default=300, help="n_iter pedido na execução (padrão 300)")
    p.add_argument("--candidatos", type=int, default=None,
                   help=f"candidatos realmente avaliados (padrão {N_CANDIDATOS_LR} na LR)")
    p.add_argument("--n-splits", type=int, default=5, help="folds usados na execução (padrão 5)")
    p.add_argument("--sem-holdout", action="store_true",
                   help="a execução não tem holdout (ex.: dados de um único arquivo)")
    p.add_argument("--intervalo", type=float, default=2.0, help="segundos entre leituras")
    args = p.parse_args(argv)

    candidatos = args.candidatos or min(args.n_iter, N_CANDIDATOS_LR)
    total_busca = candidatos * args.n_splits
    total_avaliacao = args.n_splits + (0 if args.sem_holdout else 1)
    inicio_monitor = time.monotonic()
    inicio_execucao = None  # ctime do log, aproximando o início da execução

    ansi = sys.stdout.isatty()
    try:
        while True:
            if args.log.is_file():
                if inicio_execucao is None:
                    # O log nasce junto com a execução; usa-o como t0. Refletir
                    # reinícios exigiria reiniciar o monitor.
                    inicio_execucao = time.time() - args.log.stat().st_ctime
                linhas = args.log.read_text(encoding="utf-8", errors="replace").splitlines()
                estado = interpretar_log(linhas, total_busca, total_avaliacao)
            else:
                estado = Estado()  # --log ainda não existe

            total_s = (
                time.time() - inicio_execucao
                if inicio_execucao is not None
                else time.monotonic() - inicio_monitor
            )
            linha = render(estado, total_busca, total_avaliacao, total_s, str(args.log))
            if ansi:
                print("\033[2J\033[H" + linha, flush=True)
            else:
                print(linha, flush=True)

            if estado.fase == "concluido":
                print("\n— resultado (log completo em %s) —" % args.log, flush=True)
                for r in estado.resultados:
                    print(r, flush=True)
                return 0
            time.sleep(args.intervalo)
    except KeyboardInterrupt:
        print("\nmonitor encerrado; a execução continua (pare-a no terminal 1).", flush=True)
        return 130


if __name__ == "__main__":
    raise SystemExit(principal())
