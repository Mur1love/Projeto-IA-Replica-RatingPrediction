"""Monitor de execução do protocolo (RQ1): fase, %, tempo decorrido e tempo restante.

Acompanha o log do ``train_article_model.py`` rodado com ``--verbose 11``::

    # terminal 1 (a execução; pode ficar em background)
    python scripts/train_article_model.py --model logistic_regression \
        --verbose 11 > results/logistic_regression_run.log 2>&1

    # terminal 2 (o monitor; Ctrl-C encerra o monitor, não a execução)
    python scripts/monitor_regressao_logistica.py --log results/logistic_regression_run.log

Usos extras:

    python scripts/monitor_regressao_logistica.py --uma-vez      # foto única, sem loop
    python scripts/monitor_regressao_logistica.py --intervalo 5  # atualiza a cada 5 s

O monitor informa, por fase:

1. ``dados``      — carregamento do CSV (linha ``Dados: ...``);
2. ``busca``      — ``RandomizedSearchCV``: barra com o % de ajustes concluídos
   (contadores ``Done N tasks`` do joblib), ritmo em fits/min, tempo decorrido
   e ETA; enquanto nenhum lote conclui (antes da 1ª marca ``Done``), avisa que o
   primeiro lote está em execução;
3. ``avaliacao``  — 5 folds + holdout com os melhores hiperparâmetros (o contador
   do joblib reinicia a partir de 1);
4. ``concluido``  — métricas resumidas e caminhos dos artefatos (``summary -> ...``).

Tempo: o ``elapsed`` impresso pelo joblib é exato de dentro da fase; entre uma
linha e outra o monitor projeta ``elapsed + (agora − última escrita no log)``.
Antes da 1ª marca ``Done`` não há relógio do joblib; só o do monitor.

Denominadores da LR: ``min(n_iter, 50) ≈ 50`` candidatos × 5 folds = 250 fits na
busca e ``5 folds + 1 holdout = 6`` fits na avaliação (total do protocolo: 256).
``--candidatos``, ``--n-iter``, ``--n-splits`` e ``--sem-holdout`` ajustam para
outras execuções.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

N_CANDIDATOS_LR = 50  # grid do artigo: 5 C × 5 solver × 2 class_weight

_TAREFAS = re.compile(r"Done\s+(\d+)\s+tasks")
_DE = re.compile(r"Done\s+(\d+)\s+(?:of|out of)\s+(\d+)")
_DECURRIDO = re.compile(r"elapsed:\s*([\d:]+\.?\d*)s?")
_BUSCA_INICIADA = re.compile(r"Fitting\s+\d+\s+folds for each of")
_FIM_RESUMO = re.compile(r"summary\s+->\s+\S+")
_METRICA = re.compile(r"\s*(MAE|RMSE|AUC)\s+\d")


@dataclass
class Estado:
    """O que o log informa em um instante, já traduzido para progresso."""

    fase: str = "aguardando"           # aguardando|dados|busca|avaliacao|concluido
    busca_iniciada: bool = False       # True após a linha "Fitting N folds..."
    feitos_busca: int = 0
    feitos_avaliacao: int = 0
    tempo_busca_s: float | None = None       # ultimo elapsed do joblib na busca
    tempo_avaliacao_s: float | None = None   # ultimo elapsed do joblib na avaliação
    dados: str | None = None
    resultados: list[str] = field(default_factory=list)   # linhas finais métricas/artefatos


def _segundos(texto: str) -> float:
    """``0.7s``/``0.7``/``0:00:15`` -> segundos."""
    texto = texto.strip().removesuffix("s")
    if ":" in texto:
        h, m, s = texto.split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)
    return float(texto)


def interpretar_log(linhas: list[str], total_busca: int, total_avaliacao: int) -> Estado:
    """Converte as linhas do log em :class:`Estado`.

    O contador do joblib cresce na busca (até ``total_busca``) e reinicia na
    avaliação; cada queda marca o início da avaliação. Métricas e caminhos de
    artefatos aparecem só no fim e são devolvidos em ``resultados``.
    """
    estado = Estado()
    fase_busca = True  # True = contadores ativos são da busca
    for linha in linhas:
        linha = linha.rstrip()
        if not linha:
            continue

        if "Dados:" in linha and "features" in linha:
            estado.dados = linha.strip()
            estado.fase = "dados"

        if _BUSCA_INICIADA.search(linha):
            estado.busca_iniciada = True
            if not estado.resultados:
                estado.fase = "busca"

        bate = _TAREFAS.search(linha) or _DE.search(linha)
        if bate:
            feitos = int(bate.group(1))
            if fase_busca and feitos < estado.feitos_busca:
                fase_busca = False  # o contador reiniciou: avaliação
            if fase_busca:
                estado.feitos_busca = min(max(estado.feitos_busca, feitos), total_busca)
                estado.fase = "busca"
                tempo = _DECURRIDO.search(linha)
                if tempo:
                    estado.tempo_busca_s = max(
                        estado.tempo_busca_s or 0.0, _segundos(tempo.group(1))
                    )
            else:
                estado.feitos_avaliacao = max(estado.feitos_avaliacao, feitos)
                estado.fase = "avaliacao"
                tempo = _DECURRIDO.search(linha)
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


# --- apresentação ----------------------------------------------------------
def _pct(parte: int, total: int) -> float:
    return max(0.0, min(100.0, 100.0 * parte / float(total))) if total else 0.0


def _hhmmss(segundos: float) -> str:
    segundos = max(0, int(round(segundos)))
    return f"{segundos // 3600}:{segundos % 3600 // 60:02d}:{segundos % 60:02d}"


def _num(valor: float) -> str:
    """Número decimal com vírgula (convencional em pt-BR), 1 casa."""
    return f"{valor:.1f}".replace(".", ",")


def _bar(pct: float, largura: int = 24) -> str:
    cheio = int(round(pct / 100.0 * largura))
    return "█" * cheio + "░" * max(0, largura - cheio)


def _decorrido_fase(estado: Estado, agora: float, log_mtime: float | None) -> str:
    """Tempo decorrido da fase: joblib elapsed projetado até agora."""
    elapsed = (
        estado.tempo_busca_s if estado.fase == "busca" else
        estado.tempo_avaliacao_s if estado.fase == "avaliacao" else None
    )
    if elapsed is None:
        return ""
    delta = max(0.0, agora - log_mtime) if log_mtime is not None else 0.0
    return _hhmmss(elapsed + delta)


def render(estado: Estado, total_busca: int, total_avaliacao: int, log: str,
           *, monitor_s: float = 0.0, agora: float | None = None,
           log_mtime: float | None = None) -> str:
    """Bloco de 1–2 linhas com fase, %, tempo decorrido e tempo restante estimado."""
    total_protocolo = total_busca + total_avaliacao
    feitos_protocolo = estado.feitos_busca + estado.feitos_avaliacao
    pct_protocolo = _pct(feitos_protocolo, total_protocolo)

    if estado.fase == "aguardando":
        return f"aguardando o log em '{log}'... (monitor há {_hhmmss(monitor_s)})"
    if estado.fase == "dados":
        return f"fase: dados | {estado.dados or 'carregado'} | protocolo 0,0% (busca não iniciou)"

    if estado.fase == "busca":
        pct_busca = _pct(estado.feitos_busca, total_busca)
        cabeca = (
            f"fase de busca de hiperparâmetros | {pct_busca:4.1f}% {_bar(pct_busca)} "
            f"({estado.feitos_busca}/{total_busca} fits, protocolo {pct_protocolo:.1f}%)"
        )
        decorrido = _decorrido_fase(estado, agora or 0.0, log_mtime)
        if not estado.feitos_busca:
            return (f"{cabeca}\n  {('_≈' + decorrido + ' | ') if decorrido else ''}"
                    f"primeiro lote de fits em execução — o joblib só imprime uma "
                    f"marca quando um item do lote termina; nada a mostrar ainda")
        restante_busca = total_busca - estado.feitos_busca
        taxa = estado.feitos_busca / (estado.tempo_busca_s or 0.0)
        eta_busca = _hhmmss(restante_busca / taxa)
        # estimativa da avaliação: mesmo ritmo por fit (aproximação honesta)
        eta_total = _hhmmss(restante_busca / taxa + total_avaliacao / taxa)
        return (f"{cabeca}\n  {decorrido + ' | ' if decorrido else ''}"
                f"ritmo {_num(taxa * 60)} fits/min | resta: {eta_busca} (busca) | "
                f"ETA protocolo ≈ {eta_total}")

    if estado.fase == "avaliacao":
        pct_aval = _pct(estado.feitos_avaliacao, total_avaliacao)
        decorrido = _decorrido_fase(estado, agora or 0.0, log_mtime)
        eta_aval = 0.0
        if estado.feitos_avaliacao and estado.tempo_avaliacao_s:
            taxa_aval = estado.feitos_avaliacao / estado.tempo_avaliacao_s
            eta_aval = (total_avaliacao - estado.feitos_avaliacao) / taxa_aval
        return (f"fase de avaliação (5 folds + holdout) | {pct_aval:4.1f}% {_bar(pct_aval)} "
                f"({estado.feitos_avaliacao}/{total_avaliacao} fits)\n  "
                f"{decorrido + ' | ' if decorrido else ''}"
                f"busca 100% | protocolo {pct_protocolo:.1f}% | resta ≈ {_hhmmss(eta_aval)}")

    fim = (estado.tempo_busca_s or 0) + (estado.tempo_avaliacao_s or 0)
    return (f"concluído — 100% do protocolo | total ≈ {_hhmmss(fim)} "
            f"(busca {_hhmmss(estado.tempo_busca_s or 0)} + avaliação "
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
    p.add_argument("--sem-holdout", action="store_true",
                   help="a execução não tem holdout (ex.: dados de um único arquivo)")
    p.add_argument("--intervalo", type=float, default=2.0, help="segundos entre leituras (loop)")
    p.add_argument("--uma-vez", action="store_true",
                   help="imprime o estado atual e encerra (código 0 = concluído; 1 = em execução)")
    args = p.parse_args(argv)

    candidatos = args.candidatos or min(args.n_iter, N_CANDIDATOS_LR)
    total_busca = candidatos * args.n_splits
    total_avaliacao = args.n_splits + (0 if args.sem_holdout else 1)
    inicio_monitor = time.monotonic()
    ansi = sys.stdout.isatty()

    try:
        while True:
            agora = time.time()
            monitor_s = time.monotonic() - inicio_monitor
            if args.log.is_file():
                linhas = args.log.read_text(encoding="utf-8", errors="replace").splitlines()
                estado = interpretar_log(linhas, total_busca, total_avaliacao)
                linha = render(
                    estado, total_busca, total_avaliacao, str(args.log),
                    monitor_s=monitor_s, agora=agora,
                    log_mtime=args.log.stat().st_mtime,
                )
            else:
                estado = Estado()
                linha = render(estado, total_busca, total_avaliacao, str(args.log),
                               monitor_s=monitor_s)

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
