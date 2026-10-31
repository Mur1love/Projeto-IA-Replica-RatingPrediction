
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import compute_metrics

RAIZ = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "compare_normalization", RAIZ / "scripts" / "compare_normalization.py"
)
cn = importlib.util.module_from_spec(_spec)
sys.modules["compare_normalization"] = cn
_spec.loader.exec_module(cn)


def _csv_sintetico(path: Path, n_por_nota: int, seed: int) -> None:
    """CSV pequeno: cada nota tem palavras próprias + ruído, com texto bruto 'sujo'."""
    rng = np.random.default_rng(seed)
    vocab = {
        1: ["pessimo", "horrivel", "quebrou", "lixo"],
        2: ["ruim", "fraco", "demorou", "falhou"],
        3: ["razoavel", "normal", "mediano", "ok"],
        4: ["bom", "gostei", "otimo", "recomendo"],
        5: ["perfeito", "amei", "excelente", "maravilhoso"],
    }
    linhas = []
    for nota, palavras in vocab.items():
        for _ in range(n_por_nota):
            extra = rng.choice(["produto", "entrega", "preço", "vc", "pq"], 2)
            texto = " ".join(list(rng.choice(palavras, 3)) + list(extra)) + "!!!"
            linhas.append({"text": texto, "rating": nota})
    df = pd.DataFrame(linhas).sample(frac=1, random_state=seed)
    df.loc[df.index[0], "text"] = np.nan  # linha vazia: deve ser descartada
    df.to_csv(path, index=False)


# --------------------------------------------------------------------------- #
# Variantes e linhas
# --------------------------------------------------------------------------- #
def test_variantes_incluem_referencia_e_completa():
    assert cn.REFERENCE == "sem_normalizacao"
    assert cn.VARIANTS[cn.REFERENCE] is None
    assert cn.VARIANTS["normalizacao_completa"] == {}


def test_apply_variant_bruto_nao_altera_o_texto():
    s = pd.Series(["VC amei!!!", "Não gostei"])
    assert cn.apply_variant(s, None) == ["VC amei!!!", "Não gostei"]


def test_apply_variant_apenas_uma_etapa_por_vez():
    s = pd.Series(["VC amei!!!!"])
    assert cn.apply_variant(s, cn.VARIANTS["apenas_repeticoes"]) == ["vc amei!!"]
    assert cn.apply_variant(s, cn.VARIANTS["apenas_abreviacoes"]) == ["você amei!!!!"]
    assert cn.apply_variant(s, cn.VARIANTS["normalizacao_completa"]) == ["voce amei!!"]  # vc->você e depois sem acento


def test_apenas_acentos_remove_acento_sem_mexer_no_resto():
    s = pd.Series(["Não amei!!!!"])
    assert cn.apply_variant(s, cn.VARIANTS["apenas_acentos"]) == ["nao amei!!!!"]


def test_common_rows_descarta_vazios_e_nan(tmp_path):
    p = tmp_path / "x.csv"
    pd.DataFrame({"text": ["ok", "   ", None, "bom"], "rating": [1, 2, 3, 4]}).to_csv(p, index=False)
    df = cn.common_rows(cn.load_raw(p))
    assert df["text"].tolist() == ["ok", "bom"]
    assert df["rating"].tolist() == [1, 4]


# --------------------------------------------------------------------------- #
# Métricas e bootstrap
# --------------------------------------------------------------------------- #
def test_fast_metrics_igual_a_compute_metrics():
    rng = np.random.default_rng(0)
    for _ in range(10):
        y = rng.integers(0, 5, 400)
        p = np.where(rng.random(400) < 0.5, y, rng.integers(0, 5, 400))
        rapido, ref = cn.fast_metrics(y, p), compute_metrics(y, p)
        assert rapido == pytest.approx(ref, abs=1e-12)


def test_fast_metrics_classe_ausente_nao_gera_nan():
    y = np.array([0, 0, 1, 1])
    p = np.array([0, 1, 1, 0])  # classes 2..4 ausentes
    assert np.isfinite(list(cn.fast_metrics(y, p).values())).all()
    assert cn.fast_metrics(y, p) == pytest.approx(compute_metrics(y, p), abs=1e-12)


def test_bootstrap_previsoes_identicas_nao_e_significativo():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 5, 300)
    p = rng.integers(0, 5, 300)
    out = cn.paired_bootstrap(y, p, {"igual": p.copy()}, n_resamples=50, seed=1)
    for m in cn.METRICS:
        assert out["igual"][m]["ic95_low"] == 0.0 == out["igual"][m]["ic95_high"]
        assert out["igual"][m]["significativo"] is False


def test_bootstrap_modelo_claramente_melhor_e_significativo():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 5, 2000)
    ruim = rng.integers(0, 5, 2000)  # ~20% de acerto
    bom = y.copy()  # 100% de acerto
    out = cn.paired_bootstrap(y, ruim, {"bom": bom}, n_resamples=100, seed=2)
    assert out["bom"]["accuracy"]["significativo"] is True
    assert out["bom"]["accuracy"]["ic95_low"] > 0
    assert out["bom"]["mae"]["ic95_high"] < 0  # MAE cai


def test_bootstrap_e_reprodutivel_com_mesma_seed():
    rng = np.random.default_rng(3)
    y, a, b = (rng.integers(0, 5, 200) for _ in range(3))
    r1 = cn.paired_bootstrap(y, a, {"v": b}, n_resamples=30, seed=7)
    r2 = cn.paired_bootstrap(y, a, {"v": b}, n_resamples=30, seed=7)
    assert r1 == r2


# --------------------------------------------------------------------------- #
# Tabela
# --------------------------------------------------------------------------- #
def _linha(nome, acc, mae):
    return {"variante": nome, "accuracy": acc, "f1_macro": acc, "mae": mae, "rmse": mae,
            "vocabulario": 1000, "termos_por_doc": 5.0, "tempo_s": 1.0}


def test_build_comparison_calcula_delta_contra_a_referencia():
    rows = [_linha("sem_normalizacao", 0.50, 0.70), _linha("outra", 0.52, 0.65)]
    boot = {"outra": {m: {"ic95_low": 0.0, "ic95_high": 0.01, "significativo": m == "mae"}
                      for m in cn.METRICS}}
    df = cn.build_comparison(rows, boot).set_index("variante")
    assert df.loc["outra", "delta_accuracy"] == pytest.approx(0.02)
    assert df.loc["outra", "delta_mae"] == pytest.approx(-0.05)
    assert df.loc["sem_normalizacao", "delta_accuracy"] == 0
    assert df.loc["sem_normalizacao", "sig_accuracy"] == "ref"
    assert df.loc["outra", "sig_mae"] == "sim" and df.loc["outra", "sig_accuracy"] == "não"


def test_markdown_usa_formato_ptbr():
    rows = [_linha("sem_normalizacao", 0.4903, 0.6824)]
    df = cn.build_comparison(rows, {})
    md = cn.to_markdown(df)
    assert "0,4903" in md and "0,6824" in md
    assert "`sem_normalizacao`" in md


# --------------------------------------------------------------------------- #
# Execução fim a fim (dados sintéticos)
# --------------------------------------------------------------------------- #
def test_cli_protocolo_teste_gera_csv_json_e_md(tmp_path):
    treino, teste, saida = tmp_path / "tr.csv", tmp_path / "te.csv", tmp_path / "out"
    _csv_sintetico(treino, 120, seed=0)
    _csv_sintetico(teste, 40, seed=1)

    rc = cn.main(["--train", str(treino), "--test", str(teste),
                  "--out-dir", str(saida), "--n-bootstrap", "20"])
    assert rc == 0

    tabela = pd.read_csv(saida / "normalization_comparison.csv")
    assert tabela["variante"].tolist() == list(cn.VARIANTS)
    assert {"accuracy", "f1_macro", "mae", "rmse", "delta_accuracy", "sig_mae"} <= set(tabela.columns)

    payload = json.loads((saida / "normalization_comparison.json").read_text(encoding="utf-8"))
    assert payload["protocol"] == "test" and payload["reference"] == "sem_normalizacao"
    assert payload["n_train"] == 599 and payload["n_test"] == 199  # 1 NaN removido em cada
    assert (saida / "normalization_comparison.md").read_text(encoding="utf-8").startswith("| Variante")


def test_cli_apenas_acentos_e_identico_ao_bruto(tmp_path):
    """O TF-IDF já usa strip_accents='unicode': remover acento antes não muda nada."""
    treino, teste, saida = tmp_path / "tr.csv", tmp_path / "te.csv", tmp_path / "out"
    _csv_sintetico(treino, 120, seed=0)
    _csv_sintetico(teste, 40, seed=1)
    cn.main(["--train", str(treino), "--test", str(teste),
             "--out-dir", str(saida), "--n-bootstrap", "10"])
    t = pd.read_csv(saida / "normalization_comparison.csv").set_index("variante")
    for col in ("accuracy", "f1_macro", "mae", "rmse", "vocabulario"):
        assert t.loc["apenas_acentos", col] == t.loc["sem_normalizacao", col]


def test_cli_protocolo_validacao_nao_le_o_teste(tmp_path):
    treino, saida = tmp_path / "tr.csv", tmp_path / "out"
    _csv_sintetico(treino, 150, seed=0)
    rc = cn.main(["--train", str(treino), "--test", str(tmp_path / "nao_existe.csv"),
                  "--protocol", "validation", "--out-dir", str(saida), "--n-bootstrap", "10"])
    assert rc == 0
    payload = json.loads(
        (saida / "normalization_comparison_validation.json").read_text(encoding="utf-8")
    )
    assert payload["protocol"] == "validation"
    assert payload["n_train"] + payload["n_test"] == 749  # 750 - 1 NaN


def test_artefatos_versionados_sao_coerentes():
    """As tabelas finais devem existir e ter as mesmas variantes e linhas do experimento."""
    for nome, n_train, n_test in (
        ("normalization_comparison", 41171, 10292),
        ("normalization_comparison_validation", 32936, 8235),
    ):
        csv = RAIZ / "results" / "metrics" / f"{nome}.csv"
        js = RAIZ / "results" / "metrics" / f"{nome}.json"
        if not csv.exists():
            pytest.skip(f"{csv.name} ainda não foi gerado")
        assert pd.read_csv(csv)["variante"].tolist() == list(cn.VARIANTS)
        payload = json.loads(js.read_text(encoding="utf-8"))
        assert (payload["n_train"], payload["n_test"]) == (n_train, n_test)
