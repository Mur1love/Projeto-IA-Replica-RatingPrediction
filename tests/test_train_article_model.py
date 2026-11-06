"""CLIs do protocolo do artigo: train_article_model.py e train_svm_tfidf.py."""
import json
import subprocess
import sys

import pytest

from scripts import train_article_model, train_svm_tfidf


def test_cli_svm_gera_os_artefatos(article_csvs, tmp_path):
    train_csv, test_csv = article_csvs
    out = tmp_path / "saida"
    rc = train_article_model.main([
        "--model", "svm", "--train", str(train_csv), "--test", str(test_csv),
        "--n-iter", "2", "--n-jobs", "1", "--output-dir", str(out),
    ])
    assert rc == 0
    resumo = json.loads((out / "metrics" / "svm_rq1.json").read_text(encoding="utf-8"))
    assert resumo["protocol"]["n_iter_requested"] == 2
    assert resumo["data"]["n_rows"] == 320
    for nome in ("metrics/svm_rq1_folds.csv", "metrics/svm_rq1_holdout.json",
                 "figures/confusion_matrix_svm.png"):
        assert (out / nome).stat().st_size > 0


def test_cli_modo_rapido_com_sample_n(article_csvs, tmp_path):
    train_csv, test_csv = article_csvs
    out = tmp_path / "rapido"
    rc = train_article_model.main([
        "--model", "svm", "--train", str(train_csv), "--test", str(test_csv),
        "--sample-n", "150", "--n-iter", "2", "--n-jobs", "1", "--output-dir", str(out),
    ])
    assert rc == 0
    resumo = json.loads((out / "metrics" / "svm_rq1.json").read_text(encoding="utf-8"))
    assert resumo["data"]["sample_n"] == 150 and resumo["data"]["n_rows"] == 150


def test_cli_n_splits_e_seed_sao_registrados(article_csvs, tmp_path):
    train_csv, test_csv = article_csvs
    train_article_model.main([
        "--model", "svm", "--train", str(train_csv), "--test", str(test_csv), "--n-iter", "1",
        "--n-splits", "3", "--seed", "7", "--shuffle", "--n-jobs", "1",
        "--output-dir", str(tmp_path),
    ])
    p = json.loads((tmp_path / "metrics" / "svm_rq1.json").read_text(encoding="utf-8"))["protocol"]
    assert (p["n_splits"], p["seed"], p["shuffle"]) == (3, 7, True)


def test_cli_modelo_desconhecido_e_obrigatorio():
    with pytest.raises(SystemExit):
        train_article_model.main(["--model", "naive-bayes"])
    with pytest.raises(SystemExit):
        train_article_model.main([])


def test_cli_amostra_pequena_demais_da_erro_claro(article_csvs, tmp_path, capsys):
    train_csv, test_csv = article_csvs
    with pytest.raises(SystemExit) as exc:
        train_article_model.main([
            "--model", "svm", "--train", str(train_csv), "--test", str(test_csv),
            "--sample-n", "20", "--output-dir", str(tmp_path),
        ])
    assert exc.value.code == 2
    assert "n_splits" in capsys.readouterr().err


def test_cli_arquivo_inexistente(tmp_path, capsys):
    with pytest.raises(SystemExit):
        train_article_model.main(["--model", "svm", "--train", str(tmp_path / "x.csv")])
    assert "não encontrado" in capsys.readouterr().err


def test_cli_roda_como_script_pela_raiz(article_csvs, tmp_path):
    train_csv, test_csv = article_csvs
    rc = subprocess.run(
        [sys.executable, "scripts/train_article_model.py", "--model", "svm",
         "--train", str(train_csv), "--test", str(test_csv), "--n-iter", "1", "--n-jobs", "1",
         "--output-dir", str(tmp_path)],
        capture_output=True, text=True, timeout=300,
    )
    assert rc.returncode == 0, rc.stderr
    assert (tmp_path / "metrics" / "svm_rq1.json").exists()


def test_cli_svm_tfidf_com_comparacao_na_base_completa(article_csvs, tmp_path):
    train_csv, test_csv = article_csvs
    base = tmp_path / "baseline.json"
    base.write_text(json.dumps({"metrics_logreg": {
        "accuracy": 0.5, "f1_macro": 0.5, "mae": 0.7, "rmse": 1.1}}), encoding="utf-8")
    out = tmp_path / "saida"
    rc = train_svm_tfidf.main([
        "--train", str(train_csv), "--test", str(test_csv), "--processed-dir", str(tmp_path / "proc"),
        "--output-dir", str(out), "--baseline-json", str(base),
    ])
    assert rc == 0
    payload = json.loads((out / "metrics" / "svm_tfidf.json").read_text(encoding="utf-8"))
    assert payload["params"] == {"kernel": "linear", "C": 1.0}
    assert set(payload["comparison_with_baseline"]["delta_svm_minus_logreg"]) == {
        "accuracy", "f1_macro", "mae", "rmse"}


def test_cli_svm_tfidf_modo_rapido_nao_compara(article_csvs, tmp_path):
    train_csv, test_csv = article_csvs
    out = tmp_path / "saida"
    train_svm_tfidf.main([
        "--train", str(train_csv), "--test", str(test_csv), "--processed-dir", str(tmp_path / "proc"),
        "--output-dir", str(out), "--sample-n", "100",
    ])
    payload = json.loads((out / "metrics" / "svm_tfidf.json").read_text(encoding="utf-8"))
    assert payload["sample_n"] == 100 and payload["comparison_with_baseline"] is None
