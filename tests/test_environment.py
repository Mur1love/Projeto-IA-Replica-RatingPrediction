"""Testes do script de verificação do ambiente (Issue #1).

A lista MAIN_DEPENDENCIES deve ser mantida em sincronia com o mapa
DEPENDENCIES de scripts/check_environment.py.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECK_SCRIPT = REPO_ROOT / "scripts" / "check_environment.py"

MAIN_DEPENDENCIES = [
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "matplotlib",
    "jupyter",
    "pytest",
    "nltk",
]


def _run_check_script() -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(CHECK_SCRIPT)],
        capture_output=True,
        text=True,
        timeout=60,
    )


def _load_check_module():
    spec = importlib.util.spec_from_file_location("check_environment", CHECK_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_check_environment_script_exits_successfully():
    result = _run_check_script()
    assert result.returncode == 0, result.stderr


def test_check_environment_script_reports_each_main_dependency():
    result = _run_check_script()
    for dependency in MAIN_DEPENDENCIES:
        assert f"[OK] {dependency} " in result.stdout


def test_check_environment_reports_broken_package_and_keeps_checking(tmp_path, monkeypatch, capsys):
    broken = tmp_path / "pacote_corrompido.py"
    broken.write_text("raise RuntimeError('instalacao corrompida')\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    check = _load_check_module()
    monkeypatch.setattr(
        check,
        "DEPENDENCIES",
        {"pacote-corrompido": "pacote_corrompido", "numpy": "numpy"},
    )

    assert check.main() == 1
    output = capsys.readouterr().out
    assert "[FALHA] pacote-corrompido" in output
    assert "[OK] numpy" in output


def test_check_environment_reports_unknown_version_when_metadata_missing(monkeypatch, capsys):
    check = _load_check_module()
    monkeypatch.setattr(check, "DEPENDENCIES", {"sem-metadata": "math"})

    assert check.main() == 0
    assert "[OK] sem-metadata ?" in capsys.readouterr().out
