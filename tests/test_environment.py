"""Testes do script de verificação do ambiente (Issue #1)."""

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
    )


def test_check_environment_script_exits_successfully():
    result = _run_check_script()
    assert result.returncode == 0, result.stderr


def test_check_environment_script_reports_each_main_dependency():
    result = _run_check_script()
    for dependency in MAIN_DEPENDENCIES:
        assert dependency in result.stdout
