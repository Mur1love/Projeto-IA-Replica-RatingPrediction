"""CLI fim a fim do baseline (Issue 7) — RED."""

import json
import subprocess
import sys
from pathlib import Path


def test_cli_sample_gera_json_e_png(tmp_path):
    metrics_p = tmp_path / "m.json"
    fig_p = tmp_path / "cm.png"
    rc = subprocess.run(
        [
            sys.executable,
            "scripts/train_baseline.py",
            "--sample-n",
            "400",
            "--seed",
            "42",
            "--out-metrics",
            str(metrics_p),
            "--out-figure",
            str(fig_p),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert rc.returncode == 0, rc.stderr
    assert metrics_p.exists() and fig_p.exists()
    payload = json.loads(metrics_p.read_text(encoding="utf-8"))
    assert set(payload["metrics_logreg"]) == {"accuracy", "f1_macro", "mae", "rmse"}
