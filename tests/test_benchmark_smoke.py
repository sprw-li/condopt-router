"""Benchmark scripts must at least *run* (one was silently broken by an edit)."""

from __future__ import annotations

import os

import subprocess
import sys
from pathlib import Path

def _utf8_env() -> dict:
    """Child environment that forces UTF-8 stdio regardless of the console codepage."""
    return {**os.environ, "PYTHONIOENCODING": "utf-8"}



ROOT = Path(__file__).resolve().parents[1]


def _run(script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, f"scripts/{script}", *args],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
        env=_utf8_env(),
    )


def test_reliability_band_benchmark_runs(tmp_path):
    result = _run(
        "benchmark_transfer_reliability_band.py",
        "--budget", "6", "--seeds", "0", "--out", str(tmp_path / "band.csv"),
    )
    assert result.returncode == 0, result.stderr[-500:]
    assert "mechanism_alpha_max" in result.stdout


def test_transfer_top_k_sweep_runs(tmp_path):
    result = _run(
        "sweep_transfer_top_k.py",
        "--top-ks", "0", "1", "--budget", "8", "--seeds", "0",
    )
    assert result.returncode == 0, result.stderr[-500:]
    assert "imported_mean" in result.stdout

def test_audit_calibration_benchmark_runs(tmp_path):
    result = _run(
        "benchmark_audit_calibration.py",
        "--problem", "ackley10", "--budget", "8", "--source-points", "8",
        "--seeds", "0", "--out", str(tmp_path / "audit.csv"),
    )
    assert result.returncode == 0, result.stderr[-500:]
    assert "audit_mean" in result.stdout

def test_adaptive_transfer_benchmark_runs(tmp_path):
    result = _run(
        "benchmark_transfer_adaptive.py",
        "--budget", "8", "--source-points", "8", "--seeds", "0",
        "--out", str(tmp_path / "adaptive.csv"),
    )
    assert result.returncode == 0, result.stderr[-500:]
    assert "staged_audit" in result.stdout

def test_noise_floor_script_runs(tmp_path):
    result = _run(
        "noise_floor.py", "--mode", "paired",
        "--problem", "branin", "--problem-b", "sphere2d",
        "--algorithm", "bayes", "--budget", "6", "--metric", "best",
        "--seeds", "0", "1",
    )
    assert result.returncode == 0, result.stderr[-500:]
    assert "significant_95" in result.stdout

