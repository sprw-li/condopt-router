"""A6: the release gate must not turn "could not run" or "partial" into "passed"."""

from __future__ import annotations

import os

import json
import subprocess
import sys
from pathlib import Path

def _utf8_env() -> dict:
    """Child environment that forces UTF-8 stdio regardless of the console codepage."""
    return {**os.environ, "PYTHONIOENCODING": "utf-8"}



ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "release_check.py"


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
        env=_utf8_env(),
    )


def test_partial_run_reports_partial_and_exits_3():
    result = _run("--skip-slow")
    assert result.returncode == 3, result.stdout[-400:]
    assert "PARTIAL RUN" in result.stdout
    assert "NOT a release gate" in result.stdout
    # it must not claim a complete pass
    assert "passed" not in result.stdout.split("PARTIAL RUN")[-1].split("\n")[-2]


def test_allow_partial_makes_the_opt_out_explicit():
    result = _run("--skip-slow", "--allow-partial")
    assert result.returncode == 0, result.stdout[-300:]
    assert "PARTIAL RUN" in result.stdout


def test_json_reports_skipped_separately():
    result = _run("--skip-slow", "--json")
    payload = json.loads(result.stdout)
    assert payload["partial"] is True
    assert "skipped" in payload and "external_drift" in payload
    # every check carries an explicit status, never a bare boolean
    for check in payload["checks"]:
        assert check["status"] in {"ok", "fail", "skipped", "external_drift"}, check


def test_missing_eval_script_is_skipped_not_passed(tmp_path):
    result = _run("--skip-slow", "--json", "--eval-script", str(tmp_path / "missing.py"))
    payload = json.loads(result.stdout)
    eval_check = next(
        check for check in payload["checks"] if check["check"] == "eval_router"
    )
    assert eval_check["status"] == "skipped"
    assert eval_check["ok"] is False, "a gate that could not run is not a pass"
