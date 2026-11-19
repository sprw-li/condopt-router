"""A1 (backlog): ``init`` must not silently rewrites a populated campaign.

Review finding: re-running the documented ``init`` command on an existing run reset
``algorithm`` (tpe -> sobol) while keeping ``dataset.jsonl`` and writing no decision
record -- contradicting the project's "every override is logged" rule.
"""

from __future__ import annotations

import os

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

def _utf8_env() -> dict:
    """Child environment that forces UTF-8 stdio regardless of the console codepage."""
    return {**os.environ, "PYTHONIOENCODING": "utf-8"}



ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts" / "run_condopt.py"

SPACE = {
    "variables": [{"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0}]
}


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(CLI), *args],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
        env=_utf8_env(),
    )


def _make_run(tmp_path: Path, algorithm: str) -> Path:
    space_path = tmp_path / "space.json"
    space_path.write_text(json.dumps(SPACE), encoding="utf-8")
    run = tmp_path / "run"
    result = _run(
        "init", "--run", str(run), "--space", str(space_path),
        "--objective", "y:max", "--algorithm", algorithm,
        "--budget", "10", "--batch", "2",
    )
    assert result.returncode == 0, result.stderr
    return run


def _observe(run: Path) -> None:
    ask = _run("ask", "--run", str(run), "--n", "2")
    assert ask.returncode == 0, ask.stderr
    pending = json.loads((run / "pending.json").read_text(encoding="utf-8"))
    payload = [{"id": item["id"], "y": 1.0} for item in pending]
    results = run.parent / "results.json"
    results.write_text(json.dumps(payload), encoding="utf-8")
    tell = _run("tell", "--run", str(run), "--results", str(results))
    assert tell.returncode == 0, tell.stderr


def test_second_init_is_refused_by_default(tmp_path):
    run = _make_run(tmp_path, "tpe")
    _observe(run)
    again = _run(
        "init", "--run", str(run), "--space", str(tmp_path / "space.json"),
        "--objective", "y:max", "--algorithm", "sobol", "--budget", "10", "--batch", "2",
    )
    assert again.returncode == 2, again.stdout + again.stderr
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    assert config["algorithm"] == "tpe", "the refusal must not touch the config"


def test_if_exists_keep_changes_the_algorithm_and_logs_it(tmp_path):
    run = _make_run(tmp_path, "tpe")
    _observe(run)
    kept = _run(
        "init", "--run", str(run), "--space", str(tmp_path / "space.json"),
        "--objective", "y:max", "--algorithm", "sobol", "--budget", "10", "--batch", "2",
        "--if-exists", "keep",
    )
    assert kept.returncode == 0, kept.stderr
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    assert config["algorithm"] == "sobol"
    # observations survive, and the change is auditable
    assert len((run / "dataset.jsonl").read_text(encoding="utf-8").strip().splitlines()) == 2
    decisions = (run / "decisions.jsonl").read_text(encoding="utf-8")
    assert "campaign_reset" in decisions
    assert (run / "warm_start.json").exists()


def test_if_exists_reset_starts_clean(tmp_path):
    run = _make_run(tmp_path, "tpe")
    _observe(run)
    reset = _run(
        "init", "--run", str(run), "--space", str(tmp_path / "space.json"),
        "--objective", "y:max", "--algorithm", "sobol", "--budget", "10", "--batch", "2",
        "--if-exists", "reset",
    )
    assert reset.returncode == 0, reset.stderr
    assert (run / "dataset.jsonl").exists() is False or not (
        run / "dataset.jsonl"
    ).read_text(encoding="utf-8").strip()


def test_skill_documents_the_guard():
    skill = (ROOT / "skill" / "SKILL.md").read_text(encoding="utf-8")
    assert "--if-exists" in skill, "the skill must tell the agent about the init guard"
