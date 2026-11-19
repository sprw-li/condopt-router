"""A7：CLI 已知错误必须落成「退出码 + stderr 一行 JSON」，默认不打 traceback。

用真实子进程跑 ``scripts/run_condopt.py``（agent 的实际调用方式），而不是在进程内
调 ``main``：只有子进程才能证明退出码真的传到了 shell，且 stderr 的字节是可解析
的 JSON（错误包由 ``cli._write_stderr`` 固定写成 UTF-8）。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts" / "run_condopt.py"

SPACE = {"variables": [{"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0}]}
#: 退出码表见 src/condopt/cli.py 的模块 docstring 与 src/condopt/errors.py。
EXIT_INPUT = 2
EXIT_STATE = 3
EXIT_ENVIRONMENT = 4

#: ``error.type`` 是具体异常类名：CLI 抛 InputError/StateError，库层透传 ValueError/
#: FileNotFoundError（见 errors.error_payload 的说明）。退出码才是"族"。
INPUT_TYPES = {"InputError", "ValueError", "TypeError", "json.JSONDecodeError"}
ENVIRONMENT_TYPES = {"EnvironmentError_", "FileNotFoundError"}


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(CLI), *args],
        cwd=ROOT,
        capture_output=True,
        timeout=60,
    )


def _single_line_error(result: subprocess.CompletedProcess) -> dict:
    """断言 stderr 是一行合法 JSON 错误包，并返回它。"""
    text = result.stderr.decode("utf-8")
    lines = [line for line in text.splitlines() if line.strip()]
    assert len(lines) == 1, f"stderr 应为单行 JSON，实际 {len(lines)} 行：{text[:300]}"
    assert "Traceback" not in text, "默认（无 --debug）不得打印 traceback"
    payload = json.loads(lines[0])
    assert payload["ok"] is False
    assert payload["error"]["type"], payload
    assert payload["error"]["message"], payload
    assert isinstance(payload["error"]["hint"], str), payload
    return payload


def test_init_without_space_is_an_input_error(tmp_path):
    result = _run("init", "--run", str(tmp_path / "run"))
    assert result.returncode == EXIT_INPUT, result.stderr
    assert _single_line_error(result)["error"]["type"] == "InputError"
    assert "--space" in _single_line_error(result)["error"]["message"]


def test_init_with_invalid_space_is_an_input_error(tmp_path):
    space = tmp_path / "bad_space.json"
    # low > high 由 SearchSpace 拒绝（ValueError）→ 输入错误 2，而不是 traceback。
    space.write_text(
        json.dumps({"variables": [{"name": "x", "kind": "continuous", "low": 5, "high": 1}]}),
        encoding="utf-8",
    )
    result = _run("init", "--run", str(tmp_path / "run"), "--space", str(space))
    assert result.returncode == EXIT_INPUT, result.stderr
    assert _single_line_error(result)["error"]["type"] in INPUT_TYPES


def test_ask_on_uninitialized_run_is_a_state_error(tmp_path):
    result = _run("ask", "--run", str(tmp_path / "never-inited"), "--json")
    assert result.returncode == EXIT_STATE, result.stderr
    payload = _single_line_error(result)
    assert payload["error"]["type"] == "StateError"
    assert "init" in payload["error"]["message"]
    assert "init" in payload["error"]["hint"]


def test_missing_input_file_is_an_environment_error(tmp_path):
    result = _run(
        "init", "--run", str(tmp_path / "run"), "--space", str(tmp_path / "missing.json")
    )
    assert result.returncode == EXIT_ENVIRONMENT, result.stderr
    assert _single_line_error(result)["error"]["type"] in ENVIRONMENT_TYPES


def test_tell_with_unknown_pending_id_is_a_state_error(tmp_path):
    # 用库直接准备 run（省掉两次 CLI 冷启动），被测路径仍是真实子进程 tell。
    from condopt.base import Objective
    from condopt.space import SearchSpace
    from condopt.study import Study

    run = tmp_path / "run"
    study = Study.create(
        run,
        SearchSpace.from_dict(SPACE),
        Objective(name="y", direction="max"),
        budget=8,
        batch_size=2,
        seed=0,
    )
    study.ask(2)

    results = tmp_path / "results.json"
    results.write_text(json.dumps([{"id": "r9-9", "y": 1.0}]), encoding="utf-8")
    result = _run("tell", "--run", str(run), "--results", str(results))
    assert result.returncode == EXIT_STATE, result.stderr
    payload = _single_line_error(result)
    assert payload["error"]["type"] == "StateError"
    assert "unknown pending id" in payload["error"]["message"]
    assert "pending" in payload["error"]["hint"]


def test_debug_adds_the_traceback_after_the_json_line(tmp_path):
    result = _run("ask", "--run", str(tmp_path / "never-inited"), "--debug")
    assert result.returncode == EXIT_STATE, result.stderr
    text = result.stderr.decode("utf-8")
    assert "Traceback" in text
    # JSON 行仍然在最前面，agent 不需要 --debug 也能分支
    assert json.loads(text.splitlines()[0])["error"]["type"] == "StateError"
