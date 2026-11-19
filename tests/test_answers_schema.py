"""A8：``prepare`` 的 answers 有白名单 schema，未知键报错并提示最相近的合法键。

拼错的键以前被 ``prepare.parse_answers`` 静默忽略（agent 拿到"成功"但其实是错的
结果）。这些用例同时锁住三件事：difflib 候选键、必填/类型校验、合法输入不受影响
（向后兼容），以及错误经由 A7 的结构化错误包返回。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from condopt.errors import InputError
from condopt.prepare import template
from condopt.schema import KNOWN_ANSWERS_KEYS, validate_answers

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts" / "run_condopt.py"

VALID_ANSWERS = {
    "variables": [
        {"name": "T", "kind": "continuous", "low": 40, "high": 120, "unit": "degC"},
        {"name": "solvent", "kind": "categorical", "choices": ["MeCN", "EtOH"]},
    ],
    "objective": {"name": "yield", "direction": "max"},
    "budget": 24,
    "batch": 4,
    "noise": "medium",
    "smoothness": "smooth",
    "domain": "chemistry",
    "prior": 0,
    "constraints": 0,
    "notes": "小试",
}


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(CLI), *args],
        cwd=ROOT,
        capture_output=True,
        timeout=60,
    )


# ---------------------------------------------------------------------------
# 校验器本身
# ---------------------------------------------------------------------------


def test_template_is_accepted_unchanged():
    """skill 文档里的模板必须仍然通过（向后兼容）。"""
    answers = template()
    assert validate_answers(answers) is answers


def test_valid_answers_pass():
    assert validate_answers(VALID_ANSWERS) is VALID_ANSWERS


def test_space_variables_alias_still_passes():
    """``space.variables`` 是 ``variables`` 的等价写法，不能被当成缺必填项。"""
    space_only = {"space": {"variables": VALID_ANSWERS["variables"]}, "budget": 12}
    assert validate_answers(space_only) is space_only


def test_misspelled_key_lists_the_nearest_candidate():
    with pytest.raises(InputError) as error:
        validate_answers({**VALID_ANSWERS, "constaint": 3})
    message = str(error.value)
    assert "constaint" in message
    assert "constraints" in message, "必须用 difflib 给出最相近的候选键"
    assert "未知字段" in message


def test_misspelled_key_inside_a_variable_lists_the_candidate():
    answers = {
        "variables": [{"name": "T", "knd": "continuous", "low": 0, "high": 1}],
        "budget": 10,
    }
    with pytest.raises(InputError, match="kind"):
        validate_answers(answers)


def test_missing_required_field_raises():
    with pytest.raises(InputError, match="variables"):
        validate_answers({"budget": 10, "batch": 2})


def test_wrong_type_raises():
    with pytest.raises(InputError, match="answers.budget"):
        validate_answers({**VALID_ANSWERS, "budget": "twenty-four"})
    with pytest.raises(InputError, match="answers.noise"):
        validate_answers({**VALID_ANSWERS, "noise": "loud"})


def test_non_dict_answers_raise():
    with pytest.raises(InputError, match="object"):
        validate_answers([{"variables": []}])
    with pytest.raises(InputError, match="object"):
        validate_answers("variables")


def test_known_keys_cover_the_documented_fields():
    """白名单必须覆盖 prepare 实际读取的键，否则合法输入会被误拒。"""
    documented = {
        "objective", "variables", "budget", "batch", "noise", "smoothness",
        "domain", "prior", "constraints", "notes", "spec", "multi_fidelity",
        "feasible_fraction_estimate", "spec_calibrate_from",
    }
    assert documented <= set(KNOWN_ANSWERS_KEYS)


# ---------------------------------------------------------------------------
# CLI 入口（错误走 A7 的结构化错误包）
# ---------------------------------------------------------------------------


def _write_answers(tmp_path: Path, payload) -> Path:
    path = tmp_path / "answers.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_cli_reports_a_misspelled_key_as_json_input_error(tmp_path):
    answers = _write_answers(tmp_path, {**VALID_ANSWERS, "constaint": 3})
    result = _run("prepare", "--answers", str(answers), "--out", str(tmp_path / "ws"))
    assert result.returncode == 2, result.stderr
    text = result.stderr.decode("utf-8")
    lines = [line for line in text.splitlines() if line.strip()]
    assert len(lines) == 1, text[:300]
    payload = json.loads(lines[0])
    assert payload["ok"] is False
    assert payload["error"]["type"] == "InputError"
    assert "constraints" in payload["error"]["message"]
    assert not (tmp_path / "ws" / "space.json").exists(), "校验失败不得写出问题文件"


def test_cli_still_accepts_a_valid_template(tmp_path):
    answers = _write_answers(tmp_path, template())
    out = tmp_path / "ws"
    result = _run("prepare", "--answers", str(answers), "--out", str(out))
    assert result.returncode == 0, result.stderr
    assert (out / "space.json").exists()
    assert (out / "context.json").exists()
    assert (out / "objective.json").exists()


def test_cli_debug_traceback_is_also_utf8(tmp_path):
    """含中文消息的 traceback 也必须整体是 UTF-8：agent 要能直接解第一行 JSON。"""
    answers = _write_answers(tmp_path, {**VALID_ANSWERS, "constaint": 3})
    result = _run(
        "prepare", "--answers", str(answers), "--out", str(tmp_path / "ws"), "--debug"
    )
    assert result.returncode == 2
    text = result.stderr.decode("utf-8")  # 解不开就是契约被破坏
    assert "Traceback" in text
    assert json.loads(text.splitlines()[0])["error"]["type"] == "InputError"
