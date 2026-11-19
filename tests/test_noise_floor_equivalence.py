"""噪声底 TOST 等效性检验（规则 #11）。

外部审查指出：旧口径"配对 t 检验不显著 ⇒ 在噪声内"是 absence of evidence，不能当成
"两臂等价"。这组测试固定新口径的三条底线：CI 整段落在 ±δ 内才是 equivalent=true；
CI 整段在外是 equivalent=false；n<2 必须 equivalent=null，绝不许当等价；δ<=0 报错。
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

def _utf8_env() -> dict:
    """Child environment that forces UTF-8 stdio regardless of the console codepage."""
    return {**os.environ, "PYTHONIOENCODING": "utf-8"}



ROOT = Path(__file__).resolve().parents[1]


def _load_noise_floor():
    spec = importlib.util.spec_from_file_location(
        "noise_floor_under_test", ROOT / "scripts" / "noise_floor.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


NF = _load_noise_floor()


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    # 子进程默认用 Windows 本地编码（cp936）写中文，这里强制 UTF-8，避免解码歧义
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    return subprocess.run(
        [sys.executable, "scripts/noise_floor.py", *args],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
        errors="replace", env=env,
    )


def _json_row(result: subprocess.CompletedProcess) -> dict:
    rows = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
    assert rows, f"stdout 没有 JSON 行：{result.stdout!r} / {result.stderr!r}"
    return rows[-1]


# --------------------------------------------------------------------------- 单元：TOST


def test_equivalent_when_ci_inside_margin():
    """两个几乎相同的合成样本：CI 整段在 ±δ 内 ⇒ equivalent=true。"""
    deltas = [0.004, -0.003, 0.001, 0.0, -0.002, 0.003, -0.001, 0.002]
    row = NF.equivalence_test(deltas, 1.0, "测试固定 δ")
    assert row["equivalent"] is True
    assert row["ci_within_margin"] is True
    assert row["ci_outside_margin"] is False
    assert row["ci_low"] > -1.0 and row["ci_high"] < 1.0
    assert row["margin"] == 1.0
    assert row["mean_delta"] == pytest.approx(0.0005, abs=1e-4)
    assert row["ci_level"] == 0.90
    assert "等价" in row["equivalence_verdict"]


def test_not_equivalent_when_samples_far_apart():
    """两臂差很远：CI 整段落在 ±δ 外 ⇒ equivalent=false 且方向明确。"""
    row = NF.equivalence_test([10.0, 10.4, 9.8, 10.1, 10.2], 1.0, "测试固定 δ")
    assert row["equivalent"] is False
    assert row["ci_outside_margin"] is True
    assert row["ci_within_margin"] is False
    assert row["ci_low"] > 1.0
    assert row["mean_delta"] == pytest.approx(10.1, abs=1e-3)
    assert "不等价" in row["equivalence_verdict"]


def test_single_run_cannot_be_called_equivalent():
    """n<2：equivalent=null，且必须带"样本不足"的说明，不许当成等价。"""
    row = NF.equivalence_test([0.5], 1.0, "测试固定 δ")
    assert row["equivalent"] is None
    assert row["equivalent"] is not True
    assert row["ci_low"] is None and row["ci_high"] is None
    assert "样本不足" in row["equivalence_verdict"]
    assert "n=1 < 2" in row["note"]
    assert NF.equivalence_test([], 1.0)["equivalent"] is None


@pytest.mark.parametrize("bad", [0.0, -1.0, -1e-9])
def test_non_positive_margin_is_rejected(bad):
    with pytest.raises(ValueError):
        NF.equivalence_test([0.1, 0.2, 0.3], bad)
    # paired 必须在跑基准之前就拒绝非法 δ（fail fast，不浪费一次实验）
    with pytest.raises(ValueError):
        NF.paired("branin", "sphere2d", "bayes", 6, 2, [0, 1], "best", margin=bad)


def test_no_margin_available_is_not_equivalent():
    """离散度为 0 / 样本不足时推不出正 δ：只能是 null，不能默认等价。"""
    row = NF.equivalence_test([1.0, 2.0], None, "无 δ")
    assert row["margin"] is None
    assert row["equivalent"] is None
    assert "有效 δ" in row["equivalence_verdict"]
    assert row["note"] and "--equivalence-margin" in row["note"]


def test_suggested_margin_is_same_input_dispersion():
    """默认 δ 来自 --mode floor 口径的同输入重复测量离散度（合并 sd）。"""
    values = [1.0, 1.1, 0.9, 1.05, 0.95]
    margin, source = NF.suggested_margin(values, values)
    assert margin == pytest.approx(NF.statistics.stdev(values), rel=1e-9)
    assert "sd" in source
    assert NF.suggested_margin([1.0], values)[0] is None
    assert NF.suggested_margin([1.0, 1.0], [1.0, 1.0])[0] is None


# ----------------------------------------------------- 单元：paired 端到端（合成、无真实跑）


def test_paired_equivalent_with_default_margin(monkeypatch):
    """两臂几乎一致 ⇒ 默认 δ 下 equivalent=true。"""
    arm_a = [1.0, 1.1, 0.9, 1.05, 0.95]
    offsets = [0.010, 0.012, 0.008, 0.011, 0.009]

    def fake_measure(problem, algorithm, budget, batch, seed, metric):
        index = seed if seed < len(arm_a) else seed % len(arm_a)
        return arm_a[index] - (offsets[index] if problem == "b" else 0.0)

    monkeypatch.setattr(NF, "measure", fake_measure)
    row = NF.paired("a", "b", "bayes", 10, 2, [0, 1, 2, 3, 4], "best")
    assert row["margin"] > 0
    assert row["equivalent"] is True
    assert row["ci_low"] > -row["margin"] and row["ci_high"] < row["margin"]
    assert "合并 sd" in row["margin_source"]


def test_paired_far_apart_arms_are_not_equivalent(monkeypatch):
    def fake_measure(problem, algorithm, budget, batch, seed, metric):
        return 50.0 + 0.1 * seed if problem == "a" else 10.0 + 0.1 * seed

    monkeypatch.setattr(NF, "measure", fake_measure)
    row = NF.paired("a", "b", "bayes", 10, 2, [0, 1, 2, 3], "best")
    assert row["equivalent"] is False
    assert row["ci_outside_margin"] is True
    assert row["mean_delta"] == pytest.approx(40.0)


def test_paired_single_seed_reports_null(monkeypatch):
    monkeypatch.setattr(NF, "measure", lambda *a, **k: 1.0)
    row = NF.paired("a", "b", "bayes", 10, 2, [0], "best")
    assert row["n"] == 1
    assert row["equivalent"] is None
    assert row["margin"] is None
    assert "样本不足" in row["equivalence_verdict"]


# --------------------------------------------------------------- 单元：floor 向后兼容


def test_floor_keeps_old_fields_and_suggests_margin(monkeypatch):
    series = [1.0, 1.2, 0.8, 1.1, 0.9]

    def fake_measure(problem, algorithm, budget, batch, seed, metric):
        return series[seed]

    monkeypatch.setattr(NF, "measure", fake_measure)
    row = NF.floor("p", "bayes", 10, 2, [0, 1, 2, 3, 4], "best")
    for key in ("problem", "algorithm", "metric", "n", "values", "mean", "sd", "sem",
                "ci95_half_width", "detectable_delta_2arms", "relative_floor_pct"):
        assert key in row, f"floor 模式丢了旧字段 {key}"
    assert row["sd"] == round(NF.statistics.stdev(series), 4)
    assert row["equivalence_margin_suggested"] == row["sd"]
    assert "sd" in row["equivalence_margin_source"]


# ------------------------------------------------------------------------ CLI（真实小规模）


def test_cli_paired_json_has_equivalence_fields_and_legacy_fields():
    result = _run_cli(
        "--mode", "paired",
        "--problem", "branin", "--problem-b", "sphere2d",
        "--algorithm", "bayes", "--budget", "6", "--metric", "best",
        "--seeds", "0", "1", "--equivalence-margin", "500",
    )
    assert result.returncode == 0, result.stderr[-500:]
    row = _json_row(result)
    for key in ("equivalent", "ci_low", "ci_high", "margin", "mean_delta"):
        assert key in row, f"JSON 缺少等效性字段 {key}"
    # 向后兼容：旧字段与旧的人类可读摘要仍在
    for key in ("significant_95", "t_statistic", "t_critical_95", "verdict", "per_seed_delta"):
        assert key in row
    assert row["equivalent"] is True
    assert row["margin"] == 500.0
    assert "equivalent=True" in result.stderr


def test_cli_rejects_non_positive_margin():
    result = _run_cli(
        "--mode", "paired",
        "--problem", "branin", "--problem-b", "sphere2d",
        "--algorithm", "bayes", "--budget", "6", "--metric", "best",
        "--seeds", "0", "1", "--equivalence-margin", "0",
    )
    assert result.returncode != 0
    assert "必须 > 0" in result.stderr


def test_cli_floor_still_runs_and_reports_margin_source():
    result = _run_cli(
        "--mode", "floor", "--problem", "branin",
        "--algorithm", "bayes", "--budget", "6", "--metric", "best",
        "--seeds", "0", "1",
    )
    assert result.returncode == 0, result.stderr[-500:]
    row = _json_row(result)
    assert "detectable_delta_2arms" in row
    assert row["equivalence_margin_suggested"] == row["sd"]
