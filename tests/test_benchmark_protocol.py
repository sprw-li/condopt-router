"""预注册基准协议（scripts/benchmark_protocol.py）的测试。

覆盖四件事：
① IQM / bootstrap CI 在已知合成输入上的正确性（不跑真优化，保持快速）；
② Δ 的 CI 跨 0 时 verdict 必须是 ``inconclusive``；
③ 共享初始设计在同一 seed 下跨算法一致；
④ 输出 JSON 的固定字段齐全（端到端小规模跑一次）。
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "benchmark_protocol.py"


def _load_protocol() -> ModuleType:
    """按路径加载脚本模块（``scripts/`` 不是包）。"""
    spec = importlib.util.spec_from_file_location("benchmark_protocol", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def protocol() -> ModuleType:
    return _load_protocol()


# -- ① IQM 与 bootstrap CI -------------------------------------------------
def test_iqm_equals_trimmed_mean(protocol):
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    # n=8, trim=0.25 -> k=2 -> 去掉 1,2 与 7,8 -> mean(3,4,5,6)
    assert protocol.iqm(values) == pytest.approx(4.5)

    scattered = [3.0, 1.0, 9.0, 4.0, 5.0, 2.0, 8.0, 6.0, 7.0, 11.0, 10.0, 12.0]
    ordered = sorted(scattered)
    k = int(len(ordered) * 0.25)
    expected = sum(ordered[k: len(ordered) - k]) / (len(ordered) - 2 * k)
    assert protocol.iqm(scattered) == pytest.approx(expected)


def test_iqm_ignores_tail_outlier_and_nan(protocol):
    with_outlier = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1e6]
    assert protocol.iqm(with_outlier) < 6.0  # 算术均值会被 1e6 拉走

    assert protocol.iqm([5.0]) == pytest.approx(5.0)
    assert protocol.iqm([1.0, float("nan"), 3.0]) == pytest.approx(2.0)
    assert math.isnan(protocol.iqm([]))
    assert math.isnan(protocol.iqm([float("nan")]))


def test_bootstrap_ci_brackets_statistic_and_is_reproducible(protocol):
    values = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
    low, high = protocol.bootstrap_ci(values, resamples=400)
    assert low <= protocol.iqm(values) <= high
    assert protocol.bootstrap_ci(values, resamples=400) == (low, high)

    # 常数样本：区间必须退化到该常数
    low_c, high_c = protocol.bootstrap_ci([7.0] * 5, resamples=200)
    assert low_c == pytest.approx(7.0)
    assert high_c == pytest.approx(7.0)


def test_bootstrap_ci_widens_with_dispersion(protocol):
    tight = [10.0, 10.05, 10.1, 9.95, 10.02]
    wide = [1.0, 5.0, 10.0, 15.0, 20.0]
    tight_low, tight_high = protocol.bootstrap_ci(tight, resamples=400)
    wide_low, wide_high = protocol.bootstrap_ci(wide, resamples=400)
    assert (wide_high - wide_low) > (tight_high - tight_low)


# -- ② CI 跨 0 -> inconclusive --------------------------------------------
def test_verdict_inconclusive_when_ci_crosses_zero(protocol):
    assert protocol.paired_verdict(-0.5, 0.5, 0.1, 0.0) == "inconclusive"
    assert protocol.paired_verdict(-1.0, 0.2, -0.4, 0.0) == "inconclusive"
    assert protocol.paired_verdict(-0.01, 1.0, 0.5, 0.0) == "inconclusive"
    # 区间退化成一个点 0 也算跨 0
    assert protocol.paired_verdict(0.0, 0.0, 0.0, 0.0) == "inconclusive"
    # 非有限值同样不允许下结论
    assert protocol.paired_verdict(float("nan"), 1.0, 0.5, 0.0) == "inconclusive"


def test_verdict_sign_and_noise_floor(protocol):
    assert protocol.paired_verdict(0.5, 2.0, 1.0, 0.0) == "better"
    assert protocol.paired_verdict(-2.0, -0.5, -1.0, 0.0) == "worse"
    # |Δ| 在噪声底内 -> 不许叫改进
    assert protocol.paired_verdict(0.5, 2.0, 1.0, 1.0) == "inconclusive"


def test_paired_effect_overlapping_samples_is_inconclusive(protocol):
    a = [1.0, 2.0, 3.0, 4.0]
    b = [1.5, 1.5, 3.5, 3.5]
    effect = protocol.paired_effect(a, b, noise_floor=0.0, resamples=400)
    assert effect["n_pairs"] == 4
    assert effect["verdict"] == "inconclusive"
    assert effect["ci_low"] <= effect["mean_diff"] <= effect["ci_high"]

    # 每一对都稳定 +3：CI 不跨 0，必须给出方向与相对百分比
    baseline = [1.0, 2.0, 3.0, 4.0]
    shifted = [value + 3.0 for value in baseline]
    clear = protocol.paired_effect(baseline, shifted, noise_floor=0.0, resamples=400)
    assert clear["mean_diff"] == pytest.approx(-3.0)
    assert clear["verdict"] == "worse"  # A 是 baseline，比 shifted 差
    assert clear["relative_pct"] == pytest.approx(-100.0 * 3.0 / sum(shifted) * 4)


def test_paired_effect_drops_incomplete_pairs(protocol):
    effect = protocol.paired_effect(
        [1.0, float("nan"), 3.0], [2.0, 5.0, 4.0], noise_floor=0.0, resamples=200
    )
    assert effect["n_pairs"] == 2


# -- ③ 共享初始设计 --------------------------------------------------------
def test_shared_initial_design_is_deterministic_per_seed(protocol):
    problem = protocol.builtin_problems()["discrete_catalyst"]
    first = protocol.shared_initial_design(problem, seed=3, n_points=4)
    second = protocol.shared_initial_design(problem, seed=3, n_points=4)
    assert first == second
    assert protocol.design_digest(first) == protocol.design_digest(second)
    assert len(first) == 4
    # 不同 seed 必须是不同设计，否则种子集合没有意义
    assert first != protocol.shared_initial_design(problem, seed=4, n_points=4)
    assert protocol.shared_initial_design(problem, seed=3, n_points=0) == []


def test_shared_initial_design_is_identical_across_algorithms(protocol):
    problem = protocol.builtin_problems()["discrete_catalyst"]
    sobol_arm = protocol.run_arm(problem, "sobol", seed=3, budget=4, init=2, batch=1)
    tpe_arm = protocol.run_arm(problem, "tpe", seed=3, budget=4, init=2, batch=1)
    assert sobol_arm["initial_design"] == tpe_arm["initial_design"]
    assert sobol_arm["initial_design_sha"] == tpe_arm["initial_design_sha"]
    assert sobol_arm["initial_design"] == protocol.shared_initial_design(
        problem, seed=3, n_points=2
    )
    assert sobol_arm["n_evals"] == 4 and tpe_arm["n_evals"] == 4
    # 同一配置重复跑必须完全一致（确定性模拟器 + 固定 seed 的算法）
    repeat = protocol.run_arm(problem, "tpe", seed=3, budget=4, init=2, batch=1)
    assert repeat["value"] == tpe_arm["value"]


# -- ④ 报告字段与端到端 ----------------------------------------------------
def test_metric_name_and_signed_convention(protocol):
    problems = protocol.builtin_problems()
    assert protocol.metric_name(problems["discrete_catalyst"]) == "best"
    moo = problems["moo_yield_cost"]
    assert protocol.metric_name(moo) == "hypervolume"
    observations = [
        moo.make_observation({"T": 80.0, "time": 40.0}),
        moo.make_observation({"T": 90.0, "time": 60.0}),
    ]
    assert protocol.metric_value(moo, observations) >= 0.0


def test_report_has_required_fields(protocol):
    rows = [
        {"algorithm": "a", "seed": seed, "value": value, "run_kind": "main"}
        for seed, value in enumerate([1.0, 2.0, 3.0, 4.0])
    ]
    rows.append({"algorithm": "b", "seed": 0, "value": 2.0, "run_kind": "repeat"})
    rows += [
        {"algorithm": "b", "seed": seed, "value": value, "run_kind": "main"}
        for seed, value in enumerate([1.5, 2.5, 2.0, 3.0])
    ]
    report = protocol.build_report(
        "demo", "best", [0, 1, 2, 3], ["a", "b"], rows,
        {"value": 0.0, "seed": 0}, budget=10, init=3, batch=1, resamples=200,
    )
    for field in protocol.REQUIRED_FIELDS:
        assert field in report, field
    assert report["problem"] == "demo"
    assert report["metric"] == "best"
    assert report["seeds"] == [0, 1, 2, 3]
    assert report["algorithms"] == ["a", "b"]
    assert set(report["iqm"]) == {"a", "b"}
    assert set(report["ci_low"]) == {"a", "b"}
    assert set(report["ci_high"]) == {"a", "b"}
    assert report["n"] == {"a": 4, "b": 4}
    assert report["noise_floor"]["value"] == 0.0
    assert len(report["pairwise"]) == 1
    effect = report["pairwise"][0]
    assert effect["baseline"] == "a" and effect["algorithm"] == "b"
    # repeat 行不参与主汇总
    assert report["n"]["b"] == 4


def test_cli_writes_json_csv_and_summary(tmp_path, protocol):
    out_dir = tmp_path / "protocol"
    code = protocol.main([
        "--problem", "discrete_catalyst",
        "--seeds", "2",
        "--budget", "4",
        "--init", "2",
        "--bootstrap", "200",
        "--out", str(out_dir),
    ])
    assert code == 0
    payload = json.loads((out_dir / "results.json").read_text(encoding="utf-8"))
    for field in protocol.REQUIRED_FIELDS:
        assert field in payload, field
    assert payload["seeds"] == [0, 1]
    assert payload["algorithms"] == ["sobol", "tpe"]
    assert payload["metric"] == "best"
    assert set(payload["n"]) == {"sobol", "tpe"}
    assert payload["pairwise"][0]["verdict"] in {"better", "worse", "inconclusive"}

    csv_text = (out_dir / "per_seed.csv").read_text(encoding="utf-8-sig")
    header = csv_text.splitlines()[0]
    for column in ("algorithm", "seed", "value", "run_kind", "initial_design_sha"):
        assert column in header
    assert "repeat" in csv_text and "main" in csv_text

    summary = (out_dir / "summary.md").read_text(encoding="utf-8")
    assert "IQM" in summary and "noise_floor" in summary
    assert "inconclusive" in summary
