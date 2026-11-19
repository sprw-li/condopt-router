"""预注册基准协议：共享初始设计 + 显式种子集 + IQM/bootstrap CI + 配对效应量。

为什么需要这个脚本：仓库里既有的 ``scripts/benchmark_*.py`` 是**工程 smoke**
——每个算法各自随机初始设计、通常只跑 3-4 个种子、只报均值，达不到"可比较、
可复现"的协议要求（缺共享初始设计、种子数不足、缺 IQM/区间/效应量）。本脚本把
协议落成可执行代码：

* 共享初始设计：同一 seed 下所有算法看到**同一组**初始点（由 seed 派生的
  Sobol 设计点），并作为 ``prior`` 注入每个优化器；
* 种子集合显式声明：``--seeds N`` 取 ``range(N)``，实际种子列表写进输出；
* 每个算法每个种子记录最终指标（单目标 = 方向归一化后的 best，多目标 = HV）；
* 汇总用 IQM（四分位均值）而不是算术均值，并给 bootstrap 95% CI；
* 同一 seed 下配对比较 A-B：均值差 + CI + 相对百分比；
* 噪声底：用**完全相同的配置**重复跑一次，取 |重复 - 原值| 的离散度。

关键纪律：当 Δ 的 CI 跨 0（或 |Δ| 落在噪声底内）时，verdict 必须是
``inconclusive``，不许写成"更好/更差"。

运行：
    py -3 scripts/benchmark_protocol.py --seeds 3 --budget 12
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from condopt import registry  # noqa: E402
import condopt.algorithms  # noqa: E402,F401  导入即注册全部内置算法
from condopt.benchmarks import BenchmarkProblem, builtin_problems  # noqa: E402
from condopt.dataset import Observation  # noqa: E402
from condopt.pareto import summarise as pareto_summarise  # noqa: E402

def _force_utf8_stdio() -> None:
    """Emit UTF-8 regardless of the Windows console codepage (cp936 here)."""
    import sys

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except (AttributeError, ValueError):  # pragma: no cover
            pass


#: 协议版本：字段含义变化时必须递增，便于外部审查对齐。
PROTOCOL_ID = "condopt-benchmark-protocol/1"

DEFAULT_PROBLEM = "discrete_catalyst"
DEFAULT_ALGORITHMS = ("sobol", "tpe")
#: 默认种子数：协议要求显式 >= 20；测试里用 --seeds 3 保持快速。
DEFAULT_SEEDS = 20
DEFAULT_BUDGET = 30
DEFAULT_INIT = 5
DEFAULT_BATCH = 1
DEFAULT_BOOTSTRAP = 2000
#: 固定 bootstrap 随机种子：区间本身也必须可复现。
BOOTSTRAP_SEED = 20261117
#: 输出 JSON 必须含有的固定字段（外部审查按名字取值）。
REQUIRED_FIELDS = (
    "problem",
    "metric",
    "seeds",
    "algorithms",
    "iqm",
    "ci_low",
    "ci_high",
    "n",
    "noise_floor",
    "pairwise",
)

OUT_DIR = ROOT / "runs" / "benchmark_protocol"


# -- 统计量 ---------------------------------------------------------------
def iqm(values: Sequence[float], trim: float = 0.25) -> float:
    """四分位均值（interquartile mean）：去掉上下各 ``trim`` 比例后取均值。

    为什么要 IQM 而不是算术均值：单个种子的崩溃（算法在某个 landscape 上
    发散）会把算术均值整个拉走，外部审查也就无法判断"差异"到底来自多数种子
    还是某一个离群种子。IQM 对尾部不敏感，又不至于像中位数一样丢掉所有幅度
    信息。``trim=0.25`` 即标准 IQM。
    """
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return float("nan")
    arr = np.sort(np.asarray(finite, dtype=float))
    n = len(arr)
    k = int(math.floor(n * float(trim)))
    if n - 2 * k < 1:  # 点数太少，无法两侧各截断
        k = 0
    core = arr[k: n - k] if k > 0 else arr
    return float(np.mean(core))


def bootstrap_ci(
    values: Sequence[float],
    statistic: Any = iqm,
    level: float = 0.95,
    resamples: int = DEFAULT_BOOTSTRAP,
    rng_seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """对统计量做 percentile bootstrap，返回 ``(low, high)``。

    固定 ``rng_seed``：区间必须可复现，否则"结论"会随重跑变化。
    """
    finite = np.asarray(
        [float(value) for value in values if math.isfinite(float(value))], dtype=float
    )
    if finite.size == 0:
        return float("nan"), float("nan")
    if finite.size == 1:
        only = float(statistic(finite))
        return only, only
    rng = np.random.default_rng(int(rng_seed))
    draws = np.empty(int(resamples), dtype=float)
    n = finite.size
    for index in range(int(resamples)):
        draws[index] = float(statistic(finite[rng.integers(0, n, size=n)]))
    alpha = (1.0 - float(level)) / 2.0
    low = float(np.quantile(draws, alpha))
    high = float(np.quantile(draws, 1.0 - alpha))
    return low, high


def paired_verdict(
    ci_low: float, ci_high: float, mean_diff: float, noise_floor: float
) -> str:
    """由 Δ 的 CI 与噪声底给出结论，跨 0（或在噪声内）一律 ``inconclusive``。

    这是协议的硬纪律：单次测量不足以声称改进，CI 跨 0 说明"更好/更差"
    无法与随机波动区分。
    """
    if not (math.isfinite(ci_low) and math.isfinite(ci_high) and math.isfinite(mean_diff)):
        return "inconclusive"
    if ci_low <= 0.0 <= ci_high:
        return "inconclusive"
    if abs(mean_diff) <= float(noise_floor) + 1e-12:
        return "inconclusive"
    return "better" if mean_diff > 0.0 else "worse"


def paired_effect(
    a: Sequence[float],
    b: Sequence[float],
    noise_floor: float = 0.0,
    resamples: int = DEFAULT_BOOTSTRAP,
) -> dict:
    """同一 seed 下配对的 A-B 效应量：均值差 + bootstrap CI + 相对百分比。

    ``a``/``b`` 必须按种子一一对齐（调用方保证）；含 NaN 的配对整对剔除。
    """
    diffs = np.asarray(
        [
            float(x) - float(y)
            for x, y in zip(a, b)
            if math.isfinite(float(x)) and math.isfinite(float(y))
        ],
        dtype=float,
    )
    out: dict[str, Any] = {
        "n_pairs": int(diffs.size),
        "mean_diff": float("nan"),
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "relative_pct": None,
        "mean_a": float("nan"),
        "mean_b": float("nan"),
        "verdict": "inconclusive",
    }
    if diffs.size == 0:
        out["reason"] = "没有可配对的种子（两侧指标缺测）"
        return out
    mean_diff = float(np.mean(diffs))
    low, high = bootstrap_ci(diffs, statistic=lambda v: float(np.mean(v)),
                             resamples=resamples)
    mean_a = float(np.mean([float(x) for x in a if math.isfinite(float(x))]))
    mean_b = float(np.mean([float(y) for y in b if math.isfinite(float(y))]))
    out.update(
        mean_diff=mean_diff,
        ci_low=low,
        ci_high=high,
        mean_a=mean_a,
        mean_b=mean_b,
        relative_pct=(
            100.0 * mean_diff / abs(mean_b) if abs(mean_b) > 1e-12 else None
        ),
    )
    out["verdict"] = paired_verdict(low, high, mean_diff, noise_floor)
    if out["verdict"] == "inconclusive":
        out["reason"] = (
            "Δ 的 bootstrap 95% CI 跨 0，或 |Δ| 落在噪声底内"
        )
    else:
        out["reason"] = "CI 不跨 0 且 |Δ| 超过噪声底"
    return out


# -- 指标与初始设计 -------------------------------------------------------
def metric_name(problem: BenchmarkProblem) -> str:
    """按问题类型选指标名：多目标 = hypervolume，单目标 = best。"""
    return "hypervolume" if problem.objective.is_multi_objective else "best"


def _is_feasible(problem: BenchmarkProblem, observation: Observation) -> bool:
    limits = problem.objective.constraint_limits or {}
    if not limits:
        return True
    for name, limit in limits.items():
        value = observation.constraints.get(name)
        if value is None:
            return False
        op = str(limit.get("op", "<="))
        threshold = float(limit.get("value", 0.0))
        if op == "<=":
            if value > threshold:
                return False
        elif value < threshold:
            return False
    return True


def metric_value(problem: BenchmarkProblem, observations: Sequence[Observation]) -> float:
    """协议指标：多目标用参考点 HV，单目标用方向归一化后的 best（越大越好）。

    单目标值统一取 ``sign * y``（最小化问题乘 -1），这样"越大越好"在协议里
    永远成立，配对差值的符号可以直接读。约束问题只在可行点上取 best；一个
    可行点都没有时返回 NaN（聚合时整对剔除，而不是拿不可行点冒充成绩）。
    """
    if problem.objective.is_multi_objective:
        summary = pareto_summarise(list(observations), problem.objective)
        return float(summary.hypervolume or 0.0)
    objective = problem.objective
    scores = [
        float(observation.score(objective.name, objective.direction_for(objective.name)))
        for observation in observations
        if _is_feasible(problem, observation)
    ]
    return float(max(scores)) if scores else float("nan")


def shared_initial_design(
    problem: BenchmarkProblem, seed: int, n_points: int
) -> list[dict]:
    """由 seed 派生的固定初始设计点：同一 seed 下所有算法拿到同一组点。

    这是"共享初始设计"的实现：跨算法比较必须从同一起点出发，否则差异
    里混着初始设计的好坏。用 base-2 Sobol 前缀切分，保证设计本身的平衡性
    不受调用次数影响。
    """
    if int(n_points) <= 0:
        return []
    return problem.space.sample_sobol_base2(int(n_points), seed=int(seed))


def design_digest(design: Sequence[Mapping]) -> str:
    """初始设计的 sha1 摘要，用于在 CSV/JSON 里留痕"跨算法同一组点"。"""
    payload = json.dumps(
        [dict(item) for item in design], sort_keys=True, ensure_ascii=False, default=str
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def measure(
    problem: BenchmarkProblem, config: Mapping, fidelity: float = 1.0
) -> Observation:
    """按问题类型评估一个条件，返回一条 Observation（benchmark 来源）。"""
    objective = problem.objective
    constraints = (
        dict(problem.constraints_fn(config)) if problem.constraints_fn is not None else {}
    )
    fidelity = float(fidelity)
    if objective.is_multi_objective:
        if problem.objectives_fidelity_fn is not None:
            values = dict(problem.objectives_fidelity_fn(config, fidelity))
        elif problem.objectives_fn is not None:
            values = dict(problem.objectives_fn(config))
        else:  # pragma: no cover - 问题集保证多目标一定带 objectives_fn
            raise ValueError(f"multi-objective problem {problem.name!r} has no objectives_fn")
        primary = objective.name if objective.name in values else next(iter(values))
        y = float(values[primary])
        return Observation(
            x=dict(config), y=y, objectives=values, constraints=constraints,
            fidelity=fidelity, cost=fidelity, source="benchmark_protocol",
        )
    if problem.fidelity_fn is not None:
        y = float(problem.fidelity_fn(config, fidelity))
    else:
        y = float(problem.fn(config))
    return Observation(
        x=dict(config), y=y, constraints=constraints, fidelity=fidelity,
        cost=fidelity, source="benchmark_protocol",
    )


# -- 单臂执行 -------------------------------------------------------------
def run_arm(
    problem: BenchmarkProblem,
    algorithm: str,
    seed: int,
    budget: int,
    init: int,
    batch: int = DEFAULT_BATCH,
    params: Mapping | None = None,
) -> dict:
    """跑一个算法的一个种子：共享初始设计作为 prior，然后 ask/tell 到预算。

    先测共享初始设计、再把它整体 ``tell`` 给优化器（通过 ``prior``），
    这样模型在第一次 ``ask`` 之前就已经看到同一组点。
    """
    design = shared_initial_design(problem, seed, init)
    prior = [measure(problem, config) for config in design]
    optimizer = registry.build(
        algorithm,
        problem.space,
        problem.objective,
        params=params,
        seed=int(seed),
        prior=prior,
    )
    observations: list[Observation] = list(prior)
    # 采样型算法会去重：共享初始设计可能正好落在它的序列上，某一次 ask 返回空。
    # 空提议只算一次尝试并继续（序列会前进），而不是静默截断预算——截断会让
    # 某个算法的 n_evals 悄悄小于 budget，跨算法比较就失真了。
    attempts = 0
    max_attempts = max(16, 4 * int(budget))
    empty_asks = 0
    while len(observations) < int(budget) and attempts < max_attempts:
        attempts += 1
        n = min(int(batch), int(budget) - len(observations))
        if getattr(optimizer, "supports_fidelity", False):
            pairs = optimizer.ask_with_fidelity(n)
        else:
            pairs = [(config, 1.0) for config in optimizer.ask(n)]
        if not pairs:
            empty_asks += 1
            continue
        fresh = [measure(problem, config, fidelity) for config, fidelity in pairs]
        optimizer.tell(fresh)
        observations.extend(fresh)
    return {
        "algorithm": algorithm,
        "seed": int(seed),
        "value": metric_value(problem, observations),
        "n_evals": len(observations),
        "empty_asks": empty_asks,
        "initial_design_sha": design_digest(design),
        "initial_design": design,
    }


# -- 汇总与报告 -----------------------------------------------------------
def _per_seed_values(
    rows: Sequence[Mapping], algorithms: Sequence[str]
) -> dict[str, dict[int, float]]:
    """只取 ``run_kind == "main"`` 的行，按算法 -> seed -> 指标值整理。"""
    out: dict[str, dict[int, float]] = {name: {} for name in algorithms}
    for row in rows:
        if row.get("run_kind") != "main":
            continue
        name = str(row["algorithm"])
        if name in out:
            out[name][int(row["seed"])] = float(row["value"])
    return out


def build_report(
    problem_name: str,
    metric: str,
    seeds: Sequence[int],
    algorithms: Sequence[str],
    rows: Sequence[Mapping],
    noise_floor: Mapping,
    budget: int,
    init: int,
    batch: int,
    initial_design: Mapping[int, list] | None = None,
    resamples: int = DEFAULT_BOOTSTRAP,
) -> dict:
    """把 per-seed 明细汇总成固定字段的报告（字段名被外部审查依赖）。"""
    per_seed = _per_seed_values(rows, algorithms)
    floor = float(noise_floor.get("value", 0.0) or 0.0)
    iqm_by_algorithm: dict[str, float] = {}
    ci_low: dict[str, float] = {}
    ci_high: dict[str, float] = {}
    n_by_algorithm: dict[str, int] = {}
    for name in algorithms:
        values = [per_seed[name][int(seed)] for seed in seeds if int(seed) in per_seed[name]]
        finite = [value for value in values if math.isfinite(value)]
        n_by_algorithm[name] = len(finite)
        iqm_by_algorithm[name] = iqm(finite)
        low, high = bootstrap_ci(finite, resamples=resamples)
        ci_low[name], ci_high[name] = low, high
    pairwise: list[dict] = []
    if algorithms:
        baseline = algorithms[0]
        base_values = [per_seed[baseline].get(int(seed), float("nan")) for seed in seeds]
        for name in algorithms[1:]:
            other = [per_seed[name].get(int(seed), float("nan")) for seed in seeds]
            effect = paired_effect(base_values, other, noise_floor=floor, resamples=resamples)
            effect.update(baseline=baseline, algorithm=name)
            pairwise.append(effect)
    return {
        "protocol": PROTOCOL_ID,
        "problem": problem_name,
        "metric": metric,
        "seeds": [int(seed) for seed in seeds],
        "algorithms": list(algorithms),
        "iqm": iqm_by_algorithm,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n": n_by_algorithm,
        "noise_floor": dict(noise_floor),
        "pairwise": pairwise,
        "budget": int(budget),
        "initial_design_size": int(init),
        "batch_size": int(batch),
        "initial_design": {
            str(seed): list(design) for seed, design in (initial_design or {}).items()
        },
        "metric_convention": "larger_is_better（单目标已按优化方向取符号）",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": (
            "IQM + bootstrap 95% CI；Δ 的 CI 跨 0 或 |Δ| 在噪声底内时 verdict=inconclusive。"
            "本协议落地前，主 benchmark 只能称工程 smoke。"
        ),
    }


def _fmt(value: Any, digits: int = 5) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "nan"
    return f"{number:.{digits}f}"


def render_summary(report: Mapping) -> str:
    """人类可读摘要：样本量、IQM 区间、噪声底、配对结论都要能看到。"""
    lines = [
        "== 预注册基准协议摘要 ==",
        f"protocol : {report['protocol']}",
        f"problem  : {report['problem']}    metric: {report['metric']}（越大越好）",
        f"budget   : {report['budget']}      共享初始设计: {report['initial_design_size']} 点"
        f"    batch: {report['batch_size']}",
        f"seeds    : {report['seeds']}",
        f"noise_floor: {_fmt(report['noise_floor'].get('value'))}"
        f"（同配置重复跑，seed={report['noise_floor'].get('seed')}）",
        "",
        f"{'algorithm':<14}{'n':>4}{'IQM':>14}{'95% CI':>26}",
    ]
    for name in report["algorithms"]:
        lines.append(
            f"{name:<14}{report['n'][name]:>4}{_fmt(report['iqm'][name]):>14}"
            f"{('[' + _fmt(report['ci_low'][name]) + ', ' + _fmt(report['ci_high'][name]) + ']'):>26}"
        )
    lines.append("")
    lines.append("配对效应量（同一 seed 下 A-B，A 为第一个算法）：")
    if not report["pairwise"]:
        lines.append("  （只有一个算法，无需配对比较）")
    for item in report["pairwise"]:
        rel = "n/a" if item.get("relative_pct") is None else f"{item['relative_pct']:+.2f}%"
        lines.append(
            f"  {item['baseline']} - {item['algorithm']}: "
            f"Δ={_fmt(item['mean_diff'])} "
            f"CI=[{_fmt(item['ci_low'])}, {_fmt(item['ci_high'])}] "
            f"相对={rel} n_pairs={item['n_pairs']} -> {item['verdict']}"
        )
        lines.append(f"      理由: {item.get('reason', '')}")
    lines.append("")
    lines.append(
        "结论纪律：CI 跨 0 或 |Δ| 落在噪声底内一律 inconclusive；"
        "在此之前主 benchmark 只能称工程 smoke。"
    )
    return "\n".join(lines) + "\n"


# -- CLI ------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    _force_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", default=DEFAULT_PROBLEM)
    parser.add_argument("--algorithms", nargs="+", default=list(DEFAULT_ALGORITHMS))
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS,
                        help="种子个数；实际种子为 range(seed_start, seed_start+N)")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    parser.add_argument("--init", type=int, default=DEFAULT_INIT,
                        help="共享初始设计点数（跨算法一致）")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--no-noise-floor", action="store_true",
                        help="跳过噪声底重复跑（默认每个算法重复跑第一个种子一次）")
    parser.add_argument("--out", default=str(OUT_DIR))
    args = parser.parse_args(argv)

    problems = builtin_problems()
    if args.problem not in problems:
        parser.error(f"unknown problem {args.problem!r}; choose from {sorted(problems)}")
    problem = problems[args.problem]
    algorithms = list(dict.fromkeys(args.algorithms))
    unknown = [name for name in algorithms if name not in registry.names()]
    if unknown:
        parser.error(f"unknown algorithms {unknown}; available: {registry.names()}")
    count = max(1, int(args.seeds))
    seeds = list(range(int(args.seed_start), int(args.seed_start) + count))
    budget = max(2, int(args.budget))
    init = int(args.init)
    if init >= budget:
        # 初始设计吃掉整个预算就没有"优化"可言：夹住并留痕（协议不静默）。
        clamped = max(0, budget - 1)
        print(f"[warn] --init {init} >= --budget {budget}，已夹到 {clamped}", file=sys.stderr)
        init = clamped
    design = {seed: shared_initial_design(problem, seed, init) for seed in seeds}
    metric = metric_name(problem)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    initial_design: dict[int, list] = {}
    for algorithm in algorithms:
        for seed in seeds:
            arm = run_arm(problem, algorithm, seed, budget, init, args.batch)
            initial_design[seed] = arm["initial_design"]
            rows.append({**{k: v for k, v in arm.items() if k != "initial_design"},
                         "problem": args.problem, "metric": metric,
                         "budget": budget, "run_kind": "main"})

    repeats: dict[str, float] = {}
    if not args.no_noise_floor and seeds:
        noise_seed = seeds[0]
        deltas: dict[str, float] = {}
        for algorithm in algorithms:
            repeat = run_arm(problem, algorithm, noise_seed, budget, init, args.batch)
            rows.append({**{k: v for k, v in repeat.items() if k != "initial_design"},
                         "problem": args.problem, "metric": metric,
                         "budget": budget, "run_kind": "repeat"})
            main_value = next(
                float(row["value"]) for row in rows
                if row["run_kind"] == "main"
                and row["algorithm"] == algorithm
                and int(row["seed"]) == noise_seed
            )
            deltas[algorithm] = abs(float(repeat["value"]) - main_value)
        repeats = deltas
        noise_floor: dict[str, Any] = {
            "value": max(deltas.values()) if deltas else 0.0,
            "seed": noise_seed,
            "replicates": 2,
            "per_algorithm": deltas,
            "definition": "同一算法、同一 seed、同配置重复跑，|重复值 - 原值| 的最大值",
        }
    else:
        noise_floor = {
            "value": None,
            "seed": None,
            "replicates": 0,
            "per_algorithm": {},
            "definition": "未估计（--no-noise-floor）",
        }

    report = build_report(
        args.problem, metric, seeds, algorithms, rows, noise_floor,
        budget, init, args.batch, initial_design=design or initial_design,
        resamples=max(50, int(args.bootstrap)),
    )

    json_path = out_dir / "results.json"
    csv_path = out_dir / "per_seed.csv"
    summary_path = out_dir / "summary.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    columns = [
        "problem", "metric", "algorithm", "seed", "value", "n_evals",
        "run_kind", "initial_design_sha", "budget",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    summary = render_summary(report)
    summary_path.write_text(summary, encoding="utf-8")

    print(summary, end="")
    print(f"\nJSON: {json_path}\nCSV : {csv_path}\n摘要: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
