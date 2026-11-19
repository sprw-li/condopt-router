"""Estimate the noise floor before calling any difference an improvement.

Workspace rule #11: a single measurement is not evidence.  Before writing "better /
worse" a difference must clear the noise floor, and that floor has to be *measured*,
not asserted.  This script provides both halves:

* ``--mode floor``  -- repeat one configuration across seeds and report the spread
  (mean, sd, sem, and the 95% two-sided half-width), i.e. the smallest difference that
  is distinguishable from run-to-run noise at the given number of seeds;
* ``--mode paired`` -- run configuration A and B on the *same* seeds and report the
  paired difference, its sd, and the t statistic, which is the right test when both
  arms share seeds (as in the descriptor comparisons).

``--mode paired`` also answers the question the paired t test *cannot* answer.  "not
significant" is absence of evidence, not evidence of absence: a wide interval around
zero is compatible both with "the arms are the same" and with "the run is too noisy to
tell".  So the paired mode additionally runs a **TOST** equivalence test on the per-seed
deltas: the 90% CI of the mean difference must lie entirely inside ``(-δ, +δ)`` before
``equivalent=true`` is reported.  δ (``--equivalence-margin``) defaults to the
same-input repeat-measure dispersion measured by ``--mode floor`` (the pooled sd of the
two arms across seeds); the output always states where δ came from.

Metrics: ``hv`` (feasible hypervolume, MOO problems) or ``best`` (best objective,
single-objective problems).

Run:
    py -3 scripts/noise_floor.py --mode paired \
        --problem moo_solvent_constrained --problem-b moo_solvent_constrained_desc \
        --algorithm cqehvi --budget 40 --seeds 0 1 2 3 4

The JSON goes to stdout (one line, stable fields); the human-readable summary goes to
stderr so that downstream parsers keep working.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from condopt.benchmarks import builtin_problems, run_algorithm  # noqa: E402

def _force_utf8_stdio() -> None:
    """Emit UTF-8 regardless of the Windows console codepage (cp936 here)."""
    import sys

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except (AttributeError, ValueError):  # pragma: no cover
            pass


#: name -> (mean, sd, sem, half-width of the 95% CI) for a fixed configuration
T_CRITICAL_95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
}


def t_critical(n: int) -> float:
    """Two-sided 95% critical value for ``n`` samples (df = n - 1), clamped."""
    if n <= 1:
        return float("inf")
    return T_CRITICAL_95.get(n - 1, 1.96)


#: two-sided 90% critical values for the TOST equivalence interval (df = n - 1)
T_CRITICAL_90 = {
    1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015,
    6: 1.943, 7: 1.895, 8: 1.860, 9: 1.833, 10: 1.812,
}

#: TOST works on a 90% CI, i.e. 5% per side: that is the interval whose coverage
#: is required to sit inside the equivalence margin.
TOST_CI_LEVEL = 0.90


def t_critical_90(n: int) -> float:
    """Two-sided 90% critical value for ``n`` samples (df = n - 1), clamped."""
    if n <= 1:
        return float("inf")
    return T_CRITICAL_90.get(n - 1, 1.645)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def measure(problem_name: str, algorithm: str, budget: float, batch: int, seed: int,
            metric: str) -> float:
    problem = builtin_problems()[problem_name]
    if metric == "hv" and getattr(problem.objective, "is_multi_objective", False):
        momfbo = _load_module("benchmark_momfbo", ROOT / "scripts" / "benchmark_momfbo.py")
        report = momfbo.run(algorithm, budget=budget, batch_size=batch, seed=seed,
                            problem_name=problem_name)
        return float(report["hv_feasible_full"])
    result = run_algorithm(problem, algorithm, budget=int(budget), batch_size=batch, seed=seed)
    if getattr(problem.objective, "constraints", ()):
        value = result.best_feasible_score
        return float("nan") if value is None else float(value)
    return float(result.best_score)


def suggested_margin(values_a: list[float], values_b: list[float]) -> tuple[float | None, str]:
    """δ 的建议值：两配置各自**同输入重复测量**的合并 sd。

    这正是 ``--mode floor`` 的口径（同一配置跨种子重跑得到的离散度），取合并 sd 是
    为了让 δ 与两臂各自的 run-to-run 噪声尺度相当。样本不足或离散度为 0 时返回
    ``None``，因为 δ 必须 > 0，此时不允许悄悄退化成一个假的等价结论。
    """
    ns = [len(values_a), len(values_b)]
    sds = [
        statistics.stdev(values_a) if ns[0] > 1 else 0.0,
        statistics.stdev(values_b) if ns[1] > 1 else 0.0,
    ]
    if min(ns) < 2:
        return None, "样本不足：每个配置至少需要 2 个种子才能估出同输入离散度"
    pooled = math.sqrt(sum((n - 1) * sd * sd for n, sd in zip(ns, sds)) / (sum(ns) - 2))
    if pooled <= 0:
        return None, "同输入重复测量的离散度为 0（所有种子给出同一数值），推不出正的 δ"
    return pooled, (
        f"默认 δ = 两配置同输入重复测量的合并 sd = {pooled:.4f}"
        f"（--mode floor 口径：sd_a={sds[0]:.4f}, sd_b={sds[1]:.4f}）"
    )


def equivalence_test(deltas: list[float], margin: float | None,
                     margin_source: str = "") -> dict:
    """TOST：逐种子差值的 90% CI 整段落在 (−δ, +δ) 内才算 ``equivalent=True``。

    只报 p 值会把"证据不足"和"确实等价"混为一谈，因此这里报区间本身、均值差、δ，
    以及区间相对 δ 边界的三种位置（完全在内 / 完全在外 / 跨越边界），让读者能判断
    结论是"等价"、"有方向的差异"还是"这次跑不出结论"。
    """
    if margin is not None and not margin > 0:
        raise ValueError(f"--equivalence-margin 必须 > 0，收到 {margin!r}")
    n = len(deltas)
    mean_delta = statistics.mean(deltas) if n else None
    row: dict = {
        "margin": None if margin is None else round(margin, 4),
        "margin_source": margin_source,
        "ci_level": TOST_CI_LEVEL,
        "mean_delta": None if mean_delta is None else round(mean_delta, 4),
        "ci_low": None, "ci_high": None, "ci90_half_width": None,
        "t_critical_90": None,
        "equivalent": None, "ci_within_margin": None, "ci_outside_margin": None,
        "equivalence_verdict": "", "note": None,
    }
    if n < 2:
        row["equivalence_verdict"] = "样本不足（n<2），不判等价"
        row["note"] = (
            f"n={n} < 2：无法构造逐种子差的 90% 置信区间，equivalent=null"
            "（样本不足不得当成等价）"
        )
        return row

    sd = statistics.stdev(deltas)
    sem = sd / math.sqrt(n)
    critical = t_critical_90(n)
    half = critical * sem
    ci_low = mean_delta - half
    ci_high = mean_delta + half
    row["ci90_half_width"] = round(half, 4)
    row["t_critical_90"] = round(critical, 3)
    row["ci_low"] = round(ci_low, 4)
    row["ci_high"] = round(ci_high, 4)

    if margin is None:
        row["equivalence_verdict"] = "无法判定（没有有效 δ）"
        row["note"] = "δ 必须 > 0；请显式传 --equivalence-margin，或补足种子数以估出离散度"
        return row

    within = bool(ci_low > -margin and ci_high < margin)
    outside = bool(ci_high < -margin or ci_low > margin)
    row["equivalent"] = within
    row["ci_within_margin"] = within
    row["ci_outside_margin"] = outside
    if within:
        row["equivalence_verdict"] = f"等价（90% CI 整段落在 ±{margin:.4f} 内）"
    elif outside:
        direction = "A 高于 B" if mean_delta > 0 else "A 低于 B"
        row["equivalence_verdict"] = (
            f"不等价且有方向（90% CI 整段落在 ±{margin:.4f} 之外：{direction}）"
        )
    else:
        row["equivalence_verdict"] = (
            f"不确定（90% CI 跨越 ±{margin:.4f} 边界，既非等价也无方向性差异）"
        )
    return row


def floor(problem: str, algorithm: str, budget: float, batch: int,
          seeds: list[int], metric: str) -> dict:
    values = [measure(problem, algorithm, budget, batch, seed, metric) for seed in seeds]
    n = len(values)
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if n > 1 else 0.0
    sem = sd / math.sqrt(n) if n else 0.0
    half = t_critical(n) * sem
    # smallest difference between two independent arms with n seeds each
    detectable = t_critical(n) * math.sqrt(2 / n) * sd if n > 1 else float("inf")
    return {
        "problem": problem, "algorithm": algorithm, "metric": metric, "n": n,
        "values": [round(value, 4) for value in values],
        "mean": round(mean, 4), "sd": round(sd, 4), "sem": round(sem, 4),
        "ci95_half_width": round(half, 4),
        "detectable_delta_2arms": None if math.isinf(detectable) else round(detectable, 4),
        "relative_floor_pct": (
            None if not mean else round(100 * detectable / abs(mean), 3)
        ),
        # δ 的建议值 = 同输入重复测量的离散度（即上面的 sd），供 --mode paired 做 TOST
        "equivalence_margin_suggested": round(sd, 4) if n > 1 else None,
        "equivalence_margin_source": (
            "同输入重复测量的 sd（--mode floor 口径）" if n > 1
            else "样本不足：至少 2 个种子才能估出离散度"
        ),
    }


def paired(problem_a: str, problem_b: str, algorithm: str, budget: float, batch: int,
           seeds: list[int], metric: str, margin: float | None = None) -> dict:
    """同种子配对比较：旧口径给出配对 t 检验，新口径给出 TOST 等效性结论。"""
    if margin is not None and not margin > 0:
        raise ValueError(f"--equivalence-margin 必须 > 0，收到 {margin!r}")
    a = [measure(problem_a, algorithm, budget, batch, seed, metric) for seed in seeds]
    b = [measure(problem_b, algorithm, budget, batch, seed, metric) for seed in seeds]
    deltas = [x - y for x, y in zip(a, b)]
    n = len(deltas)
    mean_delta = statistics.mean(deltas)
    sd = statistics.stdev(deltas) if n > 1 else 0.0
    sem = sd / math.sqrt(n) if n else 0.0
    t_stat = (mean_delta / sem) if sem else float("inf")
    critical = t_critical(n)
    significant = abs(t_stat) > critical

    if margin is None:
        margin, margin_source = suggested_margin(a, b)
    else:
        margin_source = f"--equivalence-margin 显式给定：δ={margin}"

    row = {
        "a": problem_a, "b": problem_b, "algorithm": algorithm, "metric": metric, "n": n,
        "per_seed_delta": [round(value, 4) for value in deltas],
        "mean_delta": round(mean_delta, 4),
        "mean_a": round(statistics.mean(a), 4), "mean_b": round(statistics.mean(b), 4),
        "relative_pct": (
            None if not statistics.mean(b) else
            round(100 * mean_delta / abs(statistics.mean(b)), 3)
        ),
        "sd_delta": round(sd, 4), "sem_delta": round(sem, 4),
        "t_statistic": None if math.isinf(t_stat) else round(t_stat, 3),
        "t_critical_95": round(critical, 3),
        "significant_95": bool(significant),
        "verdict": (
            "超出噪声底" if significant
            else "在噪声内（不足以声称差异）"
        ),
    }
    row.update(equivalence_test(deltas, margin, margin_source))
    return row


def human_summary(row: dict, mode: str) -> str:
    """旧有的人类可读摘要（写 stderr，stdout 保持单行 JSON 便于下游解析）。"""
    head = f"[noise_floor] mode={mode} algorithm={row['algorithm']} metric={row['metric']}"
    if mode == "floor":
        return "\n".join([
            f"{head} problem={row['problem']} n={row['n']}",
            f"  mean={row['mean']} sd={row['sd']} sem={row['sem']} "
            f"95%CI半宽={row['ci95_half_width']}",
            f"  两臂可分辨最小差异={row['detectable_delta_2arms']} "
            f"({row['relative_floor_pct']}%)",
            f"  δ 建议值={row['equivalence_margin_suggested']}"
            f"（来源：{row['equivalence_margin_source']}）",
        ])
    return "\n".join([
        f"{head} a={row['a']} b={row['b']} n={row['n']}",
        f"  mean_a={row['mean_a']} mean_b={row['mean_b']} "
        f"mean_delta={row['mean_delta']} ({row['relative_pct']}%)",
        f"  sd_delta={row['sd_delta']} sem_delta={row['sem_delta']} "
        f"t={row['t_statistic']} 临界={row['t_critical_95']} → "
        f"significant_95={row['significant_95']}（{row['verdict']}）",
        f"  等效性(TOST)：δ={row['margin']} 来源：{row['margin_source']}",
        f"  均值差 90% CI = ({row['ci_low']}, {row['ci_high']}) "
        f"→ equivalent={row['equivalent']}（{row['equivalence_verdict']}）",
        *([f"  note: {row['note']}"] if row.get("note") else []),
    ])


def main() -> int:
    _force_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("floor", "paired"), default="floor")
    parser.add_argument("--problem", required=True)
    parser.add_argument("--problem-b", default=None, help="paired mode: second configuration")
    parser.add_argument("--algorithm", default="auto")
    parser.add_argument("--budget", type=float, default=40)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--metric", choices=("hv", "best"), default="hv")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--equivalence-margin", type=float, default=None, dest="equivalence_margin",
        help="TOST 的等效界 δ（必须 > 0）；缺省时用两配置同输入重复测量的合并 sd 作为建议值",
    )
    args = parser.parse_args()

    if args.equivalence_margin is not None and not args.equivalence_margin > 0:
        parser.error(f"--equivalence-margin 必须 > 0，收到 {args.equivalence_margin}")

    if args.mode == "floor":
        row = floor(args.problem, args.algorithm, args.budget, args.batch,
                    args.seeds, args.metric)
    else:
        if not args.problem_b:
            parser.error("--mode paired requires --problem-b")
        row = paired(args.problem, args.problem_b, args.algorithm, args.budget,
                     args.batch, args.seeds, args.metric,
                     margin=args.equivalence_margin)
    print(json.dumps(row, ensure_ascii=False))
    print(human_summary(row, args.mode), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
