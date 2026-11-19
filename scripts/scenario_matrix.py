"""Executable scenario matrix: realistic lab setups -> expected default algorithm.

Each row is a context users actually hit (dims, objective/constraint/fidelity
structure, noise).  Routing must land on the documented default; a change here
means the docs (``docs/SCENARIOS.md``) and the evidence (``docs/DEFAULTS.md``)
need updating too.

Run:
    py -3 scripts/scenario_matrix.py
    py -3 scripts/scenario_matrix.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from condopt.router import OptimizationContext, route  # noqa: E402

def _force_utf8_stdio() -> None:
    """Emit UTF-8 regardless of the Windows console codepage (cp936 here)."""
    import sys

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except (AttributeError, ValueError):  # pragma: no cover
            pass



@dataclass
class Scenario:
    name: str
    context: dict[str, Any]
    expected: str
    note: str = ""
    alternatives: tuple[str, ...] = field(default_factory=tuple)
    #: Runnable evidence: benchmark script (relative to the repo root) and the
    #: builtin problem it exercises.  Keeps every scenario traceable.
    evidence: str = ""
    problem: str = ""


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        "A. 少量连续变量单目标",
        dict(n_dims=3, n_continuous=3, budget_total=30, batch_size=3,
             n_objectives=1, noise="medium", smoothness="smooth"),
        "bayes",
        "2–6 个连续变量、目标平滑；GP 代理最稳",
        evidence="scripts/benchmark.py",
        problem="branin",
    ),
    Scenario(
        "B. 高维连续（10 维）",
        dict(n_dims=10, n_continuous=10, budget_total=60, batch_size=4,
             n_objectives=1, noise="low", smoothness="smooth"),
        "cmaes",
        "budget 60、3 seeds：cmaes −18.1 vs bayes −155.4（rosenbrock10）、−2.20 vs −5.01（ackley10）",
        evidence="scripts/benchmark_highdim.py",
        problem="rosenbrock10",
    ),
    Scenario(
        "B2. 中维连续（6 维）",
        dict(n_dims=6, n_continuous=6, budget_total=60, batch_size=4,
             n_objectives=1, noise="low", smoothness="smooth"),
        "bayes",
        "低于 8 维时 GP 仍占优，cmaes 规则不触发",
        evidence="scripts/benchmark_highdim.py",
        problem="ackley10",
    ),
    Scenario(
        "C. 混合离散（连续+分类+整数）",
        dict(n_dims=6, n_continuous=3, n_categorical=2, n_integer=1, feature_dim=12,
             budget_total=60, batch_size=4, n_objectives=1, noise="medium",
             smoothness="mixed"),
        "bayes",
        "混合空间 GP 仍可用；预算极小时 router 会退到 tpe/random",
        evidence="scripts/benchmark.py",
        problem="discrete_catalyst",
    ),
    Scenario(
        "D. 高基数分类（12 级溶剂）",
        dict(n_dims=3, n_continuous=1, n_categorical=1, feature_dim=12,
             budget_total=60, batch_size=4, n_objectives=1, noise="medium",
             smoothness="mixed", cardinality=12.0),
        "tpe",
        "one-hot 维度高时 TPE 更好；加物性描述符后特征降到 3 维会自动转向 GP 类",
        alternatives=("bayes",),
        evidence="scripts/benchmark_tier_c.py",
        problem="high_cardinality_solvent",
    ),
    Scenario(
        "E. 单目标 + 硬约束（可行域大）",
        dict(n_dims=3, n_continuous=3, budget_total=40, batch_size=3,
             n_objectives=1, n_constraints=1, noise="low"),
        "constrained",
        "溶剂约束基准：90.39 vs bayes 84.7 / tpe 85.7",
        evidence="scripts/benchmark_solvent_constrained.py",
        problem="solvent_constrained",
    ),
    Scenario(
        "F. 2 目标",
        dict(n_dims=3, n_continuous=3, budget_total=40, batch_size=4,
             n_objectives=2, noise="low", smoothness="smooth"),
        "qehvi",
        "2 目标 EHVI 与 parego 接近，默认 qehvi",
        alternatives=("parego", "qehvi_nd"),
        evidence="scripts/benchmark_3obj.py",
        problem="moo_yield_cost",
    ),
    Scenario(
        "G. 2 目标 + 硬约束（低噪声）",
        dict(n_dims=3, n_continuous=3, budget_total=40, batch_size=4,
             n_objectives=2, n_constraints=1, noise="low", smoothness="smooth"),
        "cqehvi",
        "约束 EHVI 是 2 目标 + 约束的默认",
        evidence="scripts/benchmark_3obj.py",
        problem="moo_constrained",
    ),
    Scenario(
        "H. 2 目标 + 硬约束 + 噪声",
        dict(n_dims=3, n_continuous=3, budget_total=40, batch_size=4,
             n_objectives=2, n_constraints=1, noise="high", smoothness="smooth"),
        "cqehvi",
        "cqnehvi 实测未超过 cqehvi（638/635 vs 631/626）",
        evidence="scripts/benchmark_cqehvi.py",
        problem="moo_constrained",
    ),
    Scenario(
        "I. 3–4 目标",
        dict(n_dims=4, n_continuous=4, budget_total=60, batch_size=4,
             n_objectives=3, noise="low", smoothness="smooth"),
        "qehvi_nd",
        "HV 5797 vs parego 5578（3 目标）、91244 vs 90134（4 目标）",
        evidence="scripts/benchmark_3obj.py",
        problem="moo3_constrained",
    ),
    Scenario(
        "J. 单目标 + 多保真",
        dict(n_dims=2, n_continuous=2, budget_total=30, batch_size=3,
             n_objectives=1, multi_fidelity=True, noise="medium"),
        "mfbo",
        "acquisition=ei 把 93.2 提升到 95.3",
        evidence="scripts/benchmark_mf_constrained.py",
        problem="mf_constrained",
    ),
    Scenario(
        "K. 2+ 目标 + 多保真",
        dict(n_dims=4, n_continuous=4, budget_total=40, batch_size=4,
             n_objectives=2, multi_fidelity=True, noise="medium", smoothness="smooth"),
        "mfparego",
        "四处基准一致略优于 momfbo，且全保真点更多",
        evidence="scripts/benchmark_momfbo.py",
        problem="mf_moo",
    ),
    Scenario(
        "L. 多目标 + 多保真 + 约束",
        dict(n_dims=4, n_continuous=4, budget_total=40, batch_size=4,
             n_objectives=2, n_constraints=1, multi_fidelity=True, noise="medium",
             smoothness="smooth"),
        "mfparego",
        "约束下 642.5 vs 635.6；另有可行全保真诊断兜底",
        evidence="scripts/benchmark_momfbo.py",
        problem="mf_moo_constrained",
    ),
    Scenario(
        "N. 4 目标",
        dict(n_dims=4, n_continuous=4, budget_total=60, batch_size=4,
             n_objectives=4, noise="low", smoothness="smooth"),
        "qehvi_nd",
        "4 目标 HV 型采集仍然适用（91244 vs parego 90134）",
        evidence="scripts/benchmark_3obj.py",
        problem="moo4_constrained",
    ),
    Scenario(
        "O. 4 目标 + 多保真",
        dict(n_dims=4, n_continuous=4, budget_total=50, batch_size=4,
             n_objectives=4, multi_fidelity=True, noise="medium", smoothness="smooth"),
        "mfparego",
        "mf_moo4：515205 vs momfbo 507932，全保真点 36 vs 24.3",
        evidence="scripts/benchmark_momfbo.py",
        problem="mf_moo4",
    ),
    Scenario(
        "P. 单目标 + 多保真 + 约束",
        dict(n_dims=6, n_continuous=6, budget_total=40, batch_size=3,
             n_objectives=1, n_constraints=1, multi_fidelity=True, noise="medium",
             smoothness="mixed"),
        "mfbo",
        "mfbo 声明 constraints + multi_fidelity；另有可行全保真诊断兜底",
        evidence="examples/constrained_mf_executor.py",
        problem="mf_constrained",
    ),
    Scenario(
        "Q. 小预算（8 次）单目标",
        dict(n_dims=4, n_continuous=4, budget_total=8, batch_size=2,
             n_objectives=1, noise="low", smoothness="smooth"),
        "sobol",
        "预算不足时空间填充优先于代理建模",
        evidence="scripts/benchmark.py",
        problem="branin",
    ),
    Scenario(
        "R. 大批量（batch=16）",
        dict(n_dims=5, n_continuous=5, budget_total=60, batch_size=16,
             n_objectives=1, noise="medium", smoothness="smooth"),
        "bayes",
        "支持 batch 的 GP 仍占优；批量并行由 batch_strategy 处理",
        evidence="scripts/benchmark.py",
        problem="mixed_reaction",
    ),
    Scenario(
        "M. 高噪声 + 复孔",
        dict(n_dims=2, n_continuous=2, budget_total=30, batch_size=3,
             n_objectives=1, noise="high", smoothness="smooth"),
        "bayes",
        "GP 的 alpha 用逐点噪声（objective_std / n_replicates）",
        evidence="scripts/benchmark_noise.py",
        problem="sphere2d",
    ),
    Scenario(
        "S. 2 目标 + 硬约束 + 高基数分类（描述符配对）",
        dict(n_dims=3, n_continuous=1, n_categorical=1, feature_dim=13,
             budget_total=40, batch_size=4, n_objectives=2, n_constraints=1,
             noise="medium", smoothness="smooth"),
        "cqehvi",
        "新增基准 moo_solvent_constrained：可行 HV 1060.3（one-hot）vs 1045.1（描述符，"
        "噪声内），但描述符版可行点 +25%（29.7→37.0）→ 建议“描述符 + 约束感知算法”成对使用",
        evidence="scripts/benchmark_momfbo.py",
        problem="moo_solvent_constrained",
    ),
    Scenario(
        "T. 高基数分类 + 硬约束（描述符编码）",
        dict(n_dims=2, n_continuous=1, n_categorical=1, feature_dim=3,
             budget_total=40, batch_size=4, n_objectives=1, n_constraints=1,
             noise="medium", smoothness="smooth"),
        "constrained",
        "solvent_constrained_desc：描述符把 one-hot 从 13 列降到 3 列，但路由必须**仍**选约束感知算法"
        "（实测可行率 0.550→0.875，最优值不变）",
        evidence="scripts/benchmark_solvent_constrained.py",
        problem="solvent_constrained_desc",
    ),
    Scenario(
        "U. 2 目标 + 硬约束 + 高基数分类（描述符编码）",
        dict(n_dims=2, n_continuous=1, n_categorical=1, feature_dim=3,
             budget_total=40, batch_size=4, n_objectives=2, n_constraints=1,
             noise="medium", smoothness="smooth"),
        "cqehvi",
        "moo_solvent_constrained_desc：编码换掉后仍须选 cqehvi（可行点 +25%，HV −1.4% 噪声内）",
        evidence="scripts/benchmark_momfbo.py",
        problem="moo_solvent_constrained_desc",
    ),
)


def evaluate() -> list[dict]:
    rows = []
    for scenario in SCENARIOS:
        recommendation = route(OptimizationContext(**scenario.context))
        ok = recommendation.algorithm == scenario.expected or (
            recommendation.algorithm in scenario.alternatives
        )
        evidence_exists = bool(scenario.evidence) and (ROOT / scenario.evidence).exists()
        rows.append({
            "scenario": scenario.name,
            "expected": scenario.expected,
            "actual": recommendation.algorithm,
            "ok": bool(ok) and evidence_exists,
            "note": scenario.note,
            "evidence": scenario.evidence,
            "evidence_exists": evidence_exists,
            "problem": scenario.problem,
        })
    return rows


def main() -> int:
    _force_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    rows = evaluate()
    failures = [row for row in rows if not row["ok"]]
    if args.json:
        print(json.dumps({"rows": rows, "failures": len(failures)}, ensure_ascii=False, indent=2))
    else:
        for row in rows:
            marker = "OK  " if row["ok"] else "FAIL"
            print(f"[{marker}] {row['scenario']}: expected={row['expected']} actual={row['actual']}")
        print(f"scenario_matrix: {len(rows) - len(failures)}/{len(rows)} matched")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())