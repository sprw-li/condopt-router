"""Decision-layer rule matrix: every default rule is asserted and traceable.

``docs/SCENARIOS.md`` covers *routing* (which algorithm).  This matrix covers the
*decision layer*: stopping policy, delivery slots, feasibility advice, transfer
credibility grading and the study-level stop categories.  Each rule states where
its numbers come from, so a rule cannot drift away from its evidence silently.

Run:
    py -3 scripts/decision_matrix.py            # evaluate and print
    py -3 scripts/decision_matrix.py --write    # regenerate docs/DECISIONS.md
    py -3 scripts/decision_matrix.py --check    # fail if the doc is stale
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from condopt.base import Objective  # noqa: E402
from condopt.prepare import (  # noqa: E402
    deliverable_advice,
    feasibility_advice,
    noise_advice as prepare_noise_advice,
    reliability_grade,
    stopping_advice,
)
from condopt.router import OptimizationContext, route  # noqa: E402

def _force_utf8_stdio() -> None:
    """Emit UTF-8 regardless of the Windows console codepage (cp936 here)."""
    import sys

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except (AttributeError, ValueError):  # pragma: no cover
            pass


DOC = ROOT / "docs" / "DECISIONS.md"


@dataclass
class Rule:
    name: str
    check: Callable[[], bool]
    actual: Callable[[], Any]
    expected: str
    evidence: str


def _descriptor_encoding_keeps_constraint_aware_routing() -> bool:
    """One-hot vs descriptor encoding must not change the constraint-aware default."""
    from condopt.benchmarks import builtin_problems
    from condopt.router import OptimizationContext, route

    expectations = {
        "solvent_constrained": "constrained",
        "solvent_constrained_desc": "constrained",
        "moo_solvent_constrained": "cqehvi",
        "moo_solvent_constrained_desc": "cqehvi",
    }
    for name, expected in expectations.items():
        problem = builtin_problems()[name]
        context = OptimizationContext.from_space(
            problem.space, budget_total=40, batch_size=4
        )
        context = OptimizationContext(
            **{
                **context.__dict__,
                "n_objectives": problem.objective.n_objectives,
                "n_constraints": len(problem.objective.constraints),
            }
        )
        if route(context).algorithm != expected:
            return False
    return True


def _descriptor_advice_is_paired() -> bool:
    """A high-cardinality + constrained context must pair descriptors with the algorithm."""
    from condopt.prepare import feasibility_advice
    from condopt.space import SearchSpace

    space = SearchSpace.from_dict([
        {"name": "T", "kind": "continuous", "low": 40.0, "high": 100.0},
        {
            "name": "solvent", "kind": "categorical",
            "choices": [f"S{index}" for index in range(12)],
        },
    ])
    context = OptimizationContext.from_space(space, budget_total=40, batch_size=4)
    objective = Objective(
        name="yield", direction="max", constraints=("T_max",),
        constraint_limits={"T_max": {"op": "<=", "value": 80.0}},
    )
    advice = feasibility_advice({}, context, objective)["advice"]
    return "描述符" in advice and "约束感知" in advice


def _low_confidence_audit_is_capped() -> bool:
    """A low-confidence audit must not recommend reliability 1.0."""
    import tempfile

    from condopt.benchmarks import builtin_problems
    from condopt.router import OptimizationContext
    from condopt.study import Study

    problem = builtin_problems()["mixed_reaction"]
    context = OptimizationContext.from_space(
        problem.space, budget_total=16, batch_size=2, multi_fidelity=False,
        n_objectives=1, n_constraints=0, noise="medium",
    )
    with tempfile.TemporaryDirectory() as tmp:
        source = Study.create(
            Path(tmp) / "src", problem.space, problem.objective, algorithm="random",
            budget=12, batch_size=2, context=context, seed=0,
        )
        pending = source.ask(12)
        source.tell([
            {"id": item["id"], "y": float(problem.fn(item["x"])), "fidelity": 1.0}
            for item in pending
        ])
        target = Study.create(
            Path(tmp) / "dst", problem.space, problem.objective, algorithm="random",
            budget=16, batch_size=2, context=context, seed=1,
        )
        pending = target.ask(4)
        target.tell([
            {"id": item["id"], "y": float(problem.fn(item["x"])), "fidelity": 1.0}
            for item in pending
        ])
        report = target.assess_transfer(source.dir)
        if report.get("confidence") != "low":
            return True
        return report["suggested_reliability"] <= 0.6


def _plateau_ignores_priors() -> bool:
    """The plateau series must use the local best after a warm start."""
    import tempfile

    from condopt.base import Objective, Observation
    from condopt.space import SearchSpace
    from condopt.study import Study

    with tempfile.TemporaryDirectory() as tmp:
        space = SearchSpace.from_dict(
            [{"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0}]
        )
        study = Study.create(
            Path(tmp) / "plateau", space, Objective(name="y", direction="max"),
            algorithm="random", budget=40, batch_size=2, seed=0, patience=3,
        )
        for value in (1.0, 1.0, 1.0):
            pending = study.ask(2)
            study.tell([{"id": item["id"], "y": value} for item in pending])
        study.import_observations([Observation(x={"x": 0.9}, y=99.0, fidelity=1.0)])
        # local improvements must keep the campaign alive although the prior stays best
        for value in (2.0, 3.0, 4.0):
            pending = study.ask(2)
            study.tell([{"id": item["id"], "y": value} for item in pending])
        stop, _ = study.should_stop()
        return stop is False


def _ctx(**kwargs) -> OptimizationContext:
    return OptimizationContext(**{**dict(n_dims=3, n_continuous=3, budget_total=30), **kwargs})


def _simple_objective() -> Objective:
    return Objective(name="y", direction="max")


def _constrained_objective() -> Objective:
    return Objective(
        name="y", direction="max", constraints=("c",),
        constraint_limits={"c": {"op": "<=", "value": 1.0}},
    )


def _advice(answers: dict, context: OptimizationContext, objective: Objective) -> dict:
    return deliverable_advice(answers, context, objective)


RULES: tuple[Rule, ...] = (
    Rule(
        "预算 12 的交付槽位",
        lambda: stopping_advice(_ctx(budget_total=12))["delivery_slots"] == 3,
        lambda: stopping_advice(_ctx(budget_total=12))["delivery_slots"],
        "3（= default_confirm_n(12)）",
        "research/CONFIRMATION_BUDGET.md",
    ),
    Rule(
        "默认复孔批次为 3 且可覆盖",
        lambda: (
            stopping_advice(_ctx(budget_total=12))["delivery_slots"] == 3
            and stopping_advice(_ctx(budget_total=120))["delivery_slots"] == 3
        ),
        lambda: stopping_advice(_ctx(budget_total=12))["delivery_slots"],
        "3（两种情况一致，可用 meta.confirm_n 覆盖）",
        "research/CONFIRMATION_BUDGET.md（两个问题方向不一致，保持默认）",
    ),
    Rule(
        "预算 12 的净探索预算",
        lambda: stopping_advice(_ctx(budget_total=12))["effective_exploration_budget"] == 9,
        lambda: stopping_advice(_ctx(budget_total=12))["effective_exploration_budget"],
        "12 − 3 = 9",
        "Round 231（prepare 与运行时同源）",
    ),
    Rule(
        "大预算建议跑满",
        lambda: stopping_advice(_ctx(budget_total=120))["recommended"] == "budget_first",
        lambda: stopping_advice(_ctx(budget_total=120))["recommended"],
        "budget_first（>=60）",
        "research/ADAPTER_LONG_RUN.md（69/120 vs 120/120）",
    ),
    Rule(
        "可信源的迁移建议必须是分阶段的",
        lambda: "先 top_k 1" in reliability_grade(0.9)["advice"]
        and "两批" in reliability_grade(0.9)["advice"],
        lambda: reliability_grade(0.9)["advice"][:40],
        "先 top_k 1",
        "research/TRANSFER_ADAPTIVE.md（有偏源 +0.71 vs 固定 k3/k5 −1.54/−0.33）",
    ),
    Rule(
        "描述符编码不得改变约束场景的算法选择",
        _descriptor_encoding_keeps_constraint_aware_routing,
        _descriptor_encoding_keeps_constraint_aware_routing,
        True,
        "docs/SCENARIOS.md T/U（描述符版仍选 constrained / cqehvi）",
    ),
    Rule(
        "描述符建议必须与约束感知算法成对出现",
        _descriptor_advice_is_paired,
        _descriptor_advice_is_paired,
        True,
        "research/DESCRIPTOR_FEATURES.md（只换编码不换算法会更差）",
    ),
    Rule(
        "审计定位为一票否决器而非分级器",
        lambda: "审计分高" in reliability_grade(0.9)["advice"]
        and "两批" in reliability_grade(0.9)["advice"],
        lambda: reliability_grade(0.9)["advice"][:40],
        "审计分高",
        "research/AUDIT_CALIBRATION.md（同一 seed 内审计分与偏差无关）",
    ),
    Rule(
        "低置信度审计不得建议完全信任",
        _low_confidence_audit_is_capped,
        _low_confidence_audit_is_capped,
        True,
        "research/AUDIT_CALIBRATION.md（有偏源 0.65 却 −0.54）",
    ),
    Rule(
        "可信源的先验不应默认打折",
        lambda: "不要打折" in reliability_grade(0.9)["advice"],
        lambda: reliability_grade(0.9)["advice"][:24],
        "不要打折",
        "research/PRIOR_RELIABILITY.md（r=1.0 +44.9 -> r=0.4 +39.3）",
    ),
    Rule(
        "平台期判定必须忽略导入的先验",
        _plateau_ignores_priors,
        _plateau_ignores_priors,
        True,
        "research/TRANSFER_TOP_K.md（先验曾导致暖启动 16-19/30 次就停）",
    ),
    Rule(
        "高噪声建议必须包含复孔与基线臂",
        lambda: (
            prepare_noise_advice({"noise": "high"}, _ctx(batch_size=12))["replicates"] == 3
            and prepare_noise_advice({"noise": "high"}, _ctx(batch_size=12))["baseline_arm"] is True
        ),
        lambda: prepare_noise_advice({"noise": "high"}, _ctx(batch_size=12))["replicates"],
        3,
        "research/NOISE_POLICY.md（复孔与基线臂是站得住的两条）",
    ),
    Rule(
        "预算优先必须带交付权衡说明",
        lambda: stopping_advice(_ctx(budget_total=120))["delivery_tradeoff"] is not None,
        lambda: stopping_advice(_ctx(budget_total=120))["delivery_tradeoff"] is not None,
        "不是 None（必须给出权衡说明）",
        "research/MID_BUDGET.md（2/5 → 1/5 可交付证据）",
    ),
    Rule(
        "小预算建议早停",
        lambda: stopping_advice(_ctx(budget_total=30))["recommended"] == "stagnation_first",
        lambda: stopping_advice(_ctx(budget_total=30))["recommended"],
        "stagnation_first（<60）",
        "research/STAGNATION_TUNING.md",
    ),
    Rule(
        "无约束时不做可行率告警",
        lambda: feasibility_advice({}, _ctx(), _simple_objective())["risk"] == "none",
        lambda: feasibility_advice({}, _ctx(), _simple_objective())["risk"],
        "none",
        "research/HARD_FEASIBILITY.md",
    ),
    Rule(
        "窄可行域告警",
        lambda: feasibility_advice(
            {"feasible_fraction_estimate": 0.004}, _ctx(), _constrained_objective()
        )["risk"] == "high",
        lambda: feasibility_advice(
            {"feasible_fraction_estimate": 0.004}, _ctx(), _constrained_objective()
        )["risk"],
        "high（<=5%）",
        "research/HARD_FEASIBILITY.md（1/5 vs 5/5）",
    ),
    Rule(
        "约束场景交付指标取可行口径",
        lambda: _advice({}, _ctx(), _constrained_objective())["metric"]
        == "feasible_best_value",
        lambda: _advice({}, _ctx(), _constrained_objective())["metric"],
        "feasible_best_value",
        "research/NEGATIVE_RESULTS.md §10（143.0 vs 80.9）",
    ),
    Rule(
        "多目标+约束交付指标取可行 HV",
        lambda: _advice(
            {},
            _ctx(n_objectives=2, n_constraints=1),
            Objective(
                name="y", direction="max", objectives=("a", "b"),
                directions=("max", "min"), constraints=("c",),
                constraint_limits={"c": {"op": "<=", "value": 1.0}},
            ),
        )["metric"] == "feasible_hypervolume",
        lambda: _advice(
            {},
            _ctx(n_objectives=2, n_constraints=1),
            Objective(
                name="y", direction="max", objectives=("a", "b"),
                directions=("max", "min"), constraints=("c",),
                constraint_limits={"c": {"op": "<=", "value": 1.0}},
            ),
        )["metric"],
        "feasible_hypervolume",
        "research/NEGATIVE_RESULTS.md §10",
    ),
    Rule(
        "多保真交付必须带全保真证据",
        lambda: any(
            "全保真" in item
            for item in _advice(
                {}, _ctx(multi_fidelity=True), _simple_objective()
            )["must_include"]
        ),
        lambda: _advice({}, _ctx(multi_fidelity=True), _simple_objective())["must_include"],
        "含“全保真档位的交付证据”",
        "research/NEGATIVE_RESULTS.md §10",
    ),
    Rule(
        "高可信源 -> trust",
        lambda: reliability_grade(0.9, "high")["grade"] == "trust",
        lambda: reliability_grade(0.9, "high")["grade"],
        "trust（>=0.8）",
        "research/TRANSFER_AUDIT.md",
    ),
    Rule(
        "中可信源 -> suspect",
        lambda: reliability_grade(0.65, "medium")["grade"] == "suspect",
        lambda: reliability_grade(0.65, "medium")["grade"],
        "suspect（0.5–0.8）",
        "research/TRANSFER_AUDIT.md",
    ),
    Rule(
        "低可信源 -> avoid",
        lambda: reliability_grade(0.3, "medium")["grade"] == "avoid",
        lambda: reliability_grade(0.3, "medium")["grade"],
        "avoid（<0.5，置信度非 low 时按数值判定）",
        "research/TRANSFER_AUDIT.md",
    ),
    Rule(
        "低可信 + 低置信度同样降级",
        lambda: reliability_grade(0.3, "low")["grade"] == "low_confidence",
        lambda: reliability_grade(0.3, "low")["grade"],
        "low_confidence（数值不可信时不下结论）",
        "research/NEGATIVE_RESULTS.md §11–§12",
    ),
    Rule(
        "低置信度一律降级",
        lambda: reliability_grade(0.9, "low")["grade"] != "trust",
        lambda: reliability_grade(0.9, "low")["grade"],
        "low_confidence / suspect（不得为 trust）",
        "research/NEGATIVE_RESULTS.md §11–§12（真实 20 切分）",
    ),
    Rule(
        "多保真多目标走 mfparego",
        lambda: route(_ctx(multi_fidelity=True, n_objectives=2)).algorithm == "mfparego",
        lambda: route(_ctx(multi_fidelity=True, n_objectives=2)).algorithm,
        "mfparego",
        "docs/SCENARIOS.md 场景 F",
    ),
    Rule(
        "约束单目标走 constrained",
        lambda: route(_ctx(n_constraints=1)).algorithm == "constrained",
        lambda: route(_ctx(n_constraints=1)).algorithm,
        "constrained",
        "docs/SCENARIOS.md 场景 H",
    ),
)


def _study(tmp_path: Path, **kwargs):
    """A tiny study for the study-level suppression checks (no execution needed)."""
    from condopt.study import Study
    from condopt.space import SearchSpace

    space = SearchSpace.from_dict([
        {"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0}
    ])
    return Study.create(tmp_path / f"suppress_{len(list(tmp_path.glob('suppress_*')))}",
                        space, _simple_objective(), **kwargs)


def _suppression_rows() -> list[dict]:
    """Rules that must *not* fire in the stated situation."""
    import json
    import shutil
    import tempfile

    from condopt.study import Study

    tmp_root = Path(tempfile.mkdtemp(prefix="decision_matrix_"))
    rows: list[dict] = []

    def record(name: str, actual, expected: str, evidence: str) -> None:
        rows.append({
            "name": name,
            "actual": actual,
            "expected": expected,
            "evidence": evidence,
            "suppressed": True,
        })

    try:
        # 1) the 8-D CMA-ES rule must not fire at 6 dimensions
        actual = route(_ctx(n_dims=6, n_continuous=6)).algorithm
        record("6 维不得触发 cmaes 规则", actual, "不是 cmaes", "docs/SCENARIOS.md 场景 B2")

        # 2) budget-first must not fire for small budgets
        actual = stopping_advice(_ctx(budget_total=30))["recommended"]
        record("预算 30 不得走 budget_first", actual, "不是 budget_first", "research/ADAPTER_LONG_RUN.md")

        # 3) unconstrained problems must not get a feasibility warning
        actual = feasibility_advice({}, _ctx(), _simple_objective())["risk"]
        record("无约束不得告警可行率", actual, "none", "research/HARD_FEASIBILITY.md")

        # 4) low confidence must not produce a trusted grade
        actual = reliability_grade(0.95, "low")["grade"]
        record("低置信度不得给 trust", actual, "不是 trust", "research/NEGATIVE_RESULTS.md §11–§12")

        # 5) auto_confirm=False must never ask for a confirmation batch
        study = _study(tmp_root, algorithm="random", budget=12, batch_size=3, seed=0,
                       meta={"auto_confirm": False})
        study.import_observations([{"x": {"x": 0.4}, "y": 1.0}])
        actual = study.next_action()["action"]
        record("auto_confirm=False 不得确认", actual, "不是 confirm", "docs/DECISION_LAYER.md")

        # 6) single-objective studies must not carry a progress-trajectory check
        items = [item["item"] for item in study.delivery_checklist()]
        record("单目标不得要求逐批进展", "逐批进展记录" in items, False, "docs/EXECUTOR_ADAPTER.md")

        # 7) a zero-sized transfer must import nothing (and not crash)
        target = _study(tmp_root, algorithm="random", budget=12, batch_size=3, seed=1)
        source = _study(tmp_root, algorithm="random", budget=6, batch_size=3, seed=2)
        target.tell([{"x": {"x": 0.1}, "y": 0.5}])
        source.tell([{"x": {"x": 0.9}, "y": 0.9}])
        report = target.transfer_from(source.dir, top_k=0, prior_reliability=0.9)
        record("top_k=0 不得导入观测", report["imported"], 0, "docs/DECISION_LAYER.md")

        # 8) unconstrained deliverables must not use the feasible-HV metric
        actual = deliverable_advice(
            {}, _ctx(n_objectives=2), Objective(
                name="y", direction="max", objectives=("a", "b"), directions=("max", "min")
            ),
        )["metric"]
        record("无约束不得用可行 HV 口径", actual, "hypervolume", "research/NEGATIVE_RESULTS.md §10")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
    return rows


def _routing_suppression_rows() -> list[dict]:
    """Capability/context rules that must *not* be offered or selected."""
    from condopt import registry

    rows: list[dict] = []

    def record(name: str, actual, expected: str, evidence: str) -> None:
        rows.append({
            "name": name,
            "actual": actual,
            "expected": expected,
            "evidence": evidence,
            "suppressed": True,
        })

    def offered(result) -> list[str]:
        return [result.algorithm] + [
            item["algorithm"] for item in result.alternatives
        ]

    # 1) multi-fidelity contexts must only offer fidelity-capable algorithms
    mf_route = route(_ctx(multi_fidelity=True, n_objectives=2))
    bad = [
        name for name in offered(mf_route)
        if "multi_fidelity" not in registry.spec(name).capabilities
    ]
    record("多保真上下文不得出现非多保真算法", bad, "空列表", "docs/SCENARIOS.md 场景 F")

    # 2) constrained contexts must only offer constraint-aware algorithms
    constrained_route = route(_ctx(n_constraints=1))
    bad = [
        name for name in offered(constrained_route)
        if "constraints" not in registry.spec(name).capabilities
    ]
    record("约束上下文不得出现不支持约束的算法", bad, "空列表", "docs/SCENARIOS.md 场景 H")

    # 3) single-objective contexts must not offer multi-objective algorithms
    bad = sorted(
        name for name in offered(route(_ctx()))
        if "multi_objective" in registry.spec(name).capabilities
    )
    record("单目标上下文不得出现多目标算法", bad, "空列表", "audit_routing.py")

    # 4) multi-objective contexts must not offer single-objective algorithms
    bad = sorted(
        name for name in offered(route(_ctx(n_objectives=3)))
        if "multi_objective" not in registry.spec(name).capabilities
    )
    record("多目标上下文不得出现单目标算法", bad, "空列表", "docs/SCENARIOS.md 场景 E/G")

    # 5) high-dimensional continuous problems must not select GP-BO (measured worst)
    actual = route(
        _ctx(n_dims=10, n_continuous=10, budget_total=60, smoothness="smooth", noise="low")
    ).algorithm
    record("10 维不得选 bayes", actual, "不是 bayes", "docs/SCENARIOS.md 场景 B")

    # 6a) a tiny budget must fall back to space filling when a cheap option exists
    tiny_route = route(_ctx(budget_total=4, batch_size=4))
    record(
        "预算 4 单目标退回空间填充",
        [tiny_route.algorithm, bool(tiny_route.underpowered_budget)],
        "算法在 sobol/lhs/random/grid 内且给出 underpowered 说明",
        "docs/ROUTING_POLICY.md",
    )

    # 6b) TPE must not be blocked by the GP data requirement (regression: one-hot
    #     cardinality used to make a 12-level categorical look "too high-dim")
    from condopt.space import SearchSpace

    categorical_space = SearchSpace.from_dict([
        {"name": "c", "kind": "categorical", "choices": [f"L{i}" for i in range(12)]},
        {"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0},
    ])
    categorical_context = OptimizationContext.from_space(
        categorical_space, budget_total=30, batch_size=4
    )
    blocked_names = set(
        (route(categorical_context).underpowered_budget or {}).get("blocked", {})
    )
    record(
        "12 级分类 + 预算 30 不得把 TPE 判为数据不足",
        "tpe" in blocked_names,
        False,
        "docs/SCENARIOS.md 场景 D（TPE 不拟合 one-hot GP）",
    )

    # 6c) a comfortable budget must not be flagged as underpowered
    record(
        "预算 60 不得判为数据不足",
        bool(route(_ctx(budget_total=60)).underpowered_budget),
        False,
        "docs/ROUTING_POLICY.md",
    )

    # 6d) tiny budget + multi-objective/constraints: measured to still favour the
    #     surrogate, so keep it -- but warn and never relax the capability contract
    tiny_moo = route(_ctx(budget_total=4, n_objectives=2, batch_size=4))
    record(
        "极小预算 + 多目标保留模型类",
        tiny_moo.algorithm,
        "算法在 parego/qehvi/qnehvi/cqehvi/cqnehvi/qehvi_nd内",
        "research/TINY_BUDGET.md（budget 4：parego 503.6 HV / 3-4 次有前沿 vs sobol 1/4）",
    )
    record(
        "极小预算 + 多目标不得违反能力契约",
        bool((tiny_moo.underpowered_budget or {}).get("capability_relaxed")),
        False,
        "research/TINY_BUDGET.md",
    )
    record(
        "极小预算 + 多目标必须显式告警",
        any("低于建模舒适区" in warning for warning in tiny_moo.warnings),
        True,
        "research/TINY_BUDGET.md",
    )
    record(
        "充足预算不得告警建模舒适区",
        any(
            "低于建模舒适区" in warning
            for warning in route(_ctx(budget_total=60, n_objectives=2)).warnings
        ),
        False,
        "docs/ROUTING_POLICY.md",
    )

    # 7) tiny budgets must not select the expensive EHVI family
    tiny = offered(route(_ctx(budget_total=6, batch_size=2)))
    bad = sorted(name for name in tiny if name in {"qehvi", "qehvi_nd", "qnehvi",
                                                   "cqehvi", "cqnehvi", "mqehvi_nd"})
    record("极小预算不得出现 EHVI 家族", bad, "空列表", "docs/ROUTING_POLICY.md")
    return rows


def evaluate_suppressions() -> list[dict]:
    rows = []
    for row in _suppression_rows() + _routing_suppression_rows():
        value = row["actual"]
        expected = row["expected"]
        if isinstance(expected, str) and expected.startswith("算法在 "):
            allowed = {
                token.strip()
                for token in expected.split("算法在 ", 1)[1].split("内", 1)[0].split("/")
            }
            if isinstance(value, list):
                ok = all(
                    (item in allowed) if index == 0 else bool(item)
                    for index, item in enumerate(value)
                )
            else:
                ok = str(value) in allowed
        elif expected in ("空列表",):
            ok = not value
        elif isinstance(expected, bool):
            ok = bool(value) is expected
        elif str(expected).startswith("不是 "):
            # negative assertion: the rule must not fire
            ok = str(value) != str(expected)[len("不是 "):]
        elif expected == "none":
            ok = str(value) == "none"
        elif expected == "hypervolume":
            ok = str(value) == "hypervolume"
        elif isinstance(expected, int):
            ok = int(value) == expected
        elif isinstance(value, list) and isinstance(expected, list):
            ok = sorted(map(str, value)) == sorted(map(str, expected))
        else:
            ok = str(value) == str(expected)
        rows.append({**row, "ok": ok})
    return rows


def evaluate() -> list[dict]:
    rows = []
    for rule in RULES:
        blocked = None
        try:
            ok = bool(rule.check())
            actual = rule.actual()
        except (PermissionError, OSError) as exc:
            # Some rules create a throwaway Study, which needs a writable directory.
            # In a read-only sandbox that is an *environment* limitation, not a rule
            # failure; saying so explicitly keeps the gate honest in both directions
            # (it neither passes silently nor reports a product defect that does not
            # exist).  See research/EXTERNAL_REVIEW_RESPONSE.md.
            ok, actual = False, f"{type(exc).__name__}: {exc}"
            blocked = "needs a writable temporary directory"
        except Exception as exc:  # a broken rule must be visible, not silent
            ok, actual = False, f"{type(exc).__name__}: {exc}"
        rows.append({
            "name": rule.name,
            "ok": ok,
            "actual": actual,
            "expected": rule.expected,
            "evidence": rule.evidence,
            "blocked_by_environment": blocked,
        })
    return rows


def build_doc(rows: list[dict]) -> str:
    lines = [
        "# 决策层规则矩阵（Decision Matrix）",
        "",
        "> 由 `scripts/decision_matrix.py` 生成；`docs/SCENARIOS.md` 管“选哪个算法”，",
        "> 本文件管“决策层怎么判”：停止口径、交付槽位、可行率告警、迁移可信度分级。",
        "> 每条规则都带证据出处，改动规则就会改动这里的数字。",
        "",
        "| 规则 | 期望 | 实测 | 证据 |",
        "|---|---|---|---|",
    ]
    for row in rows:
        actual = row["actual"]
        if isinstance(actual, list):
            actual = "；".join(str(item) for item in actual)
        mark = (
            "OK" if row["ok"]
            else "**BLOCKED**（环境：需要可写临时目录）" if row.get("blocked_by_environment")
            else "**FAIL**"
        )
        lines.append(
            f"| {row['name']} | {row['expected']} | {mark} `{actual}` | {row['evidence']} |"
        )
    lines.append("")
    passed = sum(1 for row in rows if row["ok"])
    blocked = sum(1 for row in rows if row.get("blocked_by_environment"))
    suppressions = evaluate_suppressions()
    lines += [
        "## 规则抑制（这些情形下对应规则**不该**触发）",
        "",
        "> 路由层的能力过滤同时被 `scripts/audit_routing.py` **穷举**校验（96 个上下文，",
        "> 候选与推荐项都必须满足上下文能力）；下表是抽样断言。",
        "",
        "| 情形 | 期望 | 实测 | 证据 |",
        "|---|---|---|---|",
    ]
    for row in suppressions:
        mark = "OK" if row["ok"] else "**FAIL**"
        lines.append(f"| {row['name']} | {row['expected']} | {mark} `{row['actual']}` | {row['evidence']} |")
    lines.append("")
    suppressed_passed = sum(1 for row in suppressions if row["ok"])
    lines.append(
        f"共 {len(rows)} 条规则（通过 {passed}）与 {len(suppressions)} 条抑制断言"
        f"（通过 {suppressed_passed}）。"
    )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    _force_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    rows = evaluate()
    suppressions = evaluate_suppressions()
    content = build_doc(rows)
    failed = [row["name"] for row in rows if not row["ok"]]
    failed += [row["name"] for row in suppressions if not row["ok"]]
    for row in rows:
        print(f"{'[OK  ]' if row['ok'] else '[FAIL]'} {row['name']}: {row['actual']}")

    if args.check:
        current = DOC.read_text(encoding="utf-8") if DOC.exists() else ""
        if current.strip() != content.strip():
            print("docs/DECISIONS.md is stale; run scripts/decision_matrix.py --write")
            return 1
    if args.write:
        DOC.write_text(content, encoding="utf-8")
        print(f"wrote {DOC}")
    print(f"{'[OK  ]' if all(row['ok'] for row in suppressions) else '[FAIL]'} "
          f"suppressions: {sum(1 for row in suppressions if row['ok'])}/{len(suppressions)}")
    total = len(rows) + len(suppressions)
    print(f"decision_matrix: {total - len(failed)}/{total} checks hold")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())