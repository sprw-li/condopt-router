"""Turn skill-questionnaire answers into condopt JSON files.

The skill asks the user 9 questions (objective, variables, budget, batch,
noise, smoothness, prior data, constraints, context).  Instead of asking the
agent to hand-write three JSON files, this module accepts one compact
``answers.json`` and produces a validated ``space.json`` / ``context.json`` /
``objective.json`` triple plus the router recommendation.

Example ``answers.json``::

    {
      "objective": {"name": "yield", "direction": "max"},
      "variables": [
        {"name": "T", "kind": "continuous", "low": 40, "high": 120, "unit": "degC"},
        {"name": "solvent", "kind": "categorical", "choices": ["MeCN", "EtOH"]}
      ],
      "budget": 24, "batch": 4,
      "noise": "medium", "smoothness": "smooth", "domain": "chemistry"
    }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

from .base import Objective
from .spec import Spec
from .router import OptimizationContext, route
from .space import SearchSpace

#: Aliases accepted in answers.json (skill-friendly names -> context fields).
ALIASES = {
    "budget": "budget_total",
    "batch": "batch_size",
    "prior": "n_observations",
    "constraints": "n_constraints",
    "fidelity": "multi_fidelity",
}

#: Fields of ``OptimizationContext`` that a user may set by hand.
CONTEXT_FIELDS = {
    "budget_total",
    "budget_remaining",
    "batch_size",
    "noise",
    "n_objectives",
    "n_constraints",
    "multi_fidelity",
    "compute",
    "smoothness",
    "n_observations",
    "domain",
    "notes",
}


def template() -> dict:
    """A filled-in example the skill can show the user before editing."""
    return {
        "objective": {"name": "yield", "direction": "max"},
        "variables": [
            {"name": "T", "kind": "continuous", "low": 40, "high": 120, "unit": "degC"},
            {"name": "time", "kind": "continuous", "low": 5, "high": 120, "unit": "min"},
            {"name": "conc", "kind": "log", "low": 0.02, "high": 1.0, "unit": "M"},
            {"name": "solvent", "kind": "categorical", "choices": ["MeCN", "EtOH", "water"]},
            {"name": "equiv", "kind": "integer", "low": 1, "high": 4},
            {"name": "inert", "kind": "bool"},
        ],
        "budget": 24,
        "batch": 4,
        "noise": "medium",
        "smoothness": "smooth",
        "domain": "chemistry",
        "prior": 0,
        "notes": "按实际实验情况修改；budget 是含先验数据在内的总评价次数",
    }


def parse_answers(answers: Mapping) -> tuple[SearchSpace, Objective, OptimizationContext]:
    if not isinstance(answers, Mapping):
        raise TypeError("answers must be a JSON object")
    payload = dict(answers)
    variables = payload.pop("variables", None)
    if variables is None and isinstance(payload.get("space"), Mapping):
        variables = payload["space"].get("variables")
    if not variables:
        raise ValueError("answers.variables 缺失：至少需要一个条件变量")
    space = SearchSpace.from_dict({"variables": variables})

    spec_payload = payload.get("spec")
    if spec_payload:
        spec = Spec.from_dict(spec_payload)
        objective = spec.scalar_objective()
    else:
        objective_payload = payload.get("objective") or {}
        if not isinstance(objective_payload, Mapping):
            raise ValueError("answers.objective 必须是对象，如 {\"name\":\"yield\",\"direction\":\"max\"}")
        objective = Objective.from_dict(objective_payload)
    missing_limits = [
        name for name in objective.constraints if name not in objective.constraint_limits
    ]
    if missing_limits:
        raise ValueError(
            f"约束 {missing_limits} 缺少 constraint_limits（如 {{\"T_max\": {{\"op\": \"<=\", \"value\": 110}}}}）"
        )

    context_payload: dict = {}
    for key, value in payload.items():
        if key == "space":
            continue
        target = ALIASES.get(key, key)
        if target in CONTEXT_FIELDS:
            context_payload[target] = value
    # Objectives/constraint counts are derived from the objective definition.
    context_payload.setdefault("n_objectives", objective.n_objectives)
    context_payload.setdefault("n_constraints", len(objective.constraints))
    if "n_objectives" in context_payload and int(context_payload["n_objectives"]) != objective.n_objectives:
        raise ValueError(
            f"n_objectives={context_payload['n_objectives']} 与 objective.objectives "
            f"({objective.n_objectives} 个) 不一致"
        )

    explicit_remaining = context_payload.get("budget_remaining") is not None
    context = OptimizationContext.from_space(space, **context_payload)
    if not explicit_remaining:
        context.budget_remaining = max(0, context.budget_total - context.n_observations)
    return space, objective, context


#: Every key ``write_problem`` returns, with what it is for.  ``docs/PREPARE_BRIEF.md``
#: documents exactly these names (checked by tests, so renames cannot silently drift).
FIELD_GUIDE = {
    "out_dir": "写出的目录",
    "space": "space.json 路径",
    "context": "context.json 路径",
    "objective": "objective.json 路径",
    "spec": "spec.json 路径（未提供 spec 时为 null）",
    "spec_calibration": "spec_calibration.json 路径（未校准时为 null）",
    "recommendation": "路由建议：算法 + 参数 + 理由/警告",
    "stopping": "停止口径建议（stagnation_first / budget_first）",
    "feasibility": "可行率预警与处置建议",
    "prior": "有先验数据时的暖启动与可信度建议",
    "deliverable": "预期交付口径：交付指标、必须包含的证据、如何自查",
    "noise_plan": "噪声处置建议：采集函数/alpha/批量/复孔/是否加基线臂",
}


def noise_advice(answers: Mapping, context: OptimizationContext) -> dict:
    """Turn the measured noise level into concrete actions.

    The skill documents "high noise -> bayes + ucb + larger alpha, or a random
    baseline"; before this the advice existed only in prose.  Noise also interacts
    with the delivery rules (a lucky single measurement is not a result), so the
    plan always mentions replicates.
    """
    level = str(answers.get("noise") or context.noise or "medium").lower()
    batch = int(getattr(context, "batch_size", 4) or 4)
    plan = {
        "level": level,
        "advice": "",
        "params": {},
        "replicates": 1,
        "baseline_arm": False,
    }
    if level == "high":
        plan.update(
            advice=(
                "噪声高：先做**复孔 + 缩小批量（≤6）**，并保留一条随机/空间填充基线臂"
                "（否则无法区分“模型有效”与“运气好”）。"
                "注意：本项目实测（mixed_reaction，σ=0/5/12）UCB + 大 kappa 反而更差，"
                "把 alpha 抬到上限也只在中等噪声下略好——切采集函数或调 alpha 前先做 A/B"
            ),
            params={},
            replicates=3,
            baseline_arm=True,
        )
    elif level == "medium":
        plan.update(
            advice=(
                "噪声中等：默认 EI 可接受，可把 alpha 抬到 0.1 量级，但收尾的确认复孔不要省"
                "（单次测量可能只是运气好）；批量建议 ≤ 6"
            ),
            params={"alpha": 0.1},
            replicates=2,
        )
    else:
        plan.update(
            advice="噪声低：可用 EI + 小 alpha，批量可以开大一些以提高样本效率",
            params={"alpha": 1.0},
            replicates=1,
        )
    if batch > 8 and level in {"medium", "high"}:
        plan["advice"] += f"；当前批量 {batch} 偏大，建议先降到 {min(batch, 6)} 或分板对照"
    return plan


def stopping_advice(context: OptimizationContext) -> dict:
    """Recommend a stopping policy for this campaign.

    Measured (``research/ADAPTER_LONG_RUN.md``, 120-evaluation budget): the
    default stagnation-first rule stopped at 69 evaluations (~57% of budget),
    while ``stop_on_stagnation=False`` used all 120 (best 108.79 vs 106.31).
    So: small/bounded budgets want the early stop; large budgets the user has
    paid for want the budget-first mode.
    """
    from .study import default_confirm_n

    budget = int(context.budget_total or 0)
    budget_first = budget >= 60
    confirm_n = default_confirm_n(budget) if budget else 3
    # The delivery rule holds back a confirmation batch (and, once the incumbent is
    # confirmed, one slot for a possible new winner), so the number of evaluations
    # actually available for exploration is smaller than the nominal budget.
    delivery_slots = confirm_n if budget > confirm_n else max(0, min(confirm_n, budget))
    effective_exploration = max(0, budget - delivery_slots)
    return {
        "recommended": "budget_first" if budget_first else "stagnation_first",
        "delivery_slots": delivery_slots,
        "effective_exploration_budget": effective_exploration,
        "how_to_enable": "Study.create(..., stop_on_stagnation=False) 或 condopt init --budget-first",
        "why": (
            f"预算 {budget}：实测默认停滞口径会在约 57% 处停止（69/120），"
            + (
                "预算较大时建议跑满，以获得更好的最优值与更充分的设计空间覆盖"
                if budget_first
                else "预算有限时建议保留提前停止，把省下的次数留给下一轮实验"
            )
        ),
        "options": {
            "stagnation_first": "连续 N 轮无显著改进即停（默认，patience=3）",
            "budget_first": "停滞仅作为建议，直到预算/成本/目标触发",
        },
        "patience": "默认 3（最小 2）；越大越晚停",
        "delivery_tradeoff": (
            "实测：budget_first 在 40–120 预算下最优值明显更高（MF+约束 92.9/98.8/100.6 vs "
            "71.8/65.6/76.9）。经修正交付检查口径后，40–90 预算下两种口径都能通过交付自查；"
            "但 120 预算上仍有异常（只用 90/120 且交付失败），需要时请显式预留确认份额。"
            if budget_first
            else None
        ),
        "delivery_note": (
            f"预算 {budget} 中有 {delivery_slots} 次保留给确认复孔/交付槽位，"
            f"可探索约 {effective_exploration} 次（最终交付点必须带确认复孔）"
            if budget
            else "建议先确定预算，确认复孔会占用其中一部分"
        ),
    }


def feasibility_advice(
    answers: Mapping,
    context: OptimizationContext,
    objective,
    observations: Sequence | None = None,
) -> dict:
    """Warn before the campaign starts when the feasible region looks tiny.

    Measured (``research/HARD_FEASIBILITY.md``): with a ~0.4% feasible box,
    constraint-aware search found a feasible point in only 1/5 runs, while
    encoding the same limits as variable bounds succeeded 5/5.  The estimate here
    uses the user's own number when given, otherwise a structural heuristic.
    """
    names = tuple(getattr(objective, "constraints", ()) or ())
    if not names:
        return {"risk": "none", "advice": "无约束，无需担心可行率"}
    supplied = answers.get("feasible_fraction_estimate")
    dims = int(context.n_dims or 0)
    if supplied is not None:
        fraction = float(supplied)
        basis = f"用户给出的可行率估计 {fraction:.1%}"
    elif len(names) >= 3 and dims <= 6:
        fraction, basis = 0.05, f"{len(names)} 条约束落在 {dims} 维空间上（结构性推测）"
    elif len(names) >= 2 and dims <= 4:
        fraction, basis = 0.15, f"{len(names)} 条约束 / {dims} 维（结构性推测）"
    else:
        fraction, basis = None, f"{len(names)} 条约束 / {dims} 维，未发现明显窄域信号"

    # <= 5% mirrors the runtime rule (step flags observed feasible fraction < 0.05
    # and the measured 1/5 vs 5/5 evidence), so a 5% estimate is already "high".
    risk = "high" if (fraction is not None and fraction <= 0.05) else (
        "medium" if (fraction is not None and fraction < 0.15) else "low"
    )
    advice = (
        "建议把已知硬限制写进变量范围或固定因子（真实工艺规程本来就该锁死），"
        "不要指望约束采集去“找”可行域"
        if risk == "high"
        else "约束机制适用于“可行域较大、只是局部越界”的情形；若实测可行率偏低，"
        "step 会给出 encode_limits_as_bounds 建议"
    )
    # Condition-dependent encoding advice (research/DESCRIPTOR_FEATURES.md, 5 seeds):
    # physical descriptors raised the feasible rate 0.55 -> 0.875 with the
    # constraint-aware optimizer, but *lowered* it (0.285 -> 0.150) for a
    # constraint-blind one.  So the encoding suggestion is only safe together with a
    # constraint-aware algorithm.
    # ``OptimizationContext`` carries aggregate counts (not the space itself):
    # n_categorical + cardinality are what the router uses, and n_descriptor_vars
    # tells us whether descriptors are already in play.
    # With a continuous variable present ``cardinality`` is infinite (None), so use the
    # one-hot blow-up as the high-cardinality signal: feature_dim - n_dims is the extra
    # columns the surrogate must model.
    one_hot_blowup = int(getattr(context, "feature_dim", 0) or 0) - int(
        getattr(context, "n_dims", 0) or 0
    )
    wide_categorical = (
        int(getattr(context, "n_categorical", 0) or 0) >= 1
        and not int(getattr(context, "n_descriptor_vars", 0) or 0)
        and one_hot_blowup >= 8
    )
    if wide_categorical:
        advice += (
            "；本问题含高基数分类变量：有绑定约束时**同时**用物理描述符编码与约束感知算法"
            "（constrained/cqehvi）可提高**可行点覆盖**（单目标可行率 0.55→0.875；MOO 可行点 +25%），"
            "但**最优值/HV 不会因此变好**（MOO 实测 −1.4%），且只换编码不换算法会更差（bayes 0.285→0.150）"
        )
    coverage = None
    if observations:
        from .feasibility import coverage_note

        coverage = coverage_note(list(observations), objective)
        if coverage:
            # "未测到" must never be read as "满足" (A12).
            advice += "；" + str(coverage["advice"])
    return {
        "risk": risk,
        "estimated_feasible_fraction": None if fraction is None else round(float(fraction), 4),
        "basis": basis,
        "advice": advice,
        "coverage": coverage,
        "reference": "research/HARD_FEASIBILITY.md（0.4% 可行域：约束搜索 1/5 vs 写进范围 5/5）",
    }


def deliverable_advice(answers: Mapping, context: OptimizationContext, objective) -> dict:
    """What the campaign will actually deliver, and what evidence it must include.

    Measured context: raw hypervolume over all observations overstates what ships
    when constraints exist (143.03 vs deliverable 80.94), and multi-fidelity
    deliveries need at least one *feasible full-fidelity* observation
    (``research/NEGATIVE_RESULTS.md`` §10/§11, ``docs/SCENARIOS.md``).
    """
    constraints = tuple(getattr(objective, "constraints", ()) or ())
    multi_objective = bool(getattr(objective, "is_multi_objective", False))
    multi_fidelity = bool(context.multi_fidelity)
    if multi_objective:
        metric = "feasible_hypervolume" if constraints else "hypervolume"
        definition = (
            "可行非支配前沿的体积（约束下不要把 raw HV 当交付指标）"
            if constraints
            else "非支配前沿的体积（HV）"
        )
    else:
        metric = "feasible_best_value" if constraints else "best_value"
        definition = (
            "满足全部约束的最优目标值" if constraints else "最优目标值（含确认复孔）"
        )
    from .study import default_confirm_n

    budget = int(context.budget_total or 0)
    confirm_n = default_confirm_n(budget) if budget else 3
    confirmation_plan = {
        "n": confirm_n,
        "share_of_budget": round(confirm_n / budget, 3) if budget else None,
        "advice": (
            f"收尾会自动做 {confirm_n} 次确认复孔"
            + (f"（约占预算 {confirm_n / budget:.0%}）" if budget else "")
            + ("；小预算下已按下限 2 次" if confirm_n == 2 else "")
        ),
    }
    must_include = []
    if constraints:
        must_include.append("至少一条可行的全保真（或最终保真）观测")
    if multi_fidelity:
        must_include.append("全保真档位的交付证据，而不是只有低保真推测")
    if multi_objective:
        must_include.append("逐批进展（HV 曲线）与前沿点数")
    else:
        must_include.append("最优条件的复孔确认（避免单次幸运值）")
    return {
        "metric": metric,
        "definition": definition,
        "must_include": must_include,
        "confirmation_plan": confirmation_plan,
        "how_to_check": "condopt digest --run <run_dir>（含可交付前沿/执行日志/停止原因）",
        "reference": "docs/SCENARIOS.md 的“指标口径”小节；research/NEGATIVE_RESULTS.md §10",
    }


def reliability_grade(value: float | None, confidence: str = "normal") -> dict:
    """Turn a ``transfer-audit`` suggestion into an actionable grade.

    Thresholds come from ``research/TRANSFER_AUDIT.md``: a consistent source scores
    1.0 while shifted/shuffled/flipped sources land in 0.3-0.6 (one shuffled seed
    reached 0.6), so 0.5-0.8 must be treated as "suspect", not "fine".
    """
    if value is None:
        return {"grade": "unknown", "advice": "先跑 condopt transfer-audit 拿到建议值"}
    value = float(value)
    if confidence == "low":
        # <5 comparable conditions: the estimate is noisy, so treat it as advice,
        # not a verdict (real ART data: suggested 0.3 while warm start won 2/2).
        return {
            "grade": "low_confidence",
            "advice": (
                f"可信度 {value:.2f}，但可比条件太少（<5）：建议值不可当硬约束；"
                "先用小 top_k（1–2）试一次，或补目标观测后再评估"
            ),
        }
    if value >= 0.8:
        return {
            "grade": "trust",
            "advice": (
            f"可信度 {value:.2f}：按原值暖启动（不要打折），但**审计分高 ≠ 收益为正**"
            "（实测源偏差仅 −3 就翻负）——先 top_k 1、跑两批再看审计决定是否放大："
            "实测该做法在有偏源上 +0.71（固定 k=3/k=5 是 −1.54/−0.33），好源上与固定 k=5 持平；"
            "否决线取 0.4–0.6，低于 0.4 起不到保护作用、≥0.8 会连好源一起否决；"
            "**审计不否决 ≠ 源可靠**（实测有偏源在一个问题上 0/4 被否决、分阶段因此 −1.03，"
            "不如 fixed_k1 的 −0.32）——没有独立证据时保持 top_k=1 最稳"
            "（research/TRANSFER_ADAPTIVE.md）。打折的代价同样真实：r=1.0 → +44.9，r=0.4 → +39.3"
        ),
        }
    if value >= 0.5:
        return {
            "grade": "suspect",
            "advice": (
                f"可信度 {value:.2f}：可疑区间（打乱源曾给出 0.6）；"
                "把 top_k 压到 1–2，或先补几条目标观测再评估"
            ),
        }
    return {
        "grade": "avoid",
        "advice": (
            f"可信度 {value:.2f}：不要用大 top_k 暖启动；"
            "宁可从零开始，或把 hard 限制写进变量范围（见 research/HARD_FEASIBILITY.md）"
        ),
    }


def prior_advice(
    answers: Mapping, context: OptimizationContext, objective=None
) -> dict:
    """What to do when the campaign starts from existing data.

    Measured (``research/TRANSFER_BENCHMARK.md``): a mismatched source transfers
    *negatively* and the damage grows with ``top_k`` (top_k=1: -369 -> top_k=8:
    -1190), while lowering ``prior_reliability`` to 0.25-0.5 recovers most of it
    (-884 -> -698). ``transfer-audit`` separates a good source (suggests 1.0) from
    a shuffled one (0.3-0.6).
    """
    n_prior = int(answers.get("prior") or context.n_observations or 0)
    if n_prior <= 0:
        return {"has_prior": False, "advice": "无先验数据；正常从零开始采样"}
    multi_objective = bool(getattr(objective, "is_multi_objective", False))
    if multi_objective and bool(context.multi_fidelity):
        return {
            "has_prior": True,
            "n_prior": n_prior,
            "mode": "pareto_front_mf",
            "advice": (
                "多保真 + 多目标：暖启动收益微弱且要付预算成本（迁移点按次数占目标预算）；"
                "小 top_k 基本中性，top_k≈8 并包含低保真点时略正（+0.35%）；"
                "千万不要调低 reliability（−0.95%）"
            ),
            "how_to_enable": (
                "Study.create(..., transfer_from=<源 run>, prior_reliability=1.0)；"
                "不开就是默认行为"
            ),
            "how_to_read_reliability": {
                "trust": ">=0.8",
                "suspect": "0.5–0.8",
                "avoid": "<0.5",
            },
            "evidence": (
                "等预算对照（5 seeds）：不开 626.92；k3 r=1.0 626.27；k5 r=1.0 626.68；"
                "k8 r=1.0 含低保真 **629.09（+0.35%）**；k5 r=0.5 **620.94（−0.95%）**；"
                "迁移点按次数占用目标预算（research/TRANSFER_MOO.md）"
            ),
        }
    if multi_objective:
        return {
            "has_prior": True,
            "n_prior": n_prior,
            "mode": "pareto_front",
            "advice": (
                "多目标：用 transfer_from 迁移源战役的**非支配前沿**（不是主目标 top_k），"
                "top_k 取 3–5；跨问题时先 condopt transfer-audit 判可信度"
            ),
            "how_to_enable": (
                "Study.create(..., transfer_from=<源 run>, prior_reliability=<audit 建议值>)；"
                "transfer_from 会自动按前沿选点（selection=pareto_front）"
            ),
            "evidence": (
                "等预算对照（5 seeds）：可行 HV +0.4–0.5%（642.66 → 645.9），"
                "前沿点数 +20%（9.6 → 11.6）；reliability 1.0 与 0.5 无差别（research/TRANSFER_MOO.md）"
            ),
        }
    return {
        "has_prior": True,
        "n_prior": n_prior,
        "advice": (
            "用 transfer_from 暖启动而不是把先验拼进本次数据：先跑 "
            "condopt transfer-audit <源 run> 判断来源可信度（一致源会建议 1.0，"
            "打乱源 0.3–0.6），再按建议设置 --prior-reliability；top_k 建议 1–3"
        ),
        "how_to_enable": (
            "Study.create(..., transfer_from=<源 run>, prior_reliability=<audit 建议值>) "
            "或 condopt import --prior-reliability <值>"
        ),
        "how_to_read_reliability": {
            "trust": ">=0.8（一致源实测 1.0）",
            "suspect": "0.5–0.8（打乱源曾给出 0.6，按可疑处理）",
            "avoid": "<0.5（shifted/flipped 实测 0.3）",
        },
        "evidence": (
            "坏源实测：top_k 1→8 从 −369 恶化到 −1190；reliability 降到 0.25–0.5 "
            "可回升到 −698（TRANSFER_BENCHMARK.md）"
        ),
    }


def write_problem(answers: Mapping, out_dir: str | Path) -> dict:
    """Validate answers, write the three JSON files, and route the problem."""
    space, objective, context = parse_answers(answers)
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    space_path = target / "space.json"
    context_path = target / "context.json"
    objective_path = target / "objective.json"
    spec_path = target / "spec.json"
    spec_payload = answers.get("spec")
    calibration_path = target / "spec_calibration.json"
    calibration_report = None
    if spec_payload:
        spec = Spec.from_dict(spec_payload)
        source_run = answers.get("spec_calibrate_from")
        if source_run:
            # Reuse measured history to calibrate scales/tolerances before the
            # next campaign starts (the explicit spec stays the fallback).
            from .study import Study

            source = Study.load(source_run)
            spec, calibration_report = spec.calibrate_observations(
                source.dataset.observations,
                by_plate=bool(answers.get("spec_calibrate_by_plate")),
                min_samples=int(answers.get("spec_calibrate_min_samples") or 3),
                tolerance_sigmas=float(answers.get("spec_calibrate_tolerance_sigmas") or 2.0),
            )
            calibration_path.write_text(
                json.dumps(calibration_report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        spec_path.write_text(
            json.dumps(spec.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    space_path.write_text(
        json.dumps(space.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    context_path.write_text(
        json.dumps(context.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    objective_path.write_text(
        json.dumps(objective.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    recommendation = route(context)
    return {
        "out_dir": str(target),
        "space": str(space_path),
        "context": str(context_path),
        "objective": str(objective_path),
        "spec": str(spec_path) if spec_payload else None,
        "spec_calibration": str(calibration_path) if calibration_report is not None else None,
        "recommendation": recommendation.to_dict(),
        "stopping": stopping_advice(context),
        "feasibility": feasibility_advice(answers, context, objective),
        "prior": prior_advice(answers, context, objective),
        "deliverable": deliverable_advice(answers, context, objective),
        "noise_plan": noise_advice(answers, context),
    }
