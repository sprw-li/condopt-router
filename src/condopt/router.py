"""Algorithm router: choose an optimizer and its parameters from context.

The router is deliberately a *transparent rule engine*, not a black box.
Each algorithm candidate accumulates score contributions and every
contribution carries a human readable reason.  The skill layer shows the
recommendation, its reasons and its alternatives to the user/AI; an
override is allowed but must be logged with a justification (see
``Study.log_decision``), so the hard algorithm layer stays auditable.

Design rationale
----------------
* Rules beat a learned policy at this stage: no training data exists, and
  a wrong recommendation must be explainable and debuggable.
* The router only recommends *registered* algorithms with *validated*
  parameters -- it can never invent a knob the algorithm does not have.
* ``route_round`` adds mid-run re-routing: if a model-based algorithm
  stalls, the router inserts a model-free exploration batch instead of
  letting the surrogate exploit a wrong region forever.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from . import registry
from .space import SearchSpace

NOISE_LEVELS = ("none", "low", "medium", "high")
SMOOTHNESS_LEVELS = ("smooth", "mixed", "rough")
COMPUTE_LEVELS = ("low", "medium", "high")
DOMAINS = ("chemistry", "materials", "process", "generic")

#: Optional per-algorithm score multipliers for calibrated routing.
#: Empty dict = the hand-tuned defaults; factors only scale the final score.
_ALGORITHM_WEIGHTS: dict[str, float] = {}


def set_algorithm_weights(weights: Mapping[str, float] | None) -> None:
    """Set calibration multipliers (``None`` resets to the defaults)."""
    global _ALGORITHM_WEIGHTS
    cleaned: dict[str, float] = {}
    known = set(registry.names())
    for name, value in (weights or {}).items():
        if name not in known:
            raise ValueError(f"unknown algorithm {name!r} in weights; known: {sorted(known)}")
        factor = float(value)
        if not math.isfinite(factor) or factor < 0:
            raise ValueError(f"weight for {name!r} must be a finite value >= 0")
        cleaned[name] = factor
    _ALGORITHM_WEIGHTS = cleaned


def algorithm_weights() -> dict[str, float]:
    return dict(_ALGORITHM_WEIGHTS)


@dataclass
class OptimizationContext:
    """Everything the router needs to know about the optimization problem."""

    n_dims: int = 0
    n_continuous: int = 0
    n_integer: int = 0
    n_log: int = 0
    n_categorical: int = 0
    n_ordinal: int = 0
    n_bool: int = 0
    #: How many categorical variables carry physical descriptors
    n_descriptor_vars: int = 0
    #: Numeric feature dimension seen by the GP (numeric + one-hot); derived
    #: from the search space when available.  The router must use this rather
    #: than ``n_dims``, because one-hot categorical variables explode it.
    feature_dim: int | None = None
    budget_total: int = 30
    budget_remaining: int | None = None
    batch_size: int = 1
    noise: str = "medium"
    n_objectives: int = 1
    n_constraints: int = 0
    multi_fidelity: bool = False
    compute: str = "medium"
    smoothness: str = "mixed"
    n_observations: int = 0
    domain: str = "generic"
    all_discrete: bool = False
    cardinality: float | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if self.n_dims <= 0:
            self.n_dims = (
                self.n_continuous
                + self.n_integer
                + self.n_log
                + self.n_categorical
                + self.n_ordinal
                + self.n_bool
            )
        if self.n_dims <= 0:
            raise ValueError("context needs n_dims > 0 (or per-kind counts)")
        if self.noise not in NOISE_LEVELS:
            raise ValueError(f"noise must be one of {NOISE_LEVELS}")
        if self.smoothness not in SMOOTHNESS_LEVELS:
            raise ValueError(f"smoothness must be one of {SMOOTHNESS_LEVELS}")
        if self.compute not in COMPUTE_LEVELS:
            raise ValueError(f"compute must be one of {COMPUTE_LEVELS}")
        if self.domain not in DOMAINS:
            raise ValueError(f"domain must be one of {DOMAINS}")
        if self.budget_total < 1:
            raise ValueError("budget_total must be >= 1")
        if self.budget_remaining is None:
            self.budget_remaining = self.budget_total
        self.budget_remaining = int(min(self.budget_remaining, self.budget_total))
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.n_objectives < 1:
            raise ValueError("n_objectives must be >= 1")

    # -- constructors -----------------------------------------------------
    @classmethod
    def from_space(cls, space: SearchSpace, **kwargs) -> "OptimizationContext":
        counts = space.kind_counts()
        card = space.cardinality()
        payload = dict(
            n_dims=space.dim,
            n_continuous=counts["continuous"],
            n_integer=counts["integer"],
            n_log=counts["log"],
            n_categorical=counts["categorical"],
            n_ordinal=counts["ordinal"],
            n_bool=counts["bool"],
            feature_dim=space.feature_dim,
            n_descriptor_vars=sum(1 for variable in space.variables if variable.descriptors),
            all_discrete=space.all_discrete(),
            cardinality=None if math.isinf(card) else card,
        )
        payload.update(kwargs)
        return cls(**payload)

    @classmethod
    def from_dict(cls, payload: Mapping) -> "OptimizationContext":
        allowed = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        unknown = [k for k in payload if k not in allowed]
        if unknown:
            raise ValueError(f"unknown context fields: {unknown}")
        return cls(**{k: v for k, v in payload.items() if k in allowed})

    def to_dict(self) -> dict:
        out = {}
        for name in self.__dataclass_fields__:  # type: ignore[attr-defined]
            out[name] = getattr(self, name)
        return out

    @property
    def discrete_frac(self) -> float:
        discrete = self.n_categorical + self.n_ordinal + self.n_bool
        return discrete / self.n_dims if self.n_dims else 0.0

    @property
    def effective_dim(self) -> int:
        """Dimension the surrogate model actually sees (>= n_dims)."""
        return int(self.feature_dim or self.n_dims)

    @property
    def remaining(self) -> int:
        return int(self.budget_remaining if self.budget_remaining is not None else self.budget_total)

    @property
    def n_prior(self) -> int:
        return int(self.n_observations)


@dataclass
class Recommendation:
    """Router output: a concrete algorithm + validated parameters."""

    algorithm: str
    params: dict
    confidence: str
    rationale: list[str]
    warnings: list[str] = field(default_factory=list)
    alternatives: list[dict] = field(default_factory=list)
    candidate_scores: dict[str, float] = field(default_factory=dict)
    #: set when the budget is too small to fit a surrogate (see ``route``)
    underpowered_budget: dict | None = None
    design: dict = field(default_factory=dict)
    plan: list[dict] = field(default_factory=list)
    context: dict = field(default_factory=dict)

    def validate(self) -> None:
        """Check the recommendation against the registry (raises on mismatch)."""
        registry.validate_params(self.algorithm, self.params)

    def to_dict(self) -> dict:
        return {
            "algorithm": self.algorithm,
            "params": dict(self.params),
            "confidence": self.confidence,
            "rationale": list(self.rationale),
            "warnings": list(self.warnings),
            "alternatives": list(self.alternatives),
            "candidate_scores": dict(self.candidate_scores),
            "underpowered_budget": self.underpowered_budget,
            "design": dict(self.design),
            "plan": list(self.plan),
            "context": dict(self.context),
        }

    def to_text(self) -> str:
        lines = [f"推荐算法：{self.algorithm}（置信度 {self.confidence}）"]
        if self.rationale:
            lines.append("理由：")
            lines.extend(f"  - {r}" for r in self.rationale)
        if self.params:
            lines.append("参数：")
            for key, value in self.params.items():
                lines.append(f"  - {key} = {value}")
        if self.design:
            lines.append(
                "初始设计：{algorithm} × {n} 点（后 {later} 次实验交给 {main}）".format(
                    algorithm=self.design.get("algorithm", "sobol"),
                    n=self.design.get("n", 0),
                    later=self.design.get("remaining", "?"),
                    main=self.design.get("main", self.algorithm),
                )
            )
        if self.underpowered_budget:
            info = self.underpowered_budget
            print(
                f"  预算 {info['budget']} < 建模下限 {info['min_model_budget']}："
                + (
                    f"改用空间填充算法（{', '.join(info['space_filling'])}）"
                    if info["space_filling"]
                    else "该上下文没有便宜的可用算法，建议提高预算"
                )
            )
        if self.alternatives:
            lines.append("备选：")
            for alt in self.alternatives:
                lines.append(f"  - {alt['algorithm']}（{alt['why']}）")
        if self.warnings:
            lines.append("注意：")
            lines.extend(f"  - {w}" for w in self.warnings)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# scoring rules
# ---------------------------------------------------------------------------
def _score_parego(ctx: OptimizationContext) -> tuple[float, list[str], list[str]]:
    score, reasons, warnings = 35.0, [], []
    if ctx.n_objectives <= 1:
        return score - 200.0, reasons, ["单目标问题不需要多目标算法"]
    score += 70
    reasons.append(f"{ctx.n_objectives} 个目标，ParEGO 是当前内置的多目标算法")
    if ctx.n_objectives == 2:
        score += 5
    elif ctx.n_objectives >= 4:
        score -= 15
        warnings.append(
            "目标数 >3 时 ParEGO 的前沿覆盖下降；3–4 目标可考虑 qehvi_nd（预算 >=40）"
        )
    if ctx.budget_total >= 20:
        score += 8
        reasons.append(f"预算 {ctx.budget_total} 次，够跑一组折中解")
    elif ctx.budget_total < 12:
        score -= 10
        warnings.append("预算偏小，多目标前沿可能只覆盖 1–2 个折中点")
    if ctx.noise in ("medium", "high"):
        score += 5
        reasons.append("目标带噪声，GP 代理可复用信息")
    if ctx.effective_dim > 20:
        score -= 25
        warnings.append(f"有效特征维 {ctx.effective_dim} > 20，ParEGO 的 GP 不可靠")
    if ctx.compute == "low":
        score -= 5
    if ctx.n_constraints > 0:
        score -= 5
        warnings.append(
            "ParEGO 的约束按 P(feasible) 近似；2 目标 + 约束用 cqehvi，3–4 目标可用 qehvi_nd"
        )
    return score, reasons, warnings


def _score_cqehvi(ctx: OptimizationContext) -> tuple[float, list[str], list[str]]:
    score, reasons, warnings = 35.0, [], []
    if ctx.n_objectives != 2:
        return score - 200.0, reasons, ["cqEHVI 当前只支持 2 目标"]
    if ctx.n_constraints <= 0:
        return score - 200.0, reasons, ["没有硬约束时用 qehvi/qnehvi/ParEGO，不需要 cqehvi"]
    score += 75
    reasons.append("恰好 2 目标 + 硬约束：可行域内精确 EHVI，超体积改进只奖励可行点")
    score += 70
    reasons.append(
        f"{ctx.n_constraints} 个约束进入可行性概率（PoF），无可行点时先找可行条件"
    )
    if ctx.budget_total >= 40:
        score += 15
        reasons.append(f"预算 {ctx.budget_total} 次足够覆盖可行前沿")
    elif ctx.budget_total < 25:
        score -= 20
        warnings.append("预算偏小，两阶段（找可行 → 覆盖前沿）可能只覆盖 1–2 个折中点")
    if ctx.noise in ("none", "low"):
        score += 5
        reasons.append("低噪声，EHVI 与可行性概率的 MC 估计可靠")
    elif ctx.noise == "high":
        score -= 15
        warnings.append(
            "高噪声：目标/约束 GP 的 MC 估计方差大，建议先聚合复孔（cqehvi 不建模噪声约束）"
        )
    if ctx.n_constraints > 5:
        score -= 10
        warnings.append("约束 >5 个，联合可行性估计方差大；建议合并/降维约束")
    if ctx.effective_dim > 20:
        score -= 25
        warnings.append(f"有效特征维 {ctx.effective_dim} > 20，GP 不可靠")
    return score, reasons, warnings


def _score_constrained(ctx: OptimizationContext) -> tuple[float, list[str], list[str]]:
    score, reasons, warnings = 35.0, [], []
    if ctx.n_constraints <= 0:
        return score - 200.0, reasons, ["无约束问题不需要约束算法"]
    score += 70
    reasons.append(f"{ctx.n_constraints} 个约束，constrained 先找可行点再做带可行性的 EI")
    if ctx.n_objectives > 1:
        score -= 30
        warnings.append(
            "多目标 + 约束：2 目标请用 cqehvi（可行域内 EHVI），"
            ">2 目标暂退回主目标 + 约束优化"
        )
    if ctx.budget_total >= 20:
        score += 8
        reasons.append(f"预算 {ctx.budget_total} 次，够两阶段（找可行 → 优化）")
    elif ctx.budget_total < 12:
        score -= 10
        warnings.append("预算偏小，可能来不及找到可行点")
    if ctx.noise in ("medium", "high"):
        score += 5
        reasons.append("带噪约束用 MC 积分，比 EI×PoF 更稳")
    if ctx.effective_dim > 20:
        score -= 25
        warnings.append(f"有效特征维 {ctx.effective_dim} > 20，GP 不可靠")
    if ctx.compute == "low":
        score -= 5
    return score, reasons, warnings


def _score_qehvi(ctx: OptimizationContext) -> tuple[float, list[str], list[str]]:
    score, reasons, warnings = 35.0, [], []
    if ctx.n_objectives != 2:
        return score - 200.0, reasons, ["qEHVI 当前只支持 2 目标"]
    score += 75
    reasons.append("恰好 2 目标，qEHVI 的分区公式可精确算超体积改进")
    if ctx.budget_total >= 40:
        score += 15
        reasons.append(f"预算 {ctx.budget_total} 次足够覆盖前沿")
    elif ctx.budget_total < 25:
        score -= 20
        warnings.append("预算偏小，ParEGO 的标量化更稳")
    if ctx.noise in ("none", "low"):
        score += 5
        reasons.append("低噪声，EHVI 的 MC 估计可靠")
    elif ctx.noise == "high":
        score -= 25
        warnings.append("高噪声应用 qnehvi（2 目标嵌套 MC），当前 qEHVI 的 EHVI 会被噪声带偏")
    else:
        score -= 10
    if ctx.effective_dim > 20:
        score -= 25
        warnings.append(f"有效特征维 {ctx.effective_dim} > 20，GP 不可靠")
    return score, reasons, warnings


def _score_cmaes(ctx: OptimizationContext) -> tuple[float, list[str], list[str]]:
    score, reasons, warnings = 25.0, [], []
    if ctx.n_objectives != 1:
        return score - 200.0, reasons, ["CMA-ES 只优化单目标"]
    if ctx.n_continuous != ctx.n_dims:
        return score - 200.0, reasons, [
            "CMA-ES 需要全连续变量；含分类/整数时请用 TPE/采样"
        ]
    if ctx.smoothness != "smooth":
        return score - 200.0, reasons, ["非光滑目标上 CMA-ES 不占优"]
    if ctx.n_dims < 8:
        return score - 200.0, reasons, ["低维连续优先用 GP / 空间填充"]
    # Measured (budget 60, batch 6, 3 seeds) on 10-D smooth problems:
    # cmaes -18.1 vs bayes -155.4 (rosenbrock10) and -2.20 vs -5.01 (ackley10),
    # so the 8-14 dimension band must beat the GP default, not merely qualify.
    if ctx.n_dims >= 15:
        score += 60.0
    else:
        score += 45.0
    if ctx.budget_total >= 40:
        score += 10.0
        reasons.append(f"预算 {ctx.budget_total} 足够跑若干代种群")
    reasons.append(
        f"连续光滑、维度 {ctx.n_dims}：GP 代理不可靠，CMA-ES 直接自适应协方差"
    )
    warnings.append("CMA-ES 不建代理；噪声高时建议复孔或用 qNEHVI")
    return score, reasons, warnings


def _score_qehvi_nd(ctx: OptimizationContext) -> tuple[float, list[str], list[str]]:
    score, reasons, warnings = 35.0, [], []
    if ctx.n_objectives not in (3, 4):
        return score - 200.0, reasons, ["qehvi_nd 只用于 3–4 目标；2 目标用 qehvi/cqehvi"]
    score += 70.0
    # Benchmarks (moo3_constrained, budget 40, 2 seeds) after PI prefilter +
    # dominance skip: qehvi_nd feasible HV 5797 vs parego 5578, feasible points
    # 24.5 vs 21, ~30 s per run.  With that evidence qehvi_nd is the automatic
    # choice for 3-4 objectives at a sufficient budget.
    reasons.append(f"{ctx.n_objectives} 目标：MC-EHVI 直接最大化超体积改进")
    if ctx.budget_total >= 40:
        score += 20.0
        reasons.append(f"预算 {ctx.budget_total} 可支撑 MC 采样开销")
    else:
        score -= 25.0
        warnings.append("预算 <40 时 MC 开销不划算，优先 parego")
    if ctx.noise == "low":
        score += 10.0
        reasons.append("低噪声下逐步 GP 后验可靠")
    elif ctx.noise == "high":
        score -= 20.0
        warnings.append("噪声偏高时考虑 parego / 复孔")
    if ctx.effective_dim > 15:
        score -= 30.0
        warnings.append(f"有效特征维 {ctx.effective_dim} > 15，逐目标 GP 不可靠")
    if ctx.n_constraints > 0:
        score += 5.0
        reasons.append("含硬约束：候选按 P(feasible) 过滤（可行性覆盖更好）")
    return score, reasons, warnings


def _score_cqnehvi(ctx: OptimizationContext) -> tuple[float, list[str], list[str]]:
    score, reasons, warnings = 35.0, [], []
    if ctx.n_objectives != 2:
        return score - 200.0, reasons, ["cqnehvi 只支持 2 目标"]
    if ctx.n_constraints <= 0:
        return score - 200.0, reasons, ["没有硬约束时用 qnehvi/qehvi/parego"]
    score += 75.0
    score += 70.0
    # Benchmarks (moo_constrained, sigma=3 and sigma=6, budget 40): cqehvi
    # feasible HV 638/635 vs cqnehvi 631/626 -- the nested noisy MC did not pay
    # off here, so cqehvi stays the automatic choice and cqnehvi is opt-in.
    score -= 45.0
    warnings.append(
        "带噪嵌套 MC 在 moo_constrained 上未超过 cqehvi（638/635 vs 631/626）；"
        "默认 2 目标 + 约束仍用 cqehvi，cqnehvi 作为手动选项"
    )
    if ctx.budget_total >= 40:
        score += 10.0
    return score, reasons, warnings


def _score_mqehvi_nd(ctx: OptimizationContext) -> tuple[float, list[str], list[str]]:
    score, reasons, warnings = 35.0, [], []
    if not ctx.multi_fidelity:
        return score - 200.0, reasons, ["mqehvi_nd 需要多保真（低保真粗筛）"]
    if ctx.n_objectives < 2 or ctx.n_objectives > 4:
        return score - 200.0, reasons, ["mqehvi_nd 支持 2–4 个目标"]
    score += 45.0
    # Measured (budget 40, 3 seeds): mf_moo HV mqehvi_nd 626.9 (min_full 0.25) /
    # 647.9 (0.6) vs momfbo 672.3; mf_moo_constrained 607.1 vs 635.6. The
    # hypervolume variant is kept as an experimental third candidate only.
    score -= 35.0
    reasons.append("多保真 + 2–4 目标：HV 型采集（E[HV 改进]/成本^α），实验性备选")
    if ctx.n_objectives >= 3:
        score += 5.0
        reasons.append("3+ 目标下 HV 型采集通常优于随机标量化")
    if ctx.budget_total >= 40:
        score += 5.0
    if ctx.effective_dim > 25:
        score -= 25.0
        warnings.append(f"有效特征维 {ctx.effective_dim} > 25，MC-EHVI 代价高")
    return score, reasons, warnings


def _score_mfparego(ctx: OptimizationContext) -> tuple[float, list[str], list[str]]:
    score, reasons, warnings = 35.0, [], []
    if not ctx.multi_fidelity:
        return score - 200.0, reasons, ["mfparego 需要多保真（低保真粗筛）"]
    if ctx.n_objectives < 2:
        return score - 200.0, reasons, ["单目标多保真用 mfbo 更直接"]
    score += 45.0
    # Evidence across four independent settings all favour mfparego, so it is
    # the default for multi-fidelity multi-objective:
    #   mf_moo   (30 evals): 672.3 vs momfbo 666.2; (40 evals): 676.1 vs 670.3
    #   mf_moo_constrained (40): 642.5 vs 635.6 (feasible HV)
    #   mf_moo3  (40): 27399 vs 26869, full-fidelity points 35.3 vs 26.0
    #   calibration labels (12 evals, 12 problems): mfparego wins mf_moo / mf_moo3
    score += 10.0  # must stay above momfbo (35+45+5 with constraints)
    reasons.append(
        "多保真 + 多目标：ParEGO 标量化 + fidelity GP；四处基准（含 2/3 目标、约束、"
        "预算 12–40）均略优于 momfbo 且全保真点更多"
    )
    if ctx.n_objectives >= 3:
        score += 20.0
        reasons.append("3+ 目标差距更大（mf_moo3 27399 vs 26869）")
    return score, reasons, warnings


def _score_momfbo(ctx: OptimizationContext) -> tuple[float, list[str], list[str]]:
    score, reasons, warnings = 35.0, [], []
    if not ctx.multi_fidelity or ctx.n_objectives <= 1:
        return score - 200.0, reasons, ["多目标 + 多保真同时成立时才需要 momfbo"]
    score += 45.0
    reasons.append("多保真 + 多目标：随机标量化后按保真度维做 EI/成本优化")
    if ctx.n_constraints > 0:
        score += 5.0
        reasons.append("含硬约束：候选按 P(feasible) 过滤")
    return score, reasons, warnings


def _score_mfbo(ctx: OptimizationContext) -> tuple[float, list[str], list[str]]:
    score, reasons, warnings = 35.0, [], []
    if not ctx.multi_fidelity:
        return score - 200.0, reasons, ["没有低成本粗筛通道时不需要多保真"]
    score += 90
    reasons.append("有低成本粗筛（短时试跑/低精度仿真），可先低保真探索再全保真确认")
    if ctx.budget_total >= 20:
        score += 5
    if ctx.effective_dim > 20:
        score -= 25
        warnings.append(f"有效特征维 {ctx.effective_dim} > 20，GP 不可靠")
    return score, reasons, warnings


def _score_qnehvi(ctx: OptimizationContext) -> tuple[float, list[str], list[str]]:
    score, reasons, warnings = 35.0, [], []
    if ctx.n_objectives != 2:
        return score - 200.0, reasons, ["qNEHVI 当前只支持 2 目标"]
    score += 75
    reasons.append("恰好 2 目标且有噪声，qNEHVI 对前沿不确定性做嵌套 MC")
    if ctx.budget_total >= 40:
        score += 15
    elif ctx.budget_total < 25:
        score -= 20
    if ctx.noise == "high":
        score += 10
    elif ctx.noise == "medium":
        score += 5
    elif ctx.noise == "low":
        score -= 20
    else:
        score -= 25
    if ctx.effective_dim > 20:
        score -= 25
        warnings.append(f"有效特征维 {ctx.effective_dim} > 20，GP 不可靠")
    return score, reasons, warnings


def _rules() -> dict:
    return {
        "bayes": _score_bayes,
        "parego": _score_parego,
        "qehvi": _score_qehvi,
        "qnehvi": _score_qnehvi,
        "constrained": _score_constrained,
        "cqehvi": _score_cqehvi,
        "mfbo": _score_mfbo,
        "momfbo": _score_momfbo,
        "mfparego": _score_mfparego,
        "mqehvi_nd": _score_mqehvi_nd,
        "cqnehvi": _score_cqnehvi,
        "qehvi_nd": _score_qehvi_nd,
        "cmaes": _score_cmaes,
        "tpe": _score_tpe,
        "sobol": _score_sobol,
        "lhs": _score_lhs,
        "random": _score_random,
        "grid": _score_grid,
    }


def _score_bayes(ctx: OptimizationContext) -> tuple[float, list[str], list[str]]:
    score, reasons, warnings = 45.0, [], []
    if ctx.n_objectives > 1:
        score -= 200
        warnings.append("多目标请走 parego/qehvi/qnehvi/cqehvi；bayes 只优化主目标")
    if ctx.n_constraints > 0:
        score -= 8
        warnings.append("约束目前用可行性过滤/惩罚近似，严格约束优化在路线图中")
    if (
        ctx.smoothness == "smooth"
        and ctx.n_continuous == ctx.n_dims
        and ctx.n_dims >= 8
    ):
        # Same evidence as above: on smooth high-dimensional landscapes the GP
        # surrogate is the weak link, so cmaes should take over.
        score -= 15
        warnings.append(
            f"维度 {ctx.n_dims} 且光滑连续：GP 代理不可靠，CMA-ES 实测更好"
        )
    eff = ctx.effective_dim
    dim_label = f"特征维 {eff}" if eff != ctx.n_dims else f"维度 {eff}"
    small_categorical = (
        ctx.all_discrete
        and ctx.n_dims <= 8
        and ctx.feature_dim is not None
        and ctx.feature_dim <= 60
        and ctx.budget_total >= 50
    )
    if eff <= 12:
        score += 18
        reasons.append(f"{dim_label} ≤ 12，GP 代理可靠")
    elif eff <= 15:
        score += 8
        reasons.append(f"{dim_label} 处于 GP 舒适区边缘")
    elif eff <= 20:
        score -= 10
        warnings.append(f"{dim_label} 偏高，GP 需要更多初始点")
    else:
        if small_categorical:
            # 5-8 categorical factors with a moderate one-hot width: GP can model
            # interactions (literature labels + paper-campaign demo), whereas
            # TPE's independent densities cannot.
            score -= 15
            reasons.append(
                f"{dim_label} 但只有 {ctx.n_dims} 个类别变量（one-hot 宽度 ≤60），"
                "GP 仍可建模变量交互"
            )
        else:
            score -= 60
            warnings.append(f"{dim_label} > 20（one-hot 会放大维度），GP 代理不可靠，优先采样/树模型")
    if small_categorical:
        score += 35
        reasons.append("小维纯类别 + 预算充足：文献与代理战役显示 GP-BO 不劣于独立密度模型")
    if ctx.smoothness == "smooth":
        score += 12
        reasons.append("目标光滑，代理模型插值可靠")
    elif ctx.smoothness == "rough":
        score -= 12
        warnings.append("目标粗糙/不连续，GP 的平滑假设可能不成立")
    if ctx.noise in ("high", "medium"):
        score += 5
        reasons.append(f"有{ctx.noise}噪声，代理模型可复用信息、抑制噪声")
    if ctx.budget_total >= 20:
        score += 12
        reasons.append(f"总预算 {ctx.budget_total} 次，足够 BO 冷启动")
    elif ctx.budget_total >= 12:
        score += 4
    elif ctx.budget_total >= 8:
        score -= 10
        warnings.append("预算 8–11 次偏紧，BO 的初始设计会占掉较大比例")
    else:
        score -= 40
        warnings.append("预算 < 8 次，样本效率优势来不及体现")
    if ctx.batch_size > 16:
        score -= 8
        warnings.append("批量 > 16，qEI 的贪心近似质量下降")
    if ctx.compute == "low":
        score -= 8
        warnings.append("算力受限时每次采集的候选池要缩小")
    if ctx.n_prior >= 6:
        score += 10
        reasons.append(f"已有 {ctx.n_prior} 条先验数据，GP 可直接暖启动")
    if ctx.n_prior and ctx.n_prior < 1.5 * eff:
        score -= 10
        warnings.append(
            f"观测数 {ctx.n_prior} < 1.5×特征维 {eff:.0f}，GP 超参容易过拟合；"
            "可先跑采样算法，或固定超参/改用树模型"
        )
    if ctx.domain == "chemistry":
        score += 5
        reasons.append("反应条件通常平滑，BO 在化学优化里样本效率高")
    if ctx.multi_fidelity:
        score -= 5
        warnings.append("多保真优化尚未内置，按单保真处理")
    if ctx.all_discrete and ctx.cardinality is not None and ctx.cardinality <= 200:
        score -= 15
        warnings.append("空间是有限小集合，枚举/网格更直接")
    return score, reasons, warnings


def _score_tpe(ctx: OptimizationContext) -> tuple[float, list[str], list[str]]:
    score, reasons, warnings = 28.0, [], []
    if ctx.n_categorical + ctx.n_ordinal > 0:
        score += 8
        reasons.append("含分类/有序变量，TPE 的独立密度模型不需要 one-hot 距离")
    if ctx.discrete_frac >= 0.5:
        score += 20
        reasons.append("离散/分类变量占主导，树/密度模型比 GP 更稳")
    eff = ctx.effective_dim
    if 15 < eff <= 60:
        score += 10
        reasons.append(f"特征维 {eff} 超 GP 舒适区，TPE 对维度不敏感")
    elif eff > 60:
        score += 15
        reasons.append(f"特征维 {eff} 很高，TPE 仍可用而 GP 几乎失效")
    if ctx.budget_total <= 25:
        score += 6
        reasons.append(f"预算 {ctx.budget_total} 次偏小，TPE 在小样本下更稳")
    if ctx.all_discrete and ctx.cardinality is not None and ctx.cardinality <= 200:
        score -= 5
        warnings.append("有限小离散空间，枚举/网格更直接")
    if ctx.smoothness == "smooth" and ctx.discrete_frac == 0 and ctx.budget_total >= 20:
        score -= 14
        warnings.append("连续光滑低维时 GP-BO 的样本效率更高")
    if ctx.noise == "high":
        score += 4
        reasons.append("高噪声下密度模型不易被单个离群点带偏")
    if (
        ctx.all_discrete
        and ctx.n_dims <= 8
        and ctx.feature_dim is not None
        and ctx.feature_dim <= 60
        and ctx.budget_total >= 50
    ):
        score -= 15
        warnings.append(
            "小维纯类别空间（≤8 变量、one-hot 宽度 ≤60）：TPE 的独立密度忽略变量交互，"
            "GP-BO 更合适（文献标签 + 论文战役证据）"
        )
    return score, reasons, warnings


def _score_sobol(ctx: OptimizationContext) -> tuple[float, list[str], list[str]]:
    score, reasons, warnings = 25.0, [], []
    if ctx.budget_total < 10:
        score += 30
        reasons.append(f"预算只有 {ctx.budget_total} 次，均匀覆盖比建模更稳")
    elif ctx.budget_total <= 20:
        score += 8
    elif ctx.budget_total > 60:
        score -= 10
    if ctx.n_dims > 20:
        score += 20
        reasons.append(f"维度 {ctx.n_dims} 高，低差异序列不需要建模")
    if ctx.batch_size >= 8:
        score += 12
        reasons.append(f"批量 {ctx.batch_size} 大，Sobol 天然适合一次并行给点")
    if ctx.all_discrete:
        score += 8
        reasons.append("全离散空间，序列点可直接映射")
    if ctx.n_prior == 0:
        score += 15
        reasons.append("尚无数据，适合做初始空间填充设计")
    if ctx.compute == "low":
        score += 8
        reasons.append("算力低时几乎零开销")
    if ctx.smoothness == "smooth" and ctx.budget_total >= 20:
        score -= 8
        warnings.append("目标光滑且预算充足时，BO 更省样本")
    return score, reasons, warnings


def _score_lhs(ctx: OptimizationContext) -> tuple[float, list[str], list[str]]:
    score, reasons, warnings = 22.0, [], []
    if ctx.batch_size >= 8:
        score += 16
        reasons.append(f"批量 {ctx.batch_size} 大，LHS 保证每维分层均匀")
    elif ctx.batch_size == 1:
        score -= 8
        warnings.append("单点串行时 LHS 优势有限")
    if ctx.budget_total <= 8:
        score += 22
        reasons.append(f"极小预算 {ctx.budget_total} 次，一次性均匀布点")
    if ctx.all_discrete:
        score += 6
        reasons.append("离散空间可分层抽样")
    if ctx.n_prior == 0:
        score += 8
        reasons.append("尚无数据，适合做初始设计")
    if ctx.n_dims > 15:
        score -= 6
        warnings.append("维度 > 15 时 LHS 优化准则的收益下降")
    return score, reasons, warnings


def _score_random(ctx: OptimizationContext) -> tuple[float, list[str], list[str]]:
    score, reasons, warnings = 15.0, [], []
    if ctx.n_dims > 25:
        score += 22
        reasons.append(f"维度 {ctx.n_dims} 很高，随机搜索是强基线")
    elif ctx.n_dims > 15:
        score += 6
    if ctx.budget_total < 6:
        score += 25
        reasons.append(f"预算只有 {ctx.budget_total} 次，任何模型都来不及学")
    if ctx.compute == "low":
        score += 10
        reasons.append("算力低，零建模开销")
    if ctx.smoothness == "rough":
        score += 8
        reasons.append("目标粗糙，随机采样不易被错误假设误导")
    if ctx.noise == "high":
        score += 5
        reasons.append("高噪声下随机基线的期望表现往往不差")
    if ctx.all_discrete:
        score += 4
    if ctx.n_prior > 0:
        score -= 5
    return score, reasons, warnings


def _score_grid(ctx: OptimizationContext) -> tuple[float, list[str], list[str]]:
    score, reasons, warnings = 5.0, [], []
    card = ctx.cardinality
    if card is not None:
        if ctx.all_discrete and card <= 200:
            score += 55
            reasons.append(f"全离散且仅 {int(card)} 种组合，穷举比建模更可靠")
        elif card <= 200:
            score += 20
            reasons.append(f"组合数 {int(card)} 可控，网格设计能给出全局图景")
        elif card <= 2000:
            score += 8
        else:
            score -= 10
        if card <= ctx.remaining:
            score += 25
            reasons.append(f"预算 {ctx.remaining} 次足以覆盖全部 {int(card)} 种组合")
    if ctx.n_dims > 4:
        score -= 20
        warnings.append("维度 > 4 时网格点数指数爆炸")
    if not ctx.all_discrete:
        score -= 5
    if card is not None and card > ctx.remaining * 2:
        score -= 15
        warnings.append("预算不足以覆盖网格，会大量重复点")
    return score, reasons, warnings


# ---------------------------------------------------------------------------
# parameter policies
# ---------------------------------------------------------------------------
def _bayes_params(ctx: OptimizationContext) -> dict:
    remaining = ctx.remaining
    eff = ctx.effective_dim
    auto_init = max(3, min(eff + 1, 20))
    n_initial = min(auto_init, max(3, int(0.4 * remaining)))
    if ctx.n_prior >= 3:
        n_initial = min(n_initial, max(3, auto_init))
    kernel = "matern32" if ctx.smoothness == "rough" else "matern52"
    if ctx.smoothness == "smooth" and ctx.n_dims <= 8:
        kernel = "rbf"
    if ctx.noise == "high":
        acquisition, alpha, kappa = "ucb", 5e-3, 2.5
    elif ctx.noise == "none":
        acquisition, alpha, kappa = "ei", 1e-8, 2.0
    else:
        acquisition, alpha, kappa = "ei", 1e-4, 2.0
    if ctx.batch_size >= 4:
        batch_strategy = "local_penalty"
    elif ctx.batch_size > 1:
        batch_strategy = "kriging_believer"
    else:
        batch_strategy = "greedy"
    if ctx.compute == "low":
        n_candidates = 512
    elif ctx.compute == "high" and ctx.n_dims <= 8:
        n_candidates = 2048
    else:
        n_candidates = 1024
    local_frac = 0.4 if ctx.n_dims <= 6 else 0.25
    return {
        "kernel": kernel,
        "length_scale": 0.3,
        "ard": True,
        "alpha": alpha,
        "n_initial": n_initial,
        "n_candidates": n_candidates,
        "acquisition": acquisition,
        "kappa": kappa,
        "xi": 0.01,
        "batch_strategy": batch_strategy,
        "local_frac": local_frac,
        "local_sigma": 0.15,
    }


def _clamp_params(ctx: OptimizationContext, params: dict) -> dict:
    """Apply the initial-design cap to the parameters the optimizer actually gets."""
    if "n_initial" in params and ctx.remaining is not None:
        params = dict(params)
        params["n_initial"] = _clamp_initial_design(
            int(ctx.remaining), int(params["n_initial"] or 0) or 3
        )
    return params


def _params_for(algorithm: str, ctx: OptimizationContext) -> dict:
    if algorithm == "bayes":
        return _bayes_params(ctx)
    if algorithm == "parego":
        eff = ctx.effective_dim
        m = max(2, ctx.n_objectives)
        n_initial = min(max(2 * (eff + 1), 2 * m), 20, max(3, int(0.5 * ctx.remaining)))
        return {
            "rho": 0.05,
            "weight_alpha": 1.0,
            "acquisition": "logei",
            "n_initial": n_initial,
            "n_candidates": 2048 if ctx.compute == "high" else 1024,
            "alpha": 1e-4,
            "length_scale": 0.3,
        }
    if algorithm == "constrained":
        params = _bayes_params(ctx)
        params.update(
            {
                "mc_draws": 256 if ctx.compute == "high" else 128,
                "constraint_margin": 0.0,
                "feasibility_threshold": 0.02,
            }
        )
        return params
    if algorithm == "qehvi":
        params = _bayes_params(ctx)
        eff = ctx.effective_dim
        params.update(
            {
                "mc_draws": 256 if ctx.compute == "high" else 128,
                "n_initial": min(max(2 * (eff + 1), 4), 20, max(3, int(0.5 * ctx.remaining))),
            }
        )
        return params
    if algorithm == "qnehvi":
        params = _bayes_params(ctx)
        eff = ctx.effective_dim
        params.update(
            {
                "mc_draws": 128,
                "n_base": 64 if ctx.compute == "high" else 32,
                "n_y": 128 if ctx.compute == "high" else 64,
                "noise_std": 0.0,
                "n_candidates": 512,
                "n_initial": min(max(2 * (eff + 1), 4), 20, max(3, int(0.5 * ctx.remaining))),
            }
        )
        return params
    if algorithm == "cqehvi":
        params = _bayes_params(ctx)
        eff = ctx.effective_dim
        params.update(
            {
                "mc_draws": 256 if ctx.compute == "high" else 128,
                "constraint_margin": 0.0,
                "feasibility_threshold": 0.02,
                "n_initial": min(max(2 * (eff + 1), 4), 20, max(3, int(0.5 * ctx.remaining))),
            }
        )
        return params
    if algorithm in ("mfbo", "momfbo"):
        eff = ctx.effective_dim
        return {
            "kernel": "matern52",
            "length_scale": 0.3,
            "ard": True,
            "alpha": 1e-4,
            "n_initial": min(max(4, 2 * (eff + 1)), 20, max(4, int(0.4 * ctx.remaining))),
            "n_candidates": 512 if ctx.compute == "low" else 1024,
            "acquisition": "ei",
            "kappa": 2.0,
            "cost_exponent": 1.0,
            "fidelity_levels": "0.25,0.5,1.0",
            "local_frac": 0.3,
            "local_sigma": 0.15,
            # Sweep on mf_moo_constrained (budget 40, 5 seeds) feasible HV:
            # 0.25 -> 636.4, 0.4 -> 639.7, 0.8 -> 640.7; 0.4 keeps the most
            # (26.6) feasible full-fidelity points, so it is the balanced default.
            "min_full_fraction": 0.4 if algorithm == "momfbo" else 0.25,
        }
    if algorithm == "qehvi_nd":
        eff = ctx.effective_dim
        return {
            "kernel": "matern52",
            "length_scale": 0.3,
            "ard": True,
            "alpha": 1e-4,
            "n_initial": min(max(2 * (eff + 1), 4), 20, max(3, int(0.4 * ctx.remaining))),
            "n_candidates": 192 if ctx.compute == "low" else 256,
            "mc_draws": 24 if ctx.compute == "low" else 32,
            "mc_prefilter": 48,
            "local_frac": 0.3,
            "local_sigma": 0.15,
        }
    if algorithm == "mfparego":
        return _params_for("mfbo", ctx)
    if algorithm == "mqehvi_nd":
        params = _params_for("qehvi_nd", ctx)
        params.update({
            "cost_exponent": 1.0,
            "fidelity_levels": "0.25,0.5,1.0",
            "min_full_fraction": 0.25,
        })
        return params
    if algorithm == "cqnehvi":
        return _params_for("cqehvi", ctx)
    if algorithm == "cmaes":
        dim = max(2, ctx.n_dims)
        return {
            "sigma0": 0.1,
            "population_size": max(4, int(4 + 3 * math.log(dim))),
            "c_mu": 0.5,
        }
    if algorithm == "tpe":
        return {
            "gamma": 0.25,
            "n_candidates": 512 if ctx.compute == "high" else 256,
            "n_initial": 0,
            "bandwidth_factor": 1.0,
            "exploration_frac": 0.1 if ctx.discrete_frac < 0.5 else 0.15,
            "prior_weight": 1.0,
        }
    if algorithm == "sobol":
        return {"scramble": True, "dedupe": True}
    if algorithm == "lhs":
        return {"optimization": "random-cd" if ctx.remaining >= 16 else "none"}
    if algorithm == "random":
        return {"dedupe": True, "max_attempts_factor": 20}
    if algorithm == "grid":
        return {"max_points": max(2, min(ctx.remaining, 500)), "levels": 5}
    raise KeyError(f"no parameter policy for algorithm {algorithm!r}")


def _clamp_initial_design(remaining: int, n_initial: int) -> int:
    """Keep a real share of the budget for the *main* algorithm.

    Every model-based algorithm starts from a space-filling design.  If the design
    consumes the whole budget the reported algorithm never runs at all -- measured:
    budget 20 with ``n_initial=20`` produced the plan
    ``sobol n=20 + tpe n=0``, i.e. "router says tpe" while 20 Sobol points execute.
    So the design is capped at ``remaining - main_floor``, where ``main_floor`` is at
    least one evaluation and at least a third of the budget when that is affordable.
    """
    if remaining <= 1:
        return max(1, remaining)
    main_floor = max(1, min(max(3, remaining // 3), remaining - 1))
    capped = min(int(n_initial), remaining - main_floor)
    return max(1, capped)


def _build_plan(ctx: OptimizationContext, algorithm: str, params: dict) -> tuple[dict, list[dict]]:
    """Initial-design split and the phase plan shown to the user."""
    remaining = ctx.remaining
    plan: list[dict] = []
    design: dict = {}
    if algorithm == "cmaes":
        population = int(params.get("population_size", 0)) or max(
            4, int(4 + 3 * math.log(max(2, ctx.n_dims)))
        )
        design = {
            "algorithm": "cmaes",
            "n": min(population, remaining),
            "main": "cmaes",
            "remaining": max(0, remaining - population),
        }
        plan = [{"phase": "evolution", "algorithm": "cmaes", "n": min(population, remaining)}]
    elif algorithm in (
        "bayes",
        "tpe",
        "parego",
        "constrained",
        "qehvi",
        "qnehvi",
        "cqehvi",
        "mfbo",
        "momfbo",
        "qehvi_nd",
        "cqnehvi",
    ):
        n_init = int(params.get("n_initial", 0))
        if n_init <= 0:
            n_init = max(3, min(ctx.effective_dim + 1, 20))
        n_init = _clamp_initial_design(remaining, n_init)
        design = {
            "algorithm": "sobol",
            "n": n_init,
            "main": algorithm,
            "remaining": max(0, remaining - n_init),
        }
        plan.append({"phase": "initial_design", "algorithm": "sobol", "n": n_init})
        plan.append(
            {
                "phase": "main_loop",
                "algorithm": algorithm,
                "n": max(0, remaining - n_init),
                "note": "每轮 ask→实测→tell，连续 2–3 轮无改进可触发重路由",
            }
        )
    elif algorithm == "grid":
        card = int(ctx.cardinality) if ctx.cardinality and math.isfinite(ctx.cardinality) else remaining
        design = {"algorithm": "grid", "n": min(card, remaining), "main": "grid", "remaining": 0}
        plan.append({"phase": "exhaustive_screen", "algorithm": "grid", "n": design["n"]})
    else:
        design = {"algorithm": algorithm, "n": remaining, "main": algorithm, "remaining": 0}
        plan.append({"phase": "sampling", "algorithm": algorithm, "n": remaining})
    return design, plan


def _required_capabilities(ctx: OptimizationContext) -> set[str]:
    """Capabilities an algorithm must declare to be a valid candidate."""
    required: set[str] = set()
    if ctx.n_objectives > 1:
        required.add("multi_objective")
    if ctx.n_constraints > 0:
        required.add("constraints")
    if ctx.multi_fidelity:
        required.add("multi_fidelity")
    if ctx.batch_size > 1:
        required.add("batch")
    if ctx.n_categorical + ctx.n_ordinal > 0:
        required.add("categorical")
    return required


def _capability_filter(ctx: OptimizationContext) -> set[str] | None:
    """Return the algorithms that declare every required capability.

    Returns ``None`` when no registered algorithm satisfies the requirements;
    the caller then falls back to scoring everything (with warnings) instead
    of returning an empty recommendation.
    """
    required = _required_capabilities(ctx)
    if not required:
        return None
    allowed = {
        name
        for name in _rules()
        if required <= set(registry.spec(name).capabilities)
    }
    return allowed or None


def route(ctx: OptimizationContext) -> Recommendation:
    """Score all registered algorithms and return the best recommendation."""
    scores: dict[str, float] = {}
    reasons_by_algo: dict[str, list[str]] = {}
    warnings_by_algo: dict[str, list[str]] = {}
    allowed = _capability_filter(ctx)
    fidelity_constraint_warning = None
    fidelity_multiobjective_warning = None
    if allowed is None and ctx.multi_fidelity and ctx.n_objectives > 1:
        reduced = _required_capabilities(ctx) - {"multi_fidelity"}
        reduced_allowed = {
            name
            for name in _rules()
            if reduced <= set(registry.spec(name).capabilities)
        }
        if reduced_allowed:
            allowed = reduced_allowed
            fidelity_multiobjective_warning = (
                "多保真 + 多目标暂无联合算法：优先保证 Pareto 多目标目标，"
                "放弃保真度通道（成本模型不生效）"
            )
    if allowed is None and ctx.multi_fidelity and ctx.n_constraints > 0:
        # No algorithm declares both capabilities. Hard constraints are safety
        # critical, so keep them and drop the fidelity channel with a warning.
        reduced = _required_capabilities(ctx) - {"multi_fidelity"}
        reduced_allowed = {
            name
            for name in _rules()
            if reduced <= set(registry.spec(name).capabilities)
        }
        if reduced_allowed:
            allowed = reduced_allowed
            fidelity_constraint_warning = (
                "多保真 + 硬约束暂无联合算法：优先保证硬约束，放弃保真度通道（成本模型不生效）"
            )
    for name, rule in _rules().items():
        try:
            registry.spec(name)
        except KeyError:  # pragma: no cover - a rule for an unbuilt algorithm
            continue
        if allowed is not None and name not in allowed:
            continue
        score, reasons, warnings = rule(ctx)
        scores[name] = round(float(score), 2)
        reasons_by_algo[name] = reasons
        warnings_by_algo[name] = warnings

    # A surrogate needs enough observations to fit: with 4 evaluations a GP-ParEGO
    # run has fewer points than free parameters.  Algorithms whose model cannot be
    # fitted at this budget are penalised *per algorithm kind* (a TPE that bins
    # categorical features needs far less data than a GP over one-hot features).
    underpowered = None
    budget = int(getattr(ctx, "budget_total", 0) or 0)
    dim = max(1, int(ctx.effective_dim))
    cheap_names = ("sobol", "lhs", "random", "grid")
    if budget:
        gp_like = {
            "bayes", "constrained", "qehvi", "qnehvi", "cqehvi", "cqnehvi",
            "qehvi_nd", "mfbo", "mfparego", "momfbo", "mqehvi_nd", "parego",
        }

        def min_model_budget(name: str) -> int:
            # Floor for "can this model be fitted at all", *not* the statistically
            # comfortable sample size: the GP implementations freeze their
            # hyperparameters when n < 2*(d+1), so a moderate dimension is still
            # usable.  Capped so that ordinary lab budgets (>=12) never fall back
            # to space filling just because the feature expansion is wide.
            if name in gp_like:
                return max(4, min(12, dim))
            if name == "tpe":
                return max(4, min(8, dim // 2))
            return 0

        blocked = {
            name: min_model_budget(name)
            for name in scores
            if budget <= min_model_budget(name)
        }
        if blocked:
            remaining = [name for name in scores if name not in blocked]
            relaxed: list[str] = []
            if not remaining and budget <= 2:
                # Only for degenerate budgets (<= 2 evaluations) does space filling
                # beat the model class: measured on moo_constrained / mf_constrained
                # with budgets 4-8, a surrogate still produced deliverable evidence
                # (3-4/4 runs with a feasible front) while plain sobol often ended
                # with no feasible point at all (1/4).  See research/TINY_BUDGET.md.
                for name in cheap_names:
                    if name in scores or name not in _rules():
                        continue
                    scores[name] = 0.0
                    reasons_by_algo[name] = [
                        f"预算 {budget} < 建模下限 {max(blocked.values())}："
                        "没有满足能力要求的便宜算法，改用空间填充取样"
                    ]
                    warnings_by_algo[name] = [
                        "该算法不建模约束/多目标结构：取点不针对性，结果仍按约束过滤与 Pareto 排序"
                    ]
                    relaxed.append(name)
                remaining = [name for name in scores if name not in blocked]
            underpowered = {
                "budget": budget,
                "min_model_budget": max(blocked.values()),
                "blocked": blocked,
                "space_filling": sorted(
                    name for name in remaining if name in set(cheap_names)
                ),
                "capability_relaxed": bool(relaxed),
                "relaxed": relaxed,
            }
            if remaining == [] and not relaxed:  # pragma: no cover - guarded above
                pass
            if remaining:
                for name, required in blocked.items():
                    scores[name] = -1e9
                    reasons_by_algo.setdefault(name, []).append(
                        f"预算 {budget} < 该算法建模下限 {required}：数据不足以拟合代理"
                    )
            else:
                # Constrained / multi-objective contexts with no cheap option: keep
                # the surrogate (measured to be the better tiny-budget choice) but
                # say that the budget is below its comfort zone.
                best_name = max(scores, key=lambda k: scores[k])
                warnings_by_algo.setdefault(best_name, []).append(
                    f"预算 {budget} 低于建模舒适区 {underpowered['min_model_budget']}："
                    "代理可能欠拟合；实测该情形下模型类仍比纯空间填充更能给出可交付证据，"
                    "若想试空间填充可用 algorithm='sobol' 显式指定"
                )

    ranked = sorted(scores, key=lambda k: scores[k], reverse=True)
    for name in scores:
        factor = _ALGORITHM_WEIGHTS.get(name)
        if factor is None or factor == 1.0:
            continue
        value = scores[name]
        if factor <= 0:
            scores[name] = -1e9  # explicitly disabled
        elif value > 0:
            scores[name] = round(value * factor, 2)
        else:
            # Negative (penalty) scores must not get boosted by multiplying.
            scores[name] = round(value / factor, 2)
    ranked = sorted(scores, key=lambda k: scores[k], reverse=True)
    best = ranked[0]
    margin = scores[best] - scores[ranked[1]] if len(ranked) > 1 else 99.0
    confidence = "high" if margin >= 20 else ("medium" if margin >= 8 else "low")
    if ctx.n_objectives > 3:
        confidence = "low"

    objective_count_warning = None
    if best in ("qehvi", "qnehvi", "cqehvi") and ctx.n_objectives != 2:
        # These optimizers implement exactly two objectives; falling back keeps
        # the study runnable instead of raising inside build().
        if ctx.n_constraints > 0:
            substitute = "constrained"
            objective_count_warning = (
                f"{best} 只支持 2 目标；当前 {ctx.n_objectives} 目标 + 约束暂无联合算法，"
                "已退回主目标 + 硬约束（constraints 生效，其余目标仅记录在观测里）"
            )
        else:
            substitute = "parego"
            objective_count_warning = (
                f"{best} 只支持 2 目标；当前 {ctx.n_objectives} 目标已改用 parego"
                "（Pareto 折中，不是单一最优）"
            )
        original = best
        best = substitute
        scores.setdefault(best, 0.0)
        ranked = [best] + [name for name in ranked if name != best]
        margin = scores[best] - scores[ranked[1]] if len(ranked) > 1 else 99.0
        confidence = "high" if margin >= 20 else ("medium" if margin >= 8 else "low")
        if ctx.n_objectives > 3:
            confidence = "low"
        reasons_by_algo.setdefault(best, [])
        warnings_by_algo.setdefault(best, [])
        reasons_by_algo[best] = list(reasons_by_algo[best]) or [
            f"{original} 不支持 {ctx.n_objectives} 目标，改用 {best}"
        ]

    params = _clamp_params(ctx, _params_for(best, ctx))
    design, plan = _build_plan(ctx, best, params)
    if best in (
        "bayes",
        "tpe",
        "parego",
        "constrained",
        "qehvi",
        "qnehvi",
        "cqehvi",
        "mfbo",
        "momfbo",
        "qehvi_nd",
        "cqnehvi",
    ):
        params["n_initial"] = design["n"]

    alternatives = []
    for name in ranked[1:3]:
        why = (reasons_by_algo[name] or warnings_by_algo[name] or ["分数接近"])[0]
        alternatives.append({"algorithm": name, "score": scores[name], "why": why})

    rationale = list(reasons_by_algo[best])
    if not rationale:
        rationale = ["其余候选得分更低，保留最稳健选择"]
    warnings = list(warnings_by_algo[best])
    if fidelity_multiobjective_warning:
        warnings.insert(
            0,
            f"没有算法同时声明能力 {sorted(_required_capabilities(ctx))}，"
            "已退回优先满足多目标的子集打分",
        )
        warnings.insert(1, fidelity_multiobjective_warning)
    if objective_count_warning:
        warnings.insert(0, objective_count_warning)
    if fidelity_constraint_warning:
        warnings.insert(
            0,
            f"没有算法同时声明能力 {sorted(_required_capabilities(ctx))}，"
            "已退回优先满足硬约束的子集打分",
        )
        warnings.insert(1, fidelity_constraint_warning)
    required = _required_capabilities(ctx)
    if required and allowed is None:
        warnings.insert(
            0,
            f"没有算法同时声明能力 {sorted(required)}，已退回全量打分；结果可能不满足约束/多目标要求",
        )
    if ctx.n_objectives > 1 and best == "parego" and not any("Pareto" in warning for warning in warnings):
        warnings.insert(
            0,
            "ParEGO 返回一组折中解（Pareto 前沿），不是单一最优；报告请用 Pareto 视图",
        )
    if best == "tpe" and ctx.n_descriptor_vars > 0:
        warnings.insert(
            0,
            "空间含描述符特征：GP 类算法（bayes/constrained/parego/mfbo/momfbo）更契合；"
            "Tier C 实测 TPE 在描述符下明显变差，可考虑显式指定 GP 类算法",
        )
    if best == "cqehvi":
        warnings.insert(
            0,
            "cqEHVI 只支持 2 目标；每个回填观测都必须带 constraints 值，"
            "报告用可行域内的 Pareto 前沿（不可行点不进入前沿与 HV）",
        )
    if ctx.n_objectives > 2 and not any("qEHVI" in warning for warning in warnings):
        warnings.append(">2 目标时前沿覆盖有限，qEHVI/cqEHVI 仅支持 2 目标，3D EHVI 暂缓")
    if best != "bayes" and ctx.n_objectives == 1 and ctx.smoothness == "smooth" and ctx.budget_total >= 20:
        warnings.append("若目标确实光滑，下一轮可尝试切到 bayes")

    rec = Recommendation(
        algorithm=best,
        params=params,
        confidence=confidence,
        rationale=rationale,
        warnings=warnings,
        alternatives=alternatives,
        candidate_scores=scores,
        design=design,
        plan=plan,
        context=ctx.to_dict(),
        underpowered_budget=underpowered,
    )
    rec.validate()
    return rec


def route_round(
    ctx: OptimizationContext,
    recent_scores: Sequence[float] | None = None,
    patience: int = 3,
    improve_tol: float = 1e-6,
) -> Recommendation:
    """Re-route for the next round, inserting exploration when stalled.

    ``recent_scores`` are the best-so-far scores per round (higher is always
    better).  If the best has not improved for ``patience`` consecutive
    rounds, the router recommends a model-free exploration batch and records
    why.  The caller decides whether to accept the switch.
    """
    rec = route(ctx)
    if not recent_scores or len(recent_scores) < patience + 1:
        return rec
    window = list(recent_scores)[-(patience + 1) :]
    if window[-1] <= window[0] + improve_tol:
        exploration = "sobol" if ctx.n_dims <= 20 else "random"
        rec = Recommendation(
            algorithm=exploration,
            params=_params_for(exploration, ctx),
            confidence="medium",
            rationale=[
                f"最近 {patience} 轮最优值没有改善（{window[0]:.4g} → {window[-1]:.4g}）",
                f"插入一批 {exploration} 全局探索，跳出当前代理模型的错误吸引域",
            ],
            warnings=["探索批次结束后建议回到 bayes 继续利用"],
            alternatives=[
                {"algorithm": rec.algorithm, "score": rec.candidate_scores.get(rec.algorithm), "why": "原推荐，先让位给探索"}
            ],
            candidate_scores=rec.candidate_scores,
            design={"algorithm": exploration, "n": min(ctx.remaining, max(ctx.batch_size, 4)), "main": rec.algorithm, "remaining": 0},
            plan=[{"phase": "exploration_restart", "algorithm": exploration, "n": min(ctx.remaining, max(ctx.batch_size, 4))}],
            context=ctx.to_dict(),
        )
    return rec


def context_from_space(space: SearchSpace, **kwargs) -> OptimizationContext:
    return OptimizationContext.from_space(space, **kwargs)
