# CAPABILITIES_SPEC — 算法能力位、能力过滤与"先过滤后打分"

> 维护者：algo-research ｜ v1（2026-11-05，Round 2）
> 对应 `REVIEW_v01.md` P1-9 / P0-3 / P1-10；参照 Ax `GenerationStrategy`、SMAC3 `Facade`、BayBE `Recommender`、Nevergrad `parametrization` 的工程写法（R8/R9/R17）。
> 目标：**把"某个算法能不能干这件事"变成机器可读的声明**，router 先按硬能力过滤、再按偏好打分；LLM/skill 只能从过滤后的候选里选。

---

## 1. 现状与问题

```python
# src/condopt/base.py:103
capabilities: frozenset[str] = frozenset()
# bayes:  {"batch","mixed","categorical","continuous","noisy","expensive","model_based"}
# random: {"batch","mixed","categorical","high_dim","cheap","model_free"}
# grid:   {"mixed","categorical","exhaustive","cheap","model_free"}
```

问题：

1. **词汇不统一、无结构**：`"noisy"`、`"expensive"`、`"cheap"` 混在一起（有的描述问题、有的描述算法成本）；无法回答"支持约束吗""支持多目标吗""最大 batch 多少"。
2. **没有被使用**：`router._rules()` 是硬编码的 5 个函数，加算法要改三处（`_rules`/`_params_for`/`_build_plan`），`capabilities` 只用于 `spec --json` 展示。
3. **无法表达"不可行"**：多目标时 router 靠 `-200` 分"劝退" bayes，然后静默选 sobol（`REVIEW_v01.md` P1-10）；约束、多保真同理。
4. **与参数策略耦合**：`_params_for` 用 `if algorithm == "bayes"` 分支，能力信息无法驱动参数。

---

## 2. 能力模型：硬能力（hard）、软偏好（soft）、元数据（meta）

| 类别 | 用途 | router 行为 |
|---|---|---|
| **hard** | 算法**必须**具备才可被选中（如"支持多目标"） | 不满足 → **过滤掉**，并记录 `filtered_out` 原因 |
| **soft** | 算法**更擅长**（如"支持 batch"、"适合混合空间"） | 不满足 → 扣分/warning，仍可被选中 |
| **meta** | 数值约束（`max_dim_soft`、`min_observations`、`suggest_cost`） | 用于打分/告警，不硬过滤 |

设计原则（对齐 R8/R17）：

- **hard 只放"语义上做不到"的**（没有多目标实现就不能算 Pareto；不支持 batch 就不能一次给 n>1 个点）。
- **soft 放"质量差异"**（GP 也能吃 categorical，但高基数时不如 TPE——这是打分问题，不是可行性问题）。
- **meta 不参与可行性**，但参与"是否需要 warning / 建议切换"。

---

## 3. JSON Schema（写进 `spec --json` 的输出）

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "AlgorithmCapabilities",
  "type": "object",
  "additionalProperties": false,
  "required": ["space", "batch", "constraints", "objectives", "fidelity", "noise", "cost_aware"],
  "properties": {
    "space": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "continuous": {"type": "boolean", "default": true},
        "integer":    {"type": "boolean", "default": true},
        "log":        {"type": "boolean", "default": true},
        "categorical":{"type": "boolean", "default": false},
        "ordinal":    {"type": "boolean", "default": false},
        "boolean":    {"type": "boolean", "default": true},
        "conditional":{"type": "boolean", "default": false},
        "max_cardinality_soft": {"type": ["integer", "null"], "default": null,
          "description": "单变量最大类别数建议值，如 GP one-hot 建议 <= 8；超过只降分不硬过滤"}
      }
    },
    "batch": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "supported": {"type": "boolean", "default": true},
        "max_batch_soft": {"type": ["integer", "null"], "default": null},
        "async_ok": {"type": "boolean", "default": false}
      }
    },
    "constraints": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "supported": {"type": "boolean", "default": false},
        "noisy_constraints": {"type": "boolean", "default": false}
      }
    },
    "objectives": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "max_objectives": {"type": "integer", "default": 1},
        "multi_objective": {"type": "boolean", "default": false},
        "pareto_report": {"type": "boolean", "default": false}
      }
    },
    "fidelity": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "supported": {"type": "boolean", "default": false},
        "levels": {"type": "array", "items": {"type": "number"}, "default": [1.0]},
        "continuous_input": {"type": "boolean", "default": false}
      }
    },
    "noise": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "models_observation_noise": {"type": "boolean", "default": false},
        "replicate_aggregation": {"type": "boolean", "default": false}
      }
    },
    "cost_aware": {"type": "boolean", "default": false},
    "deterministic": {"type": "boolean", "default": false},
    "needs_prior": {"type": "boolean", "default": false},
    "suggest_cost": {"enum": ["cheap", "medium", "expensive"], "default": "medium"},
    "max_dim_soft": {"type": ["integer", "null"], "default": null},
    "min_observations": {"type": "integer", "default": 0}
  }
}
```

**字段语义约定**

- `space.*`：**能处理**该类型变量，不代表擅长。`max_cardinality_soft` 是"超过后效果下降"的建议值，用于软打分。
- `objectives.max_objectives`：支持的目标数上限（单目标算法 = 1；ParEGO = 1（标量化后）；qNEHVI = 任意但要注明建议 ≤4）。
- `fidelity.levels`：离散保真度档位；`continuous_input=true` 表示"保真度作为连续输入"（Kandasamy 2020 路线）。
- `noise.models_observation_noise`：是否真的对噪声建模（qNEI/带噪 EI），而不是"能跑但不管噪声"。**这一位决定高噪声场景的硬过滤**（见 §5）。
- `deterministic`：同一 `(space, seed, prior)` 下 ask 序列是否可逐位复现（用于审计/回归）。
- `suggest_cost`：每次 `ask` 的 CPU 成本档位（cheap = Sobol/random；medium = TPE/forest；expensive = GP + 超参优化 + 采集优化）。

---

## 4. Python 侧实现（最小改动）

```python
# src/condopt/capabilities.py
from dataclasses import dataclass, field, asdict
from typing import Any, Mapping

@dataclass(frozen=True)
class SpaceCaps:
    continuous: bool = True
    integer: bool = True
    log: bool = True
    categorical: bool = False
    ordinal: bool = False
    boolean: bool = True
    conditional: bool = False
    max_cardinality_soft: int | None = None

@dataclass(frozen=True)
class BatchCaps:
    supported: bool = True
    max_batch_soft: int | None = None
    async_ok: bool = False

@dataclass(frozen=True)
class ObjectCaps:
    max_objectives: int = 1
    multi_objective: bool = False
    pareto_report: bool = False

@dataclass(frozen=True)
class FidelityCaps:
    supported: bool = False
    levels: tuple[float, ...] = (1.0,)
    continuous_input: bool = False

@dataclass(frozen=True)
class NoiseCaps:
    models_observation_noise: bool = False
    replicate_aggregation: bool = False

@dataclass(frozen=True)
class Capabilities:
    space: SpaceCaps = field(default_factory=SpaceCaps)
    batch: BatchCaps = field(default_factory=BatchCaps)
    constraints_supported: bool = False
    noisy_constraints: bool = False
    objects: ObjectCaps = field(default_factory=ObjectCaps)
    fidelity: FidelityCaps = field(default_factory=FidelityCaps)
    noise: NoiseCaps = field(default_factory=NoiseCaps)
    cost_aware: bool = False
    deterministic: bool = False
    needs_prior: bool = False
    suggest_cost: str = "medium"
    max_dim_soft: int | None = None
    min_observations: int = 0

    def satisfies(self, req: "Capabilities") -> tuple[bool, list[str]]:
        """检查 hard 能力；返回 (ok, missing_reasons)。"""
        missing: list[str] = []
        r, s = req.space, self.space
        for name in ("continuous", "integer", "log", "categorical", "ordinal", "boolean", "conditional"):
            if getattr(r, name) and not getattr(s, name):
                missing.append(f"space.{name}")
        if req.batch.supported and not self.batch.supported:
            missing.append("batch.supported")
        if req.constraints_supported and not self.constraints_supported:
            missing.append("constraints.supported")
        if req.noisy_constraints and not self.noisy_constraints:
            missing.append("constraints.noisy")
        if req.objects.multi_objective and not self.objects.multi_objective:
            missing.append("objectives.multi_objective")
        if req.objects.max_objectives > self.objects.max_objectives:
            missing.append(f"objectives.max_objectives<{req.objects.max_objectives}")
        if req.fidelity.supported and not self.fidelity.supported:
            missing.append("fidelity.supported")
        if req.noise.models_observation_noise and not self.noise.models_observation_noise:
            missing.append("noise.models_observation_noise")
        if req.cost_aware and not self.cost_aware:
            missing.append("cost_aware")
        return (not missing), missing

    def to_dict(self) -> dict[str, Any]: return asdict(self)
    @classmethod
    def from_dict(cls, payload: Mapping) -> "Capabilities": ...
```

`AlgorithmSpec` 改动：

```python
@dataclass(frozen=True)
class AlgorithmSpec:
    ...
    caps: Capabilities = Capabilities()
    # 兼容旧字段：capabilities: frozenset[str] 保留，由 caps 派生（能力位字符串）
```

**兼容策略**：`frozenset` 字段保留一版（`spec --json` 里同时输出 `capabilities` 旧字段与 `caps` 新对象），下个 minor 版本删除旧字段；router 只读 `caps`。

---

## 5. `required_capabilities(ctx)` —— 从问题上下文推导硬需求

```python
def required_capabilities(ctx: OptimizationContext) -> tuple[Capabilities, list[str]]:
    """返回 (hard_requirements, notes)；notes 用于 route 输出。"""
    space = SpaceCaps(
        continuous=ctx.n_continuous > 0,
        integer=ctx.n_integer > 0,
        log=ctx.n_log > 0,
        categorical=ctx.n_categorical > 0,
        ordinal=ctx.n_ordinal > 0,
        boolean=ctx.n_bool > 0,
        conditional=ctx.has_conditional,          # 新增字段（v0.2）
    )
    batch = BatchCaps(supported=ctx.batch_size > 1)
    objects = ObjectCaps(multi_objective=ctx.n_objectives > 1,
                         max_objectives=ctx.n_objectives)
    fidelity = FidelityCaps(supported=ctx.multi_fidelity)
    noise = NoiseCaps(models_observation_noise=ctx.noise == "high")
    req = Capabilities(space=space, batch=batch, objects=objects, fidelity=fidelity,
                       constraints_supported=ctx.n_constraints > 0,
                       noise=noise, cost_aware=ctx.multi_fidelity)
    return req, []
```

**逐条依据**

| 需求 | 触发条件 | 依据 |
|---|---|---|
| `space.categorical` | 空间含 categorical/ordinal/bool | 无此位的算法（纯数值 BO 库）会在解码时失败/越界 |
| `batch.supported` | `batch_size > 1` | 无 batch 的算法只能逐点给，等于串行 |
| `constraints.supported` | `n_constraints > 0` | 不支持约束的算法无法把可行性计入采集函数（R10） |
| `objectives.multi_objective` | `n_objectives > 1` | 不能静默转单目标（`REVIEW_v01.md` P1-10；R11/R12/R21） |
| `fidelity.supported` | `multi_fidelity` | 无保真度接口的算法会把不同保真度当成同分布（R18 的 emulator 协议） |
| `noise.models_observation_noise` | `noise == "high"` | "能跑"≠"建模噪声"；高噪下 EI 会过拟合幸运观测（R10/R12/R29） |
| `cost_aware` | `multi_fidelity` | 多保真必须按成本选点，否则等于浪费 |

**重要**：`categorical`、`batch` 这类需求也可以**降级**（soft）——例如 batch=2 时允许不支持 batch 的算法逐点给两次。实现上把"可降级"写成：

```python
RELAX_ORDER = ["fidelity", "cost_aware", "noise.models_observation_noise",
               "batch", "constraints", "objectives.multi_objective", "space"]
```

`relax()` 按此顺序逐个把需求降为 false，直到 `feasible` 非空；每降一级写一条 `degraded: "..."` 到 `route().warnings`，并在 `Recommendation.degraded` 字段里返回（LLM/skill 必须显式告知用户）。**`objectives.multi_objective` 与 `space` 最后才降级**（前者是语义错误，后者会崩）。

---

## 6. 先过滤后打分（伪代码）

```python
def route(ctx: OptimizationContext) -> Recommendation:
    req, _ = required_capabilities(ctx)
    specs = registry.specs()                       # 所有已注册算法

    # 1) 硬过滤
    feasible, filtered_out = [], []
    for spec in specs:
        ok, missing = spec.caps.satisfies(req)
        (feasible if ok else filtered_out).append((spec, missing))
    degraded: list[str] = []
    if not feasible:                               # 2) 按顺序降级
        req, degraded = relax(req)
        feasible, filtered_out = retry_filter(specs, req)

    # 3) 软打分（只对 feasible）
    scored = []
    for spec, _ in feasible:
        score, reasons, warnings = score_preferences(spec, ctx)     # 规则/权重
        score += soft_capability_bonus(spec.caps, ctx)              # §7
        score -= meta_penalty(spec.caps, ctx)                       # max_dim_soft/min_obs/suggest_cost
        scored.append(Scored(spec.name, score, reasons, warnings))

    ranked = sorted(scored, key=lambda s: -s.score)
    best = ranked[0]
    params = params_for(best.name, ctx)                            # 每个算法自带 policy
    rec = Recommendation(algorithm=best.name, params=params, ...,
                         degraded=degraded,
                         filtered_out=[{"algorithm": s.name, "missing": m}
                                       for s, m in filtered_out])
    rec.validate()                                                # 参数 schema 校验
    return rec
```

**实现细节**

1. **`params_for` 改为算法自带**：`AlgorithmSpec.param_policy: Callable[[OptimizationContext], dict]`（可选）。这样"加算法 = 加一个类 + 一个 policy"，router 不再有 `if algorithm == "bayes"`。
2. **`score_preferences` 只做偏好**（可校准权重，见 `BENCHMARK_SPEC.md` §7），不再处理"能不能"。
3. `filtered_out` 必须进 `Recommendation.to_dict()`，让 skill/LLM 解释"为什么没选 X"。
4. 兜底：若降级后仍无 feasible → 返回 `Recommendation(algorithm="random", degraded=[...])` 并显式 `raise` 或 `warnings` 标记，**绝不静默**。

---

## 7. 软能力加分（建议映射）

| 软能力 | 场景 | 加分建议 |
|---|---|---|
| `space.categorical` | categorical 占比 > 30% 或最大基数 > 8 | +8（TPE/forest 类）/ 0（GP 类，另扣分） |
| `batch.supported` + `max_batch_soft ≥ batch` | batch 大 | +6 |
| `noise.models_observation_noise` | noise ∈ {medium, high} | +10 |
| `fidelity.continuous_input` | 多保真且档位连续 | +8 |
| `suggest_cost == "cheap"` | compute=low 或 budget 小 | +5 |
| `max_dim_soft` | `effective_dim > max_dim_soft` | -(超出比例 × 20) |
| `min_observations` | `n_obs < min_observations` | -25（或过滤） |

**注意**：这些加分与 `BENCHMARK_SPEC.md` §7 的可校准权重是同一套 term，实现时用 `ScoreTerm` 结构统一，不要两套硬编码。

---

## 8. 与 LLM / skill 的 JSON 契约

`condopt spec --json` 输出（节选）：

```json
{
  "name": "bayes", "family": "model_based",
  "caps": {
    "space": {"continuous": true, "integer": true, "log": true, "categorical": true,
              "ordinal": false, "boolean": true, "conditional": false, "max_cardinality_soft": 8},
    "batch": {"supported": true, "max_batch_soft": 16, "async_ok": false},
    "constraints_supported": false,
    "objects": {"max_objectives": 1, "multi_objective": false, "pareto_report": false},
    "fidelity": {"supported": false, "levels": [1.0], "continuous_input": false},
    "noise": {"models_observation_noise": false, "replicate_aggregation": false},
    "cost_aware": false, "deterministic": true, "suggest_cost": "expensive",
    "max_dim_soft": 12, "min_observations": 3
  }
}
```

`route --json` 输出新增：

```json
{
  "algorithm": "bayes", "params": {...},
  "degraded": [],
  "filtered_out": [
    {"algorithm": "grid", "missing": ["batch.supported"]},
    {"algorithm": "lhs",  "missing": ["objectives.multi_objective"]}
  ],
  "capability_requirements": {"space.categorical": true, "batch.supported": true, "objectives.multi_objective": false, "noise.models_observation_noise": false}
}
```

**LLM/skill 约束**（写进 `skill/SKILL.md`）：LLM 只能在 `feasible ∪ {降级后的候选}` 中选择；若想让 LLM 推荐一个被硬过滤的算法，必须让用户显式确认 `degraded` 后果。

---

## 9. 现有算法的能力声明（迁移表）

| 算法 | space | batch | constraints | multi_objective | fidelity | noise modeling | suggest_cost | max_dim_soft |
|---|---|---|---|---|---|---|---|---|
| `bayes`（GP-BO） | 全 true，categorical 仅 ≤ `max_cardinality_soft=8`（超出降分） | ✓，soft max 16 | ✗（v0.2 加 PoF） | ✗（ParEGO 后 = 1） | ✗（v0.2 连续输入） | 部分（α 固定，未 qNEI → `false`，高噪时被 requirement 过滤掉，除非降级） | expensive | 12 |
| `sobol` | 全 true | ✓ | ✗ | ✗ | ✗ | ✗ | cheap | null |
| `lhs` | 全 true | ✓ | ✗ | ✗ | ✗ | ✗ | cheap | null |
| `random` | 全 true | ✓ | ✗ | ✗ | ✗ | ✗ | cheap | null |
| `grid` | 全 true（离散为主） | ✗（一次全给不算 batch 优化，但可枚举） | ✗ | ✗ | ✗ | ✗ | cheap | 4 |
| `tpe`（下轮） | 全 true，categorical ✓（强项） | 部分（constant liar） | ✗ | 部分（MOTPE） | ✗ | 部分 | medium | null |
| `parego`（下轮） | 全 true | ✓ | ✗ | ✓（标量化，多目标） | ✗ | 部分 | expensive | 12 |

**注意**：按上表，当前"高噪声 + 多目标 + 约束"场景会**没有任何算法通过硬过滤** → router 必然走降级路径并给出显式 warning。这正是想要的行为（比静默降级好），但要做产品决策：v0.1 允许降级并提示，v0.2 补齐 `tpe`/`parego`/`qnehi` 后自然消失。

---

## 10. 测试与验收

1. **声明完整性**：遍历 `registry.specs()`，断言每个 spec 的 `caps` 非空且 `suggest_cost ∈ {cheap,medium,expensive}`；`objectives.multi_objective` 与 `max_objectives>1` 一致。
2. **过滤正确性**：对 `n_objectives=2` 的 context，`route().algorithm` 必须 `caps.objects.multi_objective` 为真，或 `rec.degraded` 包含 `objectives.multi_objective`（当前实现下是后者）。
3. **降级顺序**：构造"无算法支持多目标"的 registry，断言降级顺序与 `RELAX_ORDER` 一致，且 `degraded` 非空。
4. **参数策略解耦**：删掉一个算法的 `param_policy` 后 `route` 应抛清晰错误（而不是 `KeyError`）。
5. **可解释性**：`filtered_out` 里每条 missing 都能对应到 `required_capabilities` 的一个 rule（快照测试）。
6. **向后兼容**：旧 `capabilities` frozenset 仍输出；`spec --json` 新旧字段同时存在一个版本周期。

---

## 11. 参考

- R8（SMAC3：Facade 把"场景"和"能力组合"绑在一起，`MultiFidelityFacade`/`MultiObjectiveFacade` 是能力过滤的产品化形式）
- R17（BayBE：`Recommender` 的可用性由空间/约束决定；`SearchSpace` 负责可行性）
- R9（Optuna：Sampler 与 Pruner 组合，参数化能力由 `suggest_*` API 表达）
- R10（带噪约束 → `noisy_constraints` 位）
- R12/R21（多目标 → `multi_objective`/`pareto_report` 位）
- R14/R5（高维 → `max_dim_soft` 与后续 `tuRBO`/`cmaes` 的 `space` 能力位）
