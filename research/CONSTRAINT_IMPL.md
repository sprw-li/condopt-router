# CONSTRAINT_IMPL — 约束优化落地方案（可直接照抄）

> 维护者：algo-research ｜ v1（2026-11-05，Round 4）
> 目标：给 condopt v0.3 加"约束"而**不破坏单目标路径**。对照现状：`Objective.constraints` 只是名字元组（`base.py:152-153`），`Study` 原样存入 `constraints=payload.get(...)`（`study.py:476`），没有阈值/方向/噪声，也没有任何算法声明 `"constraints"` 能力 → router 的 `_capability_filter` 会返回 `None` 并**静默**回退到全量打分（`router.py:596-611`）。
> 文献：R34（EI-CDF：`EI·PoF` 的原始文献）、R10（Letham：带噪约束）、R2（GP/alpha 工程）、R24（约束/噪声的方法学定位）、R23（谨慎用固定 κ UCB）。
> 与 `CONSTRAINTS_NOISE_IMPL.md`（Round 3）的关系：本文是**数据模型 + 与现有代码的集成**，Round 3 文是公式/伪代码细节。

---

## 1. 数据模型

### 1.1 `Constraint` 定义（新文件 `src/condopt/constraints.py`）

```python
from dataclasses import dataclass, field
from typing import Mapping

@dataclass(frozen=True)
class Constraint:
    """一条不等式约束：value 满足方向方向即可行。

    内部统一为 "g(x) <= 0 可行"：
        direction == "<="  ->  g = value - threshold
        direction == ">="  ->  g = threshold - value
    """
    name: str
    threshold: float = 0.0
    direction: str = "<="          # "<=" | ">="
    noisy: bool = False            # 观测是否带测量噪声（决定 PoF vs 联合 MC）
    margin: float = 0.0            # 保守裕度（g + margin <= 0 才可行）
    unit: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ValueError("constraint name must be a non-empty string")
        if self.direction not in ("<=", ">="):
            raise ValueError(f"constraint {self.name!r}: direction must be '<=' or '>='")
        object.__setattr__(self, "threshold", float(self.threshold))
        object.__setattr__(self, "margin", float(self.margin))

    def to_g(self, value: float) -> float:
        """原始观测 → 内部可行性裕度（<=0 可行）。"""
        v = float(value)
        return (v - self.threshold) if self.direction == "<=" else (self.threshold - v)

    def from_g(self, g: float) -> float:
        return float(g) + self.threshold if self.direction == "<=" else self.threshold - float(g)

    def feasible(self, value: float) -> bool:
        return self.to_g(value) + self.margin <= 0.0

    def to_dict(self) -> dict:
        return {"name": self.name, "threshold": self.threshold,
                "direction": self.direction, "noisy": self.noisy,
                "margin": self.margin, "unit": self.unit,
                "description": self.description}

    @classmethod
    def from_dict(cls, payload) -> "Constraint":
        if isinstance(payload, str):                 # 兼容旧格式：只有名字
            return cls(name=payload)
        return cls(**{k: payload.get(k) for k in
                      ("name", "threshold", "direction", "noisy", "margin", "unit", "description")
                      if payload.get(k) is not None})
```

### 1.2 挂到 `Objective`（**向后兼容**）

`Objective.constraints` 现在是 `tuple[str, ...]`。改成"名字仍是主字段，规格单独存"：

```python
@dataclass
class Objective:
    name: str = "score"
    direction: str = "max"
    objectives: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()                    # 保留：名字列表（旧 JSON 可读）
    directions: tuple[str, ...] = ()
    constraint_specs: tuple[Constraint, ...] = ()        # 新增：完整规格

    def __post_init__(self):
        ...  # 原有校验
        # 1) 字符串/字典统一成 Constraint
        specs = tuple(Constraint.from_dict(c) for c in self.constraint_specs)
        # 2) 若只给了 constraints 名字（旧格式），补默认规格（threshold=0, "<="）
        if not specs and self.constraints:
            specs = tuple(Constraint(name=n) for n in self.constraints)
        if specs:
            self.constraint_specs = specs
            self.constraints = tuple(c.name for c in specs)   # 名字从 specs 派生，保持一致
        if len({c.name for c in self.constraint_specs}) != len(self.constraint_specs):
            raise ValueError("duplicate constraint names")

    def constraint(self, name: str) -> Constraint: ...
    @property
    def n_constraints(self) -> int: return len(self.constraint_specs)

    def to_dict(self):
        return {..., "constraints": list(self.constraints),
                "constraint_specs": [c.to_dict() for c in self.constraint_specs]}
```

**兼容性规则**

1. 旧 `objective.json`（只有 `constraints: ["c1"]`）→ 自动变成 `Constraint("c1", threshold=0, "<=")`；单目标路径完全不动。
2. 新格式 `constraint_specs` 优先；两者都写时以 `constraint_specs` 为准并忽略 `constraints` 名字（或校验一致后报错）。
3. `Objective.to_dict()` 同时输出两个字段 → CLI/skill 旧解析不坏。

### 1.3 `results.json` / `Observation` 的约束观测

**现状**：`Study._to_observation` 直接 `constraints=payload.get("constraints", {})`（`study.py:476`），要求是 `{name: float}`。

**扩展**（接受两种写法）：

```jsonc
// 旧（仍支持）
{"id": "r1-1", "y": 93.2, "constraints": {"purity": 95.4}}
// 新（带噪声/复孔）
{"id": "r1-1", "objectives": {"yield": 93.2},
 "constraints": {"purity": {"value": 95.4, "std": 0.6, "n": 3}}}
```

`Study._to_observation` 里加一个 `_parse_constraints(payload)`：数字 → `value, std=0, n=1`；对象 → 取 value/std/n。`Observation` 增加：

```python
constraint_std: dict[str, float] = field(default_factory=dict)
n_replicates: int = 1
```
（`to_dict`/`from_dict` 同步；JSONL 旧行缺字段时默认 0 / 1。）

**可行性口径**（写进文档，避免歧义）：

- 默认按**观测值**判可行：`constraint.feasible(obs.constraints[name])`；
- 安全关键场景用**保守判据**：`to_g(value) + margin + κ·std ≤ 0`（κ 由 `Constraint.margin` 或全局 `constraint_margin` 参数给）；
- 报告里同时给 `g` 的均值/标准差与 `P(feasible)`。

---

## 2. 采集函数：PoF 加权 vs Letham 联合 MC

### 2.1 推荐默认（决策表）

| 条件 | 默认 | 理由 |
|---|---|---|
| 约束 `noisy=False`，且经验可行比例 ≥10% | **EI-CDF**（`ei_cdf`） | 简单、稳定；R34 的原始场景 |
| 任一约束 `noisy=True`（有复孔 std），或可行比例 <10% | **联合 MC**（`noisy_ei_c`） | `EI·PoF` 在噪声/小可行域下有偏且下溢（R10） |
| 约束是"离散允许组合/黑箱规则"（不可建模） | 候选池过滤 + 罚函数 | 没有可用的 GP 模型 |
| 多目标 + 约束 | ParEGO 标量化后按上面规则；或 v0.4 qNEHVI-constrained | R12 |

**默认参数**：`constraint_margin=0.0`（正常）/`1.0`（安全边界，单位=约束单位）；`n_base=64`；`n_y=128`；`feasibility_floor=0.02`（低于它的候选从池中剔除）。

### 2.2 EI-CDF（R34）

```
g_j(x) ~ GP(μ_j, σ_j)             # 对「可行性裕度 g」建模，不是原始约束值
EI_c(x) = EI(x) · Π_j Φ( -μ_j(x) / σ_j(x) )        # σ_j→0 时用 1{g_j<=0}
```

实现要点（挂进 `GaussianProcessBO._acquisition`，新增 `acquisition="ei_cdf"`）：

```python
def _constraint_models(self):
    """每个约束一个 GP；训练目标是 g = Constraint.to_g(value)。"""
    models = {}
    X = self._X                       # 与父类相同的 feature 矩阵
    units = self._units
    for spec in self.objective.constraint_specs:
        g, mask = [], []
        for obs, u in zip(self._observations, units):
            if spec.name in obs.constraints:
                g.append(spec.to_g(obs.constraints[spec.name]))
                mask.append(True)
            else:
                mask.append(False)
        g = np.asarray(g)
        models[spec.name] = self._fit(X[np.asarray(mask)], g)
    return models

def _acac_ei_cdf(self, model_y, features):
    acq = self._acquisition_single(model_y, features)     # 原 EI/logEI（best 基于标量 y）
    for cm in self._constraint_models().values():
        mu_g, sd_g = cm.predict(features, return_std=True)
        sd_g = np.maximum(sd_g, 1e-12)
        acq = acq * norm.cdf((-mu_g - float(self.params["constraint_margin"])) / sd_g)
    return acq
```

- 注意：**对 `g` 建模**（不是对原始值），这样"≤/≥"统一；`margin` 直接加到 `g` 上。
- `alpha`：有复孔时 `alpha = mean(std_c²/n) / std_g²`；无复孔用默认 1e-4 并允许用户覆盖。
- 过小的 `Φ` 乘积会下溢 → 在候选池层面先算 `P(feasible)`，`<feasibility_floor` 的点直接 `acq = -inf`。

### 2.3 联合 MC（Letham 风格，R10）

```python
def _acq_noisy_ei_c(self, model_y, features, base_features):
    """EI_c = E[max(0, f(x) - f*) · 1{g(x) <= 0}]，f* 也随机。"""
    n_base = int(self.params["n_base"]); n_y = int(self.params["n_y"])
    Fb = self._sample_multi(model_y, base_features, n_base)      # (n_base, n_b)
    f_star = Fb.max(axis=1)
    Y = self._sample_multi(model_y, features, n_y)               # (n_y, n_cand)
    G = np.stack([self._sample_multi(cm, features, n_y)
                  for cm in self._constraint_models().values()], axis=0)  # (J, n_y, n_cand)
    total = 0.0
    for k in range(n_base):
        imp = np.maximum(0.0, Y - f_star[k])
        feas = np.all(G + self.params["constraint_margin"] <= 0.0, axis=0)
        total += np.mean(np.where(feas, imp, 0.0))
    return total / n_base
```

- `_sample_multi(model, X, n)`：`mu, cov = model.predict(X, return_cov=True)`；一次 Cholesky 抽 `n` 条（与 `SPRINT2_SPEC.md` §3 的 `_sample_posterior_multi` 相同）。
- `base_features` = 已观测点的 features（可全量；`n>200` 时抽样）。
- 批量：`q>1` 时对 q 取 max（qEI 语义）或 q=1 + 局部惩罚；推荐后者（省算力）。

### 2.4 集成方式（不破坏单目标路径）

**推荐：新增独立算法类 `ConstrainedBO(GaussianProcessBO)`，注册名 `bayes_c`**，而不是把约束塞进 `bayes`：

```python
@register
class ConstrainedBO(GaussianProcessBO):
    spec = AlgorithmSpec(
        name="bayes_c", family="model_based",
        summary="带约束的 GP-BO：EI-CDF 或带噪约束联合 MC（Letham 风格）。",
        when="有昂贵黑盒约束（纯度/安全/收率下限），约束与目标一起测量时。",
        params=GaussianProcessBO.spec.params + (
            ParamSpec("constraint_acquisition", "choice", "auto",
                      choices=("auto", "ei_cdf", "noisy_ei_c")),
            ParamSpec("constraint_margin", "float", 0.0, low=0.0, high=5.0),
            ParamSpec("n_base", "int", 64, low=8, high=512),
            ParamSpec("n_y", "int", 128, low=16, high=1024),
            ParamSpec("feasibility_floor", "float", 0.02, low=0.0, high=0.5),
        ),
        capabilities=frozenset({"batch", "mixed", "categorical", "continuous",
                                 "noisy", "expensive", "model_based", "constraints"}),
        references=("Gardner et al., Bayesian Optimization with Inequality Constraints, ICML 2014",
                    "Letham et al., Constrained Bayesian Optimization with Noisy Experiments, 2019"),
    )
```

- `auto` 选择：`any(c.noisy for c in objective.constraint_specs)` 或历史可行比例 `< 0.1` → `noisy_ei_c`，否则 `ei_cdf`。
- **能力位**：只有 `bayes_c` 声明 `"constraints"` → router 的能力过滤（`router.py:580-611`）在 `n_constraints>0` 时只留下它（目前会因无人声明而 fallback，见 REVIEW_v03 P1）。
- 单目标无约束 → 用户仍选 `bayes`；`parego` 需要支持约束时同样加 `constraints` 能力位并复用 `_acq_constrained`（v0.4）。

---

## 3. 与 `Study` / CLI / results 的兼容

| 位置 | 改动 | 兼容策略 |
|---|---|---|
| `Study._to_observation` | 解析 `constraints` 的数值/对象两种格式，写 `constraint_std`/`n_replicates` | 数字仍可用 |
| `Study.ask` | 构建 optimizer 时 `objective` 已含 `constraint_specs`；无改动 | — |
| `Study.report` | 增加"可行性"块：每个约束的 μ±σ、`P(feasible)`、可行点数 | 无约束时不显示 |
| `Study.pareto_summary` | 若 `objective.constraint_specs` 非空：默认只保留可行观测（`feasible=True`），另加 `n_infeasible` | 单目标/无约束不变 |
| `cli tell` | 结果文件支持对象格式；CLI 提示新 schema | 旧文件仍通过 |
| `prepare.py` | `answers.constraints` 接受 `[{name,threshold,direction,noisy}]` 或 `["c1"]`；写入 `objective.json` 的 `constraint_specs`；`n_constraints` 从 specs 派生 | 旧的 `constraints: 2` 计数仍可用 |
| `benchmarks.py` | `BenchmarkProblem` 增加 `constraint_fn`（Tier E 已定义） | 无约束问题不变 |

**`prepare.py` 具体改法**（当前 `ALIASES["constraints"]="n_constraints"`，传列表会类型错）：

```python
constraint_payload = payload.pop("constraints", None)
if isinstance(constraint_payload, list):
    objective_payload = dict(objective_payload)
    objective_payload["constraint_specs"] = constraint_payload
elif constraint_payload is not None:
    context_payload["n_constraints"] = int(constraint_payload)     # 只给计数（无规格）
# 之后 objective = Objective.from_dict(objective_payload)；n_constraints 用 len(objective.constraint_specs)
```

---

## 4. 验收测试（PR 必带）

```python
def test_constraint_to_g_direction():        # "<=" 与 ">=" 的 g 符号、feasible 判定
def test_objective_backcompat_names_only():  # constraints=["c1"] → 一个默认 Constraint
def test_observation_parses_both_formats():  # 数字 vs {value,std,n}
def test_ei_cdf_matches_manual_on_grid():    # 1D/2D 网格上 EI*PoF 与手算一致
def test_ei_cdf_returns_neg_inf_below_floor(): # 不可行候选被剔除
def test_noisy_ei_c_degenerates_to_ei_without_noise(): # σ=0 时与 EI 接近（相对误差<5%）
def test_constrained_bo_never_worse_on_constraint_easy(): # best_so_far ≥ 无约束 BO 的可行最优
def test_single_objective_path_unchanged():  # 无约束 study 的 ask 序列与 v0.3 逐位一致（同 seed）
def test_router_requires_constraints_capability(): # n_constraints>0 时选中声明 constraints 的算法
```

**benchmark 验收**（`BENCHMARK_TIER_CD.md` §5）：`constraint_easy`（可行 25%、最优 95.0）上 best ≥ 94.5；`constraint_hard`（可行 6.16%、约束最优 93.096）上，`noisy_ei_c` 的首个可行点评估次数显著少于 `ei_cdf`（10 seeds，p<0.05），最终 best ≥ 90。

---

## 5. 参考

R34（EI-CDF 原始文献与适用条件）、R10（带噪约束：截断后验/期望改进；本文 MC 的可落地实现）、R2（alpha/复孔工程细节）、R24（约束/噪声的方法学位置）、R23（不要用固定 κ UCB 处理约束）。
