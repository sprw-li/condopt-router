# BENCHMARK_TIER_CD — Tier C/D 的代码级问题定义（可直接落成 Python）

> 维护者：algo-research ｜ v1（2026-11-05，Round 3）
> 对应：`BENCHMARK_SPEC.md` §3（Tier C/D 的规格层），本文给**可直接粘贴进 `src/condopt/benchmarks.py`** 的 space / 函数式 / 最优值 / 核验代码。
> 所有最优值均已用脚本核验（核验方法见 §6，容差 1e-6 的数值在文内标注）。
> 依赖现状：`SearchSpace.from_dict`、`Observation(objectives=..., constraints=...)`、`BenchmarkProblem`（需按 §1 扩展）。

---

## 1. 需要先扩展的 `BenchmarkProblem` 字段

```python
@dataclass
class BenchmarkProblem:
    name: str
    space: SearchSpace
    objective: Objective
    fn: Callable[[dict], float]                 # 单目标
    # --- 新增（多目标）---
    fn_multi: Callable[[dict], dict] | None = None      # {"yield":..,"cost":..}
    pareto_front: list[tuple[float, ...]] | None = None # 参考前沿（原始目标值）
    ref_point: tuple[float, ...] | None = None          # HV 的参考点（原始单位，含方向语义）
    ideal_point: tuple[float, ...] | None = None
    # --- 新增（噪声/约束）---
    noise: str = "none"                          # "none"|"gaussian"|"multiplicative"
    noise_std: float = 0.0                       # gaussian 的绝对 σ（或 multiplicative 的 CV）
    replicates: int = 1                          # 每次评估重复次数（>1 时观测=均值，std 已知）
    fn_constraint: Callable[[dict], dict] | None = None  # {"g":...}；约定 g<=0 可行
    tags: tuple[str, ...] = ()
    optimal_config: dict | None = None
    optimum: float | None = None
```

评估器（协议层）负责加噪声与重复：

```python
def evaluate(problem, configs, rng):
    results = []
    for cfg in configs:
        vals = []
        for _ in range(problem.replicates):
            base = problem.fn_multi(cfg) if problem.fn_multi else {"y": problem.fn(cfg)}
            if problem.noise == "gaussian":
                noisy = {k: v + rng.normal(0.0, problem.noise_std) for k, v in base.items()}
            elif problem.noise == "multiplicative":
                noisy = {k: v * (1.0 + rng.normal(0.0, problem.noise_std)) for k, v in base.items()}
            else:
                noisy = dict(base)
            vals.append(noisy)
        mean = {k: float(np.mean([v[k] for v in vals])) for k in vals[0]}
        std = ({k: float(np.std([v[k] for v in vals], ddof=1)) for k in vals[0]}
               if len(vals) > 1 else {k: 0.0 for k in vals[0]})
        results.append(EvaluationResult(objectives=mean, objective_std=std,
                                        constraints=problem.fn_constraint(cfg) if problem.fn_constraint else {},
                                        cost=1.0, fidelity=1.0))
    return results
```

---

## 2. Tier C：`discrete_catalyst_v2`（全离散，穷举可核验）

### 2.1 space

```python
space = SearchSpace.from_dict([
    {"name": "metal",   "kind": "categorical", "choices": ["Ni", "Pd", "Pt"]},
    {"name": "ligand",  "kind": "categorical", "choices": ["A", "B", "C", "D"]},
    {"name": "additive","kind": "categorical", "choices": ["X1", "X2", "X3", "X4"]},
    {"name": "temp",    "kind": "integer",     "low": 40, "high": 120, "unit": "degC"},
])
# cardinality = 3*4*4*81 = 3888
```

### 2.2 函数式

```python
BASE = {"Ni": 60.0, "Pd": 72.0, "Pt": 68.0}
LIG  = {"A": 3.0, "B": 1.0, "C": 6.0, "D": 2.0}
ADD  = {"X1": 2.0, "X2": -1.0, "X3": 4.0, "X4": 0.0}

def discrete_catalyst_v2(c: dict) -> float:
    inter = 3.0 if (c["metal"] == "Pt" and c["additive"] == "X3") else 0.0
    return (BASE[c["metal"]] + LIG[c["ligand"]] + ADD[c["additive"]]
            + inter - 0.002 * (c["temp"] - 80) ** 2)
```

### 2.3 最优（穷举核验）

```
optimum = 82.0
optimal_config = {"metal": "Pd", "ligand": "C", "additive": "X3", "temp": 80}
```
注：Pt×X3 的交互 (+3) 使 Pt 组合最高 81，仍低于 Pd(72)+C(6)+X3(4)=82。

### 2.4 用途

- `grid`/穷举 vs `bayes`：`budget=64` 时 grid 应在前 64 点覆盖 `82.0`（grid 的 `max_points` 需 ≥ 3888 才能穷举，否则只能粗化——测试 router 的 `card ≤ budget` 规则）。
- 噪声变体：`noise="gaussian", noise_std=1.0, replicates=1`（Tier D 的离散版本）。

---

## 3. Tier C：`high_cardinality_solvent`（12 级分类 + 描述符，核验 92.371）

### 3.1 关键设计

12 种溶剂，每种带 **2 维描述符** `(polarity, protic)`；目标是"接近目标描述符 (0.8, 0.2) 且温度合适"。**必须提供两条特征路径**：
1. `one_hot`：12 维 one-hot（当前 `space.features` 默认路径）；
2. `descriptor`：2 维连续描述符（`space.features` 扩展后，见 `REVIEW_v01.md` P1-6 / R22 Gryffin）。

### 3.2 space

```python
SOLVENTS = {
    "S00": (0.0, 0.0), "S01": (0.2, 0.0), "S02": (0.4, 0.0), "S03": (0.6, 0.0),
    "S04": (0.8, 0.0), "S05": (1.0, 0.0), "S06": (0.0, 1.0), "S07": (0.2, 1.0),
    "S08": (0.4, 1.0), "S09": (0.6, 1.0), "S10": (0.8, 1.0), "S11": (1.0, 1.0),
}
space = SearchSpace.from_dict([
    {"name": "T", "kind": "continuous", "low": 60.0, "high": 120.0, "unit": "degC"},
    {"name": "solvent", "kind": "categorical", "choices": list(SOLVENTS)},
])
```

### 3.3 函数式

```python
TARGET = (0.8, 0.2)   # (polarity, protic)

def high_cardinality_solvent(c: dict) -> float:
    pol, prot = SOLVENTS[c["solvent"]]
    sim = math.exp(-((pol - TARGET[0]) / 0.35) ** 2 - ((prot - TARGET[1]) / 0.6) ** 2)
    return 50.0 + 25.0 * sim + 20.0 * math.exp(-((c["T"] - 90.0) / 25.0) ** 2)
```

### 3.4 最优（解析 + 穷举核验）

```
optimum = 50 + 25*exp(-1/9) + 20 = 92.3714...
optimal_config = {"T": 90.0, "solvent": "S04"}      # S04=(0.8, 0.0)，sim=exp(-1/9)=0.8948
```
- 温度项：`T=90` 时取 1；
- 最接近目标描述符的溶剂是 `S04`（距离² = 0 + 0.1111 = 0.1111）；`S10=(0.8,1)` 距离² = 0+(0.8/0.6)²=1.777，更低。

### 3.5 描述符映射（评测器侧）

```python
def solvent_descriptors(c: dict) -> list[float]:
    pol, prot = SOLVENTS[c["solvent"]]
    return [pol, prot]
```

评测时把 `space.features(config)` 替换为 `[T_norm, pol, prot]`（3 维）——这就是 Gryffin 的核心思想（R22）。**预期**：GP 在 3 维描述符空间显著优于 12 维 one-hot（后者需 ~24+ 初始点，前者 8 个点即可）。

---

## 4. Tier D：`noisy_sphere`（噪声 + 复孔，解析核验）

### 4.1 space

```python
space = SearchSpace.from_dict([
    {"name": f"x{i+1}", "kind": "continuous", "low": 0.0, "high": 1.0} for i in range(3)
])
```

### 4.2 函数式（最大化为 1，最优点 (0.3,0.3,0.3)）

```python
def noisy_sphere(c: dict) -> float:
    d = 3
    return 1.0 - sum((c[f"x{i+1}"] - 0.3) ** 2 for i in range(d)) / d
```
- 解析最优：`optimum = 1.0`，`optimal_config = {"x1": 0.3, "x2": 0.3, "x3": 0.3}`。
- 噪声变体：
  - `noise="gaussian", noise_std=0.05, replicates=1`（σ 约为值域 1.0 的 5%）；
  - `noise="gaussian", noise_std=0.05, replicates=3`（观测=3 次均值，std 已知 → 可映射 α）；
  - 高噪：`noise_std=0.15`。
- 核验：`fn(optimal_config) == 1.0`；在 3 个变体下用"暴力网格 101³ 求真实最优"复核。

### 4.3 用途

1. 检验 `alpha` 是否正确映射（对比 `replicates=1` vs `3`：后者应显著更稳）；
2. 检验 `logei` vs `ei` 的差别（σ=0.15 时更明显）；
3. 作为 Tier D 的"连续 + 噪声"最小用例，CI 跑 3 seeds × budget 24。

---

## 5. Tier D：`constraint_easy` / `constraint_hard`（约束优化）

### 5.1 共同 space

```python
space = SearchSpace.from_dict([
    {"name": "T", "kind": "continuous", "low": 40.0, "high": 120.0, "unit": "degC"},
    {"name": "time", "kind": "continuous", "low": 5.0, "high": 240.0, "unit": "min"},
])
```

### 5.2 目标（无约束最优 95.0 @ (110, 100)）

```python
def constraint_objective(c: dict) -> float:
    return 60.0 + 35.0 * math.exp(-((c["T"] - 110.0) / 20.0) ** 2
                                  - ((c["time"] - 100.0) / 45.0) ** 2)
```

### 5.3 约束（椭圆可行域，`g <= 0` 可行）

```python
def constraint_fn(c: dict, rT: float = 25.0, rt: float = 60.0) -> dict:
    g = ((c["T"] - 95.0) / rT) ** 2 + ((c["time"] - 80.0) / rt) ** 2 - 1.0
    return {"feasibility_margin": g}     # 观测到 Observation.constraints
```

| name | rT | rt | 可行比例（MC 30 万点） | 约束最优（网格 2001×4001） | 性质 |
|---|---|---|---|---|---|
| `constraint_easy` | 25.0 | 60.0 | **0.2500** | **95.0** @ (110.0, 100.0) | 最优在可行域**内**（约束不 binding） |
| `constraint_hard` | 12.5 | 30.0 | **0.0616** | **93.096** @ (105.9, 94.8) | 约束 **binding**（无约束最优在域外） |

核验代码（直接可跑）：

```python
def verify_constraint(rT, rt, n_mc=300000, seed=0):
    rng = np.random.default_rng(seed)
    feas, best, arg = 0, -1e9, None
    for _ in range(n_mc):
        c = {"T": 40 + 80 * rng.random(), "time": 5 + 235 * rng.random()}
        if constraint_fn(c, rT, rt)["feasibility_margin"] <= 0:
            feas += 1
            y = constraint_objective(c)
            if y > best: best, arg = y, dict(c)
    return feas / n_mc, best, arg

# 期望：easy → (0.2500, ~95.0, (110,100)); hard → (0.0616, ~93.1, (~105.9,~94.8))
```

**评估器要求**：约束值本身可带噪（`noise` 同时作用在目标与约束），用于测试 `CONSTRAINTS_NOISE_IMPL.md` 的带噪约束 EI。

---

## 6. Tier F：`moo_yield_cost`（双目标，参考前沿 101 点）

### 6.1 space

```python
space = SearchSpace.from_dict([
    {"name": "T", "kind": "continuous", "low": 60.0, "high": 115.0, "unit": "degC"},
    {"name": "time", "kind": "log", "low": 20.0, "high": 180.0, "unit": "min"},
    {"name": "solvent", "kind": "categorical", "choices": ["MeCN", "EtOH"]},
    {"name": "ligand", "kind": "categorical", "choices": ["L1", "L3"]},
])
objective = Objective(name="yield", direction="max",
                      objectives=("yield", "cost"), directions=("max", "min"),
                      ref_point=(30.0, 20.5), ideal_point=(110.0, 7.0))
```

### 6.2 函数式

```python
SB = {"MeCN": 5.0, "EtOH": 3.0}
LB = {"L1": 4.0, "L3": 3.0}

def moo_yield(c: dict) -> float:
    return (20.0 + 80.0 * math.exp(-((c["T"] - 85.0) / 20.0) ** 2
                                   - (math.log(c["time"] / 60.0) / 0.9) ** 2)
            + SB[c["solvent"]] + LB[c["ligand"]])

def moo_cost(c: dict) -> float:
    return 0.1 * c["T"] + 0.05 * c["time"]

def moo_yield_cost(c: dict) -> dict:
    return {"yield": moo_yield(c), "cost": moo_cost(c)}
```

### 6.3 参考前沿（**按此协议可复现**）

1. `T = linspace(60, 115, 221)`；
2. 对每个 T，在 `time = geomspace(20, 180, 161)` × 4 组 (solvent, ligand) 中取 **yield 最大**的 `(time, solvent, ligand)`；
3. 得到 221 个 `(yield, cost)` 点，做非支配过滤（yield↑、cost↓）；
4. 结果：**101 个前沿点**，端点：
   - 最高 yield：`(yield=109.0, cost=11.5)` @ `T=85, time=60, MeCN, L1`；
   - 最低 cost：`(yield=45.769, cost=9.0)` @ `T=60, time=60, MeCN, L1`。

核验代码：

```python
def moo_reference_front():
    pts = []
    for T in np.linspace(60, 115, 221):
        best = None
        for t in np.geomspace(20, 180, 161):
            for sol in SB:
                for lig in LB:
                    c = {"T": T, "time": t, "solvent": sol, "ligand": lig}
                    y = moo_yield(c)
                    if best is None or y > best[0]:
                        best = (y, moo_cost(c), t, sol, lig)
        pts.append(best[:2])
    pts.sort(key=lambda p: (-p[0], p[1]))
    front, best_cost = [], float("inf")
    for y, cost in pts:
        if cost < best_cost - 1e-12:
            front.append((y, cost)); best_cost = cost
    return front            # len == 101，(109.0, 11.5) ... (45.769, 9.0)
```

### 6.4 HV 口径（必须固定）

- 归一化：`y_norm = (y - 30) / (110 - 30)`；`c_norm = (20.5 - cost) / (20.5 - 7.0)`（成本取反后越大越好）；
- 参考点 `r = (0, 0)`；`HV` 在 2D 用扫描线精确计算；
- 报告：`HV_ratio = HV(算法前沿) / HV(参考前沿)`（参考前沿的 HV 用上面 101 点算）。

### 6.5 验收

- ParEGO：`HV_ratio ≥ 0.85`（budget 32、5 seeds）；
- 前沿覆盖：2D 归一化空间 10×10 网格，非空格 ≥ 参考前沿的 60%；
- 噪声变体（`noise="gaussian", noise_std=1.5, replicates=2`）：`HV_ratio` 下降 <10%，或与 qNEHVI 的差距可解释。

---

## 7. 核验脚本（建议直接进 `scripts/verify_benchmarks.py`）

```python
CHECKS = [
    ("discrete_catalyst_v2", 82.0,      lambda: discrete_catalyst_v2({"metal":"Pd","ligand":"C","additive":"X3","temp":80})),
    ("high_cardinality_solvent", 50 + 25*math.exp(-1/9) + 20,
                                       lambda: high_cardinality_solvent({"T":90.0,"solvent":"S04"})),
    ("noisy_sphere", 1.0,               lambda: noisy_sphere({"x1":0.3,"x2":0.3,"x3":0.3})),
    ("constraint_easy_opt", 95.0,       lambda: constraint_objective({"T":110.0,"time":100.0})),
]
for name, expected, fn in CHECKS:
    got = fn()
    assert abs(got - expected) < 1e-6, (name, got, expected)

# 约束比例/约束最优、MOO 前沿端点、mixed_reaction 102.0、discrete_catalyst 78.0 同理加断言
```

**CI 门槛**：`verify_benchmarks.py` 必须作为 `smoke` benchmark 的前置步骤（任一断言失败即 CI 红）。

---

## 8. 参考

R18（Olympus：emulator/任务定义与可核验最优）、R32（HPOBench：协议与预算口径）、R26（初始设计影响）、R22（描述符 categorical：`high_cardinality_solvent` 的动机）、R11（ParEGO：`moo_yield_cost` 的验收算法）、R10（带噪约束：`constraint_*` 的验收）。
