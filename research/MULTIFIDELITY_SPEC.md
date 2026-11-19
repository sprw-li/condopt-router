# MULTIFIDELITY_SPEC — 多保真优化的代码级规格（successive halving + 简化 BOCA）

> 维护者：algo-research ｜ v1（2026-11-05，Round 4）
> 对应：`BENCHMARK_SPEC.md` Tier F `mf_suzuki`（公式 `y_s = y − 15·(1−s)^1.5`，`cost(s)=s`）、`ALGORITHM_SURVEY.md` §10。
> 文献：R39（successive halving：资源翻倍淘汰一半）、R40（Hyperband：SH 的多 bracket 调度）、R41（BOCA：保真度作为连续近似空间）、R35（批量局部惩罚，可叠在 fidelity 选择上）。
> 结论先行：**v0.3 最划算的是"fidelity 作为 GP 输入 + 成本感知采集"**（BOCA 简化版，1–2 天）；successive halving 作为**调度器**（不依赖 GP）另做一个独立算法（半天）。

---

## 1. 数据模型：fidelity 放哪

**结论：fidelity 不进 `SearchSpace`**（条件变量是"实验条件"，保真度是"实验规模"），而是作为**观测属性 + 算法输入**。现有 `Observation.fidelity: float = 1.0` 已经存在（`dataset.py:52`），`Study._to_observation` 也已经透传（`study.py:476` 附近）——所以**不需要改观测层**，只需要：

### 1.1 `FidelitySpec`（新文件 `src/condopt/fidelity.py`）

```python
@dataclass(frozen=True)
class FidelitySpec:
    name: str = "scale"                 # 显示名：如 "trial_time" / "batch_scale"
    levels: tuple[float, ...] = (0.25, 0.5, 1.0)   # 离散档位（连续输入时可为空）
    costs: tuple[float, ...] | None = None         # 每档成本（默认 = level 本身）
    continuous: bool = False            # True：保真度作为连续输入 s∈[levels[0], 1]
    bias_sign: str = "lower_is_worse"   # 低保真默认系统性偏低（仅用于文档/初始化）

    def cost(self, s: float) -> float:
        if self.costs: 
            # 最近档位成本（离散）或线性插值（连续）
            ...
        return float(s)

    def to_dict(self) -> dict: ...
```

挂到 `Objective`（同 `constraint_specs` 的做法）或 `Study` 的 `problem.json`：

```json
{"fidelity": {"name": "trial_time", "levels": [0.25, 0.5, 1.0],
              "costs": [0.25, 0.5, 1.0], "continuous": false}}
```

**兼容**：没有 `fidelity` 字段 → 所有观测 `fidelity=1.0`，行为与现在完全一致。

### 1.2 建议接口（最小侵入）

`Optimizer.ask` 现在返回 `list[dict]`（条件）。加一个**可选**方法：

```python
class Optimizer:
    def ask_with_fidelity(self, n: int = 1) -> list[tuple[dict, float]]:
        """默认实现：所有点用全保真。多保真算法覆写。"""
        return [(cfg, 1.0) for cfg in self.ask(n)]
```

`Study.ask` 里：

```python
pairs = optimizer.ask_with_fidelity(n)
pending = [{"id": ..., "x": cfg, "fidelity": fid, "algorithm": algorithm} for cfg, fid in pairs]
```
`suggestions.csv` 加一列 `fidelity`；`results.json` 里实验者回填 `"fidelity": 0.5`（缺省 1.0，向后兼容）。`Study.remaining()` 仍按次数计；**另外**维护 `cost_spent = Σ cost(fidelity)`，报告里同时显示（`BENCHMARK_SPEC` §5.4 的 cost-aware 指标）。

---

## 2. Successive halving（SH）——无模型的调度器

### 2.1 算法（R39/R40 的最小版）

参数：`n_configs`（初始配置数）、`r_min`（最小保真度/资源）、`eta=3`（淘汰比例）、`max_fidelity=1.0`。

```
r = r_min
configs = sample(n_configs)                     # Sobol/LHS
while r <= max_fidelity and len(configs) > 1:
    for c in configs:
        evaluate(c, fidelity=r)                 # 观测带 fidelity 入库
    k = max(1, floor(len(configs) / eta))
    configs = top_k(configs, k)                 # 按该 fidelity 下的观测值排序
    r = min(r * eta, max_fidelity)
return best(configs)
```

Hyperband 版：对 `(n, r)` 的多个 bracket（`s = s_max..0`，`n_i = ⌈B/r_i · eta^s/(s+1)⌉`）并行调度；化学实验里"短时试跑→全时长"天然对应。

### 2.2 condopt 实现

```python
@register
class SuccessiveHalving(Optimizer):
    spec = AlgorithmSpec(
        name="successive_halving", family="budget_scheduler",
        summary="逐轮淘汰：先用低保真度/短时试跑筛掉大部分配置，再对存活者用高保真度验证。",
        params=(ParamSpec("n_configs", "int", 12, low=2, high=100),
                ParamSpec("eta", "int", 3, low=2, high=5),
                ParamSpec("r_min", "float", 0.25, low=0.05, high=1.0)),
        capabilities=frozenset({"multi_fidelity", "mixed", "categorical", "cheap"}),
    )
    # ask_with_fidelity 产出一个 batch 的 (config, r)；内部状态 = 当前 rung 的配置列表
```

**注意**：SH 是**离线调度**，`ask/tell` 语义需要"一个 rung 的配置全部 tell 后才能进入下一 rung"——与现有"pending 未清不能 ask"的约束天然契合（见 `REVIEW_v01.md` P0-5 的修复）。多保真算法需要声明能力位 `"multi_fidelity"`，router 的 `_required_capabilities`（`router.py:587-588`）已经会要求它。

---

## 3. 简化 BOCA：fidelity 作为 GP 输入 + 成本感知采集

### 3.1 思路（R41 的单模型简化）

- 用**一个 GP** 在 `(x, s)` 上建模：`features_mf = [features(x), s_norm]`（`s_norm = (s − s_min)/(1 − s_min)`）；
- 保真度方向的 lengthscale 通常**更大**（低/高保真度之间平滑变化），初值取其他维 lengthscale 的 2–3 倍；若数据表明偏差明显（低保真系统性偏低），在均值函数里加 `β·s` 线性项（或对 `y` 减去 `bias_est(s)`）；
- 采集：`acq_mf(x,s) = EI(x,s) / cost(s)^α`，`α∈[0.5,1]`（默认 1，即"每单位成本收益"）；
- 每轮从 `X × S` 的候选池选 `(x,s)`；`S` 用离散档位（3–4 档）或连续网格（5–9 点）。

### 3.2 代码骨架（独立类，复用 GP 工具）

```python
@register
class MultiFidelityBO(Optimizer):
    spec = AlgorithmSpec(
        name="mf_bo", family="model_based",
        summary="多保真 BO：保真度作为输入、按单位成本收益选 (条件, 保真度)。",
        params=(ParamSpec("alpha_cost", "float", 1.0, low=0.0, high=2.0),
                ParamSpec("n_candidates", "int", 1024, low=64, high=65536),
                ParamSpec("n_initial", "int", 0, ...),
                ParamSpec("bias_term", "bool", True)),
        capabilities=frozenset({"multi_fidelity", "mixed", "categorical", "continuous",
                                 "batch", "model_based", "cost_aware"}),
    )

    def _features(self, config, fidelity):
        base = self.space.features(config)                     # 复用 space 层
        s = (fidelity - self.fid.levels[0]) / (1.0 - self.fid.levels[0])
        return np.concatenate([base, [s]])

    def ask_with_fidelity(self, n=1):
        # 1) 初始设计：对每个 x 给最低保真度（省成本）
        # 2) 有模型后：候选池 = (x_grid × s_grid)，score = EI / cost(s)**alpha_cost
        ...
        return [(cfg, s) for cfg, s in chosen]

    def tell(self, observations):
        super().tell(observations)
        # X = [features(x), s_norm]；y = 观测值（可选减去 bias_est(s)）
```

**fidelity 的 bias**：默认 `bias_est(s) = mean(y | s=1) − mean(y | s)` 不可直接算（不同 x），更稳的做法是**把 s 作为输入维**让 GP 自己学偏差（R41 的做法是分层/两阶段）。若历史数据表明"低保真系统性偏低"（同一 x 下 s=1 的 y 比 s=0.25 高），可加 `β·(1−s)` 到均值；用 `bias_term=True` 控制。

### 3.3 批量

`ask_with_fidelity(n)` 用"贪心 + 局部惩罚"（`SPRINT2_SPEC.md` §2）：batch 内每个点选 `(x,s)`，惩罚只作用在 x 上（保真度不同不算重复）。为避免"全选低保真"，对 `s<0.5` 的点额外要求至少一个高保真点/批（可配置）。

---

## 4. 与 Study / benchmark 的对接

| 位置 | 改动 |
|---|---|
| `problem.json` | 加 `fidelity` 配置（可选）；无则全 1.0 |
| `Study.ask` | `ask_with_fidelity` → pending 项带 `fidelity`；`suggestions.csv` 加列 |
| `Study.tell` | results 的 `fidelity` 透传到 `Observation`（已支持） |
| `Study.summary/report` | 显示 `cost_spent`、各保真度评估次数、"单位成本 best-so-far" |
| `benchmarks.py` | 实现 `mf_suzuki`（Tier F）：`y_s = y − 15(1−s)^1.5`，`cost(s)=s` |
| `router.py` | `multi_fidelity=True` 时：`mf_bo`（默认，连续/档位）或 `successive_halving`（离散档位、强噪声）；两者声明 `multi_fidelity` 能力 |

**预算口径**：`budget_total` 仍按"评估次数"，另加 `cost_budget`；报告同时给"次数口径"和"成本口径"的 best-so-far（`BENCHMARK_SPEC` §5.4）。

---

## 5. 参数与验收

| 参数 | 默认 | 说明 |
|---|---|---|
| levels | (0.25, 0.5, 1.0) | 化学：短时/中时/全时；或小试/中试 |
| costs | = levels | 成本比可实测后填入 |
| `alpha_cost` | 1.0 | 0.5 更激进用低保真 |
| 保真度 lengthscale | 2–3× 其他维 | 偏差平滑 |
| `n_initial` | `max(2*(feature_dim+1), 6)` | 初始设计**全用最低保真度**（成本低） |
| SH：`eta`/`r_min` | 3 / 0.25 | R39 默认 |

**验收**

1. `mf_suzuki`（budget 次数=40）：`mf_bo` 的 **cost-aware best-so-far 在同等总成本下** ≥ 全保真 BO 的 1.3×（或同等 best 下成本 ≤0.7×）；
2. `successive_halving`：在"低保真与真值相关性 ρ=0.9"的合成问题上，同成本下 best ≥ 全保真随机搜索 +30%；ρ<0.5 时允许更差，但要有 warning（`BENCHMARK_SPEC` 提到的"相关性 <0.7 慎用"）；
3. `test_ask_with_fidelity_defaults_to_one`：单保真算法行为不变；
4. `test_study_records_fidelity`：pending/suggestions/results/dataset 全链路保真度一致；
5. 报告有 `cost_spent` 与两种口径的 best。

---

## 6. 参考

R39（successive halving）、R40（Hyperband：多 bracket 调度）、R41（BOCA：连续近似空间；本文单模型简化）、R35（批量局部惩罚）、R18（Olympus 的预算/成本协议）、R32（benchmark 成本口径）。
