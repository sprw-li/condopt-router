# MULTIFIDELITY_MVP — 多保真最小可行实现（最小侵入）

> 维护者：algo-research ｜ v1（2026-11-05，Round 5）
> 对应：`MULTIFIDELITY_SPEC.md`（Round 4 完整规格）、`BENCHMARK_SPEC.md` Tier F `mf_suzuki`、`router.py` 的 `multi_fidelity` 能力位（已存在，`router.py:625-626`）。
> 文献：R41（BOCA：保真度作为连续输入）、R39（successive halving）、R40（Hyperband）、R18（成本口径）。
> 目标：**不碰 `space.py`、不改 `Observation`**（`fidelity` 字段已有），只加 3 个触点：`Objective/Study` 配置、`ask_with_fidelity`、`pending`/CSV 透传。

---

## 1. 触点清单（按改动量排序）

| # | 位置 | 改动 | 是否必做 |
|---|---|---|---|
| 1 | `Objective`（`base.py:140`） | 加 `fidelity: dict \| None`（`{"levels":[...], "costs":[...], "continuous":false}`） | 必做 |
| 2 | `Optimizer`（`base.py:212`） | 加默认方法 `ask_with_fidelity(n) -> [(cfg, fid)]`，默认全 1.0 | 必做 |
| 3 | `Study.ask`（`study.py:338-350`） | `optimizer.ask_with_fidelity(n)`；pending 项加 `"fidelity": fid` | 必做 |
| 4 | `Study._write_suggestions`（`study.py:633-638`） | CSV 加 `fidelity` 列 | 必做 |
| 5 | `Study.tell` | results 的 `fidelity` 已透传到 `Observation`（`study.py:478`） | 无需改 |
| 6 | `Study.summary/report` | 加 `cost_spent`、按 fidelity 分组的评估次数 | 建议 |
| 7 | `benchmarks.py` | 加 `mf_suzuki`（Tier F） | 必做（验收用） |

**为什么不把 fidelity 放进 `SearchSpace`**：条件变量是"实验条件"（要写给实验员），保真度是"实验规模"（决定成本），混在一起会让 space/CLI/skill 语义混乱；且 `Observation.fidelity` 已经在数据层，放 space 等于双份真相。

---

## 2. 数据与接口（可直接粘贴）

### 2.1 `Objective.fidelity`

```python
# base.py, Objective 内
fidelity: dict | None = None        # {"levels":[0.25,0.5,1.0], "costs":[0.25,0.5,1.0],
                                     #  "continuous": false, "name":"trial_time"}
# __post_init__ 校验
if self.fidelity is not None:
    levels = [float(x) for x in self.fidelity.get("levels", [1.0])]
    if not levels or min(levels) <= 0 or max(levels) > 1.0:
        raise ValueError("fidelity.levels must be in (0, 1]")
    if sorted(levels) != levels:
        raise ValueError("fidelity.levels must be sorted ascending")
    self.fidelity = dict(self.fidelity, levels=tuple(levels))
# to_dict/from_dict 带上 fidelity
```

### 2.2 `Optimizer.ask_with_fidelity`

```python
def ask_with_fidelity(self, n: int = 1) -> list[tuple[dict, float]]:
    """Default: single-fidelity. Multi-fidelity algorithms override this."""
    return [(cfg, 1.0) for cfg in self.ask(n)]
```

### 2.3 `Study.ask` 的 pending（`study.py:338-350`）

```python
pairs = optimizer.ask_with_fidelity(n)
batch_id = len(self._rounds) + 1
pending = [
    {"id": f"r{batch_id}-{i+1}", "batch": batch_id, "algorithm": algorithm,
     "x": cfg, "fidelity": float(fid)}
    for i, (cfg, fid) in enumerate(pairs)
]
```

`results.json` 回填（已支持）：`{"id": "r1-1", "y": 93.2, "fidelity": 0.5}`；缺省 1.0。

### 2.4 成本记账

`Study.remaining()` 仍按**评估次数**；新增只读属性：

```python
@property
def cost_spent(self) -> float:
    levels = (self.objective.fidelity or {}).get("levels", [1.0])
    costs = (self.objective.fidelity or {}).get("costs", list(levels))
    table = dict(zip(levels, costs))
    return float(sum(table.get(round(o.fidelity, 9), o.fidelity) for o in self.dataset))
```
报告同时给"次数口径"和"成本口径"的 best（`BENCHMARK_SPEC` §5.4）。

---

## 3. 为什么这样最省：一次跑通链路

```
用户 answers.json（含 fidelity 档位）
  → prepare.py：objective.json 写 fidelity 配置
  → Study.create：objective 带 fidelity
  → Study.ask：mf 算法返回 (cfg, fid)
  → 实验员/机器人按 fid 做实验（短时/小量）
  → results.json 带 fidelity
  → Study.tell → Observation.fidelity → dataset.jsonl
  → 下一轮 ask：算法用 (x, fidelity) 建 GP
```

单保真场景：`fidelity` 缺省 → `ask_with_fidelity` 全 1.0 → 所有现有测试/行为不变。

---

## 4. MVP 算法 A：BOCA 简化版（`mf_bo`）

### 4.1 模型

- 一个 GP，输入 `features_mf = [space.features(x), s_norm]`，`s_norm = (fid − min_level)/(1 − min_level)`；
- 保真度维 lengthscale 初值 = 其他维的 2–3 倍（低/高保真度之间更平滑）；
- 若 `bias_term=True`：训练目标 `y − bias_est(fid)`，`bias_est` 先用"同 x 不同 fidelity 的观测差"估计；数据不足时置 0（GP 自己学）。
- 采集：`acq(x,s) = EI(x,s) / cost(s)^α`，`α=1` 默认。

### 4.2 伪代码

```python
class MultiFidelityBO(GaussianProcessBO):
    spec = AlgorithmSpec(name="mf_bo", family="model_based",
        summary="多保真 BO：保真度作为 GP 输入，按单位成本收益选 (条件, 保真度)。",
        params=GaussianProcessBO.spec.params + (
            ParamSpec("alpha_cost", "float", 1.0, low=0.0, high=2.0),
            ParamSpec("bias_term", "bool", True),
        ),
        capabilities=GaussianProcessBO.spec.capabilities | {"multi_fidelity", "cost_aware"})

    def ask_with_fidelity(self, n=1):
        if len(self._y) < self._n_initial():
            # 初始设计全用最低档（省成本）
            return [(cfg, self._levels[0]) for cfg in self._initial_design(n)]
        out = []
        for _ in range(n):
            cand = self._candidate_pool_mf()          # x 池 × s 档位
            mu, sd = self._mf_model().predict(cand.features, return_std=True)
            ei = self._ei(mu, sd, best=self._y.max())
            acq = ei / np.power(cand.costs, float(self.params["alpha_cost"]))
            pick = argmax(acq)
            out.append((cand.configs[pick], cand.fidelities[pick]))
        return out
```

**关键实现点**（不做就会退化）：

1. `_candidate_pool_mf` = 条件候选池（复用 `_candidate_pool`）× 档位；每个组合都要 `features_mf`。
2. 成本指数 `α=1` 时，低成本档位天然被偏好；**但至少要有一个高保真点/若干轮**（防止 GP 永远没见过 s=1）：设 `min_full_fraction=0.1`，每 10 个建议至少有 1 个全保真。
3. 观测噪声：同 §5（REPLICATES_ALPHA）的 alpha；低保真通常噪声更大，`alpha` 可按 fidelity 分组估计（MVP 可先用全局值）。
4. **不要**把不同 fidelity 的观测直接当独立同分布：GP 的 `s` 输入维承担这个建模（这也是为什么 lengthscale 要给大）。

---

## 5. MVP 算法 B：successive halving 调度器（`successive_halving`）

```python
class SuccessiveHalving(Optimizer):
    spec = AlgorithmSpec(name="successive_halving", family="budget_scheduler",
        summary="逐轮淘汰：先低保真筛掉大部分配置，再对存活者做高保真验证。",
        params=(ParamSpec("n_configs","int",12,low=2,high=100),
                ParamSpec("eta","int",3,low=2,high=5),
                ParamSpec("r_min","float",0.25,low=0.05,high=1.0)),
        capabilities=frozenset({"multi_fidelity","mixed","categorical","cheap"}))

    # 状态：self._rung_configs, self._rung_fidelity, self._rung_results
    def ask_with_fidelity(self, n=1):
        if self._rung_configs is None:                       # 初始化：一个 rung
            self._rung_configs = self.space.sample_lhs(int(self.params["n_configs"]), rng=self.rng)
            self._rung_fidelity = float(self.params["r_min"])
        # 每次只能把当前 rung 剩下的配置发出去（Study 的 pending 机制保证批内 tell）
        return [(cfg, self._rung_fidelity) for cfg in self._rung_configs[:n]]

    def tell(self, observations):
        super().tell(observations)
        # 收齐当前 rung 后：按观测值排序，保留 top 1/eta，fidelity *= eta
        if len(observations) < len(self._rung_configs):
            return
        ranked = sorted(zip(self._rung_configs, ...), key=..., reverse=True)
        keep = max(1, len(ranked)//int(self.params["eta"]))
        self._rung_configs = [cfg for cfg, _ in ranked[:keep]]
        self._rung_fidelity = min(1.0, self._rung_fidelity * int(self.params["eta"]))
```

**接口约定**：SH 的 `ask_with_fidelity` 一个 rung 一个 rung 发；`Study.ask` 在 pending 未清时会拒绝（当前行为），正好匹配"rung 内全部 tell 完再进下一 rung"。Hyperband = 多个 `(n_i, r_i)` bracket 的 SH 并集（v1 先不做）。

---

## 6. 验收用例

| # | 用例 | 期望 |
|---|---|---|
| 1 | 单保真 study（无 fidelity 配置） | `ask_with_fidelity` 全 1.0；现有 102 测试不变 |
| 2 | fidelity 透传 | pending/suggestions.csv/results/dataset 的 fidelity 一致；`cost_spent` 正确 |
| 3 | `mf_bo` on `mf_suzuki`（`y_s = y − 15(1−s)^1.5`，`cost=s`），成本预算 = 全保真 15 次的成本 | 同等总成本下 best 比"全保真 BO"高 ≥10%，或达到同 best 成本 ≤80% |
| 4 | `mf_bo` 的低保真比例 | 总建议中 s<1 占比 >50%，但每 10 个点至少 1 个 s=1 |
| 5 | `successive_halving` 相关性 ρ=0.9 合成问题 | 同成本下 best ≥ 随机搜索 +30% |
| 6 | 低保真误导（ρ=0.3） | 允许更差，但 `report` 有 warning（相关性检查留给用户/上层） |
| 7 | `router` | `multi_fidelity=True` 时 `mf_bo`/`successive_halving` 通过能力过滤；单保真算法被过滤（`multi_fidelity` 位） |

---

## 7. 实施顺序（2–3 天）

1. 触点 #1–#4（半天，纯透传）→ 单测 #1/#2；
2. `mf_suzuki` benchmark（半天）；
3. `mf_bo`（1 天）→ 验收 #3/#4；
4. `successive_halving`（半天）→ 验收 #5；
5. router 参数策略 + 报告成本口径（半天）。

---

## 8. 参考

R41（BOCA：保真度作为连续输入、两阶段选择）、R39/R40（SH/Hyperband 预算调度）、R18（成本/预算口径）、R2（GP alpha 与噪声）、`MULTIFIDELITY_SPEC.md`（完整规格与 BOCA 理论）。
