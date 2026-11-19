# MFBO_COST_API — `remaining_cost()` 与 `best-only-full-fidelity` 的设计与验收

> 维护者：algo-research ｜ v1（2026-11-05，Round 9）
> 对应：`mfbo.py`（v0.8：初始设计已全最低档，`best=max(self._y)` 仍混保真）、`study.py`（`remaining()` 只按次数）、`benchmarks.cost_aware_best`。
> 文献：R41（BOCA：成本感知采集）、R39/R40（SH/Hyperband 的预算口径）、R18（Olympus 的 budget/cost 协议）、R32（benchmark 成本口径）、R42（HV/真值口径）。
> **现状实测（v0.8）**：预算 cost=24 的 `mfbo` run 实际花 **25.25（+5.2%）**；`best = max(self._y)` 仍可能被低保真的高噪声值污染（`REVIEW_v08` P2-5）。

---

## 1. 目标与不变量

1. **成本是预算**：多保真算法要能回答"我还剩多少成本"、"这个 batch 是否付得起"；`Study.remaining()`（次数）保留给单保真。
2. **`best` 只用全保真**：采集函数的"当前最优"必须是**可交付的全保真值**；低保真值只用于建模，不用于基线。
3. **不破坏单保真**：无 `fidelity` 配置时所有成本 = 1.0 次数 = 成本，行为与现在一致。

---

## 2. 数据模型

### 2.1 `Objective.fidelity`（已在 v0.6 定义，补全 `costs` 语义）

```json
"fidelity": {"name": "scale", "levels": [0.25, 0.5, 1.0],
             "costs": [0.25, 0.5, 1.0], "continuous": false}
```
- `costs` 与 `levels` 等长；缺省 `costs = levels`；
- 校验：`levels` 升序、(0,1]、`costs` 非负、长度一致。

### 2.2 `Study` 新接口

```python
class Study:
    @property
    def cost_spent(self) -> float:
        """Σ cost(observation.fidelity)；无 fidelity 配置时 = len(dataset)。"""
        spec = self.objective.fidelity or {}
        table = dict(zip(spec.get("levels", (1.0,)), spec.get("costs", spec.get("levels", (1.0,)))))
        return float(sum(table.get(round(o.fidelity, 9), o.fidelity) for o in self.dataset))

    @property
    def cost_budget(self) -> float:
        spec = self.objective.fidelity or {}
        if not spec:
            return float(self.budget)
        costs = spec.get("costs", spec.get("levels", [1.0]))
        return float(self.budget) * float(max(costs))    # 见 §2.3 口径

    def remaining_cost(self) -> float:
        return max(0.0, self.cost_budget - self.cost_spent)

    def remaining(self) -> int:                     # 保留：次数口径
        ...
```

**预算口径（必须写进文档）**：`budget`（次数）与 `cost_budget`（成本）是**两个独立预算**；多保真 study 用 `cost_budget`，单保真用 `budget`。建议 CLI 增加 `--cost-budget`；缺省 `cost_budget = budget × max(costs)`（等价"全部用全保真的成本"）。

### 2.3 `ask_with_fidelity` 的成本规划

```python
def ask_with_fidelity(self, n=1):
    remaining_cost = ...  # Study 传入或由 optimizer 维护
    pairs = []
    for _ in range(n):
        config, level = self._propose(pairs)
        level = self._fit_to_budget(level, remaining_cost - self._planned_cost(pairs))
        pairs.append((config, level))
    return pairs

def _fit_to_budget(self, level: float, budget_left: float) -> float:
    """若 level 的成本超出剩余预算，降到付得起的最高档；都付不起则用最低档。"""
    affordable = [lv for lv in self._levels() if self._cost(lv) <= budget_left + 1e-12]
    if not affordable:
        return float(self._levels()[0])
    return float(max(affordable)) if self._cost(level) > budget_left else float(level)
```

**批量**：batch 内先按 `n` 个点规划成本（最坏情况 `n×cost(level)`），超出则整批降档；仍超出则减小 batch（返回少于 n，`Study` 允许）。

### 2.4 `best` 只用全保真

```python
def _best_full_fidelity(self) -> float | None:
    full = [self._y[i] for i, o in enumerate(self._observations) if o.fidelity >= 1.0 - 1e-9]
    return max(full) if full else None

# mfbo._propose:
best = self._best_full_fidelity()
if best is None:
    # 还没有全保真点：用"最高保真观测 + 该档偏差估计"的保守值
    best = max(self._y) - self._bias_estimate(fidelity_of_max)
    if best is None:  # 无任何观测
        return random ^ min_level
```
- `_bias_estimate(s) = mean(full - low)` 用同 x 不同 fidelity 的配对估计；无配对时用 0 + 保守裕度（如 5% 值域）。
- 也可把"当前最大值来自低保真"显式写进 `decisions`，供报告解释。

### 2.5 `benchmarks.cost_aware_best` 对齐

```python
def cost_aware_best(observations, budget: float | None = None) -> dict:
    """按观测顺序累计成本；budget=None 返回全程 best_full_prefix/成本曲线点。"""
    spent, best_full, curve = 0.0, None, []
    for o in observations:
        spent += mf_suzuki_cost(o.fidelity)
        if o.fidelity >= 1.0 - 1e-9:
            best_full = o.y if best_full is None else max(best_full, o.y)
        curve.append({"cost": spent, "best_full": best_full})
    return {"best_full_fidelity": best_full, "total_cost": spent, "curve": curve,
            "best_at_budget": last_best_at(curve, budget)}
```
删除/降级 `best_value_per_cost = max(value/spent)`（它不是可比较指标，`REVIEW_v08` P2-6）。

---

## 3. 验收测试

```python
def test_cost_spent_matches_sum_of_costs():
    # 3 个观测 fidelity 0.25/0.5/1.0 → cost_spent == 1.75
def test_remaining_cost_never_negative():
    # 预算耗尽后 ask_with_fidelity 返回最低档（或空），不抛 ZeroDivision/负预算
def test_ask_with_fidelity_respects_cost_budget():
    # cost_budget=2, batch=4 → 建议总成本 ≤2（或返回少于 4 个）
def test_mfbo_never_overshoots_cost_budget():
    # 实测 v0.8 25.25/24（+5.2%）→ 修复后 ≤24 或 ≤24*1.02
def test_mfbo_best_ignores_low_fidelity_observations():
    # 构造：低保真 y=200（含 bias+噪声）、全保真 y=100 → acquisition 的 best 应是 100
def test_mfbo_initial_design_lowest_level():
    # v0.8 已修：前 n 个建议 fidelity 全 == min(levels)
def test_cost_aware_best_curve_monotone_cost():
    # curve 的 cost 单调递增；best_full 非降
```

**benchmark 验收**：`mf_suzuki`、cost 预算 24、2 seeds：`mfbo` 的 `total_cost ≤ 24*1.02`，且 `best_full_fidelity` 不劣于全保真 BO 的 95%（v0.8 实测均值 +8.3%，修复后应保持）。

---

## 4. 与 `Study`/CLI 的接线

| 位置 | 改动 |
|---|---|
| `Study.ask` | 传给 `ask_with_fidelity` 的剩余成本：`self.remaining_cost()`；pending 里记录 `cost_est` |
| `Study.tell` | 观测的 fidelity 已透传；更新 `cost_spent`（属性计算，无需存） |
| `Study.report` | 显示 `cost_spent / cost_budget`、按 fidelity 分组次数、best_full |
| CLI | `init --cost-budget`、`ask` 打印"本批预计成本"、`report --json` 含 cost 字段 |
| `router` | `multi_fidelity=True` 时把 `cost_aware` 能力位纳入（已有 `mfbo` 声明） |

**迁移**：旧 study（无 `fidelity` 配置）→ `cost_spent = len(dataset)`，`cost_budget = budget`，行为不变。

---

## 5. 参考

R41（BOCA：成本感知/保真度）、R39/R40（成本预算与 SH/Hyperband 口径）、R18（Olympus budget/cost 协议）、R32（benchmark 成本口径）、`MULTIFIDELITY_MVP.md`、`MF_BENCHMARK.md`、`REVIEW_v08.md`（P1-4/P1-5/P2-5/P2-6）。
