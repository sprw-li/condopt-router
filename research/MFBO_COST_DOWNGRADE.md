# MFBO_COST_DOWNGRADE — `remaining_cost` 传入 optimizer 与降档/减批的实现规格

> 维护者：algo-research ｜ v1（2026-11-05，Round 12）
> 对应：`REVIEW_v11.md` P0-1（预算未被消费）、`MFBO_COST_API.md`（完整 API 设计）、`study.py:350` 的 `ask_with_fidelity` 调用点。
> 文献：R41（BOCA：成本感知选保真度）、R39/R40（预算调度必须在选点前）、R18（Olympus budget/cost 协议）、R32（成本口径）。
> 硬约束：**不破坏 `ask/tell` 契约**（`ask_with_fidelity` 是可选方法；单保真算法完全不受影响）。

---

## 1. 传参方案（三种，推荐 ③）

| 方案 | 做法 | 优点 | 缺点 |
|---|---|---|---|
| ① 构造参数 | `registry.build(..., cost_budget=...)` → `Optimizer.__init__` 增加可选 kwarg | 显式、无状态歧义 | 需改 `registry.build` 与所有算法签名（向后兼容默认 None，改动点 3 处） |
| ② call 参数 | `ask_with_fidelity(n, remaining_cost=None)` | 最贴近语义（每轮剩余成本会变） | Study 需按 `hasattr`/签名适配；第三方算法签名不一致 |
| ③ **构造参数 + 可选 call 覆盖** | `Optimizer.cost_budget` 属性（构造时设），`ask_with_fidelity` 也可接收 `remaining_cost` 覆盖 | 一轮内稳定、跨轮可更新、兼容 ③ | 两处入口需测 |

推荐 **③**：`Study` 构造 optimizer 时把 `cost_budget` 设进去（与 `prior` 同层），每轮 `ask` 前用 `remaining_cost()` 覆盖（因为已花费的部分变化）。

```python
# base.py
class Optimizer:
    def __init__(self, space, objective=None, params=None, seed=0, prior=None,
                 cost_budget: float | None = None):
        ...
        self.cost_budget = cost_budget      # 可选：多保真算法使用

# registry.build 增加 cost_budget=None 透传（一行）

# study.py::ask（v0.10.1 的 build 调用处）
optimizer = registry.build(..., cost_budget=self.meta.get("cost_budget"),
                           prior=observations)
...
pairs = optimizer.ask_with_fidelity(n, remaining_cost=self.remaining_cost()) \
        if getattr(optimizer, "supports_fidelity", False) else None
```
兼容：`ask_with_fidelity` 的签名加 `remaining_cost: float | None = None`（默认 None = 不限）。若某算法未覆写该方法，`Optimizer.ask_with_fidelity` 默认实现忽略该参数。

---

## 2. `mfbo` 侧实现

### 2.1 成本表

```python
# mfbo.py
def _cost(self, level: float) -> float:
    """成本模型：cost(level) = level**cost_exponent（默认线性）。
    若 Objective.fidelity 提供 costs 表则查表（更精确）。"""
    spec = self.objective.fidelity or {}
    costs = dict(zip(spec.get("levels", ()), spec.get("costs", ())))
    if level in costs:
        return float(costs[level])
    return float(level) ** float(self.params["cost_exponent"])
```

### 2.2 降档 + 减批

```python
def ask_with_fidelity(self, n: int = 1, remaining_cost: float | None = None) -> list[tuple[dict, float]]:
    if n <= 0:
        return []
    left = self._remaining_budget(remaining_cost)
    # 1) 初始设计（最低档，成本很低）：先按剩余成本允许的数量发
    design_left = self._initial_design_left()
    out: list[tuple[dict, float]] = []
    if design_left > 0 and left is not None:
        min_cost = min(self._cost(lv) for lv in self._levels())
        k = min(n, design_left, int(left // max(min_cost, 1e-9)))
        if k == 0:
            return []                       # 付不起任何一个点：显式返回空
        out.extend(self._initial_design(k))
        left -= sum(self._cost(f) for _, f in out)
        n -= k
    # 2) model-based：按剩余成本选点
    for _ in range(n):
        config, level = self._propose(out)
        if left is not None:
            level = self._fit_to_budget(level, left)
            if level is None:
                break                       # 剩余预算不足以做最低档：提前结束
        out.append((config, level))
        left -= self._cost(level)
    return out

def _remaining_budget(self, override: float | None) -> float | None:
    if override is not None:
        return max(0.0, float(override))
    if self.cost_budget is not None:
        # 注意：optimizer 只知道自己见过的成本，Study 传入的 override 更准
        spent = sum(self._cost(f) for f in self._fidelity) if hasattr(self, "_fidelity") else 0.0
        return max(0.0, float(self.cost_budget) - spent)
    return None

def _fit_to_budget(self, level: float, left: float) -> float | None:
    """选 ≤left 的最高可负担档；一个都付不起返回 None。"""
    affordable = [lv for lv in self._levels() if self._cost(lv) <= left + 1e-12]
    if not affordable:
        return None
    if self._cost(level) <= left + 1e-12:
        return float(level)
    return float(max(affordable))
```

### 2.3 三个细节（不处理会引入新 bug）

1. **整批规划**：`ask_with_fidelity(n)` 返回的 q 个点总成本可能超预算；上面逐点递减已保证，但初始设计应 *整体* 检查（`k = left // min_cost`），否则前 k 个全用最低档也可能超（初始设计全最低档，所以只需 `k ≤ left/min_cost`）。
2. **批内一致性**：`min_full_fraction` 强制全保真点时，要预留其成本；先算 `reserved = n_full × cost(1.0)`，若 `left < reserved` 则**减少 n_full**（不要被迫超支）。
3. **返回空 vs 抛错**：`Study.ask` 目前假设返回非空；若 `ask_with_fidelity` 返回 `[]`，Study 应抛 `PendingError`/"cost budget exhausted"（明确错误，不要静默）。在 `Study.ask` 加：
   ```python
   if not pairs:
       raise ValueError("cost budget exhausted; increase --cost-budget or reduce batch")
   ```

---

## 3. 与 `Study` 的接线（3 行 + 1 个异常）

```python
# study.py ask()
pairs = (
    optimizer.ask_with_fidelity(n, remaining_cost=self.remaining_cost())
    if getattr(optimizer, "supports_fidelity", False)
    else [(cfg, 1.0) for cfg in optimizer.ask(n)]
)
if not pairs:
    raise ValueError("no affordable point: remaining_cost exhausted")
```
`remaining_cost()` 已在 v0.10.1 可用（有 `meta["cost_budget"]` 时）；无预算时返回 None → 行为与现在一致。

---

## 4. 5 条 pytest 验收

```python
def test_ask_with_fidelity_respects_remaining_cost():
    # cost_budget=2.0, batch=4, levels 0.25/0.5/1.0 → 总成本 ≤2.0（降档/减批）
def test_ask_with_fidelity_returns_empty_when_budget_exhausted():
    # remaining_cost=0.1 < min cost → 返回 []；Study.ask 抛明确 ValueError
def test_min_full_fraction_reserves_cost():
    # min_full_fraction=0.5, n=2 → 至少 1 个全保真；预算足够时不降档，预算不足时 n_full 减少
def test_mfbo_never_overshoots_cost_with_study():
    # Study.create(cost_budget=12); 跑到结束 → cost_spent ≤ 12*1.02（v0.8 实测 13.25/12 → 必须过）
def test_single_fidelity_unaffected():
    # bayes 的 ask_with_fidelity(3) 全 1.0；remaining_cost 参数被忽略
```

**benchmark 验收**：`mf_suzuki`、cost 预算 12、batch 2、2 seeds：`cost_spent ≤ 12×1.02`，且 `best_full_fidelity` 不低于 v0.10.1 的 95%（修复不应损失样本效率）。

---

## 5. 参考

R41（BOCA：成本感知）、R39/R40（预算调度）、R18（budget/cost 协议）、R32（成本口径）、`MFBO_COST_API.md`、`REVIEW_v11.md`。
