# REVIEW_v14 — 对 v0.10.4（成本语义收尾：override / 软契约 / 审计）的审查

> 审查者：algo-research ｜ 日期：2026-11-05 ｜ 快照：v0.10.4（`mfbo.py:164-209`、`study.py:362,376-384`、`base.py:269`）
> 格式：**`文件:行` → 问题 → 文献依据 → 修复建议 → 优先级**。P0 = 正确性/结果不可信；P1 = 显著影响样本效率或与宣称不符；P2 = 工程质量。
> **已确认修好**：`ask_with_fidelity(n, cost_remaining=...)` call-level override（`mfbo.py:164-168`）；`last_cost_trimmed` 记录（`:208`）且 `Study.ask` 写 `cost_trim` 决策（`study.py:376-384`，含 requested/returned/trimmed）；`Optimizer.cost_remaining` 基类默认 None（`base.py:269`）；`min_full_fraction` 改为**软契约**（预算紧时允许降档，文档/测试标注）；`_cost` 统一；148 tests。

---

## 0. 总结

| 编号 | 位置 | 一句话 | 优先级 |
|---|---|---|---|
| P1-1 | `mfbo.py:124-126` | `_cost` 仍只用 `level**cost_exponent`，**忽略 `Objective.fidelity.costs` 表**（v0.10.3 P1-2 未变）→ 用户配置真实成本表不生效、无 warning | P1 |
| P1-2 | `mfbo.py:178-194` + 文档 | "软契约"已确认，但**没有为 `n_full` 预留成本**：预算紧时 `cap` 可能低于 1.0，强制全保真点全被降档；若文档承诺"至少 1 个全保真（预算允许时）"，逻辑应显式检查并记录 `reduced_full` | P1（文档/行为一致性） |
| P2-1 | `study.py:296-384` | `cost_trim` 决策的 `reason` 用 `f"…{n}…"`，而 `n` 是**本轮请求数**；`params` 里 `min_full_fraction` 等来自 `self.params`（Study 级）而非 optimizer 实际参数——审计信息可能不精确 | P2 |
| P2-2 | `mfbo.py:161-162` | `ask()` 仍丢弃 fidelity（直接调用者会按全保真执行） | P2 |
| P2-3 | `mfbo.py:211` | `_initial_design_left` 用 `self.t`（有效 tell 数）而非"已建议数"；trim 后 `t` 不变 → 下一轮可能继续发初始点 | P2 |
| P2-4 | `study.py:271,279` | `cost_spent()` 方法 vs `remaining` property；`report` 文本仍不显示成本预算（summary JSON 已有） | P2 |
| P2-5 | `study.py:516-520` | 旧 study（v0.10.1 前）观测 cost=0 与新观测 cost=1.0 混合 → `cost_spent` 跨版本不一致；无 schema_version | P2 |
| P2-6 | `qehvi/qnehvi` | `_acquisition` 尚未抽取（grep 无 `_acquisition`/`_greedy_batch`）→ 快 CI #2 仍不能写严；`QNEHVI_FAST_CI_FINAL.md` 已给 diff | P2 |
| P2-7 | `router.py:807-813` | 权重正乘/负除/`-1e9` 禁用（v0.8 遗留） | P2 |

**总体**：v0.10.4 把成本闭环的**行为契约**补齐了——call-level override 让"直接使用 optimizer"也能传预算，`cost_trim` 让裁剪可审计，`min_full_fraction` 明确为软契约。剩余唯一影响正确性的是 `_cost` 仍忽略 `costs` 表；其余是文档/审计精度/旧数据兼容的 P2。**成本闭环可以从本轮起视为"主路径完成"**，后续把精力转到快 CI 与 records v2。

---

## 1. P1 详述

### P1-1 `mfbo.py:124-126` — `costs` 表仍被忽略

```python
def _cost(self, level: float) -> float:
    return float(level) ** float(self.params["cost_exponent"])
```
`Objective.fidelity.costs`（若存在）不参与。两种口径不一致：用户配置 `costs=(0.1, 0.3, 1.0)` 时预算仍按 `level**exp` 计算 → 可能超支（若真实成本更高）或过度保守。**修复**（10 行）：
```python
def _cost(self, level: float) -> float:
    spec = self.objective.fidelity or {}
    costs = dict(zip(spec.get("levels", ()), spec.get("costs", ())))
    if level in costs:
        return float(costs[level])
    return float(level) ** float(self.params["cost_exponent"])
```
skill 文档写明优先级：`costs` 表 > `cost_exponent`。**依据**：R18（cost 协议由用户数据定义）、R41（BOCA：成本模型是采集的一部分）、R50（配置可追溯）。**验收**：`test_mfbo_cost_table_priority`。

### P1-2 `mfbo.py:178-194` — 软契约下的 `n_full` 预留

`min_full_fraction` 现在是软契约（预算紧时降档允许），但实现上**先强制全保真、后统一 cap**：如果 `cap < 1.0`，所有强制点都被降档，`n_full` 计算完全没起作用。若文档/测试宣称"预算允许时至少 1 个全保真"，应在 cap 前检查：
```python
full_cost = self._cost(1.0)
if budget is not None and full_cost <= budget:
    n_full = max(1, min(n_full, int(budget // full_cost)))   # 至少 1 个可负担
else:
    n_full = 0
```
并在 `last_cost_trimmed`/decisions 记录 `reduced_full`。**依据**：R41、R18。

---

## 2. P2（择机）

- **P2-1** `cost_trim` 决策的 payload 用 optimizer 实际参数（`optimizer.params`）与真实 `requested`（`last_requested`），而非 Study 级 dict。
- **P2-2** `ask()` 文档明确"仅单保真/内部使用"，或对多保真 study 抛 `NotImplementedError`。
- **P2-3** `_initial_design_left` 用"已建议数"（`len(self._X)` 或单独计数）而非 `t`。
- **P2-4** `cost_spent` 统一 property；`report` 文本显示 `cost_spent/cost_budget/remaining_cost`。
- **P2-5** 旧 study：dataset `schema_version` + report 标注"v0.10.1 前观测 cost=0"。
- **P2-6** qehvi/qnehvi 抽取（`QNEHVI_APPLY_PLAN.md` 给了应用顺序/回滚/验证）。
- **P2-7** router 权重语义。

---

## 3. 证据（代码核对）

| 检查 | 结果 |
|---|---|
| call-level override | `mfbo.py:164-168` ✓（`cost_remaining` 参数优先） |
| trim 审计 | `mfbo.py:208` `last_cost_trimmed`；`study.py:376-384` 写 `cost_trim` decisions ✓ |
| 基类默认 | `base.py:269` `self.cost_remaining = None` ✓ |
| 软契约 | `mfbo.py:178-194`（先强制后 cap）✓；`n_full` 预留缺失（P1-2） |
| `_cost` 表 | 仍 `level**cost_exponent`（P1-1） |
| `_acquisition` | grep 无 → 未落（P2-6） |

---

## 4. 修复顺序

1. **P1-1**（`costs` 表优先级 + skill 文档）——唯一影响预算正确性的点。
2. **P1-2**（`n_full` 可负担检查 + 决策记录）——软契约的行为一致性。
3. **P2-6**（qNEHVI 快 CI 应用，按 `QNEHVI_APPLY_PLAN.md`）。
4. P2 清理（审计 payload、report 成本文本、旧数据迁移、router 语义）。

**回归门槛**：148+ tests；`test_mfbo_cost_table_priority`、`test_cost_trim_decision_payload`、`test_min_full_fraction_soft_contract`。

---

## 5. 参考

R41（BOCA：成本/保真度决策）、R18（budget/cost 协议）、R39/R40（预算调度）、R50（配置/审计）、R57（Garnett：成本感知设计）、`REVIEW_v13.md`、`MFBO_COST_DOWNGRADE.md`。
