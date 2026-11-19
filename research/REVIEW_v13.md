# REVIEW_v13 — 对 v0.10.3（统一成本模型 / skill 文档同步）的审查

> 审查者：algo-research ｜ 日期：2026-11-05 ｜ 快照：v0.10.3（`mfbo.py:124-126,162-196,255`、`base.py:269`、`study.py:115-121,280,362,371-395`、`skill/references/JSON_SCHEMAS.md:108`）
> 格式：**`文件:行` → 问题 → 文献依据 → 修复建议 → 优先级**。P0 = 正确性/结果不可信；P1 = 显著影响样本效率或与宣称不符；P2 = 工程质量。
> **已确认修好**：`mfbo._cost(level)=level**cost_exponent`（`mfbo.py:124-126`）统一用于预算判断（`:180-187`）与采集（`:255`）；`Optimizer.cost_remaining` 进基类（`base.py:269`，默认 None）；skill `JSON_SCHEMAS.md:108` 已写 `--cost-budget`/`cost_spent/remaining_cost` 口径；146 tests。

---

## 0. 总结

| 编号 | 位置 | 一句话 | 优先级 |
|---|---|---|---|
| P1-1 | `mfbo.py:167-183` | `min_full_fraction` 先强制 `n_full` 个全保真、再被 `min(fidelity, cap)` 降档 → **强制比例在预算紧时名存实亡**；且没有为 `n_full` 预留成本 | P1 |
| P1-2 | `mfbo.py:124-126` vs `base.Objective.fidelity.costs` | `_cost` 只用 `level**cost_exponent`，**完全忽略 `Objective.fidelity.costs` 表**（`MFBO_COST_API.md` 曾把 `costs` 作为"更精确"的选项）→ 配置了真实成本表也不生效，且无 warning | P1 |
| P1-3 | `mfbo.py:154` + `base.py:269` | `ask_with_fidelity` 仍**无 `remaining_cost` call override**；直接使用 optimizer（不经 Study）时 `cost_remaining=None` → **预算静默失效**（不报错、不约束） | P1 |
| P1-4 | `study.py:371-395` | `ask_with_fidelity` 可能返回 `< n`（trim）或抛错；`Study.ask` 不记录裁剪（无 decisions/warning），用户看到"建议变少"却不知原因 | P1 |
| P2-1 | `study.py:271,279` | `cost_spent()` 方法 vs `remaining` property；`report` 文本不显示成本预算（summary JSON 已有） | P2 |
| P2-2 | `study.py:516-520` | 旧 study（v0.10.1 前）观测 cost=0 与新观测 cost=1.0 混合 → `cost_spent` 跨版本不一致 | P2 |
| P2-3 | `mfbo.py:151-152` | `ask()` 丢弃 fidelity（直接调用者会按全保真执行） | P2 |
| P2-4 | `mfbo.py:199-202` | `_initial_design_left` 用 `self.t`（有效 tell 数）而非"已建议数"；与 `_init_queue` 按需切片叠加时，预算 trim 后 `t` 不变 → 下一轮可能继续发初始点 | P2 |
| P2-5 | `qehvi/qnehvi` | `_acquisition` 尚未抽取（`QNEHVI_FAST_CI_FINAL.md` 给了最终 diff） | P2 |
| P2-6 | `router.py:807-813` | 权重正乘/负除/`-1e9` 禁用（v0.8 遗留） | P2 |

**总体**：v0.10.3 把"成本模型"统一到 `_cost()`，这是本轮最关键的修复——预算判断与采集函数终于用同一套成本；`cost_remaining` 进基类也消除了"直接使用 optimizer 崩"的问题。剩余风险集中在**两个语义未定义点**：①`min_full_fraction` 在预算约束下的含义（硬保证 vs 软偏好）；②`Objective.fidelity.costs` 与 `cost_exponent` 的优先级（现在前者被忽略）。这两项不需要改算法，只需明确契约并写入 skill 文档。

---

## 1. P1 详述

### P1-1 `mfbo.py:167-183` — `min_full_fraction` 的预算语义

```python
n_full = max(1, ceil(min_full_fraction * n))       # 先决定
for index in range(n):
    ...
    if index < n_full: fidelity = 1.0               # 强制全保真
...
cap = affordable[-1] if affordable else levels[0]
out = [(config, min(fidelity, cap)) for ...]        # 全保真可能被降成 cap
```
预算紧时 `cap < 1.0` → 强制全保真失效；但参数名/文档暗示"至少有一个全保真点"。**修复（二选一，建议 A）**：

```python
# A) 硬保证 + 预留成本
reserved = n_full * self._cost(1.0)
if budget is not None and reserved > budget:
    n_full = max(0, int(budget // max(self._cost(1.0), 1e-12)))
    # 记录 decisions: reduced_n_full
# 分配时先放 n_full 个全保真，再用剩余预算放其余点
```
```python
# B) 软偏好：重命名为 `prefer_full_fraction`，预算不足时允许降档，并在 report 说明
```
**依据**：R41（BOCA：保真度选择与成本约束是同一决策）、R18（budget/cost 协议）。

### P1-2 `mfbo.py:124-126` — `costs` 表被忽略

```python
def _cost(self, level):
    return float(level) ** float(self.params["cost_exponent"])
```
`Objective.fidelity.costs`（若配置）完全不参与。两种口径不一致会让用户困惑（"我配置了真实成本表，为什么预算按公式算"）。**修复（建议）**：
```python
def _cost(self, level: float) -> float:
    spec = self.objective.fidelity or {}
    costs = dict(zip(spec.get("levels", ()), spec.get("costs", ())))
    if level in costs:
        return float(costs[level])
    return float(level) ** float(self.params["cost_exponent"])
```
并在 skill 文档写明优先级：`costs` 表 > `cost_exponent` 公式。**依据**：R18（cost 协议应由用户数据定义）、R50（配置应可追溯）。

### P1-3 `mfbo.py:154` — 无 call override；直接使用预算静默失效

`cost_remaining` 默认 None（`base.py:269`）→ 直接 `registry.build("mfbo",...)` 后 `ask_with_fidelity` 不做任何预算约束，也不 warning。**修复**：`ask_with_fidelity(n, remaining_cost=None)`，`remaining_cost` 优先；若 `cost_remaining is None` 且 `remaining_cost is None` → 不约束（文档写明），但 `Study` 路径永远传值。**依据**：R50（配置/预算显式化）。

### P1-4 `study.py:371-395` — 裁剪不记录

`pairs` 可能少于 `n`。**修复**：`Study.ask` 里
```python
if len(pairs) < n:
    self.log_decision(action="cost_trim", algorithm=algorithm,
                      payload={"requested": n, "returned": len(pairs),
                               "remaining_cost": self.remaining_cost()})
```
并在 CLI 输出 `trimmed to k due to cost budget`。**依据**：R50（审计）、R32（可复现）。

---

## 2. P2（择机）

- **P2-1** `cost_spent` 统一 property；`report` 文本显示 `cost_spent/cost_budget/remaining_cost`（summary JSON 已有，文本层未用）。
- **P2-2** 旧 study cost 迁移：dataset `schema_version` + report 标注。
- **P2-3** `mfbo.ask()` 文档明确"仅单保真/内部使用"，或对多保真 study 抛 `NotImplementedError`。
- **P2-4** `_initial_design_left` 与 `_init_queue` 的一致性（用已建议数）。
- **P2-5** qehvi/qnehvi `_acquisition` 抽取（最终 diff 见 `QNEHVI_FAST_CI_FINAL.md`）。
- **P2-6** router 权重语义（负分加性、禁用从 scores 删除）。

---

## 3. 证据（代码核对）

| 检查 | 结果 |
|---|---|
| `_cost` 统一 | `mfbo.py:124-126`；用于 `:180,187,255` ✓ |
| `cost_remaining` 基类 | `base.py:269` 默认 None ✓ |
| Study 注入 | `study.py:362` 每轮赋值 ✓ |
| skill 文档 | `JSON_SCHEMAS.md:108` 写明 `--cost-budget`/`cost_spent/remaining_cost` ✓ |
| `min_full_fraction` | `:167-183` 先强制后降档（P1-1） |
| `costs` 表 | 未被 `_cost` 使用（P1-2） |
| trim 记录 | `study.py:371-395` 无 decisions（P1-4） |

---

## 4. 修复顺序

1. **P1-2**（`costs` 表优先级 + skill 文档）——口径一致性，10 行。
2. **P1-1**（`min_full_fraction` 预留成本/软偏好）——多保真承诺。
3. **P1-3/P1-4**（call override、trim 记录）——可审计。
4. **P2-5**（qNEHVI 快 CI 补丁）与 P2 清理。

**回归门槛**：146+ tests；`test_mfbo_cost_table_priority`、`test_min_full_fraction_respects_budget`、`test_study_logs_cost_trim`；`mf_suzuki` 超支 ≤2%。

---

## 5. 参考

R41（BOCA：保真度/成本同一决策）、R18（budget/cost 协议）、R39/R40（预算调度）、R50（配置/审计）、R32（可复现）、R57（Garnett：BO 的成本/采集设计）、`REVIEW_v12.md`、`MFBO_COST_DOWNGRADE.md`。
