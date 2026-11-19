# REVIEW_v15 — 对 v0.10.5（`fidelity_costs` 表 / 全保真计数审计）的审查

> 审查者：algo-research ｜ 日期：2026-11-05 ｜ 快照：v0.10.5（`mfbo.py:133-151,183-230`、`study.py:376-390`）
> 格式：**`文件:行` → 问题 → 文献依据 → 修复建议 → 优先级**。P0 = 正确性/结果不可信；P1 = 显著影响样本效率或与宣称不符；P2 = 工程质量。
> **已确认修好**：`_cost` 支持 `fidelity_costs` 表（`mfbo.py:133-151`，按最近 level 查表，回退 `level**cost_exponent`）；`last_full_requested/returned` 记录（`:114-115,228-229`）并写入 `cost_trim` 决策（`study.py:383-389`）；149 tests。

---

## 0. 总结

| 编号 | 位置 | 一句话 | 优先级 |
|---|---|---|---|
| P1-1 | `mfbo.py:148-151` | 成本表用 **最近 level** 匹配（`min(table, key=abs)`）：表里缺某档时会**静默取到别档的成本**（如只配 `0.5:0.5,1.0:1.0`，`0.25` 会按 0.5 的成本算）；无 warning、无校验 | P1 |
| P1-2 | `mfbo.py:139-151` vs `Objective.fidelity.costs` | `fidelity_costs` 只从 **params 字符串**读（`"0.25:0.25,1:1"`），**`Objective.fidelity.costs`（spec）仍被忽略**；两处配置来源无优先级/冲突提示 | P1 |
| P1-3 | `mfbo.py:197-213` + `study.py:383-389` | `min_full_fraction` 软契约的审计已补（requested/returned），但**没有为 `n_full` 预留成本**：`cap<1.0` 时强制全保真点全部降档，`full_returned=0` 只能"事后解释"，不能"事前保证（预算允许时至少 1 个）" | P1（软契约边界） |
| P2-1 | `mfbo.py:140-147` | 表解析静默跳过非法 chunk / 缺失档位；无 `warnings.warn`，错误配置难发现 | P2 |
| P2-2 | `study.py:378-389` | `cost_trim` 的 `reason` 用 Study 级 `params`（`self.params`），非 optimizer 实际参数；审计 payload 的 `requested/returned` 已是精确值 ✓ | P2 |
| P2-3 | `mfbo.py:180-181` | `ask()` 仍丢弃 fidelity（直接调用者会按全保真执行） | P2 |
| P2-4 | `mfbo.py:232` | `_initial_design_left` 用 `self.t`（有效 tell 数）而非已建议数；trim 后 `t` 不变 → 下轮可能继续发初始点 | P2 |
| P2-5 | `study.py:271,279` | `cost_spent()` 方法 vs `remaining` property；`report` 文本仍不显示成本预算 | P2 |
| P2-6 | `study.py:516-520` | 旧 study（v0.10.1 前）观测 cost=0 与新观测 cost=1.0 混合 → 跨版本 cost 不一致 | P2 |
| P2-7 | `qehvi/qnehvi` | `_acquisition` 仍未抽取（`QNEHVI_ACQUISITION_EXTRACT.md` 给了逐行替换清单） | P2 |
| P2-8 | `router.py:807-813` | 权重正乘/负除/`-1e9` 禁用（v0.8 遗留） | P2 |

**总体**：v0.10.5 补齐了成本闭环的最后两块拼图——**可配置成本表**与**全保真计数审计**，行为面基本完整。剩余风险集中在"成本表匹配语义"（最近档 vs 精确档、params vs objective spec）与"软契约的事前保证"。这两个都是口径问题，不是算法问题；修法各 10–15 行。

---

## 1. P1 详述

### P1-1 `mfbo.py:148-151` — 最近 level 匹配会静默错配

```python
if table:
    nearest = min(table, key=lambda candidate: abs(candidate - float(level)))
    return float(table[nearest])
return float(level) ** float(self.params["cost_exponent"])
```
若 `fidelity_levels="0.25,0.5,1.0"` 但 `fidelity_costs="0.5:0.5,1.0:1.0"`（漏了 0.25），`_cost(0.25)` 返回 0.5 → 预算被**系统性高估**（少发点）或低估（看方向）。**修复**：
```python
def _cost(self, level: float) -> float:
    table = self._cost_table()          # 解析一次，缓存
    if table:
        key = min(table, key=lambda k: abs(k - level)) if level not in table else level
        if abs(key - level) <= 1e-9:
            return table[key]
        warnings.warn(f"fidelity {level} missing from fidelity_costs; using {level**exp}",
                      RuntimeWarning, stacklevel=2)
    return level ** float(self.params["cost_exponent"])
```
并加 `test_cost_table_exact_match`、`test_cost_table_missing_level_warns`。

### P1-2 `mfbo.py:139` vs `Objective.fidelity.costs` — 两个来源无优先级

`_cost` 只读 `params["fidelity_costs"]`；`Objective.fidelity` 的 `costs` 字段（spec 里存在）完全不参与。用户若在 objective JSON 里配了 costs，会以为生效但实际按公式算。**修复**（建议）：优先级 **params 表 > objective.fidelity.costs > exponent**，并在 `_cost` 里三者合并 + 文档写明；或直接删除 objective 的 `costs` 字段用法（避免歧义）。

### P1-3 `mfbo.py:197-213` — 软契约的事前保证

`n_full` 在 cap 前决定，`cap<1` 时全部降档；`last_full_returned=0` 只是"事后记录"。若文档/验收宣称"预算允许时至少 1 个全保真"，应改成：
```python
full_cost = self._cost(1.0)
if budget is not None:
    n_full = min(n_full, int(budget // max(full_cost, 1e-12))) if full_cost <= budget else 0
```
（即"能负担才承诺"），并把 `n_full` 写进 `last_full_requested`（已有）。若维持"纯软"语义，则文档要明确"不保证全保真点存在"。

---

## 2. P2（择机）

- **P2-1** 表解析：非法 chunk/缺失档位 `warnings.warn`；`fidelity_levels` 与表键一致性校验。
- **P2-2** `cost_trim` 的 `reason`/`params` 用 optimizer 实际参数（`optimizer.params`）。
- **P2-3** `mfbo.ask()` 文档明确"仅单保真/内部使用"。
- **P2-4** `_initial_design_left` 用已建议数。
- **P2-5** `cost_spent` property 化；report 显示成本预算。
- **P2-6** 旧 study `schema_version` + report 标注。
- **P2-7** qehvi/qnehvi 抽取（逐行清单见 `QNEHVI_ACQUISITION_EXTRACT.md`）。
- **P2-8** router 权重语义。

---

## 3. 证据（代码核对）

| 检查 | 结果 |
|---|---|
| 成本表 | `mfbo.py:133-151`：`fidelity_costs` 解析 + 最近档 + 回退公式 ✓ |
| 全保真计数 | `:114-115` 初始化；`:228-229` 记录 ✓ |
| 审计 payload | `study.py:383-389` 含 `full_requested/full_returned` ✓ |
| 最近档风险 | `min(table, key=abs)` 无精确匹配校验（P1-1） |
| objective costs | 未被 `_cost` 使用（P1-2） |
| `_acquisition` | 仍无（P2-7） |

---

## 4. 修复顺序

1. **P1-1**（精确匹配 + warning）——成本表正确性。
2. **P1-2**（优先级/文档）——口径一致性。
3. **P1-3**（软契约事前保证或文档明确）——验收口径。
4. **P2-7**（qNEHVI 应用，按 `QNEHVI_ACQUISITION_EXTRACT.md`）与 P2 清理。

**回归门槛**：149+ tests；`test_cost_table_exact_match`、`test_cost_table_missing_level_warns`、`test_min_full_fraction_soft_contract`（已存在，补"预算足够时 full_returned≥1"）。

---

## 5. 参考

R41（BOCA：成本模型即采集一部分）、R18（budget/cost 协议）、R50（配置/审计）、R57（Garnett：成本感知设计）、R58（Frazier：多保真/成本教程）、`REVIEW_v14.md`、`MFBO_COST_DOWNGRADE.md`。
