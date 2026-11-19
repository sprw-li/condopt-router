# REVIEW_v12 — 对 v0.10.2 成本闭环的审查

> 审查者：algo-research ｜ 日期：2026-11-05 ｜ 快照：v0.10.2（`study.py:115-121,275-284,352-372,602-625`、`mfbo.py:154-192`、`cli.py:173,367`）
> 格式：**`文件:行` → 问题 → 文献依据 → 修复建议 → 优先级**。P0 = 正确性/结果不可信；P1 = 显著影响样本效率或与宣称不符；P2 = 工程质量。
> **已确认闭环**：`Study.create(..., cost_budget=)` 一等参数（`study.py:115-121`）→ `meta["cost_budget"]`；CLI `--cost-budget`（`cli.py:173,367`）；`remaining_cost()` 有 `meta` 时返回；`Study.ask` 每轮把 `optimizer.cost_remaining = self.remaining_cost()`（`study.py:362`）；`mfbo.ask_with_fidelity` 降档（`cap`）、裁剪（trim）、耗尽/不可负担均抛 `ValueError`（`mfbo.py:171-191`）；summary 含 `cost_spent/remaining_cost`（`study.py:614-615`）；全保真观测 `cost=fidelity`（1.0）。

---

## 0. 总结

| 编号 | 位置 | 一句话 | 优先级 |
|---|---|---|---|
| P0-1 | `mfbo.py:176-183` | 降档/裁剪把 **`level` 当成本**（`level <= budget`、`spent + fidelity`），忽略 `cost_exponent` 与 `Objective.fidelity.costs` 表 → 成本模型不是"cost=s"时**预算会被违反或过度保守** | **P0** |
| P1-1 | `base.py::Optimizer` + `mfbo.py:171` | `self.cost_remaining` 只在 `Study.ask` 里动态赋值（`study.py:362`），**基类没有该属性**；直接 `registry.build("mfbo", ...); ask_with_fidelity()` 会 `AttributeError` | P1 |
| P1-2 | `mfbo.py:154-192` | `ask_with_fidelity` 仍**没有 `remaining_cost` 参数**（与 `MFBO_COST_DOWNGRADE.md` 的"构造参数 + call override"设计不一致）；且 `ask()` 丢弃 fidelity（直接调用者会按全保真执行） | P1 |
| P1-3 | `mfbo.py:163-170` | `min_full_fraction` 在预算约束前先强制 `n_full` 个全保真，随后降档 `min(fidelity, cap)` 可能把全保真也降掉 → **强制比例名存实亡**；且没有为 `n_full` 预留成本 | P1 |
| P1-4 | `study.py:371-374` | `ask_with_fidelity` 可能返回**少于 n 个**（裁剪）或抛错；`Study.ask` 没有 `len(pairs) < n` 的显式处理/日志（预算紧时用户看到"建议 2 个"却不知道被裁剪） | P1 |
| P2-1 | `study.py:271,279` | `cost_spent()` 是方法、`remaining` 是 property；summary 里已给出 cost 字段，但 `report` 文本只显示次数预算，不显示成本预算 | P2 |
| P2-2 | `study.py:362` | `cost_remaining` 是"外部注入属性"，不在 `Optimizer.state_dict`/文档中；重启/直接构建时行为不同 | P2 |
| P2-3 | `study.py:516-520` | 旧 study（v0.10.1 前）观测 `cost=0` 与新观测 `cost=1.0` 混合 → `cost_spent` 跨版本不一致；无 schema_version/迁移 | P2 |
| P2-4 | `mfbo.py:196-198` | `_initial_design_left` 用 `self.t`（有效 tell 数）而非"已建议数"，与 `_init_queue` 的按需切片叠加时可能多发/少发初始点（预算紧时被 trim 后 `t` 不变，下一轮继续发） | P2 |
| P2-5 | `qehvi/qnehvi` | `_model_ask` 里拟合/采集/贪心混在一起，快 CI #2 无法直接测（见 `QNEHVI_FAST_CI_PATCH.md`） | P2 |
| P2-6 | `router.py:807-813` | 权重正乘/负除 + `-1e9` 禁用（v0.8 遗留，仍未改） | P2 |

**总体**：v0.10.2 把"成本闭环"的**主路径**打通了（API/CLI/注入/降档/裁剪/报错/汇总），这是本轮最实质的进展。剩余风险集中在**两个细节**：①成本模型被简化为 `cost=fidelity`（与 `cost_exponent`/`costs` 表不一致时预算失真）；②`cost_remaining` 是外部注入属性而非一等状态，直接使用 optimizer 的路径会崩。两者都是 1 小时内可修的"接线级"问题。

---

## 1. P0 详述

### P0-1 `mfbo.py:176-183` — 用 level 当成本

```python
affordable = [level for level in levels if level <= budget]   # ← 假设 cost(level)=level
cap = affordable[-1] if affordable else levels[0]
out = [(config, min(fidelity, cap)) for config, fidelity in out]
...
if spent + fidelity <= budget + 1e-12:                        # ← 同样假设
```
而 `_propose` 的成本感知用的是 `costs = level ** cost_exponent`（`mfbo.py:223-226`），且 `Objective.fidelity.costs` 可能给出非线性的真实成本。当 `cost_exponent != 1.0`（如 1.0 以外）或 `costs` 表存在时：
- 若真实成本 > level（exp>1）→ 预算被违反（实际花费超过 `cost_budget`）；
- 若真实成本 < level → 过度保守（少发点）。

**文献依据**：R41（BOCA：成本模型必须与采集函数一致）、R18（cost 协议）、R32（成本口径）。

**修复**：统一 `self._cost(level)`（从 `Objective.fidelity.costs` 查表，回退 `level**cost_exponent`），所有预算判断用它：
```python
affordable = [lv for lv in levels if self._cost(lv) <= budget]
cap = affordable[-1] if affordable else levels[0]
spent += self._cost(fidelity)
```
**验收**：`test_mfbo_budget_with_nonlinear_cost`（cost_exponent=2、budget=1.5 → 总成本 ≤1.5）；`test_mfbo_cost_table_respected`。

---

## 2. P1 详述

### P1-1 `mfbo.py:171` — `cost_remaining` 未定义在基类

`self.cost_remaining` 由 `Study.ask`（`study.py:362`）动态赋值；`Optimizer.__init__` 没有该属性（grep 无定义）→ 直接 `registry.build("mfbo", ...).ask_with_fidelity(2)` 会 `AttributeError`。**修复**：`Optimizer.__init__(..., cost_remaining: float | None = None)` 设默认 None，并在 `registry.build` 透传；`Study` 的赋值保留（每轮更新）。**验收**：`test_mfbo_direct_use_without_study`。

### P1-2 `ask_with_fidelity` 无 call override；`ask()` 丢弃 fidelity

- 设计文档推荐"构造参数 + call override"，实现只有 `self.cost_remaining`（无参数）；
- `ask()`（`mfbo.py:151-152`）返回 configs、丢弃 fidelity；直接调用 `opt.ask(4)` 的脚本会按全保真评估 → 成本/保真度不一致。
**修复**：`ask_with_fidelity(n, remaining_cost=None)`，`remaining_cost` 优先于 `self.cost_remaining`；`ask()` 文档明确"仅单保真场景/内部使用"，或让它抛 `NotImplementedError`（避免误用）。

### P1-3 `min_full_fraction` 的强制比例被降档破坏

```python
if index < n_full: fidelity = 1.0
...
out = [(config, min(fidelity, cap)) ...]   # 全保真可能被降成 cap
```
预算不足时"至少 1 个全保真"的保证消失，但参数名暗示有保证。**修复**：先为 `n_full` 预留成本（`reserved = n_full * cost(1.0)`），若 `budget < reserved` 则 `n_full = floor(budget / cost(1.0))`，并在 `decisions` 记录"reduced n_full"; 剩余预算再分配降档。**验收**：`test_min_full_fraction_respects_budget`。

### P1-4 `Study.ask` 未记录裁剪

`ask_with_fidelity` 可能返回 `< n`（trim）或抛错。`Study.ask` 直接构建 pending，没有 warning/decisions 记录"本批被裁剪到 k 个"。**修复**：若 `len(pairs) < n`，写 `decisions`（`action="cost_trim"`, payload={requested, returned, remaining_cost}）并在 CLI 输出提示；若 `pairs==[]` 抛明确错误（已有 `ValueError` 分支）。

---

## 3. P2（择机）

- **P2-1** `study.py`：`cost_spent` 统一 property；report 文本显示 `cost_spent/cost_budget/remaining_cost`（summary 已有 JSON 字段，文本层未用）。
- **P2-2** `cost_remaining` 进入 `Optimizer` 文档/`state_dict`（若重启需保持）。
- **P2-3** 旧 study 兼容：`dataset.jsonl` 加 `schema_version`；report 标注"v0.10.1 前观测 cost=0"。
- **P2-4** `_initial_design_left` 与 `_init_queue` 的一致性（用"已建议数"而非 `t`）。
- **P2-5** `qehvi/qnehvi` 的 `_acquisition` 抽取（`QNEHVI_FAST_CI_PATCH.md` 给了 diff）。
- **P2-6** `router.py` 权重语义（负分加性、禁用从 scores 删除）。

---

## 4. 证据（代码核对）

| 检查 | 结果 |
|---|---|
| `Study.create(cost_budget=)` | `study.py:115-121` ✓ |
| CLI `--cost-budget` | `cli.py:173,367` ✓ |
| 每轮注入 | `study.py:362` `optimizer.cost_remaining = self.remaining_cost()` ✓ |
| mfbo 降档/裁剪/报错 | `mfbo.py:171-191` ✓；但用 `level` 当成本（P0-1） |
| summary 成本字段 | `study.py:614-615` ✓ |
| 全保真 cost | `study.py:516-520` `cost=float(fidelity)` ✓ |
| 直接使用 optimizer | `self.cost_remaining` 未在基类定义 → `AttributeError`（P1-1） |

---

## 5. 修复顺序

1. **P0-1**（`_cost()` 统一）——影响预算正确性，10 行。
2. **P1-1/P1-2**（基类属性 + call override + `ask()` 防误用）。
3. **P1-3/P1-4**（n_full 预留、裁剪记录）。
4. **P2-5**（qehvi/qnehvi 抽取，配合快 CI）。
5. P2 清理。

**回归门槛**：144+ tests；`test_mfbo_budget_with_nonlinear_cost`、`test_mfbo_direct_use_without_study`、`test_min_full_fraction_respects_budget`、`test_study_records_cost_trim`；`mf_suzuki` 成本超支 ≤2%。

---

## 6. 参考

R41（BOCA：成本感知采集）、R18（budget/cost 协议）、R39/R40（预算调度）、R32（成本口径）、R54（HPO 评估协议）、`REVIEW_v11.md`、`MFBO_COST_DOWNGRADE.md`。
