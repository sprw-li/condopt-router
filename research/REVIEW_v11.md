# REVIEW_v11 — 对 v0.10.1（全保真 cost=1.0 / CLI --cost-budget）的审查

> 审查者：algo-research ｜ 日期：2026-11-05 ｜ 快照：v0.10.1（`study.py:271-280,516-520`、`cli.py:173,367`、`mfbo.py:152-154`）
> 格式：**`文件:行` → 问题 → 文献依据 → 修复建议 → 优先级**。P0 = 正确性/结果不可信；P1 = 显著影响样本效率或与宣称不符；P2 = 工程质量。
> **已确认修好**：`_to_observation` 全保真观测默认 `cost=fidelity`（即 1.0，不再低估，`study.py:516-520`）；CLI `init --cost-budget` 写入 `meta["cost_budget"]`（`cli.py:173,367`）；`remaining_cost()` 有 `meta` 时返回 `budget − cost_spent`（`:275-280`）；141 tests 全绿。

---

## 0. 总结

| 编号 | 位置 | 一句话 | 优先级 |
|---|---|---|---|
| P0-1 | `mfbo.py:152-154` + `study.py:350` | `mfbo.ask_with_fidelity(n)` **不接收剩余成本**；Study 也不传 → `remaining_cost()` 有值但**没人用**，成本超支仍会发生（v0.8 实测 25.25/24，v0.10.1 未改） | **P0** |
| P1-1 | `study.py:63-100`（`create`） | `cost_budget` 只能经 CLI 的 `meta` 传入；**Python API `Study.create` 没有一等参数**，直接调用者容易漏（`remaining_cost()` 返回 None） | P1 |
| P1-2 | `calibration.py:174-224` + `benchmarks.run_algorithm` | records 仍无真噪声/MOO true-HV/约束/多保真指标（v0.10 遗留）：`collect_records(noise=...)` 只改 context；MOO 标签用 `best_score`；`mfbo` 被能力过滤跳过 | P1 |
| P1-3 | `calibration.py:84-85,211-214` | `regret` 仍用 `|best|` 归一（实测可 >1）；能力过滤"超集误杀"（`special ⊄ required`）未修，未来 constrained-MOO 会被排除 | P1 |
| P1-4 | `mfbo.py:205`（`best = _best_scalar()`） | incumbent 已用全保真 ✓，但**没有全保真观测时回退 `max(self._y)`（混保真）**；`_candidate_pool` local 中心仍用混保真 `_y`（`:274`） | P1 |
| P2-1 | `study.py:271,275` | `cost_spent()` 是方法、`remaining` 是 property；`report` 只显示 `cost_spent`，不显示 `cost_budget/remaining_cost` | P2 |
| P2-2 | `study.py:516-520` | 旧 study 的历史观测（v0.10.1 之前写入）`cost=0`，与新观测 `cost=1.0` 混合 → `cost_spent` 跨版本不一致；需 migration/标注 | P2 |
| P2-3 | `mfbo.py:154-166` | `ask()` 丢弃 fidelity（只在 `ask_with_fidelity` 返回）；直接调用 `ask()` 的脚本会把建议当全保真执行 | P2 |
| P2-4 | `router.py:807-813` | 正乘/负除 + `-1e9` 禁用（v0.8 遗留） | P2 |
| P2-5 | `qehvi/qnehvi` | `_acquisition` 仍未抽取 → 快 CI #2 无法写严 | P2 |

**总体**：v0.10.1 把"成本默认值"和"预算入口"补上了（会计终于能算对），但**没有任何组件消费预算**——`remaining_cost()` 是"只读的孤儿 API"。这是典型的"看起来闭环、实际未接线"：测试可以只测 `remaining_cost()` 返回值而全绿，但 `mfbo` 依然超支。下一步应优先做 `MFBO_COST_DOWNGRADE.md` 的接线，再谈正式校准。

---

## 1. P0 详述

### P0-1 `mfbo.py:152-154` — 预算未被消费

```python
def ask_with_fidelity(self, n: int = 1) -> list[tuple[dict, float]]:
    ...
    for index in range(n):
        config, fidelity = self._propose(out)     # 没有 remaining_cost 输入
```
- `Study.ask`（`study.py:350`）调用 `optimizer.ask_with_fidelity(n)`，没有传 `self.remaining_cost()`；
- `mfbo` 的内部 `_propose` 也不检查成本（`mfbo.py:195-243`）→ 超支不可检测；
- `Study.remaining()` 只按次数，`batch_size` 一次给 n 个点，可能一次就超成本预算。

**文献依据**：R41（BOCA：按成本选保真度是算法的一部分）、R39/R40（预算调度必须在选点前做）、R18（Olympus 的 budget/cost 协议）。

**修复**：见 `MFBO_COST_DOWNGRADE.md`（`ask_with_fidelity(n, remaining_cost=...)` + Study 传参 + optimizer 内部 `_fit_to_budget`/减批）。**验收**：`test_mfbo_never_overshoots_cost`（≤预算×1.02）。

---

## 2. P1 详述

### P1-1 `Study.create` 无 `cost_budget` 参数

CLI 可用 `--cost-budget`，但 Python API 只能 `meta={"cost_budget": ...}`；`create` 的签名里没有该参数，`remaining_cost()` 在直接调用时返回 `None`。修复：`create(..., cost_budget: float | None = None)` → `meta["cost_budget"] = cost_budget`；`Study.__init__` 同步。**依据**：R50（配置随 study 落盘/一等参数）、R17。

### P1-2 `collect_records`/`run_algorithm` 仍无真噪声/MOO/约束/MF

- `run_algorithm` 对 `problem.fn` 求值（无噪声）；
- MOO 的 `best_score` 是主目标；没有 true HV；
- 约束问题的 `feasible_best` 没有；
- `mfbo` 需要 `ask_with_fidelity` 的专用 runner，`collect_records` 走的是普通 `run_algorithm`（且能力过滤会跳过 mfbo，因为 required 不含 multi_fidelity）。
**修复**：见 `CALIBRATION_RECORDS_V2.md`（problem 元数据 + per-type runner + metric 字段 + context 推导）。

### P1-3 `regret` 归一化与能力超集误杀

`span = max(abs(best), 1e-9)` → 实测 `noise=high` 组 `mean_regret=1.28`（>1 无意义）；`special ⊄ required → skip` 会排除"能力超集"算法。修复：span 归一 + 显式互斥规则（qehvi/qnehvi 必须 MOO；mfbo 必须 multi_fidelity）。

### P1-4 `mfbo` 无全保真点时回退混保真

`_best_scalar`（`mfbo.py:250-256`）在 `full` 为空时回退 `max(self._y)` → 与 v0.9 问题相同，只是"有全保真时"正确。修复：无全保真时返回 `None` 并在 `_propose` 里先补一个全保真锚点（`return random, 1.0`）；`_candidate_pool` 的 local 中心改为"全保真 Top-k"。

---

## 3. P2（择机）

- **P2-1** `study.py`：`cost_spent` 统一 property；`report` 显示 `cost_spent/cost_budget/remaining_cost`。
- **P2-2** 旧 study 的 cost 混合：在 `dataset.jsonl` 里加 `schema_version`，或报告里标注"v0.10.1 前观测 cost=0"。
- **P2-3** `mfbo.ask()` 文档明确"仅返回条件；多保真链路由 `ask_with_fidelity`/Study 负责"。
- **P2-4** `router.py:807-813`：负分加性、禁用从 scores 删除。
- **P2-5** `qehvi/qnehvi`：抽 `_acquisition`。

---

## 4. 证据（本轮实测/代码核对）

| 检查 | 结果 |
|---|---|
| 全保真观测 cost | `study.py:516-520`：`cost = float(fidelity)`（1.0）✓ |
| CLI `--cost-budget` | `cli.py:173`：`meta={"cost_budget": args.cost_budget}`；`cli.py:367` 有参数 ✓ |
| `remaining_cost()` | `study.py:275-280`：有 meta 时 `budget − cost_spent` ✓；无 meta → None |
| mfbo 是否消费预算 | grep `mfbo.py` 仅 `ask_with_fidelity` 定义，无 `remaining_cost`/`cost_budget` → **未消费** ✗ |
| records 能力过滤 | `calibration.py:211-214` 仍 `special ⊄ required → skip`（v0.9 P1-4 未修） |
| regret 归一化 | `calibration.py:84-85` 仍 `max(abs(best), 1e-9)`（v0.9 P1-2 未修） |

---

## 5. 修复顺序

1. **P0-1**（预算接线）——`MFBO_COST_DOWNGRADE.md` 可直接照抄。
2. **P1-1**（API 一等参数）+ **P1-4**（mfbo 无全保真锚点）——小改动。
3. **P1-2**（records 扩展，`CALIBRATION_RECORDS_V2.md`）——正式 456-runs 前置。
4. **P1-3/P2**（regret、能力过滤、UI/报告）。

**回归门槛**：141+ tests；`test_mfbo_never_overshoots_cost`；`test_study_create_cost_budget`；`test_cost_spent_includes_full_fidelity`；`calibration` 的 `noise=high` 组 regret∈[0,1]。

---

## 6. 参考

R41（BOCA：成本感知采集）、R39/R40（预算调度）、R18（budget/cost 协议）、R51/R52（折外评估）、R54（HPO 综述：评估协议）、R50（配置落盘）、`REVIEW_v10.md`、`MFBO_COST_API.md`。
