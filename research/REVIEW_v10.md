# REVIEW_v10 — 对 v0.9.1–v0.10 的审查（mfbo incumbent / 成本默认 / calibration tie+noise context）

> 审查者：algo-research ｜ 日期：2026-11-05 ｜ 快照：v0.10（`mfbo.py` `_best_scalar`、`study.py::_to_observation` cost 默认、`calibration.py` tie/noise context）
> 格式：**`文件:行` → 问题 → 文献依据 → 修复建议 → 优先级**。P0 = 正确性/结果不可信；P1 = 显著影响样本效率或与宣称不符；P2 = 工程质量。
> 本轮实测：`collect_records(noise="none"/"high")` 两组核心子集（各 60 runs），数字见 `CALIBRATION_NOISE_RUN.md`。
> **已确认修好**：`mfbo._best_scalar()` 只用全保真 incumbent（`mfbo.py:250-256`，实测低保证"200"时返回 **100** ✓）；`Study._to_observation` 对 `fidelity<1` 默认 `cost=fidelity`（`study.py:516-520`）；calibration tie 计入 top1/top2（`calibration.py:73-82`）；`collect_records` 增加 `noise/smoothness/multi_fidelity` 参数（`:180-205`）。

---

## 0. 总结

| 编号 | 位置 | 一句话 | 优先级 |
|---|---|---|---|
| P0-1 | `study.py:275-280` + `mfbo.ask_with_fidelity` | `remaining_cost()` 在未设 `meta["cost_budget"]` 时返回 **None**，`Study.create`/CLI 无 `cost_budget` 入口，`mfbo` 也不消费 → **成本超支仍不会阻止**（v0.9 P0-2 只修了"默认 cost"，没修"预算/降档"） | **P0** |
| P0-2 | `study.py:516-520` | **全保真观测的 cost 默认 0**（`fidelity<1` 才记 fidelity）→ 多保真 study 的 `cost_spent` 系统性低估；若一次 run 以全保真为主，成本会计接近 0 | **P0** |
| P1-1 | `calibration.py:216-224` + `benchmarks.run_algorithm` | `collect_records(noise=...)` 只改 **context**，数据仍是 `run_algorithm` 的无噪标量评估；实测 `noise="high"` 组：in-sample default top1 **0.0**/regret **1.28** 而 LOPO 0.833 → 同一批数据在两种 context 下给出矛盾证据 | P1 |
| P1-2 | `calibration.py:84-85` | regret 仍用 `|best|` 归一（不是 span）→ 实测噪声组 `mean_regret=1.28`（>1，无意义） | P1 |
| P1-3 | `calibration.py:211-214` | "能力超集误杀"未修：`special ⊄ required → skip` 会排除未来的 constrained-MOO 算法；且 `multi_fidelity` 只能靠调用方手动传参，`mf_suzuki` 的 context 默认 `multi_fidelity=False` → mfbo 仍采不到 | P1 |
| P1-4 | `calibration.py:73-82` | tie 计入 top1 后语义变为"chosen 与最优并列即算对"；在并列多的离散问题上会抬高 top1（与"top-1 命中"不同口径），需在报告里写明 | P1 |
| P1-5 | `calibration.py:130-163` | LOPO 仅 6 折、每折训练 5 问题 → 折间权重不稳定（实测两组 context 的 tuned weights 不同：`{bayes1.444,...}` vs `{bayes1.318, sobol0.641,...}`），报告需给 min/median/max，权重向 1 正则 | P1 |
| P2-1 | `mfbo.py:252-256` | 无全保真观测时回退 `max(self._y)`（仍混保真）；`_candidate_pool` 的 local 中心仍用混合 `_y`（`:274`） | P2 |
| P2-2 | `study.py:271,275` | `cost_spent` 是方法、`remaining` 是 property；无 `cost_budget`；`report` 只报 spent，不报预算/剩余 | P2 |
| P2-3 | `calibration.py:42` | context schema 无版本校验；旧 records 缺字段会抛错 | P2 |
| P2-4 | `router.py:807-813` | 正乘/负除 + `-1e9` 禁用语义（v0.8 P1-3/P1-4）未改，权重校准仍可能利用不对称 | P2（遗留） |
| P2-5 | `qehvi/qnehvi` | `_acquisition` 抽取未做，快 CI #2 仍无法写严（`FAST_CI_SNIPPETS.md` §4 已有 10 行方案） | P2 |

**总体**：v0.10 把 v0.9 的两个最具体问题（mfbo incumbent、cost 默认）修了一半——incumbent 已正确、cheap run 的 cost 已正确，但**全保真 cost=0 + 无 cost_budget + mfbo 不消费**让"多保真预算"仍不可执行；校准侧 tie 修好了，但 **context-only noise 制造了"数据没变、证据反转"**的现象，说明校准必须用真实噪声数据（下一轮 `records` 需要 evaluator 噪声）。

---

## 1. P0 详述

### P0-1 `study.py:275-280` — `remaining_cost` 无默认预算、无人消费

```python
def remaining_cost(self) -> float | None:
    budget = self.meta.get("cost_budget")
    if budget is None:
        return None                 # ← 默认 None
    ...
```
- `Study.create` 没有 `cost_budget` 参数，CLI 也没有 `--cost-budget` → 默认永远 None；
- `mfbo.ask_with_fidelity`（`mfbo.py:149-166`）没有"剩余成本"参数 → 即使 `remaining_cost` 有值也不会降档；
- 实测：预算 cost≈24 的 mfbo run 仍会按次数发点（v0.8 实测 25.25/24，+5.2%；v0.10 未改该路径）。

**修复**（`MFBO_COST_API.md` §2.3 最小版）：
1. `Study.create(..., cost_budget=...)` → `meta["cost_budget"]`；缺省 = `budget × max(costs)`（等价全保真成本）；
2. `remaining_cost()` 无显式预算时用上面的派生值（不再 None）；
3. `ask_with_fidelity(n, remaining_cost=...)`：`_fit_to_budget(level, left)` 降档；batch 内先规划成本、超出减批；
4. `report` 显示 `cost_spent/cost_budget/remaining_cost`。

**验收**：`test_mfbo_never_overshoots_cost`（≤budget×1.02）、`test_remaining_cost_derived_default`。

### P0-2 `study.py:516-520` — 全保真 cost 默认 0

```python
cost = payload.get("cost")
if cost is None:
    # full-fidelity observations default to 0 (legacy behaviour)
    cost = float(fidelity) if float(fidelity) < 1.0 - 1e-9 else 0.0
```
单保真 study 里"cost 恒 0"是合理的（legacy）；但**多保真 study 里全保真点是最贵的（cost(1)=1）**，记 0 会让 `cost_spent` 只累加 cheap run，系统性低估（例如 10 个全保真 + 5 个 0.25 → 真实 11.25，记录 1.25）。实测：pending fidelity 0.25 时 cost=0.25 ✓，但 full 点会记 0。

**修复**：
```python
if self.objective.fidelity:          # 多保真 study：所有档位都按 cost 模型计
    costs = dict(zip(levels, costs_list))
    cost = float(payload.get("cost") or costs.get(fidelity, fidelity))
else:
    cost = float(payload.get("cost") or 0.0)   # 单保真 legacy
```
或从 `Objective.fidelity.costs` 查表（推荐）。**验收**：`test_cost_spent_includes_full_fidelity`（fidelity 0.25/1.0 → cost_spent=1.25）。

---

## 2. P1 详述

### P1-1 `collect_records(noise=...)` 只改 context，不改数据

`run_algorithm` 仍 `problem.evaluate(config)`（无噪声）→ records 的 `score` 是无噪的，而 context 的 `noise` 只影响 router 打分。实测两组（同一批无噪数据、同一 seeds）：

| context | labels | in-sample default | in-sample tuned | LOPO |
|---|---|---|---|---|
| `noise="none", smooth="smooth"` | bayes 5/6, lhs 1/6 | top1 0.833 / top2 0.833 / regret 0.0054 | 0.833 / 0.833 / 0.0019 | 0.667 / 0.833 / 0.0850 |
| `noise="high", smooth="rough"` | 同上 | **top1 0.0** / top2 0.833 / **regret 1.28** | 0.833 / 1.0 / 0.0054 | 0.833 / 0.833 / 0.0019 |

同一批数据因 context 不同产生**相反结论**，说明"噪声 context"不能被当作校准数据的一部分，除非 evaluator 真的加噪（或 per-problem 记录真实噪声水平）。**修复**：`BenchmarkProblem` 增加 `noise_std/n_replicates/tags`，`run_algorithm` 用它们生成带噪观测 + true 值；`collect_records` 从 problem 元数据推导 context（而不是全局参数）。

### P1-2 `calibration.py:84-85` — regret 归一化错误

`span = max(abs(best), 1e-9)`：`best` 接近 0（sphere2d 最优≈0）或为负时，regret 爆炸（实测 1.28）。应改为 `(best − chosen)/(best − worst + eps)`，与 `BENCHMARK_SPEC.md` §5.1 一致。

### P1-3 `calibration.py:211-214` — 超集误杀 + MF context 缺失

```python
special = capabilities & {"multi_objective", "constraints", "multi_fidelity"}
if special and not special <= required:
    continue
```
- 未来的 constrained-MOO 算法在 MOO context（required={"multi_objective"}）会被跳过；
- `mf_suzuki` 的 `required` 不含 `multi_fidelity`（未从 problem 推导）→ `mfbo` 永远被跳过（除非调用方显式 `multi_fidelity=True` 且对该问题其它算法也传 true）。
**修复**：`required` 从 problem 元数据推导（含 multi_fidelity）；过滤逻辑改为"算法声明了 context 不需要的**特殊**能力"仅对已知互斥项（qehvi/qnehvi 需 MOO）显式判断。

### P1-4 `calibration.py:73-82` — tie 计入 top1 的口径

`is_best = chosen_score >= best - 1e-9` → 并列最优即算 top1。这与"命中标签算法"不同；在离散/小空间（多算法同分）会显著抬高 top1（实测 `noise="none"` 组 top1 从旧口径 0.833→0.833 但 tie 折的 top1 由 0→1）。**修复**：报告拆 `top1_strict`（chosen == label）与 `top1_tie`（并列最优）；LOPO 主指标用 strict，tie 单独列。

### P1-5 `calibration.py:130-163` — LOPO 折间不稳定

6 折、每折训练 5 问题 → tuned weights 折间差异大（实测两组 context 全局 tuned：`{bayes1.444, lhs1.358, random1.021, sobol0.859, tpe1.111}` vs `{bayes1.318, lhs0.938, random1.221, sobol0.641, tpe0.764}`）。**修复**：权重向 1 正则 `w = 1 + λ(w_raw−1)`（λ=0.5）；报告 min/median/max；问题数 <10 时标注"仅 smoke"。

---

## 3. P2（择机）

- **P2-1** `mfbo.py:252-256`：无全保真点时回退混合 `max(_y)`；候选池 local 中心（`:274`）也用混保真 `_y` → 应用 `_best_scalar`/全保真点做中心。
- **P2-2** `study.py`：统一 `cost_spent` 为 property、补 `cost_budget`、report 显示预算/剩余。
- **P2-3** `calibration.py:42`：context schema version。
- **P2-4** `router.py:807-813`：负分加性、禁用从 scores 删除（v0.8 遗留）。
- **P2-5** `qehvi/qnehvi`：`_acquisition` 抽取（#2 快 CI 前置）。

---

## 4. 证据（本轮实测）

```
[noise=none smooth=smooth] runs=60 collect=3.83s
  in-sample default: top1 0.833 top2 0.833 regret 0.00544
  in-sample tuned  : top1 0.833 top2 0.833 regret 0.00190  weights={bayes1.444,lhs1.358,random1.021,sobol0.859,tpe1.111}
  LOPO             : top1 0.667 top2 0.833 regret 0.08495
[noise=high smooth=rough] runs=60 collect=3.75s
  in-sample default: top1 0.000 top2 0.833 regret 1.28256
  in-sample tuned  : top1 0.833 top2 1.000 regret 0.00544  weights={bayes1.318,lhs0.938,random1.221,sobol0.641,tpe0.764}
  LOPO             : top1 0.833 top2 0.833 regret 0.00190
mfbo _best_scalar (低保真 y=200, 全保真 y=100) = 100.0  ✓
Study 成本默认：fidelity 0.25 → cost 0.25 ✓；fidelity 1.0 → cost 0 ✗（P0-2）
remaining_cost（无 meta cost_budget）= None ✗（P0-1）
```

---

## 5. 修复顺序

1. **P0-2**（全保真 cost 从 fidelity 推导）→ **P0-1**（cost_budget/remaining_cost 默认 + mfbo 降档）——多保真结论的前提。
2. **P1-1**（真实噪声 records）→ **P1-3**（required 从 problem 推导）——正式 456-runs 前提。
3. **P1-2/P1-4/P1-5**（regret 归一化、tie 双口径、权重正则）——校准数字可信度。
4. P2 清理。

**回归门槛**：140 tests 全绿 + 新增 `test_cost_spent_includes_full_fidelity`、`test_mfbo_never_overshoots_cost`、`test_lopo_top1_strict_vs_tie`；`noise="high"` 组不再出现 regret>1。

---

## 6. 参考

R51/R52（折外评估/选择偏差）、R32（统计协议）、R41（BOCA/成本）、R12（qNEHVI）、R42（HV/真值）、R50（配置/权重落盘）、R8（场景/能力过滤）。
