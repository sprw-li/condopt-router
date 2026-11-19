# REVIEW_v08 — 对 v0.7.1–v0.8（钉住/权重/真值HV/mfbo成本/校准 harness）的审查

> 审查者：algo-research ｜ 日期：2026-11-05 ｜ 快照：18:18–19:02（`study.py` 28.0KB、`router.py` 35.7KB、`calibration.py` 155 行、`scripts/calibrate_router.py` 85 行、`mfbo.py` 12.8KB、`benchmarks.py` 23.0KB）
> 格式：**`文件:行` → 问题 → 文献依据 → 修复建议 → 优先级**。P0 = 正确性/结果不可信；P1 = 显著影响样本效率或与宣称不符；P2 = 工程质量。
> **已确认修好（v0.7→v0.8）**：①显式算法钉住：`Study.create(algorithm=...)` 走 `manual_selection` 并置 `manual_override`（`study.py:149-158`），`config.json` 持久化、`Study.load` 恢复（`:196,222`），`resume_auto_route()` 可交还（`:536`）；②权重边界：正分乘、负分除、0→`-1e9` 禁用、NaN/负值拒绝（`router.py:48-51,806-813`）；③能力过滤失败时有显式 warning（`router.py:835-840`）；④mfbo 初始设计全用最低档（`mfbo.py:189`）。

---

## 0. 总结

| 编号 | 位置 | 一句话 | 优先级 |
|---|---|---|---|
| P0-1 | `calibration.py:46-119` | **校准没有 LOPO**：`evaluate_weights` 用同一批 records 既生成标签又评估，`random_search_weights` 也在同一批上选权重 → `top1 0.25→0.50` 是**样本内**数字；正式 460-runs 若直接沿用会系统性高估 | **P0** |
| P0-2 | `benchmarks.py:495-540`（`run_algorithm`）+ `calibration.py:130-155` | 通用 benchmark runner 直接对 `problem.fn` 求值（无噪声、只支持标量主目标），因此 `collect_records` **无法产生带噪/多目标/约束的 records**；ML 口径的 `best_score` 不是 HV。用它校准会在 MOO/噪声场景给出错误标签 | **P0** |
| P1-1 | `calibration.py:136-141` | `collect_records` 的 context 用 `from_space(problem.space, budget, batch, n_objectives)`，**没带 noise/smoothness/constraints/multi_fidelity** → 所有问题都被当成 `noise="medium"`、`generic`；router 的噪声相关规则在校准中失效 | P1 |
| P1-2 | `calibration.py:122-155` | `collect_records` 不按能力过滤算法：对单目标问题传 `qehvi/qnehvi` 会在 `registry.build` 抛错（m≠2），整个采集中断；脚本的 `DEFAULT_ALGORITHMS` 因此只能保守排除它们 | P1 |
| P1-3 | `router.py:801-813` | 负分算法"除 factor"的语义与正分"乘 factor"相反：`factor>1` 对负分是**减轻惩罚**、`factor<1` 是**加重惩罚**；校准脚本对两者统一搜 `[0.6,1.6]`，会利用这个不对称（尤其当模型会给某些算法恒定负分，如单目标场景的 parego=-165） | P1 |
| P1-4 | `router.py:807-808` | `factor<=0` 把分数置 `-1e9` 而不是**移除**：`candidate_scores` 里出现 `-1e9`，`alternatives`/`confidence`/报告都受影响（margin 变成 1e9，confidence 恒 high） | P1 |
| P1-5 | `calibration.py:31-39,55-60,73-77` | 标签用**算术平均**而非 IQM；regret 用 `|best|` 归一（`best` 接近 0 时爆炸），不随问题难度缩放；`top2` 只从 `alternatives`（2 个）取，等价 top-3 | P2/P1 |
| P1-6 | `scripts/calibrate_router.py:25-33` | 默认 problems/algorithms 只有 6 个单目标问题 × 6 个算法（且**不含** parego/qehvi/qnehvi/constrained/mfbo）→ 校准只覆盖采样器 vs bayes/tpe，router 实际最需要校准的多目标/约束分支完全没有样本 | P1 |
| P1-7 | `router.py`（全局 `_ALGORITHM_WEIGHTS`）+ `cli.py` | 权重仍是**进程级全局**，只有 `route`/`init` 子命令加载；`ask/tell/report` 不应用权重。Study 不把生效权重写进 `config.json` → 同一 study 不同命令行为不一致 | P1 |
| P2-1 | `router.py:801` | `ranked = sorted(...)` 在权重循环前计算、随后被 `:814` 重排覆盖 → 死代码 | P2 |
| P2-2 | `calibration.py:89-119` | `random_search_weights` 对"records 里出现过但 router 从不选"的算法也分配权重（无效果但污染 JSON）；未记录搜索种子/迭代/折外指标到输出 schema | P2 |
| P2-3 | `calibration.py:42-43` | `_context(record)` 用 `from_dict` 重建，若 records.json 里 context 字段缺失/旧版会抛错；建议 schema version 校验 | P2 |
| P2-4 | `study.py:156` | `manual_selection` 只在 `algorithm` 非 auto 时设置；`algorithm="AUTO"/" auto "` 大小写/空格未规范化 → 会被当成显式算法钉住（或抛未知算法） | P2 |
| P2-5 | `mfbo.py:205` | `best = max(self._y)` 仍混用保真度（v0.6 P1-4 未改）；初始设计已修但仍无"全保真点/批"约束 | P1（并入 `MFBO_COST_API.md`） |
| P2-6 | `benchmarks.py:191-210` | `cost_aware_best` 的 "prefix value-per-cost" 语义需明确（v0.8 文档字符串已改，但公式未在文档里写死）；`best_full_fidelity` 只在 fidelity=1 上取，若某次 run 没有任何全保真点则返回 None | P2 |

**总体**：v0.8 的两个 P0（v0.7 的显式算法覆盖、权重语义）已修好，`manual_selection` 的持久化链路完整；`calibration.py` 是可用 MVP，但**它是样本内校准且只覆盖单目标采样器组**。正式 460-runs 之前必须先补 LOPO 与"带噪/多目标/约束"的 records 采集（`CALIBRATION_RUN_PLAN.md` 给了可执行方案）。

---

## 1. P0 详述

### P0-1 `calibration.py:46-119` — 样本内校准（无 LOPO）

```python
def evaluate_weights(records, weights):
    labels = best_algorithm_per_problem(records)      # 用全部 records 生成标签
    ...
    for problem, (context, algorithms) in contexts.items():
        chosen = route(context).algorithm
        top1 += int(chosen == labels[problem])         # 同一个 problem 既训练又评估
```
`random_search_weights` 在**同一批 records** 上挑最优权重（`calibration.py:102-111`），报告里的 `top1 0.25→0.50` 是 in-sample 指标。样本内择优在 records 少时几乎必然过拟合：9 个问题 × 少数算法时，权重可以记住"每个问题该选谁"。

**文献依据**：R51（Cawley & Talbot：模型选择必须嵌套/折外评估，否则误差估计偏乐观）、R52（Varma & Simon：CV 用于模型选择时的偏差）、R32（LOPO/多 seed 协议）。

**修复**：`calibration.py` 增加

```python
def lopo_folds(records, n_folds=None) -> list[tuple[list[RunRecord], list[RunRecord]]]:
    problems = sorted({r.problem for r in records})
    return [([r for r in records if r.problem != p], [r for r in records if r.problem == p])
            for p in problems]

def lopoe_metrics(records, iterations=60, seed=0):
    folds = []
    for train, test in lopo_folds(records):
        weights = random_search_weights(train, iterations=iterations, seed=seed)["weights"]
        folds.append(evaluate_weights(test, weights))       # 注意：test 的 labels 只在 test 内算
    return aggregate(folds)     # 均值/标准差 + 每折明细
```

`scripts/calibrate_router.py` 增加 `--folds lopo`，输出 `validation` 写"每折 held-out top1/top2/regret"，并把 in-sample 指标标为 `in_sample`（不用于验收）。

**验收**：`test_lopo_no_leakage`（训练折的 problem 不出现在测试折）；正式报告只引用 `validation.lopo_*`。

### P0-2 `benchmarks.run_algorithm` + `collect_records` — records 只覆盖"无噪标量"

`run_algorithm`（`benchmarks.py:495+`）直接 `problem.evaluate(config)`（= `problem.fn`，无噪声），`best_score` 只取单个目标。`collect_records` 因此无法表达：
- 带噪问题（`noisy_sphere` 的 σ、复孔）；
- 多目标（`moo_yield_cost` 的 HV）；
- 约束（`constraint_*` 的可行比例/约束最优）；
- 多保真（`mf_suzuki` 的 cost-aware）。

校准如果只用单目标无噪 records，学到的权重只对"连续低维 + bayes vs 采样器"有效，对 router 的多目标/约束/噪声分支是**没数据的**——这正好是 v0.8 里规则最多的部分。

**文献依据**：R32（每个 benchmark 的指标口径必须与任务类型一致；MOO 用 HV/IGD，约束用 feasible-best）、R42（HV/ref）、R12（噪声多目标）。

**修复**：给 `BenchmarkProblem`/runner 增加
```python
def run_task(problem, algorithm, budget, batch, seed, noise_std=0.0, n_replicates=1):
    # 评估器：synthetic 真值 → 加噪 → 观测；同时保存 true_objectives
    # 指标：单目标 best；MOO 用 pareto.summarise(use_true=True).hypervolume；
    #       约束用 feasible_best；多保真用 cost_aware_best
```
标签订为"该 context 下指标最优算法"，而不是 `best_score` 最大。`RunRecord` 增加 `metric_name/metric_value/true_hv`。

**验收**：`test_collect_records_covers_moo_noisy_constrained`（4 类问题各至少 1 条）；错把单目标算法用于 MOO 时被能力过滤跳过（见 P1-2）。

---

## 2. P1 详述

### P1-1 `calibration.py:136-141` — context 丢失 noise/smoothness/constraints

```python
context = OptimizationContext.from_space(problem.space, budget_total=budget,
                                         batch_size=batch_size,
                                         n_objectives=problem.objective.n_objectives)
```
`from_space` 默认 `noise="medium"`、`smoothness="mixed"`、`domain="generic"`、`n_constraints=0`、`multi_fidelity=False`。于是：
- `noisy_sphere` 与 `branin` 的路由 context 完全一样（除了空间）；
- `constraint_*` 的 `n_constraints` 丢失（`problem.objective.constraints` 存在）；
- `mf_suzuki` 的 `multi_fidelity` 丢失。
**修复**：给 `BenchmarkProblem` 加 `tags`/`noise_std`/`multi_fidelity` 字段，`collect_records` 用它们填 `OptimizationContext`（或直接从 `problem` 的元数据推导）。

### P1-2 `calibration.py:142-155` — 不按能力过滤算法

`run_algorithm(problem, "qehvi", ...)` 对单目标问题会在 `QEHVI.__init__` 抛 `ValueError("qehvi currently supports exactly 2 objectives")` → `collect_records` 直接崩溃。**修复**：用 `router._required_capabilities(context)` 过滤 `algorithms`，或 `try/except` 跳过并记录 `status="skipped_capability"`。
**文献依据**：R8（SMAC3 Facade 按场景/能力组合入口）、`CAPABILITIES_SPEC.md`。

### P1-3 `router.py:801-813` — 正/负分权重语义不对称

- 正分：`score*factor`（factor>1 = 加分）；
- 负分：`score/factor`（factor>1 = 减负 → 提权；factor<1 = 加负 → 降权）；
- 两者在 `factor=1` 处连续，但"同一 factor 对正/负分方向相反"的语义要求校准器知道这一点，否则随机搜索会找到"把 parego 的 -165 除以 1.6 → -103"这种"提权"路径，而 `factor<1` 又把它压到 -165/0.6=-275。建议：**只对正分做乘法，负分只做加法平移**（`score + (factor-1)*k`），或在校准中把"是否负分"作为 per-algorithm 特征（记录该算法在 context 中的 baseline 分符号）。

### P1-4 `router.py:807-808` — 禁用算法用 `-1e9` 表示

`-1e9` 会进入 `candidate_scores`（对外 JSON/render），`margin = best - ranked[1]` 变成超大值 → `confidence` 恒 `high`；`alternatives` 里可能出现被禁用算法（若只剩下它）。**修复**：在权重循环前把 `factor<=0` 的算法从 `scores` 中**删除**（并记录 `disabled` 列表进 `Recommendation`）。

### P1-5 `calibration.py:73-77` — regret 归一化不当

```python
span = max(abs(best), 1e-9)
regrets.append(max(0.0, (best - chosen_score) / span))
```
`best` 接近 0 时（如 `sphere2d` 最优 0 附近）分母极小 → regret 爆炸，主导目标函数。应改为 `(best - chosen) / (best - worst + eps)`（worst = 该问题最差算法的分数），与 `BENCHMARK_SPEC` 一致；标签用 IQM 而非 mean。

### P1-6 `scripts/calibrate_router.py:25-33` — 默认矩阵不含多目标/约束/多保真

`DEFAULT_PROBLEMS` 只有 6 个单目标（branin/sphere2d/mixed_reaction/discrete_catalyst/suzuki_like/high_cardinality_solvent），`DEFAULT_ALGORITHMS` 只有 6 个采样器/单目标算法。**正式 460-runs 必须扩到**：+ `noisy_sphere`、`constraint_easy/hard`、`moo_yield_cost`、`discrete_catalyst_v2`、`mf_suzuki`；算法加 `parego/qehvi/constrained/mfbo`（按 P1-2 过滤）。

### P1-7 权重全局/不落 study

见 `ROUTER_WEIGHTS_ACCEPTANCE.md` §2：`ask` 不重新加载；`config.json` 无权重快照。修复：`init` 时把生效权重写进 `config.json["router_weights"]`，`ask` 从 config 读并 `set_algorithm_weights`；`report` 显示"本次 run 使用的权重版本"。

---

## 3. P2（择机）

- **P2-1** `router.py:801`：删除死代码。
- **P2-2** `calibration.py:89-119`：输出 schema 增加 `search`（iterations/seed/range/folds）；只输出 |factor-1|>1e-3 的项。
- **P2-3** `calibration.py:42`：context schema version + 缺字段回退。
- **P2-4** `study.py:156`：`algorithm.strip().lower() == "auto"`。
- **P2-5** `mfbo.py:205`：`best` 只用全保真（见 `MFBO_COST_API.md`）。
- **P2-6** `benchmarks.py:191-210`：`cost_aware_best` 文档化公式；`best_full_fidelity=None` 时报告 `n_full=0`。
- **P2-7** `calibration.py`：`evaluate_weights` 里 `set_algorithm_weights` 的全局副作用（finally 恢复）对并发测试不友好 → 改成传参式 `route(ctx, weights=...)`（长期）。

---

## 4. 证据与验证

| 检查 | 方法 | 结果 |
|---|---|---|
| 显式算法钉住 | 读 `study.py:149-158,196,222,319,536` | create→`manual_selection` + `manual_override=True`；config 持久化/恢复；`resume_auto_route` 可交还 ✓ |
| 权重边界 | 读 `router.py:48-51,806-813` | 正乘/负除/0→-1e9/NaN 拒绝 ✓；`-1e9` 污点（P1-4）、正负不对称（P1-3） |
| 校准 harness | 读 `calibration.py` 全文 | 可跑但样本内（P0-1）、context 丢元数据（P1-1）、无能力过滤（P1-2） |
| mfbo 初始设计 | 读 `mfbo.py:189` | 全 `levels[0]` ✓（v0.7 P1-5 已修） |
| true-HV | 搜索 `benchmarks.py` 未找到 `true_objectives/hv_ratio` 辅助 | runner 只评 `problem.fn`；true-HV 需随 P0-2 一起补 |

---

## 5. 修复顺序

1. **P0-1/P0-2**（LOPO + 多类型 records）——正式校准的前置。
2. **P1-1/P1-2**（context 元数据 + 能力过滤）——否则 records 覆盖不全/崩溃。
3. **P1-4/P1-3**（禁用语义、正负分权重）——权重验收前。
4. **P1-7**（权重落盘）与 **P1-5**（regret 归一化）。
5. **P2** 清理。

**回归门槛**：132 tests 全绿 + `test_lopo_no_leakage`、`test_collect_records_covers_moo_noisy_constrained`、`test_disabled_algorithm_removed_from_scores`；正式校准报告只引用 LOPO 指标。

---

## 6. 参考

R51（Cawley & Talbot：嵌套/折外评估避免乐观偏差）、R52（Varma & Simon：CV 用于模型选择时的偏差）、R32（多 seed/LOPO/统计协议）、R42（HV/真值口径）、R12（噪声多目标）、R8（场景/能力过滤）、R50（Google Vizier：算法注册/Study 生命周期，校准权重应随 study 落盘）。
