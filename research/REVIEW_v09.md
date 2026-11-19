# REVIEW_v09 — 对 v0.8→v0.9（LOPO / records 过滤 / cost API / 全保真 best）的审查

> 审查者：algo-research ｜ 日期：2026-11-05 ｜ 快照：v0.9（`calibration.py` 213 行、`study.py` cost API、`mfbo.py` 320 行、`qnehvi.py` ASCII warning）
> 格式：**`文件:行` → 问题 → 文献依据 → 修复建议 → 优先级**。P0 = 正确性/结果不可信；P1 = 显著影响样本效率或与宣称不符；P2 = 工程质量。
> 本轮实测：核心校准子集（6 问题×5 算法×budget10×batch2×2 seeds=60 runs，collect 3.1 s）——见 `CALIBRATION_RUN_RESULTS.md`。
> **已确认修好**：`calibration.lopo_validate` 折外评估（`calibration.py:122-163`，训练折 `random_search_weights` → held-out `evaluate_weights`，无泄漏）；`collect_records` 能力过滤（`:181-200`）；`Study.cost_spent/remaining_cost`（`study.py:271-280`）；qNEHVI 无噪 warning 文案改 ASCII（`qnehvi.py:126-134`，实测触发）；mfbo 初始设计最低档（`mfbo.py:189`）；权重 NaN/负数拒绝（`router.py:48-51`）。

---

## 0. 总结

| 编号 | 位置 | 一句话 | 优先级 |
|---|---|---|---|
| P0-1 | `mfbo.py:205` | v0.9 声称"best 只用全保真"，但代码仍是 `best = max(self._y)`（混全部保真度）→ EI 基线仍可能被低保真/噪声污染 | **P0** |
| P0-2 | `study.py:271-280` + `mfbo.py:149-166` | `cost_spent` 只累加 `observation.cost`（默认 0，Study 不从 fidelity 推导），`remaining_cost` 需要 `meta["cost_budget"]` 否则返回 None，且 `mfbo` 根本不消费 `remaining_cost` → 多保真成本会计**静默为零/不生效**，v0.8 的 5.2% 超支不会自动修好 | **P0** |
| P1-1 | `calibration.py:186-192` | context 仍只带 `n_objectives/n_constraints`，缺 noise/smoothness/`multi_fidelity` → 噪声问题被当 generic，`mfbo` 因 `special⊄required` 被跳过，多保真分支永远无 records | P1 |
| P1-2 | `benchmarks.run_algorithm`（421+）+ `calibration.py:201-211` | MOO 问题仍用 `best_score`（主目标）做标签，不是 true HV → `moo_yield_cost` 若加入矩阵，标签错误 | P1 |
| P1-3 | `calibration.py:31-39,71-77` | 标签用算术平均、无 tie 处理；regret 用 `|best|` 归一。实测 `discrete_catalyst_v2` 折：标签 lhs 但 `mean_regret=0`（分数并列）→ top1 与 regret 口径矛盾 | P1 |
| P1-4 | `calibration.py:197-200` | 能力过滤逻辑 `special ⊄ required → skip` 会**误杀**"能力超集"算法（如将来 constrained-MOO 算法在 MOO context 中被跳过）；应改为"required ⊆ capabilities" | P1 |
| P1-5 | `calibration.py:89-119` | 权重搜索在 train（5 个问题）上做，折间不稳定：实测全局权重 `{bayes 1.401, lhs 1.537, random 0.623, sobol 1.026, tpe 0.702}`，而 `discrete_catalyst_v2` 折变成 `{bayes 1.444, lhs 1.358, random 1.021, sobol 0.859, tpe 1.111}` → LOPO 指标方差大，报告需给折间范围 | P1 |
| P2-1 | `study.py:271,275` vs `study.py:268` | API 风格不一致：`cost_spent()` 是方法、`remaining` 是 property；无 `cost_budget` property；`report` 只显示 spent（`study.py:604`）不显示 remaining/budget | P2 |
| P2-2 | `router.py:807-813` | 正乘/负除语义与 `-1e9` 禁用仍会污染 `candidate_scores`/confidence（v0.8 P1-3/P1-4 未改） | P1（遗留） |
| P2-3 | `study.py:522` | `_to_observation` 的 `cost=payload.get("cost", 0.0)`：应从 fidelity 推导默认成本（`cost(fidelity)`），否则 mfbo 的 cost 会计永远为 0 | P1（与 P0-2 同源） |
| P2-4 | `calibration.py:42` | `context` 字典缺字段/旧版时 `from_dict` 抛错；无 schema version | P2 |
| P2-5 | `qehvi/qnehvi` | v0.8 的 `_acquisition` 抽取仍未做 → 快 CI #2（收敛）仍无法写严（见 `FAST_CI_SNIPPETS.md`） | P2 |
| P2-6 | `calibration.py:162` | LOPO 只报均值，无折间标准差/每折明细（`folds` 里有，但报告未聚合方差） | P2 |

**总体**：v0.9 的 LOPO 与能力过滤方向正确，实测 LOPO top1 0.833 vs 默认 0.667（in-sample 0.833）说明在这个小矩阵上"默认已经是 bayes 强"、校准增益主要来自 `lhs` 那一折。**但两个 P0 说明"多保真 + 成本"的链路仍未闭环**：`best` 混保真、`cost_spent` 静默为 0、`mfbo` 不用 `remaining_cost`。这两条不修，`mfbo` 的"同成本优于全保真"结论不可复现。

---

## 1. P0 详述

### P0-1 `mfbo.py:205` — `best` 仍取全部观测的最大值

```python
model = self._fit(self._X, self._y)
best = float(np.max(self._y))          # ← 包含低保真观测
```

低保真观测系统性偏低（`mf_suzuki` bias=15·(1−s)^1.5），所以多数时候 max 来自全保真；但 **噪声/异常**（如 s=0.5 的幸运高值）会把 `best` 抬高，EI 的 `imp = mu − best` 偏小 → 采集值系统性偏低、偏向探索/低效。v0.9 的 changelog 声称"best 只用全保真观测"，代码未体现。

**修复**（同 `MFBO_COST_API.md` §2.4）：

```python
def _best_full_fidelity(self) -> float | None:
    values = [obs_y for obs_y, o in zip(self._y, self._observations) if o.fidelity >= 1.0 - 1e-9]
    if values:
        return max(values)
    # 无全保真点：用最高保真档的观测 + 偏差估计（保守）
    return float(np.max(self._y)) - self._bias_estimate_safe()

# _propose:
best = self._best_full_fidelity()
if best is None:
    return self.space.sample_uniform(1, self.rng)[0], 1.0   # 先补一个全保真锚点
```
**验收**：`test_mfbo_best_uses_full_fidelity_only`（构造低���真 y=200、全保真 y=100 → acquisition 的 `best` 必须是 100）。

### P0-2 `study.py:271-280` — 成本会计静默为零 + mfbo 不消费

```python
def cost_spent(self) -> float:
    return float(sum(observation.cost for observation in self._dataset()))
def remaining_cost(self) -> float | None:
    budget = self.meta.get("cost_budget")
    if budget is None:
        return None                  # ← 默认 None：mfbo 无从得知剩余成本
    return max(0.0, float(budget) - self.cost_spent())
```

- `Observation.cost` 默认 0（`Study._to_observation` 的 `payload.get("cost", 0.0)`），且**不从 fidelity 推导** → 多保真 run 的 `cost_spent()` 恒为 0（除非调用方手动传 cost）；
- `remaining_cost()` 在未设 `meta["cost_budget"]` 时返回 `None`，而 `mfbo.ask_with_fidelity`（`mfbo.py:149-166`）没有接收/使用剩余成本的路径 → v0.8 的 5.2% 超支依然存在；
- `Study.report` 只显示 `cost_spent`（`study.py:604`），不显示预算/剩余。

**修复**（`MFBO_COST_API.md` 的最小版）：

1. `_to_observation`：若 `payload.cost` 缺省且 `objective.fidelity` 存在 → `cost = cost(fidelity)`（`Observation` 已记录 fidelity）；
2. `Study.create(..., cost_budget=...)` → `meta["cost_budget"]`；`remaining_cost()` 在无预算时回退为 `budget × max(costs_levels) − cost_spent()`（而非 None）；
3. `ask_with_fidelity` 接受 `remaining_cost`（或从 `self._cost_budget`），按 `MFBO_COST_API.md §2.3` 降档/减批；
4. `report --json` 输出 `cost_spent/cost_budget/remaining_cost`。

**验收**：`test_cost_spent_derived_from_fidelity`（fidelity 0.25/0.5/1 → 1.75）；`test_mfbo_never_overshoots_cost`（实测 v0.8 25.25/24，修复后 ≤24×1.02）。

---

## 2. P1 详述

### P1-1 `calibration.py:186-192` — context 缺 noise/smoothness/mf

`OptimizationContext.from_space(space, budget_total, batch_size, n_objectives, n_constraints)` 其余字段默认（noise="medium"、smoothness="mixed"、multi_fidelity=False）。后果：
- `noisy_sphere` 与 `branin` 的路由 context 除空间外完全相同 → 噪声规则不会被校准到；
- `mf_suzuki` 的 context `multi_fidelity=False` → `mfbo` 在 `special⊄required` 判断中被跳过 → 多保真分支无 records（实测：默认算法组里没有 mfbo 被采集）。
**修复**：`BenchmarkProblem` 增加 `tags`/`noise`/`multi_fidelity` 元数据，`collect_records` 据此填 context；或 `collect_records(..., noise=..., multi_fidelity=...)` 显式覆盖。

### P1-2 `benchmarks.run_algorithm` + `collect_records` — MOO 标签仍用 `best_score`

`run_algorithm` 返回 `best_score`（单目标 signed score）；对 `moo_yield_cost` 它只是主目标（yield），不是 HV。若把 MOO 问题加入校准矩阵（`CALIBRATION_RUN_PLAN.md` 建议加），标签会偏向"yield 最高的算法"而不是"HV 最好的算法"。**修复**：`run_algorithm` 增加 `metric="true_hv_ratio"` 分支（用 `pareto.summarise(use_true=True)` 或合成问题重算），`RunRecord` 带 `metric_name`。

### P1-3 `calibration.py:31-39,71-77` — tie 处理与 regret 归一化

实测 `discrete_catalyst_v2` 折：标签 `lhs`，但 `bayes` 与 `lhs` 的 `best_score` 并列（该问题小离散空间里多算法都能穷举到 82），`mean_regret=0` 而 `top1=0`。这说明：
- 标签应做 tie 检测（`|best − second| < 5%·span` → tie，不计 top-1，或算 top-2 命中）；
- regret 归一化用 `|best|` 在 best≈0 时爆炸（`sphere2d` 最优 0 附近已出现）；
- 标签用 mean，建议 IQM。
**修复**：与 `CALIBRATION_RUN_PLAN.md` §2 对齐（IQM + tie + span 归一）。

### P1-4 `calibration.py:197-200` — 能力过滤的"超集误杀"

```python
special = capabilities & {"multi_objective", "constraints", "multi_fidelity"}
if special and not special <= required:
    continue
```
如果算法同时声明 `multi_objective` 与 `constraints`（未来的 constrained-MOO），在 MOO context（required={"multi_objective"}）里 `special⊄required` → 被跳过，永远采集不到。正确逻辑：**只检查 required（硬需求）**；超集是允许的：
```python
if required and not required <= capabilities:
    continue
```
（`special` 过滤想避免的是"qehvi 用在单目标"——那由 `required` 里的 multi_objective 为空、而 qehvi 的 capabilities 含 multi_objective 时……其实 qehvi 在单目标 context 会被 `required` 检查放过，仍需额外规则：**算法声明了 `multi_objective` 只能在 MOO context 用**，改用 `if "multi_objective" in capabilities and "multi_objective" not in required: continue` 等逐项显式规则。）

### P1-5 `calibration.py:89-119` — 折间权重不稳定

实测：全局权重 `{bayes:1.401, lhs:1.537, random:0.623, sobol:1.026, tpe:0.702}`；`discrete_catalyst_v2` 折的权重明显不同（`{bayes:1.444, lhs:1.358, random:1.021, sobol:0.859, tpe:1.111}`）。6 个问题的 LOPO 每折只有 5 个训练问题 → 权重方差大。**报告**：给折间 min/median/max；**方法**：权重向 1.0 正则（`w = 1 + λ(w_raw−1)`，λ=0.5），或只在 ≥3 折方向一致的 feature 上启用权重。

---

## 3. P2（择机）

- **P2-1** `study.py`：统一 `cost_spent`/`cost_budget`/`remaining_cost` 为 property；`report` 显示预算/剩余。
- **P2-2** `router.py:807-813`：`factor<=0` 从 scores 删除而非 `-1e9`；负分改加性（v0.8 遗留）。
- **P2-3** `study.py:522`：cost 默认从 fidelity 推导（与 P0-2 同）。
- **P2-4** `calibration.py:42`：context schema version/缺字段回退。
- **P2-5** `qehvi/qnehvi`：抽 `_acquisition(candidates, features)` 供快 CI #2 与诊断。
- **P2-6** `calibration.lopo_validate`：返回 `std`/`per_fold` 聚合（现有 `folds` 原始数据已够，报告层补）。

---

## 4. 证据（本轮实测）

| 检查 | 结果 |
|---|---|
| 核心校准子集 | 60 records，collect **3.1 s**；labels：bayes 5/6，lhs 1/6（`discrete_catalyst_v2`） |
| in-sample default | top1 **0.667**、top2 **0.833**、mean_regret **0.0084** |
| in-sample tuned | top1 **0.833**、top2 **1.000**、mean_regret **0.0019**；weights `{bayes 1.401, lhs 1.537, random 0.623, sobol 1.026, tpe 0.702}` |
| LOPO | top1 **0.833**、top2 **0.833**、mean_regret **0.0019**（6 折，1 折失败=discrete_catalyst_v2，且该折 regret=0 与 top1=0 矛盾 → tie 处理问题） |
| Study cost API | `cost_spent()` 为方法（非 property）；无 `cost_budget`；`remaining_cost()` 在未设 meta 时返回 None |
| mfbo best | `mfbo.py:205` 仍为 `max(self._y)`（混保真） |
| qNEHVI warning | ASCII 文案已改（`qnehvi.py:131`，实测触发）✓ |

---

## 5. 修复顺序

1. **P0-1**（mfbo best 全保真）——影响多保真采集正确性，改动 10 行。
2. **P0-2**（cost 从 fidelity 推导 + cost_budget + mfbo 消费）——使成本口径闭环。
3. **P1-1/P1-2**（context 元数据 + MOO 指标）——正式 456-runs 的前置。
4. **P1-4/P1-3**（能力过滤超集、tie/regret 口径）——校准数字可信度。
5. **P1-5**（折间稳定性/正则）与 P2 清理。

**回归门槛**：137 tests 全绿 + 上述新增单测；核心子集 LOPO 与 in-sample 的差距保持"LOPO ≤ in-sample"（当前 0.833 vs 0.833，但折间不稳）；`mf_suzuki` 成本超支 ≤2%。

---

## 6. 参考

R51（Cawley & Talbot：折外/嵌套评估）、R52（Varma & Simon：选择偏差）、R32（多 seed/IQM/统计）、R42（HV/真值）、R41（BOCA：成本感知/全保真）、R12（qNEHVI）、R8（场景/能力过滤）、R50（配置/权重随 study 落盘）。
