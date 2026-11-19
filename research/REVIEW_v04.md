# REVIEW_v04 — 对 v0.4（Sprint2 / 约束 / ParEGO P0 修复）的文献级审查

> 审查者：algo-research ｜ 日期：2026-11-05 ｜ 代码快照：17:18–17:24（`bayes.py` 21.4KB、`constrained.py` 8.7KB、`parego.py` 9.7KB）
> 格式：**`文件:行` → 问题 → 文献依据 → 修复建议 → 优先级**。P0 = 正确性/结果不可信；P1 = 显著影响样本效率或与宣称不符；P2 = 工程质量。
> 前序：`REVIEW_v01.md`、`REVIEW_v03.md`。**已确认修复**：ParEGO 固定归一化窗口（P0-1）、缺目标容错（P0-2）、`Objective.ref_point` 显式化（P0-3）、ARD/局部惩罚/TS 多抽样/Sobol 2^k（Sprint2 四项）、约束 BO 落地（`constrained.py`）。

---

## 0. 总结

| 编号 | 位置 | 一句话 | 优先级 |
|---|---|---|---|
| P0-1 | `parego.py:146-166` | 归一化窗口在**第一次 `_window()` 调用**就冻结；若此时 `_Y` 行数 < `freeze_n`（批量部分 tell/初始设计未满），窗口永久偏窄/退化，之后所有标量化都错 | **P0** |
| P0-2 | `constrained.py:9-11,147-167` | 文档宣称 Letham 式"带噪约束"，实现只用 GP 后验抽样、**未加入观测噪声**，且 `f*` 取"可行观测最大值"而非随机 baseline → 噪声约束下 `P(feasible)` 过度自信、易早停 | **P0**（若以"带噪约束"对外承诺） |
| P1-1 | `parego.py:136,246-252` | `tell` 只把 valid 观测给 `super().tell`，`self.t` 小于 study 观测数 → `_initial_design_left()` 可能继续发初始设计点（重复消耗预算） | P1 |
| P1-2 | `parego.py:114-120,146-166` | `_bounds`（冻结窗口）未进 `state_dict`；`_skipped` 未上报；窗口退化（span<EPS）静默置 0 | P1 |
| P1-3 | `base.py:157-181` | 三个平行概念：`ref_point`（HV）、`objective_bounds`（ParEGO 归一化）、`constraint_limits`（约束），无一致性校验（low<high、bounds 与 ref 关系、threshold 类型） | P1 |
| P1-4 | `bayes.py:409-434` | `alpha` 仍是固定参数；`Observation` 无 `objective_std/n_replicates` → 复孔/噪声未接入（`REPLICATES_ALPHA.md` 的补丁未落） | P1 |
| P1-5 | `constrained.py:147-148` | 约束 GP 复用目标 `alpha`（同一个参数）→ 约束噪声与目标噪声被强行同一；带噪约束的 `P(feasible)` 会过度自信 | P1 |
| P1-6 | `constrained.py:165-167` | `f* = max(y[feasible])` 固定（无 baseline 随机化）→ 噪声下"幸运观测"会压住后续改进（qNEI 的动机，R10/R12） | P1 |
| P2-1 | `constrained.py:29` | `_INFEASIBLE = 1e300` 是死代码 | P2 |
| P2-2 | `constrained.py:172-175` | `feasibility_threshold` 过滤后若无有限值则回退到 `feasibility`，但不写 warning/decisions → 用户不知道"这一批没有可行候选" | P2 |
| P2-3 | `bayes.py:228-244` | `_init_queue` 未持久化：每个 `Study.ask` 重建 optimizer 都会重新生成 2^k 设计并过滤已测点；`total=max(n_initial, t+k)` 随 t 增长，每次生成量变大 | P2 |
| P2-4 | `bayes.py:428` | 非冻结时 `n_restarts_optimizer=1` 偏少（建议 3，n≥15） | P2 |
| P2-5 | `parego.py:186-206` | 每个 proposal 仍重建 inner GP（v0.3 P1-1 未改）；seed 仍是长度推导 → 独立 ask 重复风险 | P2 |
| P2-6 | `parego.py:168-183` | 多目标退化（某目标 span<EPS）不给 warning；全部退化时返回常数 g | P2 |

---

## 1. P0 详述

### P0-1 `parego.py:146-166` — 冻结窗口的时机错误

```python
def _window(self):
    ...
    explicit = all(name in self.objective.objective_bounds for name in self.objective.objectives)
    if explicit: ...
    else:
        freeze_n = max(3, 2 * m)
        rows = self._Y[:freeze_n]
        lows, highs = np.min(rows, axis=0), np.max(rows, axis=0)
    self._bounds = {name: (lows[i], highs[i]) ...}      # ← 立即冻结，永不更新
    return lows, highs
```

问题：`_window()` 第一次被调用发生在**第一个 model-based proposal**（`_scalar_values` 内），此时 `_Y` 可能只有初始设计的一部分（例如 Study 分批 `ask` 后**部分 tell**；v0.4 已允许部分结果，见 `REVIEW_v01` P0-5 修复）。若此时行数 < `freeze_n`：窗口由更少/更偏的点决定；更糟的是若这些点恰好几乎相同（例如都来自初始设计的前几个局部点），`span < _EPS` → 该目标整列被置 0，**永久**丢失该目标的优化方向。

实测（用 v0.3 的公式复算冻结窗口行为）：给定 2 个点 `(90,10),(80,12)` 冻结后，任何后续新点都不会改变 `g` 的历史值（这是 P0-1 修复的目的，正确）；但若用于冻结的 2 个点恰好是 `(90,10),(90,11)`，则目标 1 的 span=0 → 该列永久为 0 → 等价于单目标。**这是"修复引入的新风险"**：从"每轮漂移"变成"一次定终身"。

**文献依据**：R11（ParEGO 的标量化需要**有意义的**目标范围；原文用已知/估计的界）；`PAREGO_SPEC.md` §2.2（固定窗口的前提是"窗口本身可信"）。

**修复（最小改动）**

```python
def _window(self):
    m = self.objective.n_objectives
    if len(self._bounds) == m:
        return ...                                  # 已冻结
    explicit = all(name in self.objective.objective_bounds for name in self.objective.objectives)
    if not explicit:
        freeze_n = max(3, 2 * m)
        if len(self._Y) < freeze_n:                 # ← 数据不够：不冻结
            rows = self._Y
            lows, highs = rows.min(axis=0), rows.max(axis=0)
            # 保守扩边：span=0 的目标用"全局目标值 ±1"占位，避免永久 0
            span = highs - lows
            bad = span < 1e-9
            lows = np.where(bad, lows - 0.5, lows)
            highs = np.where(bad, highs + 0.5, highs)
            return lows, highs                       # 每次调用重算，直到数据够
        ...
        self._bounds = {...}                         # 冻结
```

**验收**：单测 `test_parego_window_waits_for_enough_rows`：先 tell 2 个点 → 调 `_window()` 不冻结（`_bounds` 为空）；tell 到 `2m` 个点后再调用 → `_bounds` 非空，且之后新观测不改变窗口。

### P0-2 `constrained.py:9-11,147-167` — "带噪约束"名不符实

文档（`constrained.py:9-15`）明确写了 "Letham-style correction to the naive EI×∏PoF" 和 "noisy measurements"。但实现：

- `constraint_models = [self._fit(self._X, self._C[:, j]) ...]`（line 148）→ 每个约束一个 GP，**训练目标是观测到的约束值**，`alpha` 用目标参数（见 P1-5）；
- `constraint_draws = self._sample_posterior_multi(model, features, draws)`（line 153）→ 抽样只包含**后验认知不确定性**；观测噪声（复孔/测量误差）没有加入；
- `feasible &= constraint_draws <= threshold - margin`（line 158）→ 用后验**真值**判可行；而实验者看到的是**带噪测量值**，两者不是一回事；
- `best = max(self._y[feasible])`（line 165）→ `f*` 固定为已观测最优，噪声下这是"幸运观测"（R10/R12 的核心问题）。

**文献依据**：R10（Letham 明确指出必须对"约束噪声"与"目标噪声"分别建模；`EI·PoF` 在噪声下有偏）、R12（qNEHVI：baseline 必须随机化）、R2（alpha 与噪声）。

**修复（与 `CONSTRAINT_IMPL.md` 对齐）**

1. 给 `Observation` 加 `constraint_std: dict[str,float]`/`n_replicates`（`REPLICATES_ALPHA.md`）；
2. 抽样带噪约束：`c_noisy = c_true_draw + rng.normal(0, σ_c/√k, size)`；`feasible` 用 `c_noisy <= threshold - margin`；
3. 目标 baseline 随机化：`f_star = max over feasible baseline draws`，improvement = `max(0, y_draw - f_star)`（与 qNEI 相同）；
4. 若短期不做：**改文档**，明确当前实现是 "noiseless-constraint EI_c（epistemic-only）"，`Constraint.noisy` 不支持 → 避免过度承诺。

**验收**：`constraint_hard` + 约束噪声 σ=0.5 的 benchmark 上，带噪版 `P(feasible)` 校准（预测 vs 经验频率的 Brier<0.2）优于现状；至少一条单测 `test_constrained_uses_constraint_noise`。

---

## 2. P1 详述

### P1-1 `parego.py:136` — `t` 与 study 观测数脱节导致重复初始设计

`super().tell(valid)` 只把完整观测计入 `self.t`；`_initial_design_left() = n_initial − t`。Study 重建 optimizer 时传入的 `prior=dataset`（含被跳过的观测），于是 `t` 可能小于 `len(dataset)` → ParEGO 认为初始设计没做完，继续发 Sobol 点（与已测点重复，且 `_propose` 只在 model-based 分支去重，初始设计分支没有去重）。
**修复**：`super().tell(observations)`（全部观测）用于计数，`_Y` 只取 valid；或让 `_initial_design_left` 用 `len(self._xs) + len(self._Y)` 之外的独立计数。
**依据**：R8（RunHistory 的 status 计数与"已完成"语义一致）。

### P1-2 `parego.py` — 状态/告警缺失

- `_bounds` 不进 `state_dict`：若进程重启时传入的 `prior` 与上次不同（部分 tell 后重启），窗口会在新数据上重新冻结 → 历史 `g` 仍可能跨进程不一致。
- `_skipped` 只在内部计数，不写 report/decisions；跳过比例高时用户无感。
- 退化列静默置 0（`parego.py:176-178`），无 warning。
**修复**：`state_dict` 加 `{"bounds": self._bounds, "skipped": self._skipped}`；`report` 显示 `skipped_incomplete`；退化列写 warning。

### P1-3 `base.py` — 三个目标相关 schema 无一致性校验

`Objective` 同时有 `ref_point`（HV 参考点）、`objective_bounds`（ParEGO 归一化窗口）、`constraint_limits`（约束 `{op,value}`）。校验只有长度检查（`base.py:183-186`）。缺少：`bounds[low] < bounds[high]`；`ref_point` 与 `bounds` 的关系（ref 应不优于 bounds.low）；`constraint_limits` 的 `op ∈ {"<=",">="}`、`value` 有限。**修复**：`__post_init__` 集中校验，报错信息指出字段名；`from_dict` 兼容 `{"op":"<=","value":...}` 与 `{"threshold":...}` 两种写法（`CONSTRAINT_IMPL.md` §1）。
**依据**：R17（BayBE campaign 配置在加载时做完整校验）。

### P1-4 `bayes.py:409-434` — 复孔/噪声仍未接入

`alpha=float(self.params["alpha"])` 固定；`Observation` 无 `objective_std/n_replicates`。当前"高噪声用较大 alpha"只能靠手动调参。**修复**：`REPLICATES_ALPHA.md` 的 `_alpha_for(y)` + `Observation` 三字段。
**依据**：R2（重复测量）、R10。

### P1-5 `constrained.py:147-148` — 约束 GP 复用目标 alpha

约束与目标的噪声量级通常不同（如纯度测量 σ=0.3，收率 σ=1.5）。用同一 `alpha` → 某一侧过拟合/过平滑。**修复**：`_fit` 增加可选 `alpha=None`；约束 GP 用 `constraint_std` 估计的 alpha（`REPLICATES_ALPHA.md` §2.3）。

### P1-6 `constrained.py:165-167` — `f*` 未随机化

噪声下应 `f* = max over baseline posterior draws`（qNEI 语义，R10/R12）；当前固定用观测最大 → 幸运观测后 EI_c 变 0，停止探索。**修复**：与 §P0-2 的 baseline 抽样一起做。

---

## 3. P2（择机）

- **P2-1 `constrained.py:29`**：`_INFEASIBLE = 1e300` 未使用，删除或用于 `acquire` 的哨兵。
- **P2-2 `constrained.py:172-175`**：`feasibility_threshold` 回退无 warning；写入 `decisions`（"batch had no candidate above floor"）。
- **P2-3 `bayes.py:228-244`**（Sprint2 的 Sobol 2^k）：`_init_queue` 是实例状态，Study 每轮重建 → 每轮重算 2^k 设计（`total` 随 `t` 增长）。建议 `state_dict` 持久化队列；或把初始设计下沉到 Study（一轮生成、跨轮消费）。
- **P2-4 `bayes.py:428`**：非冻结时 restart=1；建议 `3 if len(y)>=15 else 1`，成本可控。
- **P2-5 `parego.py:186-206`**：per-proposal GP 重建与 seed 推导（v0.3 P1-1 遗留）；顺手把 `bayes._fit` 抽成模块级 `fit_gp()` 共用。
- **P2-6 `parego.py:168-183`**：全部目标 span<EPS 时 `g≡0`，应抛 `ValueError("all objectives constant")`；单目标常数列加 warning。
- **P2-7**（新增观察）：`render.py` 61KB 且直接读 `Objective.ref_point`——UI 层与算法层共用 ref 口径是好事，但需在 `docs` 里写明"HV 用 ref_point，ParEGO 归一化用 objective_bounds"，避免两套口径混用。

---

## 4. 证据与验证

| 检查 | 方法 | 结果 |
|---|---|---|
| ParEGO 归一化已冻结 | 读 `parego.py:146-183` + 复算 | 显式 bounds 或前 `2m` 行冻结；但冻结时机见 P0-1 |
| 缺目标容错 | 读 `parego.py:123-144` | `_skipped` 计数、`tell` 不抛异常 ✓（v0.3 P0-2 已修） |
| ARD | 读 `bayes.py:36-61,414-419` | 每特征 lengthscale + 维度缩放初值 ✓ |
| 局部惩罚 | 读 `bayes.py:269-320` | `local_penalty` 批量策略 ✓ |
| TS 批量 | 读 `bayes.py:347,487-500` | `_sample_posterior_multi`（一次 Cholesky 多抽样）✓ |
| Sobol 2^k | 读 `bayes.py:228-244` | `space.sample_sobol_base2` + 队列 ✓ |
| ref_point | 读 `pareto.py:128-135`、`base.py:159` | 显式 `Objective.ref_point` 优先、signed 换算 ✓（v0.3 P0-3 已修） |
| 约束 MC | 读 `constrained.py:147-175` | 独立 GP + MC 可行性/EI_c；**无观测噪声、`f*` 固定**（P0-2） |

---

## 5. 修复顺序建议

1. **P0-1**（窗口冻结时机）——影响所有 ParEGO 结果，改动 10 行。
2. **P0-2 的一部分**（文档澄清 + `f*` 随机化）——若短期不做带噪约束，先把文档改成 "noiseless-constraint EI_c"，避免对外承诺不符；`f*` 随机化是 qNEI 的最小步骤。
3. **P1-1 / P1-2**（计数与状态）——避免重复预算与跨进程不一致。
4. **P1-4/P1-5**（复孔 alpha）——`REPLICATES_ALPHA.md` 实施后一起解决。
5. **P1-3**（Objective 校验）——减少配置错误。
6. P2 清理。

**回归门槛**：`pytest` 全绿 + 上述单测；`moo_yield_cost` 5 seeds 的 HV 轨迹（固定 ref）与修复前相比不劣化；`constraint_hard` 的可行性校准（Brier）不差于修复前。

---

## 6. 参考

R11（ParEGO 标量化/窗口）、R10（带噪约束与 EI_c）、R12（qNEHVI baseline 随机化）、R2（复孔/alpha）、R8（RunHistory 状态语义）、R17（配置加载校验）、R21/R42（HV/ref 口径）。
