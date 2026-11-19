# REVIEW_v05 — 对 v0.5（qehvi.py / ParEGO 窗口修复 / constrained 文档 / Pareto ref）的审查

> 审查者：algo-research ｜ 日期：2026-11-05 ｜ 快照：17:28（qehvi.py 176 行、pareto.py 9.2KB、parego.py 10.1KB、constrained.py 8.8KB、dataset.py 7.4KB、bayes.py 21.4KB）
> 格式：**`文件:行` → 问题 → 文献依据 → 修复建议 → 优先级**。P0 = 正确性/结果不可信；P1 = 显著影响样本效率或与宣称不符；P2 = 工程质量。
> **已确认修复（本轮）**：ParEGO 窗口过早冻结（`parego.py:146-174`，`len(_Y) < freeze_n` 时不冻结）；`constrained.py` 文档已改为"observation noise is not modelled explicitly（复孔需先聚合）"；`Objective.ref_point` 在 `pareto.summarise`（`pareto.py:189-196`）与 qehvi（`qehvi.py:103-115`）中生效；复孔字段（`dataset.py:53-55`）与 `bayes._fit` 的逐点 alpha（`bayes.py:434-445`）已接入。

---

## 0. 总结

| 编号 | 位置 | 一句话 | 优先级 |
|---|---|---|---|
| P0-1 | `qehvi.py:108-115,130-131` | ref 回退值在"首次可用数据"时冻结；若之后没有任何观测严格支配 ref，`partition_2d` 返回空 → 直接退回 Sobol **且无 warning**，预算被静默浪费 | **P0** |
| P0-2 | `qehvi.py:69-87` + `bayes.py:197-223` | qehvi 的 `tell` 只把含两个目标的观测给父类；父类 `t`/`_y`/`_noise_var` 只覆盖 valid 子集，而 Study 的 `len(dataset)` 更大 → `_initial_design_left()` 判定与实际不一致（同 `parego.py` P1-1），且 `_skipped` 不上报 | **P0** |
| P1-1 | `qehvi.py:136,143` | 每次 ask 拟合 **3 个 GP**（2 目标 + 1 个只为取 lengthscale），第三个完全可复用 `models[0]`；n=200 时额外 ~30% 计算 | P1 |
| P1-2 | `qehvi.py:121,147-161` | 候选池的 local 中心来自父类 `self._y`（**主目标**），多目标下会把候选池拉向主目标最优区，损害前沿覆盖 | P1 |
| P1-3 | `qehvi.py:103-115` | 提供 `Objective.ref_point` 时**没有**校验它"严格劣于所有观测"；若用户给错（如 ref 高于最差点），`partition_2d` 静默过滤 → HV/EHVI 行为不可预期 | P1 |
| P1-4 | `router.py:270-293` | `_score_qehvi` 不考虑 `ref_point` 是否存在；无 ref 时 qehvi 用数据推导 ref（不可跨 run 比较）仍被推荐 | P1 |
| P1-5 | `dataset.py:53` | 复孔字段 `objective_std` 是**标量**（单一目标），多目标/多约束各目标噪声无法表达；`bayes._fit` 的逐点 alpha 只对应主目标 | P1 |
| P1-6 | `constrained.py:147-167`（v0.4 遗留） | `f*` 仍固定为"可行观测最大值"，噪声下仍会在幸运观测后早停；文档已澄清"noiseless constraint"，与实现一致，但**能力声明仍写 `noisy`**（`capabilities` 来自父类） | P1 |
| P2-1 | `parego.py:146-174` | 窗口冻结时机已修，但 `_bounds` 仍不进 `state_dict`；退化列（span<EPS）静默置 0；`_skipped` 不上报 | P2 |
| P2-2 | `pareto.py:189-196` | `ref_point` 存在时不再写"数据推导 ref 不可比"的提示；`summarise` 的 `reference_point` 返回 signed 值，UI 若展示原始单位需再换算 | P2 |
| P2-3 | `qehvi.py:125-141` | `mc_draws` 同时用于单点 EHVI 与批量；q≥4 时建议降为 64 并靠局部惩罚，文档/参数注释需说明 | P2 |
| P2-4 | `qehvi.py:168-176` | `state_dict` 只存 `ref`，不存 `_Y`/`_xs`（依赖 prior 重建，可接受）；若 restart 时 prior 与上次不同，ref 回退值会重新冻结（跨进程不一致） | P2 |
| P2-5 | `pareto.py:101-123` | `partition_2d` 对"非支配集内含低于 ref 的点"依赖内部过滤（line 110），但上层 `non_dominated_mask` 已把低于 ref 的点算进前沿；`n_front` 报告会偏大 | P2 |
| P2-6 | `constrained.py:9-13`（已澄清） | 文档改好了，但**能力位**仍继承父类的 `noisy`；建议改名为 `noisy` 仅在真正支持观测噪声后声明，或明确 `noisy` = "能处理带噪观测（聚合后）" 的语义并写进 `CAPABILITIES_SPEC` | P2 |

**总体**：qEHVI 的核心数学（分区/DHV/MC）实现正确（与 `qEHVI_IMPL.md` 的 9 个 DHV 用例一致），ParEGO 窗口修复与 constrained 文档修正到位，复孔 alpha 已接入。**主要风险是"回退路径的静默性"**：ref 失效退回 Sobol、无 ref 仍被推荐、缺目标观测导致计数不一致——这些不会让测试变红，但会在真实 campaign 中悄悄浪费预算。

---

## 1. P0 详述

### P0-1 `qehvi.py:108-115,130-131` — ref 回退 + 空分区 → 静默退化为 Sobol

```python
if self._ref is None:
    signed = signed_scores(self._Y, self.objective)
    lows = np.min(signed, axis=0); highs = np.max(signed, axis=0)
    span = np.maximum(highs - lows, 1e-9)
    self._ref = lows - 0.05 * span          # ← 首次调用即冻结
...
boxes = partition_2d(signed[mask], ref)
if not boxes:                                # 没有任何点严格支配 ref
    return self.space.sample_sobol(n, rng=self.rng)   # 静默
```

两种失效路径：

1. 首次调用发生在 `_model_ask`（要求 `len(self._y) >= 3`），此时 `_Y` 可能只有 3 行 → ref 由很少的点决定并**永久冻结**（与 v0.4 ParEGO 同型问题）；
2. 任何一轮若所有观测都不严格支配 ref（例如用户给的 `ref_point` 太高，或数据整体退化），`partition_2d` 返回空 → **每一轮都退回 Sobol**，不报错、不写 decisions，`Study.report` 也看不出。

**文献依据**：R21（EHVI 要求 ref 严格被支配；BoTorch 会校验 ref 并抛错）、R42（HV 的 ref 必须严格劣于前沿）、R17（配置错误应在加载/首次使用时显式报错）。

**修复**

```python
def _signed_references(self):
    if self.objective.ref_point:
        raw = ...; ref = raw * signs
        signed = signed_scores(self._Y, self.objective)
        if not np.all(np.any(signed > ref, axis=0)) or not np.any(np.all(signed > ref, axis=1)):
            raise ValueError(f"ref_point {self.objective.ref_point} does not dominate any observed point")
        return ref
    if self._ref is None or not self._ref_frozen:
        if len(self._Y) < max(3, 2 * self.objective.n_objectives):
            raise RuntimeError("qehvi: need >=2*m observations to freeze a data-derived ref; "
                               "provide Objective.ref_point")
        ... freeze ...
        self._ref_frozen = True
    return self._ref
```
并在 `_model_ask` 的空 boxes 分支：`warnings.warn("qehvi: no observation dominates ref; falling back to Sobol")` + 写入 `decisions`（或直接 `raise` 让 router 重路由）。

**验收**：`test_qehvi_raises_on_bad_ref_point`、`test_qehvi_waits_before_freezing_ref`；`Study` 的 `decisions.jsonl` 里出现 `qehvi_ref_fallback` 记录。

### P0-2 `qehvi.py:69-87` — 缺目标观测与父类计数不一致

`tell` 只对 valid 子集调用 `super().tell(valid)`；`Optimizer.tell` 的 `t`、`GaussianProcessBO` 的 `_y`/`_noise_var` 都只覆盖 valid。Study 每轮重建 optimizer 时 `prior=dataset`（含被跳过的行）→ `t < len(dataset)` → `_initial_design_left()` 可能继续发初始设计点（重复消耗预算），且 `_skipped` 从不进 report。

**修复**：与 ParEGO 同策：计数用全部观测（`super().tell(observations)`），数值矩阵只取 valid；`_skipped` 写入 `state_dict`/`decisions`；若跳过比例 >30% 报 warning。

**文献依据**：R8（RunHistory 状态机：FAILED 也计数）、`REVIEW_v04.md` P1-1（同一问题在 ParEGO 已指出）。

---

## 2. P1 详述

### P1-1 `qehvi.py:136,143` — 三次 GP 拟合

```python
for index in range(2):
    model = self._fit(self._X, self._Y[:, index])       # 2 次
...
scale = self._local_penalty_scale(self._fit(self._X, self._y))   # 第 3 次，仅为 lengthscale
```

第三个 GP 用主目标 `self._y` 拟合，只为取 lengthscale 中位数——与 `models[0]`（第 1 个目标）几乎相同，直接复用即可（或者用两个 lengths 的中位数）。n=1024 候选、n=200 观测时，每次 `_fit` 含超参优化 ~0.2–0.5 s，省 1/3。

### P1-2 `qehvi.py:121` — 候选池 local 中心用主目标

`GaussianProcessBO._candidate_pool` 用 `np.argsort(-self._y)` 选 top-5 作局部中心；qehvi 的 `self._y` 是**主目标** score（`tell` 里父类按 `objective.name` 计算）。多目标下这会系统性地把 40% 候选点放在"主目标最优区"，EHVI 采到的前沿覆盖会偏向主目标端。
**修复**：qehvi 覆写 `_candidate_pool` 的 local 中心为"Pareto 集的点"或"每个目标端各取 top-2"；或临时把 `self._y` 设为"归一化后的 hypervolume 贡献"（不推荐，改 pool 更干净）。
**依据**：R21（前沿覆盖依赖候选分布）、R11（ParEGO 的权重轮换也是为了覆盖不同区域）。

### P1-3 `qehvi.py:103-115` — ref_point 无校验

用户给 `ref_point` 时应校验"严格劣于所有已观测点"（逐分量至少一个点全维占优），否则 `partition_2d` 静默过滤。见 P0-1 修复代码。**依据**：R42/R21。

### P1-4 `router.py:270-293` — 无 ref 仍推荐 qehvi

`_score_qehvi` 只看 m==2/budget/noise，不看 `Objective.ref_point`。无 ref 时 qehvi 用数据推导 ref（跨 run 不可比），而 ParEGO 不受影响。
**修复**：`_score_qehvi` 加 `if not ctx.has_ref_point: score -= 15; warnings.append("缺少 ref_point：HV 不可跨 run 比较")`（`OptimizationContext` 加 `has_ref_point` 字段，`prepare.py` 从 objective 透传）。

### P1-5 `dataset.py:53` — 复孔字段是标量

`objective_std: float` 只能表达单一目标的噪声；跨目标/约束的噪声无法表达。`bayes._fit` 的逐点 alpha 也只对应主目标。
**修复**：改为 `objective_std: dict[str, float]`（旧 JSON 数字 → `{primary: value}`），`constraint_std: dict`；`bayes._fit` 增加 `target_name` 参数按目标取噪声。这是 qNEHVI/带噪约束的前置。
**依据**：R2（重复测量/噪声）、R10（带噪约束）。

### P1-6 `constrained.py:147-167` — `f*` 固定（文档已澄清，能力位仍写 noisy）

文档已明确"observation noise is not modelled explicitly"，与实现一致 ✓；但 `capabilities` 继承父类的 `noisy`（`constrained.py:68`），router/技能层会把它当"支持带噪观测"的算法。**修复**：能力位语义写进 `CAPABILITIES_SPEC`（`noisy` = "能处理带噪观测（aggregate replicates 后）"），或把 `noisy` 换成 `noise_robust`（只有真正建模观测噪声的算法才声明）。同时 `f*` 随机化留作 v0.6。

---

## 3. P2（择机）

- **P2-1 `parego.py:146-174`**：`_bounds` 进 `state_dict`；`_skipped` 进 report/decisions；退化列警告。
- **P2-2 `pareto.py:189-196`**：`ref_point` 存在时 `summarise` 的 `reference_point` 是 signed 值；`render.py` 若展示原始单位需 `* signs` 逆变换，文档写清楚。
- **P2-3 `qehvi.py:125-141`**：`mc_draws` 在 q≥4 时降到 64（配合局部惩罚），参数描述补充。
- **P2-4 `qehvi.py:168-176`**：`ref` 持久化已有；若上次运行是数据推导 ref、restart 时 prior 变化 → 重新冻结（可接受，但应写 decisions）。
- **P2-5 `pareto.py:101-123`**：`n_front` 报告应基于"支配 ref 的前沿"（`non_dominated_mask` + `all(P>ref)`），否则显示的前沿点数含无法贡献 HV 的点。
- **P2-6 `constrained.py:9-13`/`capabilities`**：见 P1-6。

---

## 4. 证据与验证

| 检查 | 方法 | 结果 |
|---|---|---|
| 分区/DHV 数学 | 读 `pareto.py:101-159` + `qEHVI_IMPL.md` 的 9 个用例 | 公式一致；`dhv_2d_batch` 对 `(S,m,2)` 向量化 ✓ |
| range/ref | 读 `pareto.py:189-196`、`qehvi.py:103-115` | 显式 `ref_point` 优先 ✓；回退冻结问题见 P0-1 |
| ParEGO 窗口 | 读 `parego.py:146-174` | `len(_Y)<freeze_n` 不再冻结 ✓（v0.4 P0-1 已修） |
| constrained 文档 | 读 `constrained.py:1-18` | 已澄清"no observation noise modelling" ✓（v0.4 P0-2 部分修） |
| 复孔 alpha | 读 `dataset.py:53-55`、`bayes.py:197-223,434-445` | 逐点 alpha = (σ²/k)/var(y) 接入 ✓；但 std 是标量（P1-5） |
| qehvi 计数 | 读 `qehvi.py:69-87` | 只 `super().tell(valid)` → 计数不一致（P0-2） |

---

## 5. 修复顺序

1. **P0-1**（ref 校验 + 空分区不静默）——影响所有 qEHVI 结果。
2. **P0-2**（计数/跳过上报）——与 ParEGO 同一模式，一次改两个类。
3. **P1-1/P1-2**（第三个 GP、candidate pool 中心）——样本效率与前沿覆盖。
4. **P1-4**（router 的 ref 检查）——避免无 ref 时推荐 qEHVI。
5. **P1-5**（std 字典化）——qNEHVI 与带噪约束的前置。
6. P2 清理。

**回归门槛**：108 测试全绿 + 上述新增单测；`moo_yield_cost` 的 HV_ratio 不劣化（当前 0.95）；新增 `test_qehvi_no_ref_raises_or_warns` 后 router 在"无 ref_point"时不选择 qehvi。

---

## 6. 参考

R21（qEHVI：分区/ref 要求）、R42（HV 与固定参考点）、R12（qNEHVI：噪声多目标）、R2（复孔/alpha）、R10（带噪约束）、R8（RunHistory 状态语义）、R17（配置校验）。
