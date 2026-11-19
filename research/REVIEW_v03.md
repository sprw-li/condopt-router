# REVIEW_v03 — 对 condopt v0.3 多目标实现的文献级审查

> 审查者：algo-research ｜ 日期：2026-11-05 ｜ 代码快照：15:17–15:24（`parego.py` 214 行、`pareto.py` 177 行、`prepare.py` 143 行、`router.py` 719 行）
> 格式：**`文件:行` → 问题 → 文献依据 → 修复建议 → 优先级**。P0 = 正确性/可比性错误；P1 = 显著影响样本效率或可解释性；P2 = 工程质量。
> 关联：`PAREGO_SPEC.md`（Round 3 规格）、`CONSTRAINT_IMPL.md`（本轮约束模型）、`MULTI_OBJECTIVE_AND_NOISE.md`（qEHVI/概率 Pareto）。

---

## 0. 总结

| 编号 | 位置 | 一句话 | 优先级 |
|---|---|---|---|
| P0-1 | `parego.py:133-148` | ParEGO 用**动态 min/max** 归一化，新增一个观测会改变所有历史点的标量化目标（实测 0.525→0.3625） | **P0** |
| P0-2 | `parego.py:126,131` + `pareto.py:24-36` | 缺任一目标值的观测（失败/部分 tell/约束-only）会让 `tell` 抛 `ValueError`，整条 campaign 崩溃 | **P0** |
| P0-3 | `pareto.py:125-135` + `study.py:518-525` | HV 参考点**由数据推导**（`lows−0.05·span`），不同算法/seed/轮次的 HV 不可比；报告里的 HV 数值会误导（实测同一前沿：动态 ref HV=26.15 vs 固定 ref=1100） | **P0** |
| P1-1 | `parego.py:182-195` | 每个 proposal 新建并拟合一个 inner GP；seed 由数据长度推导，**两次独立 `ask(1)` 会得到同一个点**（pending 未排除） | P1 |
| P1-2 | `parego.py:196-206` | 批量只做"重复点重采样"，**无多样性机制**（无 local penalization/λ 去重），batch 点可任意接近 | P1 |
| P1-3 | `base.py:139-209` | `Objective` 无 `ref_point`/`ideal_point`/`constraint_specs` → ParEGO 无法固定归一化窗口、约束无法表达（见 `CONSTRAINT_IMPL.md`） | P1 |
| P1-4 | `router.py:596-611,619-630` | 能力过滤找不到候选时**静默回退**全量打分，无 `degraded`/warning；当前 `n_constraints>0` 必然触发（无算法声明 constraints） | P1 |
| P1-5 | `router.py:213-241,636-637` | 多目标路径只有 ParEGO（能力过滤后唯一候选）；无 qEHVI 切换阈值；`m>3 → confidence=low` 是硬编码 | P1 |
| P1-6 | `prepare.py:96-110` | `constraints` 若按 skill 问卷传**列表**会被塞进 `n_constraints` → `TypeError`（int 比较）；无 Constraint 规格解析 | P1 |
| P1-7 | `parego.py:133-148` | 某目标恒定时该列置 0 但不重归一化 λ、也不告警；全部目标恒定 → g 恒 0，GP 退化为常数 | P1 |
| P2-1 | `pareto.py:52-86` | `m>2` 的 O(n²) 支配检测未处理 NaN；`m>3` 无 HV 近似 | P2 |
| P2-2 | `pareto.py:125-128` | `m>2` 时 `hypervolume=None`，报告静默不显示 | P2 |
| P2-3 | `parego.py:115-118,380` | `state_dict` 未存 proposal 计数/λ 历史；续跑后"已用 proposal 数"不可知 | P2 |
| P2-4 | `router.py:244-253` | `_rules()` 与 `registry` 可能漂移（注册了但无规则的算法永远不会被选中） | P2 |
| P2-5 | `router.py:641-642` | `params["n_initial"]=design["n"]` 只对 bayes/tpe 生效；parego 依赖两者恰好相等（当前相等但脆弱） | P2 |
| P2-6 | `prepare.py:96-102` | 未识别的 answers 字段被静默丢弃（拼写错误不可见） | P2 |
| P2-7 | `parego.py:165-170` | 直接使用 `ParEGO`（非经 router）时初始设计不受"≤40% 预算"约束 | P2 |
| P2-8 | `parego.py:172-206` | λ 未记录到 `state_dict`/decisions → 无法审计"这一轮朝哪个方向优化" | P2 |

**总体**：ParEGO 的"零新内核"集成方式正确（复用 `GaussianProcessBO` 的 GP/候选池），Pareto 工具与 router 的多目标分支结构清晰。**主要问题是"归一化窗口"和"HV 参考点"两个口径问题**——它们不影响代码跑通，但会让"多目标优化结果"不可比、不可复现，必须在扩展 qEHVI 之前修掉。

---

## 1. P0 详述

### P0-1 `parego.py:133-148` — 动态 min/max 归一化改变历史标量化目标

```python
def _normalized(self):
    for j in range(m):
        low, high = float(np.min(column)), float(np.max(column))   # ← 每轮重算
        out[:, j] = (column - low) / (high - low)  # 或 (high-column)/(high-low)
```

每次 `_propose` 都重新计算 min/max，而 `_scalar_values()`（`parego.py:208-211`）用它对**所有历史观测**重新标量化 → 新增一个观测（首/末极值变化）后，**历史点的 `g` 值整体改变**，GP 的训练目标（以及 EI 的 `best`）在轮次间不连续。

**实测**（λ=(0.5,0.5), ρ=0.05，两个目标）：

```
第 1 轮数据 [(90,10),(80,12)]：g = [0.525, 0.525]
第 2 轮加入 (95,8) 后，同样两个历史点：g = [0.3625, 0.525]
```

后果：①同一批数据在不同轮次下"改进量"含义不同 → EI 的 `best` 不可比，可能误判停滞/改进；②`route_round` 的停滞检测（基于 best-so-far）被污染；③benchmark 的轨迹不可复现地依赖"何时加入极值点"。

**文献依据**：R11（ParEGO 原文用固定的目标范围/已知上下界做标量化）；`PAREGO_SPEC.md` §2.2（固定窗口；动态窗口会破坏 EI 的可比性）。

**修复（两选一，推荐 A）**

```python
# A) 固定窗口（推荐）：在 Objective 里加 ref_point / ideal_point（见 P1-3），
#    首次 tell 时冻结并写入 state：
def _maybe_init_normalizer(self):
    if self._normalizer is not None:
        return
    if self.objective.ref_point:
        lo = np.asarray(self.objective.ref_point, float)
    else:                                    # 无 ref：用首个窗口 + 余量，之后冻结
        lo = self._Y.min(axis=0) - 0.05 * (self._Y.max(axis=0) - self._Y.min(axis=0))
    hi = (np.asarray(self.objective.ideal_point, float) if self.objective.ideal_point
          else self._Y.max(axis=0) + 0.05 * (self._Y.max(axis=0) - self._Y.min(axis=0)))
    # 注意：方向翻转在此完成（min 目标 → 分数取负，统一"越大越好"）
    self._normalizer = (lo, hi)
```
```python
# B) 若要保留动态行为：至少显式暴露 normalization="dynamic" 参数 + warning，
#    并把"每轮变化的 g 历史"写进 decisions，供审计与 benchmark 关掉。
```
**验收**：单测 `test_parego_normalization_frozen`：加入第 3 个观测后，前 2 个观测的标量化值与上一轮**逐位相同**；`moo_yield_cost` 的 HV 轨迹在 5 seeds 下方差下降。

### P0-2 `parego.py:126,131` + `pareto.py:24-36` — 缺目标值的观测使 `tell` 崩溃

```python
values = np.vstack([observation_values(obs, self.objective) for obs in observations])
```
`observation_values`（`pareto.py:24-36`）对每个目标名找不到值就 `raise ValueError`。真实实验里常见：①某次实验失败只回 `y` 不回副目标；②`tell` 部分结果（v0.3 已允许部分 tell，见 `REVIEW_v01.md` P0-5 的修复）；③多目标 study 里有人只测主目标。

后果：一条坏观测 = 整个 `tell` 失败，后续所有轮次卡住（比丢弃这条数据严重得多）。

**文献依据**：R8（SMAC3 RunHistory 用状态机记录 FAILED/CRASHED，而不是让优化器崩溃）、R10（约束/目标缺失需显式处理）。

**修复**：在 `ParEGO.tell` 里做掩码：

```python
def tell(self, observations):
    observations = list(observations)
    rows, xs = [], []
    for obs in observations:
        try:
            v = observation_values(obs, self.objective)
        except ValueError:
            self._skipped += 1                 # 记录，不中断
            continue
        if not np.all(np.isfinite(v)):
            self._skipped += 1
            continue
        rows.append(v); xs.append(dict(obs.x))
    super().tell([o for o in observations])    # t 计数仍按收到的观测
    if not rows:
        return
    self._Y = np.vstack(rows) if len(self._Y) == 0 else np.vstack([self._Y, rows])
    self._xs.extend(xs)
```
并在 `report`/`decisions` 里写 `skipped_incomplete=N`；若跳过比例 >30% 则 warning。

**验收**：`test_parego_tell_tolerates_missing_objective`、`test_parego_tell_tolerates_partial_batch`。

### P0-3 `pareto.py:125-135` + `study.py:518-525` — HV 参考点由数据推导，HV 不可比

```python
if reference is None:
    lows = np.min(scores, axis=0); highs = np.max(scores, axis=0)
    reference_point = lows - 0.05 * span        # ← 依赖当前数据
```
不同算法/seed/轮次的 `lows` 不同 → **HV 数值在算法间不可比**；`Study.report`（`study.py:593-600`）直接显示该 HV。实测（同一前沿，两个不同 ref）：

```
数据推导 ref=(79.25, 7.8) → HV=26.15
固定 ref=(0,0)               → HV=1100.0
```

**文献依据**：R11/R21（HV 必须在固定参考点下比较；`BENCHMARK_SPEC.md` §5.2/§6.4 已规定"ref 固定、先归一化"）；R32（benchmark 协议要求跨 run 可比）。

**修复**：

1. `Objective` 增加 `ref_point`（原始目标单位，含方向语义）；`summarise` 优先用它；
2. `Study` 在 `problem.json`/`config.json` 里持久化 ref（创建时确定，之后冻结）；
3. `Study.report` 在无 ref 时显示 "HV（相对当前数据，不可跨 run 比较）" 或干脆不显示；
4. `prepare.py` 从 answers 读 `ref_point`，缺省用启发式并在 `objective.json` 写明（`warnings: ["ref_point 使用默认值，跨 run 不可比"]`）；
5. benchmark 用 `BENCHMARK_TIER_CD.md` §6.4 的固定 ref（`(30,20.5)` 归一化后 `(0,0)`）。

**验收**：`test_hv_uses_objective_ref_point`、`test_study_persists_ref_point`；`moo_yield_cost` 上同一前沿用固定 ref 的 HV 在 5 seeds 下一致（±1e-9）。

---

## 2. P1 详述

### P1-1 `parego.py:182-195` — 每个 proposal 重新建/拟合 inner GP；seed 由长度推导

- 每提案一次 `GaussianProcessBO(...prior=...)` → 每次 fit 全量 GP（ARD + 超参优化），batch=8 时 8 次；n=100 时一轮 ask 可能数秒（可接受但可优化）。
- `seed=self.seed + 104729*(len(self._xs)+1) + len(batch)`：**同一 `_xs` 长度、同一 batch 长度会得到同一 seed** → 两次独立 `ask(1)`（中间没有 tell，例如 pending 场景/外部实验延迟）会得到**相同的推荐点**；且 `existing` 只排除已 tell 的点，不排除 pending。
- 修复：用持久 proposal 计数器 `self._n_proposed += 1` 作为 seed 的一部分；把 pending 点（`Study.pending` 或 optimizer 自维护的 `_proposed`）纳入去重。
- 文献依据：R3（批量/异步推荐需 pending 感知）、R9（Optuna trial 状态机）。

### P1-2 `parego.py:196-206` — 批量无多样性机制

只检查"不与已测点/同批点重复"，没有距离/惩罚项；batch=8 时点可能扎堆（尤其离散空间）。`PAREGO_SPEC.md` §6 推荐：每点独立 λ + 局部惩罚（`SPRINT2_SPEC.md` §2 的 `1−exp(−d²/2ℓ²)`）。
**修复**：在 `_propose` 返回前对 `batch` 里已有点做局部惩罚（或让 `ask` 在 `_proposal` 层面把已选点乘惩罚）；测试：`test_parego_batch_min_distance`。
**文献依据**：R35（local penalization）、R11（ParEGO 批量扩展常用不同 λ+分散）。

### P1-3 `base.py:139-209` — `Objective` 缺 `ref_point`/`ideal_point`/`constraint_specs`

ParEGO 的 P0-1 与约束功能都卡在这里。**修复**：按 `PAREGO_SPEC.md` §1 与 `CONSTRAINT_IMPL.md` §1.2 加字段（向后兼容：空=旧行为），`from_dict`/`to_dict` 同步；`Objective` 校验长度一致。
**文献依据**：R11（固定范围标量化）、R34/R10（约束规格）、R17（BayBE 把 domain/ref 写入 campaign 配置）。

### P1-4 `router.py:596-611,619-630` — 能力过滤静默回退

```python
allowed = {... if required <= set(registry.spec(name).capabilities)}
return allowed or None          # ← 无人满足 → None → 静默全量打分
```
当前 `n_constraints>0` → `required={"constraints"}` → **没有算法声明** → `None` → 所有算法照常打分（bayes 仅 -8 分），警告里也只说"近似"。应在 `None` 时返回 `degraded` 列表并写第一条 warning（"无算法声明 required capabilities: constraints"），`Recommendation.degraded` 字段（`CAPABILITIES_SPEC.md`）落地前可先塞进 `warnings[0]`。
**文献依据**：R8（SMAC3 Facade 在能力不满足时明确拒绝/降级）、`CAPABILITIES_SPEC.md` §6 降级顺序。

### P1-5 `router.py:213-241,636-637` — 多目标只有 ParEGO，无 qEHVI 切换条件

`_capability_filter` 下只有 `parego` 声明 `multi_objective` → 多目标永远 ParEGO。`_score_parego` 对 m≥4 只警告。缺少的切换条件（写入 `qEHVI_SPEC.md` §5）：m∈{2,3} 且 budget≥40 且噪声低 → qEHVI 应优于 ParEGO；m>3 → 标量化/NSGA-II。`confidence="low" if m>3` 是硬编码，应与 Pareto 前沿覆盖诊断（如每目标极值覆盖）挂钩。
**修复**：等 `qehvi` 注册后加能力位；`confidence` 改为基于"前沿覆盖指标"（`pareto.py` 的 `n_front`/各目标极值是否被触及）。

### P1-6 `prepare.py:96-110` — constraints 列表导致 TypeError

`ALIASES["constraints"]="n_constraints"`，若 answers 写 `"constraints": [{"name":"purity","threshold":95,"direction":">="}]` → `context_payload["n_constraints"]` 是 list → `OptimizationContext.__post_init__` 里 `int` 比较报 `TypeError`。
**修复**：见 `CONSTRAINT_IMPL.md` §3 的 `prepare.py` 改法（列表 → `constraint_specs`；数字 → 计数；`n_constraints` 从 specs 派生）。

### P1-7 `parego.py:133-148` — 恒定目标列/全恒定静默

`if high-low < EPS: out[:,j]=0.0` 后没有 warning，且 λ 仍在该维分配（虽然乘 0 无影响，但"目标恒定"本身是数据/配置错误）。全部恒定 → `g≡0`，GP 学常数，`ask` 仍返回点（浪费预算）。
**修复**：`tell` 时检测并写 warning；若全部目标方差 <EPS → 抛 `ValueError("所有目标都是常数，无法做多目标优化")`。

---

## 3. P2（择机）

- **P2-1 `pareto.py:52-86`**：`m>2` 的 O(n²) 支配检测在 n>2000 时慢；建议按第一目标排序后扫描（O(n log n + n·k)）或分块；NaN 行需先剔除。
- **P2-2 `pareto.py:125-128`**：`m>2` 不计算 HV（`None`）→ 报告应显示"m>3 暂不计算 HV"而不是空白；后续用 MC 近似。
- **P2-3 `parego.py:115-118`**：`state_dict` 加 `{"n_proposed": ..., "lam_history": ...}`；否则续跑后 seed/审计信息丢失。
- **P2-4 `router.py:244-253`**：加 import-time 断言：`set(registry.names()) == set(_rules())`（或记录未被路由的算法并 warn），防止新算法"注册了但不可选"。
- **P2-5 `router.py:641-642`**：`_build_plan` 对 parego 也应写 `params["n_initial"] = design["n"]`（当前恰好相等，脆弱）。
- **P2-6 `prepare.py:96-102`**：未识别字段加 warning（`unknown answer keys: [...]`），避免用户把 `"budget": "24"` 拼错成 `"budgets"` 后静默用默认值。
- **P2-7 `parego.py:165-170`**：`n_initial` 上限 20 且无"≤40% budget"约束（router 有，直接实例化没有）；把该逻辑下沉到 `Optimizer` 或参数校验。
- **P2-8 `parego.py:172-206`**：λ 未写入 `state_dict`/`decisions` → 无法审计"轮次方向"；建议每轮记 `{"lam": ..., "rho": ...}`（体积小、审计价值高）。

---

## 4. 证据附录（本审查实际运行）

| 检查 | 方法 | 结果 |
|---|---|---|
| 动态归一化影响 | 复算同两个历史点在加入第 3 点前后的标量化值 | `[0.525,0.525]` → `[0.3625,0.525]`（λ=(0.5,0.5), ρ=0.05） |
| HV 参考点可比性 | `pareto.summarise`（数据推导 ref） vs `hypervolume_2d(front, fixed=(0,0))` | 26.15 vs 1100.0（同一前沿） |
| 支配/HV 正确性 | 手工构造 2D 前沿（(1,3),(2,2),(3,1)） | `hypervolume_2d` 与手算面积一致 |
| 代码结构 | 只读 `parego.py`/`pareto.py`/`prepare.py`/`router.py` | 无写操作 |

---

## 5. 修复顺序（建议）

1. **P0-1 + P1-3**（一起做：`Objective` 加 `ref_point/ideal_point`，ParEGO 冻结窗口）——影响所有多目标结果的可比性。
2. **P0-3**（HV 固定 ref + Study 持久化 + report 口径）——benchmark/报告的可信度。
3. **P0-2**（缺目标值容错）——鲁棒性，避免真实实验把 campaign 卡死。
4. **P1-4/P1-6**（能力过滤 warning + constraints 解析）——为 `CONSTRAINT_IMPL.md` 铺路。
5. **P1-1/P1-2**（pending 去重 + 批量多样性）——样本效率。
6. P2 清理。

**回归门槛**：修完后 `pytest` 全绿 + `moo_yield_cost` 5 seeds 的 HV 轨迹用固定 ref 计算，ParEGO 的 HV_ratio ≥0.85 且跨 seed 方差下降。

---

## 6. 参考

R11（ParEGO 原始定义/固定范围）、R21（EHVI/HV 固定 ref）、R32（benchmark 协议可比性）、R8（RunHistory 的 failed 状态）、R35（批量局部惩罚）、R10（缺失/带噪观测）、R17（campaign 配置持久化 ref）、R9（trial 状态机与 pending）。
