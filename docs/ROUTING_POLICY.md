# 路由策略（ROUTING_POLICY）

本文档解释 `condopt.router` 如何从问题上下文得到「算法 + 参数」，以及为什么这样设计。
代码位置：`src/condopt/router.py`。规则是确定性的、可测试的；每条候选的得分都能拆成人话。

### 编码建议（描述符）

高基数分类（one-hot 膨胀 ≥8 列）+ 绑定约束时，给分类变量配 `descriptors` **并同时**用约束感知算法
（`constrained`/`cqehvi`）：实测可行覆盖 +25%~+59%，但**最优值/HV 不变或略降**（MOO −1.4%，噪声内）；
只换编码不换算法会更差（`bayes` 可行率 0.285→0.150）。依据：`research/DESCRIPTOR_FEATURES.md`。


## 高维规则的地貌依赖（读这条再决定要不要覆盖）

8–14 维切 `cmaes` 的经验来自 10 维 rosenbrock/ackley。6/8/10/12 维扫描
（`research/DIM_BOUNDARY.md`）显示它其实**由地貌决定**：

- 窄谷型（Rosenbrock）：6 维起 cmaes 就更好，差距随维度放大 → **已知窄谷可提前显式选 cmaes**；
- 多峰型（Ackley）：6/8/12 维 bayes 反而更优，10 维只是打平 → router 的 8 维规则是
  “高维 GP 不可靠”的兜底，不等于“8 维以上 cmaes 一定更好”。

## 极小预算：代理拟合下限

模型类算法需要足够观测才能拟合，router 按**算法类型**判断而不是一刀切：

| 类型 | 建模下限 |
|---|---|
| GP 类（bayes/constrained/parego/qEHVI 家族/mfbo 家族） | `max(4, min(12, effective_dim))` |
| TPE | `max(4, min(8, effective_dim // 2))`（按箱分桶，不拟合 one-hot GP） |
| 空间填充/进化（sobol/lhs/random/grid/cmaes） | 0 |

下限取“能否拟合”而不是“统计上舒适”的样本量：GP 实现在 `n < 2(d+1)` 时会冻结超参数，因此中等维度仍可用；上限 12 保证常规实验预算（≥12）不会因为特征展开宽而退回空间填充。

低于下限（`budget <= 下限`）的算法会被压到负分，并在 `Recommendation.underpowered_budget` 里列出被屏蔽的算法与
建议的空间填充选项。

约束/多目标场景本来**没有**同时满足能力的便宜算法。**实测后**（`research/TINY_BUDGET.md`，
4 seeds）的结论是：不要退回空间填充 —— 纯 `sobol` 在「MOO + 约束」下 3 个预算里都只有
**1/4** 的运行产生非空可行前沿，而代理类 3–4/4；`mf_constrained` 上 `constrained`
（31.7/31.8/42.6）也普遍优于 `sobol`（27.7/37.3/37.3）与 `mfbo`（21.0/24.6/24.6）。

因此最终策略是**保留代理类推荐 + 显式告警**：

- `underpowered_budget` 仍列出“低于建模下限被屏蔽的算法”；
- 若该上下文没有满足能力的便宜算法，则保留代理并告警
  “预算低于建模舒适区，可能欠拟合；实测该情形下模型类仍比纯空间填充更能给出可交付证据，
  若想试空间填充可用 `algorithm='sobol'` 显式指定”；
- 只有预算低到 **≤2** 次（模型完全无从谈起）才自动放宽到空间填充；
- 需要“稳定给出前沿”的便宜退路时，`lhs` 是实测最稳的选择（4/4）。

这不是把约束丢掉：约束与多目标始终由 study 端记录、过滤与排序。

## 1. 输入：OptimizationContext

| 字段 | 影响 |
|---|---|
| `n_dims` / 各类变量计数 | 决定是否需要空间填充、是否可穷举 |
| `feature_dim`（由 `--space` 自动推导） | **GP 真正看到的维度**（数值 + one-hot）；bayes 的维度分档与初始设计点数都按它算 |
| `budget_total` / `budget_remaining` | 决定初始设计占比、是否值得建模 |
| `batch_size` | 决定批量采集策略；批量大时偏向采样或 constant-liar |
| `noise` | 高噪声 → UCB + 大 `alpha`；也提高 BO 的相对价值 |
| `smoothness` | smooth → BO；rough → matern32/采样 |
| `n_objectives` | >1 → 多目标路由（2 目标且预算足/低噪 → qEHVI，否则 ParEGO）；2 目标 + 硬约束 → cqEHVI |
| `n_constraints` | >0 → 约束路由（单目标 `constrained`；2 目标 + 约束 `cqehvi`，先可行后优化） |
| `multi_fidelity` | true → 多保真路由（`mfbo`，fidelity 作 GP 输入） |
| `compute` | 低算力 → 缩小候选池 / 偏向采样 |
| `n_observations` | ≥6 → BO 暖启动加分 |
| `domain` | chemistry 默认平滑先验 +5 |
| `all_discrete` / `cardinality` | 小离散空间 → grid 决定性优势 |

## 2. 打分规则（当前版本）

每个算法从基础分出发，按条件累加/扣分。最终排序取最高分，第二名用于置信度与备选。

### bayes（基础 45）

> 维度分档用 `effective_dim = feature_dim or n_dims`（分类变量 one-hot 后可能远大于变量个数）。

| 条件 | 分值 | 说明 |
|---|---|---|
| 有效特征维 ≤12 | +18 | GP 舒适区 |
| 12 < 有效维 ≤15 | +8 | 边缘 |
| 15 < 有效维 ≤20 | −10 | 需更多初始点 |
| 有效维 >20 | −60 | 代理不可靠，退回采样/树模型 |
| smoothness=smooth | +12 | 平滑假设成立 |
| smoothness=rough | −12 | 平滑假设易破 |
| noise ∈ {medium,high} | +5 | 代理可复用信息 |
| 预算 ≥20 | +12 | 足够冷启动 |
| 预算 12–19 | +4 | |
| 预算 8–11 | −10 | 初始设计挤占 |
| 预算 <8 | −40 | 来不及学 |
| batch >16 | −8 | qEI 贪心近似变差 |
| compute=low | −8 | 候选池缩小 |
| 先验数据 ≥6 | +10 | 暖启动 |
| 0 < 先验数据 < 1.5×有效维 | −10 | 超参易过拟合，建议先采样或固定超参 |
| domain=chemistry | +5 | 化学先验 |
| 多目标 | −200 | 用 parego/qehvi/qnehvi/cqehvi |
| 全离散且组合 ≤200 | −15 | 枚举更直接 |
| 有约束 | −8 | 约束近似 |
| 多保真 | −5 | 用 mfbo |

### parego（基础 35）

> 多目标问题会先按能力过滤：只有声明 `multi_objective` 的 `parego` 能通过。

| 条件 | 分值 |
|---|---|
| `n_objectives > 1` | +70（否则 −200） |
| 恰好 2 目标 | +5 |
| 4+ 目标 | −15（前沿覆盖下降；qEHVI/cqEHVI 只支持 2 目标） |
| 预算 ≥20 | +8 |
| 预算 <12 | −10 |
| 噪声 medium/high | +5 |
| 有效特征维 >20 | −25 |
| compute=low | −5 |
| 有约束 | −5（ParEGO 不处理约束；2 目标用 cqehvi） |

### qehvi（基础 35）

> 2 目标场景下与 `parego` 竞争；预算充足 + 低噪声时 qEHVI 胜出。

| 条件 | 分值 |
|---|---|
| `n_objectives == 2` | +75（否则 −200） |
| 预算 ≥40 | +15 |
| 预算 <25 | −20 |
| 噪声 none/low | +5 |
| 噪声 medium | −10 |
| 噪声 high | −25（应上 qNEHVI） |
| 有效特征维 >20 | −25 |

### qnehvi（基础 35）

> 2 目标 + 噪声 medium/high 时优先；低噪声让位给 `qehvi`。

| 条件 | 分值 |
|---|---|
| `n_objectives == 2` | +75（否则 −200） |
| 预算 ≥40 | +15 |
| noise high | +10 |
| noise medium | +5 |
| noise low | −20 |
| 有效特征维 >20 | −25 |

### mfbo（基础 35）

> 有 `multi_fidelity` 时能力过滤只留 `mfbo`。

| 条件 | 分值 |
|---|---|
| `multi_fidelity=true` | +90（否则 −200） |
| 预算 ≥20 | +5 |
| 有效特征维 >20 | −25 |

### cqehvi（基础 35）

> 2 目标 + 硬约束时能力过滤只留 `cqehvi`（同时声明 `multi_objective` + `constraints`）。

| 条件 | 分值 |
|---|---|
| `n_objectives == 2` | +75（否则 −200） |
| `n_constraints > 0` | +70（否则 −200） |
| 预算 ≥40 | +15 |
| 预算 <25 | −20（两阶段可能来不及覆盖前沿） |
| 噪声 none/low | +5 |
| 噪声 high | −15（MC 估计方差大，先聚合复孔） |
| 约束 >5 个 | −10（联合可行性方差大） |
| 有效特征维 >20 | −25 |

### constrained（基础 35）

> 单目标 + 约束时能力过滤只留 `constrained`；2 目标 + 约束会被 `cqehvi` 接管。

| 条件 | 分值 |
|---|---|
| `n_constraints > 0` | +70（否则 −200） |
| 预算 ≥20 | +8 |
| 预算 <12 | −10 |
| 噪声 medium/high | +5 |
| 有效特征维 >20 | −25 |
| compute=low | −5 |
| 多目标 + 约束 | −30（2 目标应上 cqehvi；>2 目标暂退回主目标 + 约束并警告） |

### 小维纯类别例外（v0.11，文献+代理战役）

当 `all_discrete` 且 `n_dims <= 8` 且 `feature_dim <= 60` 且预算 `>=50` 时，**bayes 加 35 分、tpe 减 15 分**：
小维类别空间里 GP 仍能建模变量交互，而 TPE 的独立密度忽略交互。依据：
2026 ART HTE 复现（TPE 100 次停在 91%，GP-BO 命中 99%）+ 356 条文献标签（categorical_heavy：tpe 10 / bayes 9）。
高基数、高维 one-hot、小预算仍保持 TPE 倾向。

### tpe（基础 28）

| 条件 | 分值 |
|---|---|
| 含分类/有序变量 | +8 |
| 离散/分类变量占主导（≥50%） | +20 |
| 15 < 有效特征维 ≤60 | +10 |
| 有效特征维 >60 | +15 |
| 预算 ≤25 | +6 |
| 高噪声 | +4 |
| 全离散且组合 ≤200 | −5（grid 更直接） |
| 连续光滑且预算 ≥20 | −14（GP-BO 更省样本） |

### sobol（基础 25）

| 条件 | 分值 |
|---|---|
| 预算 <10 | +30 |
| 预算 10–20 | +8 |
| 预算 >60 | −10 |
| 维度 >20 | +20 |
| batch ≥8 | +12 |
| 全离散 | +8 |
| 无先验数据 | +15 |
| compute=low | +8 |
| 光滑且预算 ≥20 | −8 |

### lhs（基础 22）

| 条件 | 分值 |
|---|---|
| batch ≥8 | +16 |
| batch = 1 | −8 |
| 预算 ≤8 | +22 |
| 全离散 | +6 |
| 无先验数据 | +8 |
| 维度 >15 | −6 |

### random（基础 15）

| 条件 | 分值 |
|---|---|
| 维度 >25 | +22 |
| 15 < 维度 ≤25 | +6 |
| 预算 <6 | +25 |
| compute=low | +10 |
| smoothness=rough | +8 |
| noise=high | +5 |
| 全离散 | +4 |
| 有先验数据 | −5 |

### grid（基础 5）

| 条件 | 分值 |
|---|---|
| 全离散且组合 ≤200 | +55 |
| 组合 ≤200（非全离散） | +20 |
| 组合 200–2000 | +8 |
| 预算能覆盖全部组合 | +25 |
| 维度 >4 | −20 |
| 非全离散 | −5 |
| 组合 > 2×剩余预算 | −15 |

## 3. 能力过滤（先过滤再打分）

`route()` 在打分前先按 `Capabilities` 过滤候选算法：

| 上下文条件 | 要求能力 |
|---|---|
| `n_objectives > 1` | `multi_objective` |
| `n_constraints > 0` | `constraints` |
| `multi_fidelity` | `multi_fidelity` |
| `batch_size > 1` | `batch` |
| 含分类/有序变量 | `categorical` |

- 当前内置 18 个算法：`parego`/`qehvi`/`qnehvi`/`qehvi_nd`/`cqehvi`/`cqnehvi`/`momfbo`/`mfparego`/`mqehvi_nd` 声明 `multi_objective`，
  `constrained`/`cqehvi`/`parego`/`mfbo`/`momfbo` 声明 `constraints`，
  `mfbo`/`momfbo` 声明 `multi_fidelity`，其余声明 `batch/categorical/mixed`；
  能力过滤覆盖上述全部场景，仅极少数无解组合会退回子集打分并给能力警告。
- 新增算法只要在 `AlgorithmSpec.capabilities` 声明能力，就会自动进入过滤逻辑。

## 4. 置信度与备选

- `margin = 第一名 − 第二名`；`≥20` → high，`≥8` → medium，否则 low。
- 备选取第 2、3 名，附一句各自的 top 理由。
- `low` 置信度意味着规则冲突或边界情况，skill 必须向用户说明并建议小批量试探。
- 多目标（`n_objectives > 1`）不再强制降为 low；只有目标数 >3 时才降级（ParEGO 前沿覆盖有限）。

## 5. 参数策略（按算法）

### bayes

```
eff       = feature_dim（数值 + one-hot 后的真实维度）
n_initial = min(max(3, eff+1), 20, max(3, 0.4 × 剩余预算))
kernel    = smooth 且 eff≤8 → rbf；rough → matern32；否则 matern52
alpha     = none 1e-8 / low 1e-6 / medium 1e-4 / high 5e-3
acquisition = high 噪声 → ucb(κ=2.5)；否则 ei
batch_strategy = batch≥4 → constant_liar；batch 2–3 → kriging_believer；否则 greedy
n_candidates   = compute low 512；high 且 eff≤8 2048；否则 1024
local_frac     = eff≤6 0.4；否则 0.25
```

GP 超参优化只在观测数 ≥ max(6, 2×eff) 时开启，否则冻结初始长度尺度（避免小样本过拟合）。

### 其余

- `parego`: `n_initial = min(max(2*(eff+1), 2*m), 20, 0.5×剩余预算)`；
  `rho=0.05`、`weight_alpha=1.0`、`acquisition=logei`、`alpha=1e-4`
- `qehvi`: 复用 bayes 参数 + `mc_draws=128/256`；`n_initial = min(max(2*(eff+1),4),20,0.5×剩余预算)`
- `qnehvi`: qehvi 参数 + `n_base=32/64`、`n_y=64/128`、`n_candidates=512`、`noise_std=0`（用观测 std）
- `mfbo`: `fidelity_levels="0.25,0.5,1.0"`、`cost_exponent=1.0`、`min_full_fraction=0.25`；
  `n_initial = min(max(4,2*(eff+1)),20,0.4×剩余预算)`、`n_candidates=512/1024`
- `constrained`: 复用 bayes 参数 + `mc_draws=128/256`、`constraint_margin=0`、`feasibility_threshold=0.02`
- `cqehvi`: 复用 bayes 参数 + `mc_draws=128/256`、`constraint_margin=0`、`feasibility_threshold=0.02`；
  `n_initial = min(max(2*(eff+1),4),20,0.5×剩余预算)`
- `sobol`: `scramble=true, dedupe=true`
- `lhs`: 预算 ≥16 → `optimization=random-cd`，否则 `none`
- `random`: `dedupe=true`
- `grid`: `max_points = min(剩余预算, 500)`，`levels=5`

## 6. 初始设计与阶段计划

- 选 `bayes` 时，前 `n_initial` 个点由 `sobol` 生成（空间填充），之后交给 GP 采集。
  `Recommendation.design` 明确写出 `{"algorithm":"sobol","n":7,"main":"bayes","remaining":17}`，
  `plan` 给出阶段列表，skill 直接展示给用户。
- 选 `grid` 时 `design` 给出可枚举点数；选采样算法时整段预算都给该算法。

## 7. 中途重路由（route_round）

- 输入：每轮“累计最优值”序列（`rounds.json` 的 `best`）。
- 若最近 `patience`（默认 3）轮最优值没有改善（容差 1e-6），router 推荐**一批**
  全局探索（有效维 ≤20 用 `sobol`，否则 `random`），并记录原因。
- Study 的 `auto_route` 在每次 `ask` 前调用；探索批次结束后自动回到原算法（配置不变）。
- 重路由不是“换算法”，而是“插入探索”；只有用户/AI 通过 `override` 才会真正换算法。
- 一旦发生 `override`，`manual_override=true`，自动重路由暂停，直到显式 `condopt auto`。

## 8. 覆写与审计

- `condopt override --run ... --algorithm ... --reason "..."` 必须给理由，否则报错。
- 决策写入 `decisions.jsonl`：`initial_route / auto_reroute / override / resume_auto_route`，
  含算法、参数、理由、上一步配置、候选分数。
- `condopt auto --run ...` 恢复自动重路由，也写一条审计记录。
- **显式指定算法**：`Study.create(algorithm="mfbo")` 会在创建时钉住该算法（写 `manual_selection` 决策，
  `auto_route` 暂停）；要恢复自动路由请运行 `condopt auto`。这样算法验收/基线对照不会被静默改掉。
- **校准权重（v0.7）**：`route/init --weights weights.json`，JSON 形如
  `{"weights": {"bayes": 1.2, "random": 0.8}}`；正分乘 factor、负分除 factor、0 表示禁用，
  空权重 = 手工默认（零行为变化）。权重来自 LOPO-CV 校准，可回退、需在 decisions 里记录来源。
- 目的：AI 可以灵活，但实验战役结束后任何人能复盘“为什么第 5 轮换了算法”。

## 9. 已知限制与扩展方式

限制：

- 规则权重来自经验与文献，尚未用 benchmark 自动校准；
- 3+ 目标 EHVI（递归分区）仍未实现；>2 目标 + 约束现由 `parego`（P(feasible) 择优）承担；
- 未考虑实验成本随条件变化（cost-aware）；
- 未利用历史相似任务的迁移信息。

扩展一个新算法（三步）：

1. 在 `src/condopt/algorithms/` 写 `Optimizer` 子类并 `@register`，声明 `AlgorithmSpec`
   （参数 schema、能力、参考文献、when）；
2. 在 `router._rules()` 加打分函数、`_params_for()` 加参数策略；
3. 在 `tests/test_router.py` 加“该算法应在什么上下文胜出”的用例，跑 `pytest`。

这样“算法多而杂”就变成“注册表 + 规则”，加算法不会破坏既有行为，也不影响 skill 的调用方式。

## 多保真 + 硬约束的降级规则

`mfbo` 现已声明 `multi_fidelity` + `constraints`，同时请求两者时直接路由到 mfbo。
若出现其他无解组合（如 2 目标 + 约束 + 多保真），router 仍优先保证硬约束：
退回满足 `constraints` 的子集打分（通常是 `constrained`），并给出两条 warning
（能力的通用提示 + “放弃保真度通道/成本模型不生效”）。避免 mfbo 静默忽略约束。

## 校准数据（11 problems / budget 10 / seed 0）

默认问题集现覆盖：单目标、约束、2 目标、2 目标+约束、多保真、多保真+约束。
最近一次校准：

- metric coverage: true_best 36 / feasible_best 1 / true_hv 3 / cost_aware_best 1 / cost_aware_feasible_best 1
- labels: constraint_easy→constrained, moo_yield_cost→parego, moo_constrained→cqehvi,
  mf_suzuki→mfbo, mf_constrained→mfbo
- default rules top1 0.636；in-sample 0.818；**LOPO top1 0.727 / top2 0.909 / mean_regret 0.019**
- 校准权重写入 `configs/router_weights_calibrated.json`（仅供参考，**未设为默认**，
  避免单 seed 过拟合；qehvi/cqehvi 得分偏低来自单 seed 小样本）。

多 seed（0/1/2）复测（`runs/calibration_round60`，权重另存 `configs/router_weights_calibrated_multiseed.json`）：

| 口径 | top1 | top2 | mean regret |
|---|---|---|---|
| default rules | 0.818 | 0.909 | 0.006 |
| in-sample | 0.909 | 0.909 | 0.001 |
| LOPO | 0.818 | 0.909 | 0.026 |

结论：默认规则在多 seed 下已与 LOPO 持平；单 seed 权重（如 qehvi 0.60）与多 seed（1.57）
差异巨大，说明单 seed 校准权重不应直接启用。继续保留默认规则，校准权重仅作参考。

## 多保真 + 多目标的降级

`momfbo` 现已声明 `multi_objective` + `multi_fidelity`（+ `constraints`），
多保真 + 多目标直接路由到 momfbo。仅当出现更复杂的无解组合时，router 才退回
「优先保证多目标」的子集打分并 warning。

## 目标数超出算法范围时的降级

`qehvi`/`qnehvi`/`cqehvi` 只实现 2 目标。若 `n_objectives != 2`：

- 有硬约束 → 退回 `constrained`（主目标 + 约束生效，其余目标仅记录），并 warning；
- 无约束 → 退回 `parego`（Pareto 折中），并 warning。

避免 router 选出会在 `build()` 里直接报错、导致 study 无法启动的算法。
## 路由矩阵体检

`scripts/audit_routing.py` 枚举 (维度 × 目标数 × 约束 × 多保真 × 混合空间 × 噪声) 共 96 个上下文：
逐个 route、核对所选算法的 capabilities 覆盖必需能力、再用匹配空间 build+ask。
任何“选出的算法跑不起来/能力不满足”都会非零退出；当前 96/96 通过。
### 多目标 + 多保真的默认算法

四处独立设置一致指向 `mfparego`，因此**多保真 + 多目标统一默认 `mfparego`**：

| 设置 | mfparego | momfbo | 备注 |
|---|---|---|---|
| 2 目标 + MF（30） | 672.3 | 666.2 | |
| 2 目标 + MF（40） | 676.1 | 670.3 | |
| 2 目标 + 约束 + MF（40） | 642.5 | 635.6 | 可行 HV |
| 3 目标 + MF（40） | 27399 | 26869 | 全保真点 35.3 vs 26.0 |
| 4 目标 + MF（40） | 515205 | 507932 | 全保真点 36.0 vs 24.3 |
| 校准标签（12 问题，预算 12） | 胜 | — | LOPO top1 0.75 / top2 1.00 |

校准权重的 A/B（LOPO）：top1 0.75 持平，**top2 0.917 → 1.000**；因权重只在 budget 12 的12 问题上拟合，暂不设为默认，可用 `condopt route --weights <file>` 显式启用。

`momfbo` 仍作为候选保留（手动指定时可用其调好的 `min_full_fraction=0.4`）；
`mqehvi_nd` 仍是实验性第三候选。详见
[`research/ROUTER_CALIBRATION_NEW_BENCHMARKS.md`](../research/ROUTER_CALIBRATION_NEW_BENCHMARKS.md)。

`mqehvi_nd`（HV 型）在三处均落后（626.9 / 607.1 / 25587），仅作实验性候选。

### 默认值来源

所有默认值集中列在 [`docs/DEFAULTS.md`](DEFAULTS.md)（含实测数字与证据文件）；
新增默认值时必须补一行。

### 负结果与默认值来源

router 的默认值只来自有基准支持的方案；被判负的方案记录在
[`research/NEGATIVE_RESULTS.md`](../research/NEGATIVE_RESULTS.md)（cqnehvi、Sobol 播种 CMA-ES、
mqehvi_nd、TPE+描述符、垃圾描述符对照、真实六点 LOO 失败、4 维 EHVI 性能坑）。

### 场景矩阵（可执行）

[`docs/SCENARIOS.md`](SCENARIOS.md) 列出 14 个真实场景与默认算法，由
`scripts/scenario_matrix.py` 实际调用 router 校验（已纳入 release_check）。
本轮据此修正了高维连续边界：8–14 维光滑连续改由 `cmaes` 胜出
（实测 −18.1 vs bayes −155.4），6 维以下仍为 `bayes`。

### 覆盖回归防护

`tests/test_coverage.py` 对 72 个能力上下文断言：

1. 每个上下文都有满足全部所需能力的推荐（否则能力过滤出现空洞）；
2. **多保真 + 多目标**至少有两个候选算法（防止单点依赖）。

`scripts/audit_routing.py` 额外打印每个能力组合的候选集合与 `single_candidate` 计数。

