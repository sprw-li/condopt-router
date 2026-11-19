# 算法目录（ALGORITHM_CATALOG）

> 本文件是「条件优化算法」的结构化目录：每个算法的适用条件、参数语义、建议范围、失效模式、实现位置与文献依据。
> 机器可读的真相源是代码里的 `AlgorithmSpec`（`condopt spec --json`）；本文件解释这些字段背后的理由。
> 文献编号 `[Rn]` 对应 [`research/REFERENCES.md`](../research/REFERENCES.md)。

## 0. 总览

| registry name | 家族 | 一句话 | 能力 | 实现 |
|---|---|---|---|---|
| `random` | sampling | 均匀随机采样，永远保留的基线 | batch/mixed/categorical/high_dim/cheap | `algorithms/sampling.py` |
| `sobol` | sampling | 低差异序列，均匀覆盖空间 | batch/mixed/categorical/space_filling | `algorithms/sampling.py` |
| `lhs` | sampling | 拉丁超立方，单批内每维分层 | batch/mixed/space_filling | `algorithms/sampling.py` |
| `grid` | sampling | 网格/析因穷举 | batch/mixed/categorical/exhaustive | `algorithms/sampling.py` |
| `bayes` | model_based | GP 代理 + EI/UCB/PI/TS 采集 | batch/mixed/categorical/continuous/noisy/expensive | `algorithms/bayes.py` |
| `cmaes` | evolutionary | CMA-ES（rank-μ 更新）：单位立方体上维护 N(m, sigma²C)，按排名更新均值/协方差；中高维连续光滑问题可替代 GP | batch/continuous/mixed/noisy | `algorithms/cmaes.py` |

10 维对比（`scripts/benchmark_highdim.py`，budget 80、batch 8、3 seeds，指标 best_score，越高越好）：
rosenbrock10 cmaes **-14.4** vs bayes -215.0 / sobol -409.4 / random -616.2；ackley10 cmaes **-2.13** vs bayes -5.04 / tpe -7.02 / sobol -7.85 —— 10 维连续场景 CMA-ES 明显领先。
参数来自 `scripts/sweep_cmaes.py` 扫描：sigma0=0.1、c_mu=0.5（对比旧默认 sigma0=0.3：-82.3/-4.63）；曾尝试 Sobol 播种，实测变差，已放弃并在代码注释中记录。

描述符与算法搭配（Tier C 12 级溶剂，budget 20，5 seeds）：bayes 用描述符 92.02/regret 0.35 vs one-hot 91.86/0.51（略优）；tpe 用描述符 84.07/8.30 vs one-hot 90.22/2.16（明显变差）。结论：描述符优先配 GP 类算法；描述符还会缩小 feature_dim，router 在高基数分类场景可能自动从 tpe 转向 bayes（测试已固定）。
| `tpe` | model_based | 两组核密度比 l/g，天然支持混合/分类空间 | batch/mixed/categorical/continuous/noisy/cheap | `algorithms/tpe.py` |
| `parego` | model_based | 多目标：随机加权 + 增广 Tchebycheff，复用 GP+EI（观测 objective_std 逐点传入 alpha）；支持硬约束（候选按 P(feasible) 择优） |

噪声传递对比（`scripts/benchmark_noise.py`，`moo_yield_cost` 注入 σ=3 高斯噪声，budget 30，3 seeds，指标为真实目标 HV）：noise_aware=True **637.2** vs False **620.6**（+2.7%），默认开启。 batch/mixed/categorical/continuous/noisy/multi_objective/constraints | `algorithms/parego.py` |
| `constrained` | model_based | 约束：先 PoF 找可行点，再 MC 估计带可行性的 EI | batch/mixed/categorical/continuous/noisy/constraints | `algorithms/constrained.py` |
| `qehvi` | model_based | 2 目标：分区公式精确 EHVI + MC，直接优化超体积 | batch/mixed/categorical/continuous/multi_objective | `algorithms/qehvi.py` |
| `qnehvi` | model_based | 2 目标带噪：嵌套 MC 估 EHVI（baseline 前沿不确定性） | batch/mixed/categorical/continuous/multi_objective/noisy | `algorithms/qnehvi.py` |
| `cqehvi` | model_based | 2 目标 + 硬约束：可行域内 EHVI + 两阶段 PoF | batch/mixed/categorical/continuous/multi_objective/constraints | `algorithms/cqehvi.py` |
| `mfbo` | model_based | 多保真：fidelity 作 GP 输入 + `EI/cost^α`，粗筛后全保真确认；含硬约束时乘 P(feasible) | batch/mixed/categorical/continuous/noisy/multi_fidelity/constraints | `algorithms/mfbo.py` |
| `qehvi_nd` | model_based | 3–4 目标 MC-EHVI：逐目标 GP 后验采样 + 递归切片 HV 估计期望超体积改进，支持硬约束 P(feasible)（可选，非默认） | batch/mixed/categorical/continuous/noisy/multi_objective/constraints | `algorithms/ehvi_nd.py` |

3 目标 + 约束对比（`scripts/benchmark_3obj.py`，`moo3_constrained`，2 seeds，budget 40）：
加 PI 预筛 + 支配跳过前——qehvi_nd 可行 HV 5557 / 可行点 36.5，单次 126–2355 s；
加预筛后——qehvi_nd **5797 / 24.5**、约 30 s，parego 5578 / 21。

4 目标 + 约束（`moo4_constrained`，budget 40，2 seeds，`--problem moo4_constrained`）：
qehvi_nd 可行 HV **91244** / 可行点 26.5（约 19–27 s），parego 90134 / 24（约 28 s）。
关键优化 `hypervolume_contribution`：把点限制在自己的盒子里做增量 HV，4D 单次运行从 839 s 降到 ~20 s，因此 3/4 目标共用同一套 MC 预算（prefilter 48、draws 32）。
结论：预筛后 qehvi_nd 在 HV 与耗时上都可接受并反超 parego，router 对「3–4 目标 + 预算 >=40 + 低/中噪声 + 有效维 <=15」默认选它；小预算或高噪声仍走 parego。
| `cqnehvi` | model_based | 2 目标 + 硬约束 + 噪声：嵌套 MC 估计带噪 EHVI，再乘逐候选 P(feasible)；无可行点时先做可行性搜索 | batch/mixed/categorical/continuous/noisy/multi_objective/constraints | `algorithms/cqnehvi.py` |

噪声对比（`moo_constrained`，budget 40）：sigma=3 时 cqehvi 638.2 vs cqnehvi 631.3；sigma=6 时 635.0 vs 625.6（均 2–3 seeds）。带噪嵌套 MC 在此基准未取得收益，因此**默认仍用 cqehvi**，cqnehvi 作为手动选项（router warning 中提示）。
| `mqehvi_nd` | model_based | 多保真 + 2–4 目标 MC-EHVI：fidelity 作为逐目标 GP 输入，按 E[HV 改进]/成本^α 选条件与保真度（实验性） | batch/mixed/categorical/continuous/noisy/model_based/multi_objective/constraints/multi_fidelity | `algorithms/mqehvi_nd.py` |
| `mfparego` | model_based | 多保真 + 多目标：每轮随机加权（增广 Tchebycheff）标量化，再把 fidelity 作为 GP 输入，按 EI/成本^α 同时选条件与保真度 | batch/mixed/categorical/continuous/noisy/model_based/multi_fidelity/multi_objective/constraints | `algorithms/mfparego.py` |
| `momfbo` | model_based | 多目标 + 多保真：随机标量化（增广 Tchebycheff）+ fidelity 维 GP + EI/成本，支持硬约束 P(feasible)；标量化噪声逐点传入 | batch/mixed/categorical/continuous/noisy/multi_objective/multi_fidelity/constraints | `algorithms/momfbo.py` |



momfbo 参数扫描（`scripts/sweep_momfbo.py`，12 组）：`min_full_fraction=0.0` 会出现整轮无全保真点的失败（ei 下 1–2/2 seeds，HV 跌到 0–285）；保持 0.25 时 ei/logei 都稳定在 HV 662 左右。因此 `step` 新增安全网：多保真上下文且观测 ≥4 但 `n_full_fidelity == 0` 时建议 `collect_full_fidelity`。
`mf_moo` 对比（budget 20，2 seeds，`scripts/benchmark_momfbo.py`）：momfbo 全保真 HV 662.2 vs parego 668.4 —— 该基准上低保真偏差对 yield 的误导使 MF 无优势；momfbo 保留为“有多保真通道 + 多目标”时候选，默认路由仍按能力选择。
| 基准 | - | `mf_constrained`：多保真 Suzuki + T ≤ 80 °C 硬约束（约束 binding） | - | `benchmarks.py` |
| 体检 | - | `scripts/audit_routing.py`：96 个能力上下文逐个 route+核对能力+build+ask | - | `scripts/audit_routing.py` |
| 体检 | - | `scripts/audit_algorithms.py`：为每个注册算法选能力匹配的基准并跑 ask/tell，任一失败即非零退出 | - | `scripts/audit_algorithms.py` |
| 基准 | - | `moo3_constrained`：3 目标（yield/cost/purity）+ T ≤ 78 °C 硬约束 | - | `benchmarks.py` |

`mf_constrained` 对比（同资源预算 20，3 seeds，`scripts/benchmark_mf_constrained.py`）：
mfbo（约束感知多保真，acquisition=ei）best feasible yield 均值 **95.3**，constrained（全保真）**50.1**；
feasible full-fidelity rate **0.667** vs 0.483，收益来自低成本粗筛 + P(feasible)。
参数扫描见 `scripts/sweep_mfbo.py`：ei 比 logei 更稳（logei 部分 seed 整轮无可行点），min_full_fraction=0.25 / cost_exponent=1.0 保持保守默认。

选择逻辑由 `src/condopt/router.py` 决定，规则与权重见 [`ROUTING_POLICY.md`](ROUTING_POLICY.md)。

---

## 1. `bayes` —— GP 贝叶斯优化

**直觉**：用 GP 给黑箱目标建一个“均值 + 不确定性”的后验；采集函数（EI/UCB/PI/TS）在“可能更优”与“还不确定”之间平衡，每次只花一次昂贵实验去换最大的期望信息量。

**适用**：目标昂贵、预算 10–200、维度 ≤15、目标较光滑；有噪声反而更需要代理来复用信息。

**不适用**：维度 >20（普通 GP 退化，需 TuRBO 或采样）；目标不连续/有阈值；批量极大（>16 时 qEI 的贪心近似变差）；全离散且组合很少（穷举更直接）。

### 参数

| 参数 | 默认 | 范围 | 含义与调法 |
|---|---|---|---|
| `kernel` | `matern52` | matern52 / matern32 / rbf | Matérn 5/2 通用；目标更粗糙用 3/2；非常光滑且维度低才用 RBF。参照 [R2] 的核选择经验。 |
| `length_scale` | 0.3 | 0.01–10 | 特征长度（单位立方体坐标）；GP 会围绕它做极大似然优化。数据密集时更小。 |
| `alpha` | 1e-6 | 1e-12–1.0 | **观测噪声方差**（不是标准差）。none 1e-8 / low 1e-6 / medium 1e-4 / high 5e-3。噪声大时过小会过拟合噪声。 |
| `n_initial` | 0（自动） | 0–200 | 初始空间填充点数；0 表示 `max(3, min(dim+1, 20))`。router 会把它限制在预算的 40% 以内。 |
| `n_candidates` | 1024 | 64–65536 | 采集函数评估的候选池大小；越大越准越慢。算力低用 512。 |
| `acquisition` | `ei` | ei / logei / ucb / pi / ts | EI 平衡探索利用；log-EI 数值更稳；UCB 更偏探索（高噪声默认）；PI 偏利用；TS 用后验采样。 |
| `kappa` | 2.0 | 0–10 | UCB 探索系数；高噪声 2.5。 |
| `xi` | 0.01 | 0–1 | EI/PI 的改进阈值（jitter）；越大越偏探索。 |
| `batch_strategy` | `kriging_believer` | greedy / kriging_believer / constant_liar | 批量选点：逐点用“信念值”或“常数谎言”更新模型，防止一批点全扎堆。参照 [R2]。 |
| `local_frac` | 0.3 | 0–0.9 | 候选池中围绕当前最优的局部扰动比例。 |
| `local_sigma` | 0.15 | 0.01–1.0 | 局部扰动的单位立方体标准差。 |

### 失效模式与排查

| 现象 | 可能原因 | 处理 |
|---|---|---|
| 连续多轮无改进 | 代理被错误先验/错误区域困住 | `route_round` 自动插探索；或手动切 `sobol` 一批 |
| 预测与实测偏差大 | 噪声大但 `alpha` 太小；存在隐藏变量 | 增大 `alpha`；检查空间是否漏变量；核改 `matern32` |
| 每轮建议都挤在一起 | 批量策略贪心 | `batch_strategy=constant_liar`；增大 `local_frac` |
| 早期就锁死在一个尖峰 | 初始点太少/太集中 | 增大 `n_initial`；改用 `ucb` |
| 维度高时明显退化 | 普通 GP 的维度灾难 | 降维/固定部分变量；或改用采样基线；v0.2 计划 TuRBO [R14] |

### 实现要点

- 特征：数值变量归一化到 [0,1]，分类变量 one-hot（低基数）；高基数分类的“虚假距离”问题见 §5。
- 候选池：50% 均匀随机 + Sobol + `local_frac` 局部扰动；先 decode→re-encode 归并到合法配置，再剔除已评价/已选点。
- 采集：EI `imp·Φ(z)+σ·φ(z)`；log-EI 取对数；UCB `μ+κσ`；PI `Φ((μ−best−ξ)/σ)`；TS 用核矩阵 Cholesky 采后验样本。
- 批量：逐点采集，选完用 `kriging_believer`（μ）或 `constant_liar`（best−σ）作为伪观测重拟合。
- 优化器：`sklearn.gaussian_process.GaussianProcessRegressor`，`normalize_y=True`，数据 ≥6 时 1 次超参重启。

**文献**：EI/EGO [R1]；BO 工程细节与批量 [R2]；采集函数统一视角 [R3]；GP 教程与高维 [R14][R15]；混合空间编码 [R13]。

---

## 2. `tpe` —— 独立逐维 Parzen 估计（非 tree / 非条件密度）

> **命名精确性（2026-11-11 外部审查）**：本实现是**逐维独立**的 KDE + 分位分组，没有 tree-structured 的**条件空间**（active-variable mask）与联合密度；注册键 `tpe` 保持不变以免破坏接口，但文档与参数说明都不得暗示它等价于 Optuna 的 TPE。证据：`src/condopt/algorithms/tpe.py` 顶部说明 + `tests/test_algorithm_naming.py`。

**直觉**：不拟合 GP，而是把观测分成“好组”和“坏组”，分别用核密度 `l(x)`、`g(x)` 估计；
提案时从 `l(x)` 采样候选并按 `l/g` 排序。分类变量用 Laplace 平滑的类别分布，数值变量用高斯 KDE，
因此不需要 one-hot 距离、也不需要 GP 超参，小样本/混合空间更稳。参考 [R4][R9]。

**适用**：分类变量多或 one-hot 后特征维高（GP 不可靠）、观测少（<20）、目标无需光滑假设。

**不适用**：连续光滑低维（GP-BO 样本效率更高）；需要精确不确定性（TPE 没有后验方差）。

### 参数

| 参数 | 默认 | 范围 | 含义与调法 |
|---|---|---|---|
| `gamma` | 0.25 | 0.02–0.9 | 好样本分位数；越小越利用，越大越探索 |
| `n_candidates` | 256 | 32–4096 | 每次从 l(x) 采样的候选数 |
| `n_initial` | 0（自动） | 0–200 | 0 表示 `max(3, min(feature_dim+1, 15))` |
| `bandwidth_factor` | 1.0 | 0.1–5.0 | 数值维高斯核带宽缩放；数据稀疏调大 |
| `exploration_frac` | 0.1 | 0–0.5 | 直接均匀随机探索的概率（防锁死局部） |
| `min_distance` | 0.02 | 0–0.5 | 与已测点的最小单位立方体距离（防重复提同一点） |
| `prior_weight` | 1.0 | 0.1–10 | 分类变量平滑强度 |

### 失效模式

- 在连续光滑/多峰问题上容易被局部盆地吸住 → router 只在分类/高维场景推荐它；
- `gamma` 太大 → 好组混入坏点；太小 → 过拟合到 3–4 个点；
- 没有不确定性估计，不适合“必须报告置信区间”的决策。

**文献**：TPE [R4]；Optuna 的 ask/tell 与 TPE 实现 [R9]。

---

## 3. `parego` —— 多目标 ParEGO

**直觉**：每轮随机抽一个权重向量 `λ`，用增广 Tchebycheff 把 m 个目标压成一个标量：
`g_λ(z) = max_i(λ_i z_i) + ρ·Σ_i λ_i z_i`（z 为目标归一化、统一成“越大越好”后的向量）。
然后在 `g_λ` 上拟合普通 GP + EI，提议下一个点。换一组 `λ` 就换一个搜索方向，
重复下来就能覆盖非凸 Pareto 前沿（加权和做不到）。

**适用**：2–3 个目标（收率 vs 成本、活性 vs 选择性）、实验昂贵/带噪声、需要一组折中解。

**不适用**：单目标（用 bayes）；4+ 目标（前沿覆盖差，qEHVI 只支持 2 目标）；需要严格约束（2 目标 + 约束用 `cqehvi`）。

### 参数

| 参数 | 默认 | 范围 | 含义与调法 |
|---|---|---|---|
| `rho` | 0.05 | 0–0.5 | 增广项；太小前沿覆盖差，太大退化为加权和 |
| `weight_alpha` | 1.0 | 0.1–10 | Dirichlet 浓度；<1 强调端点，>1 更均匀 |
| `acquisition` | `logei` | ei/logei/ucb | 标量化后的单目标采集函数 |
| `n_initial` | 0（自动） | 0–200 | 0 表示 `max(2*(feature_dim+1), 2*m)`，上限 20 |
| `n_candidates` | 1024 | 64–65536 | 每次采集的候选池 |
| `alpha` | 1e-4 | 1e-12–1.0 | 观测噪声方差 |
| `length_scale` | 0.3 | 0.01–10 | GP 初始长度尺度 |

### 失效模式

- 目标强相关 → 前沿退化成点；目标尺度差异大 → 必须归一化；
- 权重方向覆盖不全 → 前沿有洞（缓解：权重去重/最小夹角，尚未实现）；
- 噪声大 → 先聚合重复测量，再标量化。

### 实现与验收

- 实现：`algorithms/parego.py`，复用 `GaussianProcessBO` 的 GP/候选池/采集函数（每个提议点独立 λ）。
- 报告：`Study.report()` 输出非支配点数与 2D HV；`pareto.py` 提供非支配排序与 HV。
- 基准 `moo_yield_cost`（max 收率 vs min 成本）：预算 20、批量 4、3 seeds 下 ParEGO 的平均 HV ratio 领先随机/sobol 基线（见 §9 表）。
- 参考：Knowles, ParEGO, IEEE TEC 2006；Daulton et al., qNEHVI, NeurIPS 2021 [R11][R12]。

---

## 4. `constrained` —— 约束贝叶斯优化

**直觉**：实验条件常有安全/可行性边界（`T ≤ 110`、溶解度 ≥ 0.5、成本 ≤ 上限）。
直接优化目标 + 事后过滤会浪费大量实验；`constrained` 分两阶段：

1. 还没有可行点时：最大化**可行性概率** PoF（先找到任何安全条件）；
2. 已有可行点后：最大化**带可行性的期望改进**
   `EI_c(x) = E[max(0, f(x) − f*) · 1{all constraints feasible}]`，
   用独立 GP 后验的 Monte Carlo 积分估计（比 `EI × Π PoF` 在带噪约束下更稳）。

**适用**：1–3 个约束、实验昂贵/带噪、安全边界必须满足（化学/工艺最常见）。

**不适用**：2 目标 + 约束（用 `cqehvi`，可行域内 EHVI）；>2 目标 + 约束（用 `parego`，候选按 P(feasible) 择优）；约束本身完全观测不到；约束极多（>5 个，联合可行性估计方差大）。

### 参数

| 参数 | 默认 | 范围 | 含义与调法 |
|---|---|---|---|
| `mc_draws` | 128 | 16–512 | EI_c 的 MC 采样数；高算力/多约束调大 |
| `constraint_margin` | 0.0 | 0–10 | 安全裕量：`<=` 按 `value−margin` 判可行，`>=` 按 `value+margin` |
| `feasibility_threshold` | 0.02 | 0–0.5 | 候选点最低 PoF；低于此值不参与采集 |
| 其余 | 同 `bayes` | | ARD 核、EI/log-EI、批量 local penalization 等 |

约束定义放在 `objective.json`：

```json
{"constraints": ["T_max"], "constraint_limits": {"T_max": {"op": "<=", "value": 110.0}}}
```

结果回填时每个观测带 `constraints`：

```json
[{"id": "r1-1", "y": 88.1, "constraints": {"T_max": 108.4}}]
```

### 失效模式

- 没有可行点时，优化器会先“找可行”；如果预算太小可能还没找到就结束 → 报告需给出 `n_feasible` 与首个可行点位置；
- 约束的 GP 噪声没建模时 PoF 可能过度自信；复孔/重复测量应聚合后再回填；
- 约束过多时独立 GP 近似误差增大，建议先合并/降维。

### 验收

在 `constraint_easy`（约 25% 可行）与 `constraint_hard`（约 6% 可行）上，预算 20、批量 4：
本机 seed 0 分别找到可行最优 **94.99 / 92.81**（约束最优 95.0 / 93.096），测试在 `tests/test_constrained.py` 回归。

**文献**：Gardner et al. 2014（EI-CDF）；Letham et al. 2019（带噪约束 EI）。

---

## 5. `qehvi` —— 单点候选 EHVI + 局部惩罚批量（非联合 q 点 EHVI）

**直觉**：多目标优化的目标是让 Pareto 前沿覆盖尽可能大的超体积（HV）。EHVI 直接计算
「加入候选点后 HV 的期望提升」；2 目标时旧支配域可被**精确分区**成互不重叠的矩形，
`DHV(y) = Π(y−r) − Σ_boxes Π max(0, min(y,u_b)−l_b)`，再对 GP 后验做 MC 求期望。

**适用**：恰好 2 个目标、预算 ≥40、噪声低/无；比 ParEGO 的前沿覆盖/ HV 更好。

**不适用**：3+ 目标（需递归分区）；高噪声多目标（应上 qNEHVI）；预算很小（ParEGO 更稳）。

**关键参数**：`mc_draws`（默认 128）、`n_initial`、ARD/核/批处理参数同 `bayes`。
必须给 `objective.ref_point`（原始尺度，例如 `(60, 30)`），否则 EHVI 的 HV 口径跨 run 不可比。

**实现**：`algorithms/qehvi.py`；分区与 DHV 在 `pareto.py`（`partition_2d` / `dhv_2d` / `dhv_2d_batch`）。
批量用贪心 + local penalization（不伪造观测）。

**验收**：`moo_yield_cost`、预算 40、批量 4、seed 0 时 HV ratio ≈ **0.95**（vs 506 点参考前沿）；
`tests/test_qehvi.py` 有分区/DHV 手工值与基准回归。

**带噪变体 `qnehvi`**：对当前 Pareto 前沿做 `n_base` 次 baseline 抽样（每次按观测噪声重采样目标值、
重建前沿），候选 EHVI 对 baseline 求嵌套平均；默认 `n_base=64, n_y=128`。
router 在 2 目标 + 预算 ≥40 + 噪声 medium/high 时选它；低噪声仍用 `qehvi`（更省算力）。
`tests/test_qnehvi.py` 在加噪 `moo_yield_cost` 上回归 HV ratio。

**文献**：Daulton et al., qEHVI, NeurIPS 2020；qNEHVI, NeurIPS 2021。

### 5.1 约束变体 `cqehvi` —— 2 目标 + 硬约束

**直觉**：真实的多目标战役几乎总有可行性边界（温度上限、溶解度下限、成本上限）。
先优化无约束前沿、再事后过滤，会让一半实验花在**做不出来的条件**上，报告的“最优折中”也可能不可行。
`cqehvi` 把 qEHVI 的精确 2D 分区和 `constrained` 的两阶段可行性合成一个采集函数：

```
cEHVI(x) = E[ DHV(f(x)) · 1{all g_j(x) ≤ 0} ]
```

用目标 GP 与约束 GP 的独立后验做 MC 积分估计；前 Pareto 分区**只用可行观测**构建
（不可行点不能“挡住”永远用不上的体积），并用 `feasibility_threshold` 过滤掉 PoF 过低的候选。

**两阶段**（与 `constrained` 一致）：

1. 还没有可行观测 → 最大化可行性概率 PoF（先找到任何安全条件）；
2. 已有可行观测 → 最大化上面的 `cEHVI`（只奖励可行域内的超体积改进）。

**参数**：`mc_draws`（128）、`constraint_margin`（0）、`feasibility_threshold`（0.02），其余同 `qehvi`/`bayes`
（ARD 核、`n_candidates`、local penalization）。必须给 `objective.constraints` + `constraint_limits`，
且**每个回填观测都要带 `constraints`**（缺值按“不可行”处理，不猜测）。

**失效模式**：约束太多（>5）或可行域极小且预算不足 → 第一阶段可能花光预算；
高噪声下独立 GP 的 PoF 过度自信（不建模噪声约束）→ 先聚合复孔再回填；
`objective.ref_point` 缺失时 HV 不可跨 run 比较。

**验收**：`moo_constrained`（max 收率 vs min 成本，双硬约束 T ≤ 78 °C、time ≤ 50 min，可行域 ≈13%），
预算 40、批量 4、3 seeds：

| algorithm | 可行观测数 / 40 | 可行前沿 HV ratio |
|---|---|---|
| `cqehvi` | 35 / 33 / 34 | **0.972 / 0.977 / 0.985** |
| `qehvi`（无约束） | 3 / 3 / 2 | 0.723 / 0.711 / 0.675 |
| `parego`（无约束） | 17 / 19 / 17 | 0.966 / 0.794 / 0.962 |
| `random` | 3 / 3 / 5 | 0.683 / 0.759 / 0.859 |
| `sobol` | 5 / 3 / 6 | 0.861 / 0.627 / 0.812 |

批量策略：默认 `batch_strategy=local_penalty`；可选 `kriging_believer`（后验均值伪观测）与
`constant_liar`（悲观常数伪观测）。24 次评估 / batch 8 / 2 seeds 实测：
local_penalty feasible 0.729 / HV 0.980；kriging_believer 0.729 / 0.983；
constant_liar 0.729 / 0.974 —— 当前基准下 KB 仅 +0.003 HV，CL 更差，默认仍 local_penalty（省算力）。

实现：`algorithms/cqehvi.py`（复用 `qehvi` 的分区/DHV 与 `constrained` 的可行性逻辑）；
回归测试 `tests/test_cqehvi.py`，路由用例在 `tests/test_router.py`；
可复现脚本：`py -3 scripts/benchmark_cqehvi.py --budget 40 --batch 4 --seeds 0 1 2`
（输出 CSV 到 `runs/benchmark_cqehvi/`）。

**文献**：Daulton et al., qEHVI, NeurIPS 2020；Gardner et al., EI-CDF, ICML 2014；Letham et al., 2019。

---

## 6. `mfbo` —— 多保真贝叶斯优化

**直觉**：有低成本粗筛（短时试跑、低精度仿真）时，先用低保真把候选范围缩小，再对最有希望的点做全保真。
`mfbo` 把 `fidelity` 作为 **GP 的第 d+1 个输入特征**（偏差由数据学习），
并按 `EI / cost(s)^cost_exponent` 同时选择**条件 + 保真度**；至少 `min_full_fraction` 比例的提议保持全保真。

**适用**：多保真通道存在、最终结论必须基于全保真实验；预算允许 20+ 次评价。

**不适用**：低保真与全保真结论排序不一致（偏差不可学习）；没有粗筛通道（用 `bayes`）。

**参数**：`fidelity_levels`（默认 `0.25,0.5,1.0`）、`cost_exponent`（默认 1）、`min_full_fraction`（默认 0.25）、
其余同 `bayes`（ARD 核、log-EI、n_initial/n_candidates）。

**接口**：`Optimizer.ask_with_fidelity(n) -> list[(config, fidelity)]`；`Study` 会把 fidelity 写进
`pending.json` / `suggestions.csv`，并随 `tell` 落到 `Observation.fidelity`。

**验收**：`tests/test_multifidelity.py`（接口契约、最小全保真比例、Study 落盘/回填）。

**文献**：Kandasamy et al., BOCA, ICML 2017；Jamieson & Talwalkar, AISTATS 2016。

---

## 7. `sobol` —— 低差异序列

**直觉**：用确定性的低差异序列在单位超立方体里撒点，比 i.i.d. 更均匀（分层性质），适合“还不想建模、先把空间铺开”。

**适用**：初始设计、批量并行、维度大、预算小、需要诊断空间覆盖。

**参数**：

| 参数 | 默认 | 说明 |
|---|---|---|
| `scramble` | true | 随机化 Sobol（Owen scrambling），避免对齐伪影、支持无偏估计。 |
| `dedupe` | true | 跳过已评价过的配置（序列重启/批量混用时会撞点）。 |

**注意**：n 不是 2 的幂时均匀性略降（可接受，代码已静默处理警告）[R6]。不能利用已观测信息，所以只做“探索”与初始设计。

**文献**：Sobol 1967；scipy.stats.qmc；初始设计对比 [R6]。

---

## 8. `lhs` —— 拉丁超立方

**直觉**：把每一维分成 n 层、每层恰好取一个点，保证每个变量的边缘分布都被分层覆盖；一次性给一批点时比纯随机更均匀。

**适用**：一次性批量（8–48）实验、初始筛选、每维都希望被覆盖。

**参数**：

| 参数 | 默认 | 说明 |
|---|---|---|
| `optimization` | `random-cd` | scipy 的 LHS 优化准则：`random-cd` 降低维度间相关性；`lloyd` 更均匀但更慢；`none` 最快。 |

**注意**：串行小批量时优势不大；不能自适应。

**文献**：LHS 原始论文 [R6]；scipy.stats.qmc.LatinHypercube。

---

## 9. `grid` —— 网格/析因设计

**直觉**：把每个变量离散成层，枚举全部组合；不是“优化”而是“穷举 + 对比”，在组合数很小或需要做效应分析时最可靠。

**适用**：全离散且组合数 ≤200、预算能覆盖；少量分类/整数条件的筛选实验。

**参数**：

| 参数 | 默认 | 说明 |
|---|---|---|
| `max_points` | 200 | 最多枚举点数；router 会按剩余预算设置。 |
| `levels` | 5 | 连续/整数变量每维取几层。 |

**注意**：维度 >4 时指数爆炸；预算不足以覆盖时会产生重复点（`dedupe` 不适用，因为它是确定性序列）。router 对“组合数 ≤200 且预算覆盖”给 `+55/+25` 的高分。

**文献**：Montgomery《Design and Analysis of Experiments》；析因设计经典。

---

## 10. 混合空间与噪声：设计决策

### 7.1 混合空间编码

- **数值变量**：线性/对数归一化到 [0,1]；`log` 用于浓度这类跨数量级变量。
- **低基数分类**：one-hot 特征（GP 距离不会把“MeCN vs THF”和“MeCN vs water”当等距）；单位立方体坐标用 `index/(k−1)`，便于 Sobol/LHS 采样与解码。
- **高基数分类（>6–8 类）**：one-hot 会让 GP 维度膨胀且产生“虚假距离”；已支持：给分类变量配置 `descriptors`（每个 choice 一个数值向量，如分子量/logP/介电常数），GP 特征将用描述符替代 one-hot，并按维度 min-max 归一化；允许部分 choice 缺描述符（缺失项零向量 + unknown 指示位）（旧方案：one-hot 或分层分组，参照 Gryffin 思路）（见 [R13] 与 `research/ALGORITHM_SURVEY.md`）。
- **描述符只提高“覆盖”，不提高“最好值”（Round 293–294 实测，5 / 3 seeds）**：
  - 单目标 + 绑定约束：可行率 **0.550 → 0.875**，最优值不变（+0.08%）；
  - 2 目标 + 硬约束：可行点 **29.7 → 37.0（+25%）**，可行 HV **−1.4%（噪声内）**；
  - **必须与约束感知算法成对使用**：同样的描述符编码配上不看约束的算法会更差
    （`bayes` 可行率 0.285 → 0.150、最优值 −3.0%）；`tpe` 是“可行率升、最优值 −9.2%”的权衡；
  - 依据：`research/DESCRIPTOR_FEATURES.md`（含复现命令），登记表 `descriptor_features`。
- **整数**：线性归一化后解码取整；边界处理用 clip（restart 方案见 [R5]）。
- **有序分类**：仍按 one-hot 处理（v0.1 简化），有序信息尚未进入核函数。

### 7.2 噪声

- 化学实验噪声是一等公民：复孔 CV >5% 时，普通 EI 会“追噪声”，建议 `ucb` + 增大 `alpha`，并考虑复孔/加权。
- 更严谨的做法（已实现）：**noisy EI / qNEI**（对观测噪声做期望，而非把噪声当方差）[R12] → `qnehvi`；
  约束用 Letham 式联合 MC 期望 [R10] → `constrained` / `cqehvi`，而不是 `EI × P(feasible)`。

### 7.3 约束与多目标

- **多目标（已实现）**：`parego` 用增广 Tchebycheff 标量化 + 每点独立 Dirichlet 权重，复用 GP/EI；
  `qehvi` 用精确 2D 分区算 EHVI；`qnehvi` 对前沿不确定性做嵌套 MC；`cqehvi` 在可行域内做 EHVI；
  `Study.report()` 输出非支配前沿与 2D HV。
- **约束（已实现）**：`constrained`（单目标）与 `cqehvi`（2 目标）都采用两阶段：
  无可行点时最大化 PoF，有可行点后最大化联合 MC 期望 `E[max(0,f−f*)·1{g≤0}]` / `E[DHV·1{g≤0}]`。
  两者都是独立 GP 近似（不显式建模约束噪声），约束带噪时应先聚合复孔。
- **仍缺**：3+ 目标 EHVI（递归分区，P3 暂缓）、噪声约束的显式联合建模、跨底物迁移。

---

## 11. 规划中算法（v0.6+）

> 标注「已实现」的行保留在此作为历史记录；当前实现以 §1–§9 为准。

| 算法 | 解决什么 | 关键设计 | 文献 |
|---|---|---|---|
| ARD 核（Sprint 2，已实现） | 各特征尺度差异大、one-hot 虚假距离 | 每特征 length_scale + 维度缩放先验，n<2d 时冻结 | [R15] |
| local penalization（已实现） | 批量选点扎堆 | 用 `1−exp(−d²/2ℓ²)` 惩罚替代 constant-liar 伪观测 | [R35] |
| 随机森林代理（SMAC 式） | 条件空间、离散/高基数 | 森林 + EI + intensification/RunHistory | [R7][R8] |
| CMA-ES | 高维、崎岖、连续 | λ=4+⌊3ln n⌋、路径自适应、IPOP/BIPOP 重启 | [R5] |
| bandit（UCB1/Thompson） | 纯分类、少量选项、快速淘汰 | Beta 后验、每类计数 | 见 survey |
| TuRBO | 高维 BO | 信任域自适应 + 多区域并行 | [R14] |
| 3D EHVI（P3 暂缓） | 3+ 目标 | 递归分区 + `partition_nd`/DHV 自检；触发条件见 `NEXT_PRIORITIES_V3.md` | [R12] |
| 噪声约束联合建模 | >2 目标 + 约束、约束本身带噪 | 联合 MC / 约束 GP 显式噪声，替代独立 GP 近似 | [R10] |
| 多保真 | 有仿真/短时试跑 | BOCA/KG + 保真度调度 | 见 survey |
| 暖启动/迁移 | 有历史 campaign | 每任务 GP 集成 + 似然加权 | [R16] |
| LLM 提议 | 冷启动、语义空间 | 结构化 JSON 提案 + 数值采集函数兜底 | [R19][R20] |

---

## 12. 基准与验收

当前 `scripts/benchmark.py` 内置 Tier A–G 问题（Branin/Hartmann3/Sphere、6D 混合反应、Suzuki 风格、
高基数溶剂、离散催化剂、2 目标收率-成本、2 目标 + 双硬约束），指标为多次种子下的平均最优得分（越大越好）。

**Tier A/B（预算 16、批量 2、3 seeds）**

| problem | bayes | tpe | lhs | random | sobol |
|---|---|---|---|---|---|
| branin（min，分数已符号化） | **-0.792** | -9.556 | -2.052 | -4.542 | -3.727 |
| discrete_catalyst | **77.99** | 76.60 | 74.97 | 76.86 | 76.29 |
| hartmann3 | **-0.000173** | -0.01775 | -0.0196 | -0.0288 | -0.0277 |
| mixed_reaction | **96.69** | 94.89 | 94.36 | 96.48 | 93.51 |
| sphere2d | **-0.000732** | -0.00854 | -0.0102 | -0.00909 | -0.00693 |

**Tier B/C（预算 20、批量 4、3 seeds；用 router 参数策略）**

| problem | bayes | lhs | random | sobol | tpe |
|---|---|---|---|---|---|
| suzuki_like | 66.9 | 70.35 | **71.22** | 59.86 | 53.63 |
| discrete_catalyst_v2 | **80.63** | 79.08 | 79.23 | 79.72 | 78.49 |
| high_cardinality_solvent | **91.67** | 90.26 | 90.63 | 90.10 | 90.22 |

**Tier F（多目标，预算 20–24、批量 4）**：`parego` 在 `moo_yield_cost` 上 24 次评估的 HV ratio ≈ 0.97
（参考前沿 506 点），优于随机/sobol 基线；`test_multiobjective.py` 有回归测试。

**Tier G（多目标 + 硬约束，预算 40、批量 4）**：`moo_constrained`（max 收率 vs min 成本，
T ≤ 78 °C 且 time ≤ 50 min，可行域约 13%）上对比可行观测数与可行前沿 HV ratio：

| algorithm | 可行观测数 / 40（3 seeds） | 可行前沿 HV ratio（3 seeds） |
|---|---|---|
| `cqehvi` | 35 / 33 / 34 | **0.972 / 0.977 / 0.985** |
| `qehvi` | 3 / 3 / 2 | 0.723 / 0.711 / 0.675 |
| `parego` | 17 / 19 / 17 | 0.966 / 0.794 / 0.962 |
| `random` | 3 / 3 / 5 | 0.683 / 0.759 / 0.859 |
| `sobol` | 5 / 3 / 6 | 0.861 / 0.627 / 0.812 |

结论：把硬约束交给 `cqehvi` 的采集函数，实验预算的 85%+ 落在可行域，且可行前沿覆盖最稳；
无约束算法（qehvi）把预算花在做不出来的条件上，可行 HV 明显更差。回归测试 `tests/test_cqehvi.py`。

注：

- 最优常数由 `test_benchmarks.py` 用已知最优点自检（`regret(optimum)=0`）。
- 本套件前两档偏连续/低维光滑问题，TPE 不占优是**预期行为**；Tier C 显示“预算小 + 高基数分类”时
  随机/LHS 仍可能领先，这正是下一轮 router 权重校准（LOPO-CV）要解决的问题。

结论：在低维光滑/混合问题上 BO 的样本效率优势明显；这支持 router 的默认偏好。
**注意**：这是合成函数，不代表真实化学实验；真实评估需用 Olympus [R18] 式的噪声 emulator 或公开反应数据集。

验收标准（下一轮 benchmark 扩展时执行）：

1. 每个算法在至少 3 个问题上、3 seeds 下无崩溃、无越界；
2. `bayes` 在低维光滑问题上 ≥ 随机基线；在预算 <8 或维度 >20 时不强行推荐；
3. router 的 top-1 选择在基准上与“事后最优算法”的一致率 ≥ 70%；
4. 每次 benchmark 结果落 CSV 并可复现（固定 seed）。

---

## 13. 如何新增一个算法

1. 在 `src/condopt/algorithms/` 新建模块，写 `Optimizer` 子类，用 `@register` 装饰；
2. 声明 `AlgorithmSpec`：`name/family/summary/when/params/capabilities/references`；
3. 在 `router._rules()` 注册打分函数、在 `_params_for()` 注册参数策略；
4. 在 `tests/test_algorithms.py` 加 ask/tell 契约测试、在 `tests/test_router.py` 加“该算法应在什么上下文胜出”的用例；
5. 更新本目录与 `docs/ROUTING_POLICY.md`，并在 benchmark 里跑一轮。

多保真多目标对比（budget 40）：`mf_moo` 5 seeds mfparego HV 676.1 vs momfbo 670.3；新增 `mf_moo_constrained`（两目标 + 硬约束 + 多保真）3 seeds：mfparego 可行 HV 642.5 / 可行全保真点 26.0 vs momfbo 635.6 / 22.7（+1.1%）。四处独立设置（含校准标签）一致指向 mfparego，router 现把多保真 + 多目标默认改为 **mfparego**，momfbo 保留为候选（消除单点依赖）。

`momfbo.min_full_fraction` 扫描（`mf_moo_constrained`，budget 40，5 seeds）可行 HV：0.25 → 636.4、0.4 → 639.7、0.8 → 640.7，可行全保真点 23.8 / 26.6 / 24.8；router 现默认下发 0.4（兼顾 HV 与全保真证据量）。

多保真多目标 HV 型对比（budget 40，3 seeds）：`mf_moo` mqehvi_nd 626.9（min_full_fraction=0.25）/ 647.9（0.6）vs momfbo 672.3；`mf_moo_constrained` 607.1 vs 635.6。HV 型采集在该组合下**输给**标量化策略，因此 router 仍默认 momfbo，mqehvi_nd 仅作实验性第三候选。
新增 3 目标多保真基准 `mf_moo3`（max yield / min cost / max selectivity，低保真对 yield 与 selectivity 有负偏）。（budget 40，3 seeds）可行 HV：**mfparego 27399** vs momfbo 26869 vs mqehvi_nd 25587；全保真点 35.3 / 26.0 / 20.0。3+ 目标 + 多保真因此改由 router 默认选 **mfparego**（2 目标仍为 momfbo）。
4 目标多保真基准 `mf_moo4`（max yield / min cost / max selectivity / max purity，budget 40，3 seeds）可行 HV：**mfparego 515205（36 全保真点）vs momfbo 507932（24.3）**，HV +1.4% 但全保真证据多近 48% -> 与 3 目标结论一致，3+ 目标默认 `mfparego`。
新增多目标 + 多保真 + 12 级溶剂基准 `mf_solvent` / `mf_solvent_desc`（budget 40，3 seeds）：momfbo 1195.3 / 1192.0，mfparego 1196.1 / 1196.6 —— 描述符在该组合下 HV 中性，主要收益是特征维度 13 → 3。
新增高基数 + 硬约束基准 `solvent_constrained` / `..._desc`（12 级溶剂，T ≤ 80，budget 60，3 seeds）：`constrained` 90.394 / 90.377（可行率 0.683 / **0.917**）远优于 bayes（84.7 / 82.4）与 tpe（85.7 / 79.4），同时说明描述符能把可行率提升约 34%。

## 14. 实现边界与命名精确性（外部审查回应）

**规则**：名称相同 ≠ 机制相似 ≠ 算法等价。下表是每个"文献同名"算法的**实际实现范围**，
以及尚未实现的部分；对外描述（README、skill、release notes）不得超出"实际实现"这一列。

| 算法（注册键） | 实际实现 | 与文献的差距 | 证据 |
|---|---|---|---|
| `bayes` | GP + EI/UCB/PI + local-penalization 批量、ARD、逐点噪声 | 批量策略非联合 q-EI | `algorithms/bayes.py` |
| `tpe` | 逐维独立 KDE + 分位分组 | **无 tree/条件密度**（Optuna 的 TPE 有） | `algorithms/tpe.py` |
| `qehvi` | 单点候选 MC-EHVI 贪心 + local penalty | **非联合 q 点 EHVI**（BoTorch qEHVI 是联合） | `algorithms/ehvi.py` |
| `qnehvi` | 嵌套 MC：baseline 在 `X_baseline` 上取**模型后验**（每目标一次 GP 采样，逐目标噪声经 `Observation.objectives_std`）；候选走 GP 后验 | 与 BoTorch qNEHVI 仍非逐行等价（未做固定小案例数值对照），且未实现 noisy-EHVI mixin 的白化 | `algorithms/qnehvi.py` |
| `cmaes` | rank-μ 更新 + 启发式 sigma | **无 evolution paths**（非完整 CMA-ES） | `algorithms/cmaes.py` |
| `mfbo` / `momfbo` / `mfparego` | 多保真 GP、成本感知采集 | 非官方实现，未与官方代码做数值对照 | `algorithms/mfbo.py` 等 |

**待办（来自审查，按优先级）**：

1. `qnehvi`：baseline 改为在 `X_baseline` 上取模型后验；`Observation` 支持逐目标 std；
   加"baseline 必须调用模型"的结构测试与固定小案例数值对照；
2. `tpe`：若要 tree，需定义 active-variable mask 与条件采样，否则文档永久标注"独立逐维"；
3. `qehvi`：实现联合 q acquisition 后才能去掉"单点候选"限定；
4. `cmaes`：补 evolution paths（或改名 `simplified_rank_mu_es`）；
5. 主 benchmark：满足 `research/BENCHMARK_SPEC.md` 的预注册协议（共享初始设计、≥20 seeds、
   IQM/区间/成对检验）之前，只称"工程 smoke/校准"。
