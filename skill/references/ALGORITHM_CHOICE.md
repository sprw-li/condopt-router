# 算法速查（skill 决策用）

router 已把选择逻辑编码进 `condopt route`；本文件解释**为什么**这样选，以及每条路走不通时的切换信号。
算法实现与完整参数见 `src/condopt/algorithms/`，参考文献见 `research/REFERENCES.md`。

## 一句话选择表

| 你的情况 | 用 | 不要用 |
|---|---|---|
| 昂贵、20–100 次预算、维度 ≤15、目标较光滑，连续/顺序变量为主 | `bayes`（EI/UCB） | 网格（浪费预算） |
| 维度 >20、或目标粗糙、或预算低于**该算法的建模下限**（`route` 会给出 underpowered 说明） | `sobol` / `lhs` / `random` | BO（代理拟合不出来） |
| 一次性给一大批（8–48）、每维都要覆盖 | `lhs` | BO（单点评分慢） |
| 全离散、组合数 ≤200、预算能覆盖 | `grid` | BO（建模收益低于穷举） |
| 分类变量多、或 one-hot 后特征维 >15、或观测很少 | `tpe` | GP（超参不足/one-hot 虚假距离） |
| 纯类别但变量少（≤8 个）且 one-hot 宽度≤60、预算≥50 | `bayes` | 默认 TPE（独立密度忽略变量交互） |
| 2 目标、预算 ≥40、低噪声 | `qEHVI` | 直接优化超体积，前沿覆盖优于 ParEGO |
| 2 目标 + 硬约束（温度/时间/成本上限） | `cqehvi` | 可行域内 EHVI：先 PoF 找可行条件，再只覆盖可行前沿 |
| 2 目标、预算 ≥40、噪声中高 | `qnehvi` | 嵌套 MC 处理前沿不确定性；低噪用 qehvi |
| 2 目标、预算 <40 或噪声中高 | `parego` | 标量化在小预算下更稳 |
| 有低成本粗筛（短时试跑/低精度仿真） | `mfbo` | 直接全保真跑会浪费预算 |
| 有安全/可行性约束（温度上限、溶解度、成本上限） | `constrained` | 事后过滤（浪费实验、带噪约束会失效） |
| 批量 ≥8 且必须同时给点 | `sobol` / `lhs`；或 `bayes` + `constant_liar` | 无批量策略的贪心 BO |
| 高噪声（CV>5%） | `bayes` + `ucb` + 大 `alpha`，或随机基线对照 | 小 `alpha` 的 EI |

> 选择算法只是一半：预算里要给**确认复孔/交付自查**留位置（`prepare` 的
> `effective_exploration_budget` 就是净探索次数），停止口径按预算走（≥60 建议 `budget_first`）。
> 这两条与算法无关，见 `docs/DECISIONS.md`。

## 逐个算法

### bayes（GP-BO）

- **适用**：目标昂贵（每次实验成本高）、预算 10–200、维度 ≤15、目标较光滑、有噪声更好（代理可复用信息）。
- **关键参数**：`kernel`、`alpha`（噪声方差）、`n_initial`、`acquisition`、`batch_strategy`、`n_candidates`。
- **失效模式**：维度太高（>20）代理失效；目标不连续/有阈值；初始点太少导致 GP 过拟合；`alpha` 远小于真实噪声；
  批量大而用贪心导致一轮点全扎堆在同一个峰。
- **切换信号**：连续 3 轮无改进 → router 自动插探索；预测与实测差距持续大 → 检查隐藏变量/边界，改 `matern32`；
  分类变量占主导 → 考虑 `grid`。
- **参考**：Jones et al. 1998（EI）、Srinivas et al. 2010（UCB）、Rasmussen & Williams 2006（GP）。

### sobol

- **适用**：初始设计、批量并行、维度大、预算小、需要均匀覆盖空间做诊断。
- **关键参数**：`scramble`（随机化）、`dedupe`。
- **失效模式**：n 不是 2 的幂时均匀性略降（可接受）；不能利用已观测信息；点用完不重复。
- **参考**：Sobol 1967；scipy.stats.qmc。

### lhs

- **适用**：一次性批量（8–48）实验、每维都要被分层覆盖、初始筛选。
- **关键参数**：`optimization`（`random-cd` 降低维度间相关性，`lloyd` 更均匀但更慢）。
- **失效模式**：串行小批量时优势不明显；不能自适应。
- **参考**：McKay et al. 1979。

### random

- **适用**：维度极高、预算极小、算力极低、作为任何算法的对照组。
- **失效模式**：完全不利用信息；在低维光滑问题上明显劣于 BO。
- **参考**：Bergstra & Bengio 2012（随机搜索是强基线）。

### grid

- **适用**：少量离散/粗粒度条件、组合数 ≤200、需要穷举或做析因效应分析。
- **关键参数**：`max_points`、`levels`。
- **失效模式**：维度 >4 指数爆炸；预算不足以覆盖时大量重复点。
- **参考**：Montgomery《Design and Analysis of Experiments》。

### tpe

- **例外**：纯类别空间如果变量数少（≤8）、one-hot 宽度 ≤60、预算 ≥50，优先 `bayes`（GP 可建模交互）。
  2026 ART HTE 复现：TPE 100 次停在 91%，GP-BO 命中论文 99% 条件；文献标签里 categorical-heavy 的
  tpe/bayes 也基本持平。只有高基数/高维 one-hot/小预算才继续坚持 TPE。

- **适用**：分类变量多 / one-hot 后特征维高（>15）/ 观测很少（<20）/ 目标不要求光滑。
- **关键参数**：`gamma`（好样本分位，0.25 默认）、`bandwidth_factor`、`exploration_frac`、`min_distance`。
- **失效模式**：连续光滑多峰问题易锁死在局部盆地（router 因此只在分类/高维场景推荐它）；
  没有后验方差，不适合需要置信区间的决策。
- **参考**：Bergstra et al. 2011（TPE）；Optuna 实现。

### mfbo（多保真）

- **适用**：有低成本粗筛（短时试跑/低精度仿真），最终结论要基于全保真；预算 20+。
- **关键参数**：`fidelity_levels`（默认 `0.25,0.5,1.0`）、`cost_exponent`、`min_full_fraction`（默认 0.25）。
- **接口**：`ask_with_fidelity` 给出 `(条件, fidelity)`；`Study` 写入 `pending.json`/`suggestions.csv`，
  回填时带 `fidelity` 字段（不填则沿用建议的 fidelity）。
- **失效模式**：低保真与全保真排序不一致（偏差不可学）；没有粗筛通道（用 `bayes`）。
- **参考**：Kandasamy et al., BOCA, ICML 2017。
- **硬约束**：mfbo 现支持 constraints（采集值乘 P(feasible)），可与多保真同时使用。

### cmaes（中高维连续；简化 rank-μ，无 evolution paths）

- **适用**：全连续、光滑、维度 ≥8（≥15 更明显）、预算 ≥40、单目标。
- **注意（地貌依赖）**：8 维规则在高维 GP 不可靠时兜底，不是“8 维以上 cmaes 一定更好”。
  实测（`research/DIM_BOUNDARY.md`）：窄谷型（Rosenbrock）6 维起 cmaes 就更好；
  多峰型（Ackley）6/8/12 维 bayes 反而更优，10 维只是打平。已知窄谷问题可提前显式选 cmaes。
- **做法**：rank-μ CMA-ES（无进化路径），在单位立方体上自适应均值与协方差。
- **不适用**：含分类/整数、粗糙目标、极小预算、多目标/约束。
- **失效模式**：噪声高时排名不可靠（建议复孔）；种群规模需小于预算。

### qehvi_nd（3–4 目标，可选；单点候选 EHVI + 增量 HV）

- **适用**：3–4 个目标、预算 ≥40、噪声低到中等；需要直接优化超体积或更强可行性覆盖时手动指定。
- **做法**：逐目标 GP 后验 MC 采样 + 递归切片 HV 估计 EHVI，候选按 P(feasible) 过滤。
- **与 parego 的关系**：预算 >=40、低/中噪声且有效维 <=15 时 router 默认选 qehvi_nd；
  实测预算 40 下可行 HV 5797 vs parego 5578、可行点 24.5 vs 21，单次约 30 s；
  预算 <40、高噪声或高维时仍用 parego；
- **4 目标**：同样适用（实测可行 HV 91244 vs parego 90134，单次约 20 s）；增量 HV 贡献让 3/4 目标共用同一套 MC 预算（prefilter 48 / draws 32）。

### 多保真 + 多目标的默认选择

- **2+ 目标（含约束）**：默认 `mfparego`（四处基准与校准标签一致略优，且全保真点更多）
- `momfbo` 作为候选保留（手动指定时可用其调好的 `min_full_fraction=0.4`）
- `mqehvi_nd`（HV 型）三处均落后 -> 仅研究对比用

### mqehvi_nd（2–4 目标 + 多保真，实验性）

- **做法**：fidelity 作为逐目标 GP 输入特征，按 E[HV 改进]/成本^α 同时选条件与保真度。
- **实测结论**：多保真多目标三处均落后于标量化方案 -> **暂不推荐自动选择**，
  仅用于研究对比；完整数字见 `docs/ALGORITHM_CATALOG.md`（避免两处维护同一组数据）。

### mfparego（多目标 + 多保真）

- **适用**：2+ 目标且有低保真粗筛通道；当 `momfbo` 不可用或想用 ParEGO 式随机标量化时。
- **做法**：每轮 Dirichlet 采样权重 → 增广 Tchebycheff 标量化 → fidelity 作为 GP 输入 →
  `EI / cost^α` 同时选条件与保真度。
- **实测**：`mf_moo` budget 40、5 seeds 下 HV 676.1 vs momfbo 670.3；**router 现在默认 `mfparego`**（momfbo 作为候选保留）。Round 245 又补了第 2 个独立数据集（mf_moo budget 30：674.5 vs 669.5），差距约 1%，属于“不落后”。

### cqnehvi（2 目标 + 约束 + 噪声）

- **适用**：恰好 2 个目标、有硬约束、观测噪声中高（复孔波动明显）。
- **做法**：对 Pareto 前沿做嵌套 MC（带噪 EHVI），再乘逐候选 P(feasible)；
  无可行观测时先最大化可行性概率。
- **与 cqehvi 的关系**：实测（moo_constrained，budget 40）sigma=3 时 cqehvi 638 vs cqnehvi 631、
  sigma=6 时 635 vs 626，带噪嵌套 MC 未取得收益；**默认仍用 cqehvi**，cqnehvi 仅在明确需要
  带噪前沿不确定性建模时手动指定。

### momfbo（多目标 + 多保真）

- **适用**：有低成本粗筛通道，同时需要 2+ 目标折中解；可与硬约束并用。
- **做法**：随机标量化（增广 Tchebycheff）+ fidelity 作 GP 输入 + `EI/cost^α`，
  候选按 P(feasible) 过滤。
- **接口**：与 mfbo 相同，`ask_with_fidelity` 给出 `(条件, fidelity)`。

### constrained（约束）

- **适用**：温度/溶解度/成本等安全或可行性边界必须满足；实验昂贵、带噪。
- **关键参数**：`mc_draws`、`constraint_margin`、`feasibility_threshold`；约束定义写在
  `objective.json` 的 `constraints` + `constraint_limits`（如 `{"T_max": {"op":"<=","value":110}}`）。
- **两阶段**：没有可行点时先最大化 PoF；有可行点后最大化 `E[max(0,f−f*)·1{feasible}]`。
- **失效模式**：可行域太小且预算不足（可能找不到可行点）；约束太多（>5）联合可行性估计方差大；
  复孔结果应先聚合再回填。
- **参考**：Gardner et al. 2014（EI-CDF）；Letham et al. 2019（带噪约束）。

### qehvi（2 目标 EHVI）

- **适用**：恰好 2 个目标、预算 ≥40、噪声低/无；想直接最大化超体积（前沿覆盖更均匀）。
- **关键参数**：`mc_draws`（128）、`n_initial`；需要 `objective.ref_point`（原始尺度）固定 HV 口径。
- **带噪变体 `qnehvi`**：2 目标 + 噪声 medium/high 时用它（`n_base/n_y` 控制嵌套 MC 精度；
  `noise_std` 缺省用观测 `objective_std`）。
- **失效模式**：预算小或噪声大时不如 ParEGO；3+ 目标未支持；ref_point 设错会让 EHVI 全为 0。
- **参考**：Daulton et al., qEHVI, NeurIPS 2020；qNEHVI, NeurIPS 2021。

### cqehvi（2 目标 + 硬约束）

- **适用**：恰好 2 个目标，且同时有安全/可行性硬约束（温度上限、溶解度、时间/成本上限）；
  实验昂贵、噪声低/中；想避免“事后过滤”把一半预算花在做不出来的条件上。
- **两阶段**：还没有可行观测时先最大化可行性概率 PoF；有可行观测后最大化
  `E[DHV·1{feasible}]`（只有可行域内的超体积改进被奖励），Pareto 分区只用可行点构建。
- **关键参数**：`mc_draws`（128）、`constraint_margin`（安全裕量）、`feasibility_threshold`（0.02）；
  需要 `objective.ref_point`（原始尺度）固定 HV 口径。
- **失效模式**：约束 >5 或可行域极小且预算不足（第一阶段可能花光预算）；高噪声下 PoF 过度自信
  （不建模噪声约束，先聚合复孔）；`ref_point` 设错会让 EHVI 全为 0。
- **回填要求**：每个结果必须带 `constraints`（缺值按不可行处理，不要猜）。
- **参考**：Daulton et al., qEHVI, NeurIPS 2020；Gardner et al., EI-CDF, ICML 2014；
  Letham et al., 2019（带噪约束）。

### parego（多目标）

- **适用**：2–3 个目标（收率 vs 成本、活性 vs 选择性）、实验昂贵/带噪声、需要一组折中解。
- **关键参数**：`rho`（增广项 0.05）、`weight_alpha`（Dirichlet 浓度）、`acquisition`、`n_initial`。
- **失效模式**：目标强相关 → 前沿退化为点；目标尺度差异大 → 必须给 `objective_bounds` 固定归一化；
  4+ 目标前沿覆盖差；噪声大 → 先聚合复孔再标量化。
- **报告**：`Study.report()` 输出 Pareto 前沿 + 2D HV，不要只报单一最优。
- **参考**：Knowles, ParEGO, IEEE TEC 2006；Daulton et al., qNEHVI, NeurIPS 2021。

## 当前未内置（路线图）

`cma-es`、`ga`、`bandit`（纯分类）、3+ 目标 EHVI、>2 目标 + 约束组合。
用户提出这些需求时：如实说明当前版本只支持上表 12 种，并给出替代方案（例如 3+ 目标 → `parego`；
分类很多 → `tpe`；2 目标无约束 → `qehvi`/`qnehvi`；2 目标 + 约束 → `cqehvi`），不要假装支持。

## 人机分工

- **硬规则**：router 打分、参数 schema 校验、预算检查、每轮记录——不允许 AI 绕过。
- **AI 的机动性**：解释推荐、结合领域知识覆盖 router（必须给理由）、发现数据异常时建议换算法。
- **人的裁决**：安全边界、是否接受覆写、实验是否真的做了。
