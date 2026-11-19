# 负结果清单（Negative Results）

记录“试过但没赢”的方案，避免后人重复踩坑，也说明 router 的默认值是怎么来的。
每条都给出当时的数字与当前处置。

## 索引

| # | 主题 | 一句话结论 |
|---|---|---|
| 1 | `cqnehvi`：带噪约束 2 目标没有收益 | 带噪约束 2 目标上 cqnehvi 不如 cqehvi |
| 2 | Sobol 播种的 CMA-ES | Sobol 播种的 CMA-ES 不如默认初始化 |
| 3 | `mqehvi_nd`：HV 型多保真多目标输给标量化 | mqehvi_nd 输给标量化（保留但标为实验性） |
| 4 | TPE + 物性描述符 | TPE + 物性描述符反而更差（84.1 vs 90.2） |
| 5 | 垃圾描述符不会比 one-hot 更差（削弱“描述符必须准确”的担忧） | 垃圾描述符与 one-hot 相当（不必苛求描述符精度） |
| 6 | 真实 ART 六点上的代理预测失败（样本量问题） | 真实 ART 六点上代理预测失败（样本量太小） |
| 7 | 4 维 MC-EHVI 的性能坑（已修） | 4 维 MC-EHVI 性能坑（已修复） |
| 8 | 描述符在多保真多目标下降级为“中性” | 描述符在多保真多目标上降级为中性 |
| 9 | 停滞后的“探索重启”不如直接继续（而调参优于继续） | 停滞后探索重启不如继续；调参优于继续 |
| 10 | 多保真 + 约束：全保真槽位可能全部落在不可行区（已修复交付证据，质量仍待提升） | 多保真 + 约束：全保真槽位可能全部不可行（已修复交付证据） |
| 11 | 小样本下 transfer-audit 会误判（真实数据） | 小样本下 transfer-audit 会误判（真实数据；已加低置信度保护） |
| 12 | transfer-audit 在真实小样本上没有排序能力 | transfer-audit 在真实小样本上没有排序能力（保护机制 20/20 生效） |
| 13 | 迁移否决后反复复检：6 次 0 翻案 |  |
| 14 | 分阶段迁移的保护依赖审计能否否决：一个问题上 0/4 |  |

## 1. `cqnehvi`：带噪约束 2 目标没有收益

- **做法**：把 qNEHVI 的嵌套 MC 与 cqehvi 的 P(feasible) 组合（`E[DHV] × P(feasible)`）。
- **基准**：`moo_constrained`，budget 40，2–3 seeds。
  - σ=3：cqehvi **638.2** vs cqnehvi 631.3
  - σ=6：cqehvi **635.0** vs cqnehvi 625.6
- **结论**：带噪嵌套 MC 未超过普通约束 EHVI。
- **现状**：算法保留（可手动指定），router 默认仍为 `cqehvi`，并给出 warning。

## 2. Sobol 播种的 CMA-ES

- **做法**：用 Sobol 序列初始化 CMA-ES 种群，期望改善高维早期探索。
- **基准**：`rosenbrock10` / `ackley10` 扫描。
- **结论**：不如纯随机初始化 + 调好的 `sigma0=0.1`、`c_mu=0.5`（后者把 −82.3 改善到 −14.4）。
- **现状**：不采用；保留纯 CMA-ES 默认参数。

## 3. `mqehvi_nd`：HV 型多保真多目标输给标量化

- **做法**：fidelity 作为逐目标 GP 输入，按 `E[HV 改进]/cost^α` 选条件与保真度。
- **基准**：budget 40，3 seeds。

| 问题 | mqehvi_nd | momfbo | mfparego |
|---|---|---|---|
| `mf_moo`（2 目标） | 626.9 / 647.9（min_full 0.25/0.6） | **672.3** | 676.1 |
| `mf_moo_constrained` | 607.1 | **635.6** | 642.5 |
| `mf_moo3`（3 目标） | 25587 | 26869 | **27399** |

- **结论**：多保真场景下标量化 + 低保真粗筛明显更有效（廉价有偏数据会误导 HV 采集）。
- **现状**：`mqehvi_nd` 保留为实验性候选；router 3+ 目标改用 `mfparego`。

### 3b. 4 目标多保真补充（不同预算，仅供参考）

mf_moo4（budget 40，3 seeds）可行 HV：mfparego **515205**（36 全保真点）vs momfbo 507932（24.3）。
mqehvi_nd 在 budget 20 下为 467141（成本减半，**不可与上面直接比较**），仍无优势迹象。

## 4. TPE + 物性描述符

- **做法**：Tier C（12 级溶剂）用 `polarity/protic` 描述符替代 one-hot，跑 TPE。
- **基准**：budget 20，5 seeds。
  - TPE：one-hot **90.2** vs descriptors 84.1（明显变差）
  - bayes：one-hot 91.9 vs descriptors **92.0**（略优）
- **结论**：描述符适合 GP 类算法，ParEGO/TPE 的密度/标量化假设不匹配压缩后的连续特征。
- **现状**：描述符优先搭配 GP；用 TPE 时保留 one-hot（step 会提示 `review_algorithm_for_descriptors`）。

## 5. 垃圾描述符不会比 one-hot 更差（削弱“描述符必须准确”的担忧）

- **做法**：给 60 类别 latent 描述符加 `ε·N(0,1)`，ε 从 0 到 99。
- **结果**（budget 30，3 seeds）：ε=0/0.1 → 99.93；0.25/0.5 → 99.87；1.0 → 99.75；
  ε=99（≈垃圾/one-hot 对照）→ 99.77。
- **结论**：该场景下**描述符鲁棒**，最坏不比 one-hot 差；同时也说明描述符的收益上限有限
  （不是“必须上描述符”）。
- **现状**：描述符作为可选增强；不夸大其收益。

## 6. 真实 ART 六点上的代理预测失败（样本量问题）

- **做法**：用 ART Table 2 六条真实测量的留一交叉验证（LOO），one-hot 与溶剂描述符两版。
- **结果**：
  - one-hot：MAE 11.81，Spearman −1.00，真实最优未被预测为最优
  - 仅溶剂描述符：MAE 12.14，Spearman −0.83，仍无预测能力
- **结论**：5 个训练点无法覆盖 ~30k 组合，瓶颈是数据量而非特征表示。
- **现状**：如实记录；现实路径是 `transfer_from` 暖启动（good 源实测 +37%）+ 更多真实先验点。

## 7. 4 维 MC-EHVI 的性能坑（已修）

- **现象**：`qehvi_nd` 早期版本 4 目标单次 839 s，无法实用。
- **修复**：PI 预筛 + 支配跳过 + 增量 `hypervolume_contribution`。
- **结果**：4 目标 839 s → ~20 s，且 HV 超过 parego（91244 vs 90134）。
- **现状**：优化保留在代码中；这也是“先做基准再上默认”的一个正面案例。

## 8. 描述符在多保真多目标下降级为“中性”

- **做法**：`mf_solvent`（one-hot 13 维）vs `mf_solvent_desc`（描述符 3 维），
  momfbo / mfparego，budget 40，3 seeds。
- **结果**：momfbo 1195.3 → 1192.0；mfparego 1196.1 → 1196.6（差异 <0.5%）。
- **结论**：多保真多目标场景描述符**既不帮也不伤** HV，收益主要在维度与算力；
  这再次限制“描述符必然更好”的说法。
- **现状**：描述符保持可选；文档不再宣称普适收益。
- **复核**（Round 249，代码多次改动后重跑，3 seeds）：momfbo 1195.3 → 1192.0、mfparego 1196.1 → 1196.6，与归档数字一致（差异 <0.5%），结论不变。

## 9. 停滞后的“探索重启”不如直接继续（而调参优于继续）

- **做法**：`scripts/benchmark_stagnation.py` 三种停滞处置（budget 60，3 seeds）：
  `stop`（接受平台）、`continue`（同算法继续）、`explore`（注入一批 sobol 后恢复自动路由）。
- **结果**（budget 60，3 seeds，`rosenbrock10`）：

| 处置 | best 均值 |
|---|---|
| `retune`（换 acquisition / 抬 kappa，同算法） | **−205.4** |
| `continue`（原样继续） | −319.8 |
| `explore`（sobol 一批后回路由） | −373.7（= `stop`） |

  `mixed_reaction` 上四者几乎相同（101.93–101.94）。
- **结论**：停滞时**先调探索参数**（同一模型、换采集函数或抬 kappa）收益最大；
  模型无关的探索重启没有帮助（只是多花预算）。
- **现状**：已产品化为 `Study.retune_for_exploration()` 与 `condopt retune --run <run>`，
  `next_action` 建议文案改为“调参重调 > 继续当前算法 > 探索重启”，并附实测数字。
- **追问（`patience` 敏感性）**：patience=1 会导致过早停止（−1511），=3 最优（−205），
  ≥5 几乎不触发（−306）；详见 `research/STAGNATION_TUNING.md`。

## 10. 多保真 + 约束：全保真槽位可能全部落在不可行区（已修复交付证据，质量仍待提升）

- **发现**：`examples/constrained_mf_executor.py`（`mf_constrained`，T ≤ 80，cost 预算 10）
  跑出 20 条观测（14 条 0.25 保真 + 6 条全保真），成本 9.5 未超预算，但
  **6 条全保真全部不可行**（T > 80）——交付缺少可行的全保真证据。
- **原因**：`mfbo` 的 `min_full_fraction` 只强制“这一批里有全保真”，
  不会为了约束额外筛“预测可行的候选”；在该 landscape 上约束 GP 又不足以把搜索引到 T≤80。
- **缓解（诊断）**：`step` 输出 `n_full_fidelity` / `n_feasible_full_fidelity`，
  并在“有全保真但全不可行”时给出 `collect_feasible_full_fidelity` 建议。
- **部分修复（可行性感知的全保真槽位）**：`mfbo.feasibility_aware_full=True`（默认，
  **仅在目标带约束时生效**）时，全保真槽位只在 s=1 档位上按 `acquisition × P(feasible)`
  选点，而不是把低保真提议"升级"成全保真；无约束问题（如 `mf_suzuki`）行为完全不变。
- **复核**（`scripts/benchmark_feasible_full.py`，mf_constrained，budget 24，cost 10，5 seeds）：

| `feasibility_aware_full` | 全保真条数 | 可行全保真条数 | 至少一条可行的比例 |
|---|---|---|---|
| False（旧行为） | 4.4 | 1.6 | 0.40 |
| **True（新默认）** | 4.4 | **1.8** | **0.60** |

  改善方向正确但不彻底：单次运行仍可能 0 条可行全保真（示例 seed 0 即如此），
  所以诊断 + `collect_feasible_full_fidelity` 建议仍需保留。
- **尝试过的更激进预筛（无收益）**：`feasibility_full_threshold` 要求全保真候选
  `P(feasible) ≥ 0.5 / 0.7`（无候选达标时回退全体）。mf_constrained、budget 24、5 seeds 下
  三档结果完全相同（可行全保真 1.8、至少一条可行比例 0.60）——在本任务上 GP 的
  P(feasible) 区分度不足，预筛没有改变选点。默认保持 0（关闭）。
- **修复（confirm 优先可行条件）**：当约束存在且**尚无可行的全保真观测**时，`confirm`
  改为确认“已观测到的最优可行条件”（审计 `confirmation_basis="best_feasible_observed"`），
  而不是照搬 recommendation。

  复核（`mf_constrained`，budget 24，cost 10，5 seeds）：

| 配置 | 全保真条数 | 可行全保真条数 | 至少一条可行的比例 |
|---|---|---|---|
| 修复前 | 4.4 | 1.6 | 0.40 |
| 可行性感知槽位 | 4.4 | 1.8 | 0.60 |
| **+ confirm 优先可行** | 4.0 | **3.4** | **1.00** |

  示例 seed 0 也从 0/6 可行变为 **3/6 可行**（`best_feasible_full_fidelity` 34.2）。

- **诚实的边界**：该修复保证“交付有可行的全保真证据”，但不保证该证据*最优*——
  mf_constrained 的可行最优应在 T≈80 处（约 105），而示例只拿到 34.2，
  说明在约束区域内的高质量搜索仍受预算与代理精度限制。
- **尝试过的两阶段可行性搜索（无收益）**：`two_phase_feasibility=True` 时，若尚无可行
  观测则先纯最大化 P(feasible)（phase A）。在 `mf_constrained`（T ≤ 80）与更紧的
  T ≤ 65（约 80% 空间不可行）上，开/关结果完全相同：初始 Sobol 设计在前 1–2 条观测内
  就命中可行点，phase A 从未激活。默认保留 True（语义正确、无害），但在这类问题上无收益。
- **极低可行率的对照实验**：把可行率压到 ~0.4% 后，约束搜索（含 phase A）在 5 seeds 中只有 **1/5** 找到可行点；而把同样的限制写进变量范围后 **5/5** 成功、全部实验可行（可行全保真 11 vs 0.2）。完整对照见 `research/HARD_FEASIBILITY.md`。

## 11. 小样本下 transfer-audit 会误判（真实数据）

- **发现**：真实 ART Table 2 六点数据（目标 3 条 / 源 3 条）上，暖启动 **2/2 找回真实最优
  99.0**（cold 臂只有 83.4 / 75.8），但 `transfer-audit` 给出 `suggested_reliability=0.3`
  （按分级规则属于“avoid”）。
- **原因**：只有 3 个可比条件，秩相关在小样本下方差极大。
- **现状（已缓解）**：`assess_transfer` 返回 `confidence` 与 bootstrap 90% 区间
  （200 次重采样、固定种子），可比条件 <5 或区间宽 >1.0 时判为 `low`；
  `reliability_grade` 相应给 `low_confidence`：**建议值不可当硬约束**，先用 `top_k` 1–2 试。
- **待办**：用更鲁棒的估计（例如源/目标残差建模、或要求更多目标观测后再评估），
  让小样本也能给出可用判断。

## 12. transfer-audit 在真实小样本上没有排序能力

- **做法**：ART 六点全部 C(6,3)=20 个源/目标切分，每切分跑一次 audit，标记源是否含实测最优 99。
- **结果**：源含最优的切分建议 reliability 均值 **0.44**，不含最优的 **0.65**（方向反转）；
  Spearman −0.40 vs −0.10；但 `confidence=low` 在 **20/20** 上全部触发。
- **结论**：n=3 时建议值不可用（保护机制按设计生效）；要判断源可信度，目标侧至少要
  5 个以上可比条件，否则应该靠领域判断或先补观测。
- **待办**：把“可比条件阈值”做成可配置并写入 skill 的使用说明（当前是硬编码 5）。

> 索引由 `scripts/build_negative_index.py` 生成（幂等）；`tests/test_negative_results.py` 校验。

## 13. 迁移否决后反复复检：6 次 0 翻案

**试过**：分阶段迁移被审计否决后，再跑一批就重新审计一次（最多 6 次），看能否“翻案”后放大 top_k。

**结果**（ackley10，budget 24，2 seeds）：有偏源与打乱源各 **6 次复检、0 次翻案**；
Δ 与只否决一次完全相同（有偏 +0.78/+0.63，打乱 +2.37/+1.34）。

**处置**：不把复检机制写进库（`transfer_staged` 每次调用只做一次判定）；否决一次即保持小 top_k。
代价是多算几次审计、收益为零，属于“不需要的复杂度”。

> 索引由 `scripts/build_negative_index.py` 生成（幂等）；`tests/test_negative_results.py` 校验。

## 14. 分阶段迁移的保护依赖审计能否否决：一个问题上 0/4

**试过**：把“先 k=1 → 两批 → 审计决定是否放大”当作可靠的通用配方（在 ackley10 上有偏源 +0.71）。

**结果**（discrete_catalyst，budget 24，4 seeds）：有偏源(−6) **0/4 被审计否决**（审计建议值始终 ≥0.6），
于是分阶段照样放大到 k=5，Δ = **−1.03**，比 fixed_k1（−0.32）、fixed_k3（−0.52）、fixed_k5（−0.38）都差；
用审计自身的建议值打折（staged_soft）也只有 −1.04，没有任何帮助 —— 与“审计对偏差大小天然盲”一致。

**处置**：API 与默认参数不变（ackley10 上仍有效），但**不再声称它普遍优于固定策略**；
建议里写明“审计不否决 ≠ 源可靠，拿不准就保持 k=1”。跨两个问题最稳的臂是 `fixed_k1`。

> 索引由 `scripts/build_negative_index.py` 生成（幂等）；`tests/test_negative_results.py` 校验。
