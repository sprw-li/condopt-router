# 默认值来源总表（Provenance of Defaults）

> 证据复核：`py -3 scripts/verify_evidence.py`（重跑关键基准，确认表中数字未漂；见 `research/EVIDENCE_AUDIT.md`）。
>
> 自动对账：`py -3 scripts/check_defaults_sync.py`（已纳入 release_check）会核对本表引用的参数默认值、`Study.create` 签名、router 路由结果与 study.py 中的显式阈值。

router 与算法里的每个默认值都应有实测依据；本表集中列出，避免"凭感觉调参"。
新增默认值时请同时补一行与证据文件。

## 路由默认

| 场景 | 默认 | 实测依据 | 证据文件 |
|---|---|---|---|
| 单目标 + 硬约束 | `constrained` | 溶剂场景 90.394（one-hot）/ 90.377（描述符）vs bayes 84.7；**硬可行域（6%）**：constrained 92.97（3/3 有可行点）vs bayes 87.80（仅 1/3）；约束不绑定时两者持平 | `research/CONSTRAINED_SECOND_DATASET.md`、`research/DESCRIPTOR_FEATURES.md` |
| 2 目标 + 约束 + 噪声 | `cqehvi` | σ=3：cqehvi 638.2 vs cqnehvi 631.3；σ=6：635.0 vs 625.6 | `research/NEGATIVE_RESULTS.md` §1 |
| 3–4 目标 | `qehvi_nd` | HV 5797 vs parego 5578（3 目标）、91244 vs 90134（4 目标）；单次 839 s → ~20 s | `docs/ALGORITHM_CATALOG.md` |
| 单目标 + 多保真 | `mfbo` | acquisition=ei：93.2 → 95.3 | `research/MF_BENCHMARK.md` |
| 2+ 目标 + 多保真（含约束） | `mfparego` | 四处设置一致：672.3/676.1 vs momfbo 666.2/670.3；约束 642.5 vs 635.6；3 目标 27399 vs 26869 | `research/ROUTER_CALIBRATION_NEW_BENCHMARKS.md` |
| 高维连续（8–14 维光滑，**地貌依赖**） | `cmaes`（权重上调，bayes 在高维光滑扣分） | 10 维：cmaes −18.1 vs bayes −155.4（rosenbrock10）、−2.20 vs −5.01（ackley10）；6–12 维扫描：窄谷型 6 维起 cmaes 更好（−12.0 vs −33.6），多峰型 6/8/12 维 bayes 更优 → 规则是“高维 GP 不可靠”的兜底，非纯维度决定 | `docs/SCENARIOS.md`、`research/DIM_BOUNDARY.md` |
| 分类描述符存在 | 优先 GP 类 | Tier C：bayes 92.02 vs one-hot 91.86；tpe 84.07 vs 90.22（更差） | `research/DESCRIPTOR_FEATURES.md` |
| 覆盖率回归 | 每上下文至少 2 候选 | `single_candidate` 24 → 0 | `docs/ROUTING_POLICY.md` |
| 校准数据要求 | `calibrate_router.py` 默认每问题 2 seeds | 单 seed 标签会把 sphere2d 判给 cmaes；3 seeds 复核 bayes 更优（branin −0.41 vs −5.12） | `research/ROUTER_CALIBRATION_NEW_BENCHMARKS.md` |
| 权重校准（可选） | 默认规则；校准文件可显式加载 | LOPO top1 0.750 持平、top2 0.917 → 1.000 | `research/ROUTER_CALIBRATION_NEW_BENCHMARKS.md` |

## 循环与停滞

| 默认 | 值 | 实测依据 | 证据文件 |
|---|---|---|---|
| `patience` | 3（<2 夹紧到 2） | patience=1 停止过早（−1511）；=3 最优；≥5 几乎不触发 | `research/STAGNATION_TUNING.md` |
| 停止口径 | `stop_on_stagnation=True`（停滞即停） | 需要跑满预算时设 False（budget_first）：停滞只作为建议。`prepare` 在预算 ≥60 时建议 budget_first | `research/ADAPTER_LONG_RUN.md` |
| 停止口径阈值 | 预算 ≥60 才建议 `budget_first` | MF+约束：40/60/90/120 预算下 best 92.9/98.8/100.6/102.2 vs 71.8/65.6/76.9/65.6；修正交付检查口径与基准步数上限后，两种口径在 40–120 上**交付全部通过** | `research/MID_BUDGET.md` |
| 交付自检覆盖 | 约束/多保真/多目标/单目标四种形态都必须通过「交付自查」 | 实测 12/12（无约束 MOO parego、多保真 MOO mfparego、约束 MOO cqehvi、单目标 bayes 各 3/3） | `research/DELIVERY_EVIDENCE.md` |
| 交付槽位保留 | 最多保留 1 次评估（未被新最优点用掉就浪费掉） | 保证“最终交付的那个点”总有确认复孔：budget-first 实测 29/30、23/24；换来 e2e 演练交付自查 100% 通过 | `docs/EXECUTOR_ADAPTER.md` |
| 极小预算的建模下限 | GP 类 `max(4, min(12, effective_dim))`；TPE `max(4, min(8, dim//2))` | 预算低于下限时先屏蔽该算法；单目标改走空间填充（预算 4 -> sobol）。约束/多目标没有便宜算法时**保留代理 + 告警**：实测 budget 4 MOO 下 parego 503.6 HV/3-4 次有前沿，sobol 只有 1/4；预算 ≤2 才自动放宽。TPE 不按 one-hot 维度计 | `docs/ROUTING_POLICY.md` |
| 确认复孔批次 | 3（可用 meta.confirm_n 覆盖） | branin 20 seeds：2 次复孔中位数 3.33 vs **3 次 4.08**；branin 10 seeds 也 3 次更好（5.67 vs 3.44/2.36）；但 mf_constrained 10 seeds 反向（1–2 次 47.98 vs 3 次 45.21）；交付证据在两种设置下都 10/10 → 方向不一致，保持 3，追最优值可用 confirm_n=2 | `research/CONFIRMATION_BUDGET.md` |
| 停滞处置 | `retune_for_exploration()` | rosenbrock10 −205.4 vs continue −319.8 vs 探索 −373.7；ackley10 同样 tune 更优 | `research/STAGNATION_TUNING.md` |
| 噪声处置 | 复孔 3 + 批量 ≤6 + 随机基线臂；**不默认切 UCB，先做 A/B** | 混合/离散型（mixed_reaction、suzuki_like）EI 最稳、UCB 更差；光滑低维（hartmann3）UCB 反而更好 → 地形依赖，故只保留方向一致的复孔/小批量/基线臂 | `research/NOISE_POLICY.md` |
| 噪声下停滞 | 仍用 `retune` | σ=3：−208.1 vs −278.1 | `research/STAGNATION_TUNING.md` |
| 约束下停滞 | 仍用 `retune` | 可行点 37.7 → 43.7（+16%）；MF+约束 budget 80：101.92 vs 101.06（+0.85%，多用 28% 预算） | `research/STAGNATION_TUNING.md` |

## 迁移（warm start）

| 默认 | 值 | 实测依据 | 证据文件 |
|---|---|---|---|
| `top_k` | 建议 1–3；>4 时告警；fork 默认 5 | bad 源负迁移 k=1 −369 → k=8 −1190 | `research/TRANSFER_BENCHMARK.md` |
| 迁移否决线 | 取 0.4–0.6（不建议 <0.4 或 ≥0.8） | ackley10（2 seeds）扫描：0.2 时有偏源未被否决、Δ 回到 −0.53；0.4/0.6 保护生效且好源不被否（最差 +0.71）；≥0.8 好源也被否（否决率 1.0），机制退化为“永不放大” | `research/TRANSFER_ADAPTIVE.md` |
| 分阶段迁移 | 先 top_k 1，跑两批后再决定是否放大 | ackley10（budget 24，2 seeds）：有偏源 staged +0.71（否决 2/2）vs 固定 k3/k5 的 **−1.54 / −0.33**；好源 staged +1.85 ≈ k5 +1.85，k3 +2.14（扩大 top_k 无额外收益） | `research/TRANSFER_ADAPTIVE.md` |
| 审计置信度封顶 | 低置信度审计的建议可靠度 ≤0.6 | ackley10（2 seeds）：有偏源审计 0.65（旧规则=完全信任）却带来 −0.54；相同源 0.60 还低于轻微噪声 0.65 → 低置信度不得给 1.0 | `research/AUDIT_CALIBRATION.md` |
| 迁移可信度分级 | audit ≥0.8 用原值（不打折）+ top_k 3–5；0.5–0.8 压到 top_k 1–2 并打折 0.4–0.6；<0.5 不暖启动 | 单保真源 + top_k=5（2 seeds）：cold 49.03；r=1.0/0.7 → **93.89（+44.9）**；r=0.4 → 88.29（+39.3，**损失 5.6**）→ 可信源不该打折 | `research/PRIOR_RELIABILITY.md` |
| 迁移观测的复孔语义 | 导入的观测 `n_replicates=1`，源次数存 `meta.source_n_replicates` | 源已复孔不等于目标已复孔；否则会跳过确认、交付缺证据（实测 warm 臂自查 0/4 → 4/4） | `research/TRANSFER_COST.md` |
| 迁移的元数据清洗 | 源的 purpose/plate/plate_well/batch 移入 `meta.source_meta` | 否则源的 confirmation 标记会让目标误判已确认 | `research/TRANSFER_COST.md` |
| 暖启动的预算口径依赖 | 纯次数预算：+6~+9；硬成本预算：−5（先验计入成本） | mf_constrained budget 30（5 seeds）cold 73.81 → k=3 81.10；cost_budget 24（4 seeds）none 39.87 → full 34.05 / with_low 34.82 | `research/TRANSFER_TOP_K.md`、`research/TRANSFER_COST.md` |
| 暖启动的收益（修复后） | 好源 +6~+9 分（budget 30，5 seeds）；坏源按 top_k 放大负迁移 | 修掉“先验导致平台期误判、提前停止”的缺陷前后：−31.5 → **+6.0**（k=1）；k=3 81.10、k=8 82.98 vs 冷启动 73.81 | `research/TRANSFER_TOP_K.md` |
| 真实先验暖启动 | 强烈建议（真实数据 12/12 vs 0/12） | ART Table 2 六点做源、代理 oracle 做目标：warm 12/12 命中真实最优，cold 0/12；结论不敏感于种子与 cold 起点 | `research/REAL_PRIOR_TRANSFER.md` |
| 多目标暖启动（全保真） | 建议开启（前沿迁移） |
| 多目标暖启动（多保真） | 小 k 中性；k≈8 含低保真略正 | 5 seeds：626.92（不开）/ 626.27（k3）/ 626.68（k5）/**629.09（k8 含低保真，+0.35%）**/ 620.94（r=0.5，−0.95%）；迁移点按次数占目标预算 | `research/TRANSFER_MOO.md` | 5 seeds：可行 HV +0.4–0.5%，**前沿点数 +20%**（9.6 → 11.6）；reliability 1.0 与 0.5 无差别 | `research/TRANSFER_MOO.md` |
| 暖启动记账 | `transfer_from` 返回 `budget_charged`/`cost_charged` | 迁移点按次数占用目标预算，审计可查剩余预算；`transfer_from` 可传 run 目录或已加载的 Study | `research/TRANSFER_MOO.md` |
| 选择口径 | 单目标主目标；多目标非支配前沿优先 | 主目标取 top_k 会丢掉前沿另一端 | `research/TRANSFER_BENCHMARK.md` |
| `prior_reliability` | 1.0；可疑源 0.25–0.5 | bad 源 top_k=4：−884 → −698 | `research/TRANSFER_BENCHMARK.md` |
| 可信度分级 | ≥0.8 用 / 0.5–0.8 可疑 / <0.5 避免；**可比条件 <5 或 bootstrap CI 宽 >1.0 时降级为 low_confidence（不可当硬约束）** | CLI `transfer-audit` 直接打印评级与建议 | `research/TRANSFER_AUDIT.md` |
| 可信度估计口径 | bootstrap 中位数（200 次，固定种子） | 比单点 Spearman 稳；同时给出 `suggested_reliability_point` 与 `spearman_robust` | `research/TRANSFER_AUDIT.md` |
| 可信度评估 | `transfer-audit`（目标 ≥3 条后） | 4 类源对照：一致源 **1.0**、shifted/shuffled/flipped 0.3–0.6（打乱源有一例 0.6，按“可疑”处理） | `research/TRANSFER_AUDIT.md` |

## 执行器与成本

| 默认 | 值 | 实测依据 | 证据文件 |
|---|---|---|---|
| 成本预算 | 硬上限（ask/confirm 截断） | 示例从超支 13.5 修到恰好 12.0 | `docs/EXECUTOR_INTEGRATION.md` |
| 失败孔位 | 保留 pending，可补测重跑 | 部分 tell 返回 `still_pending` 且写审计 | `docs/EXECUTOR_INTEGRATION.md` |
| 全保真槽位（仅带约束时生效） | `feasibility_aware_full=true` | mf_constrained：可行全保真条数 1.6 → 1.8，至少一条可行的比例 0.40 → 0.60；无约束问题行为不变 | `research/NEGATIVE_RESULTS.md` §10 |
| 全保真约束预筛 | `feasibility_full_threshold=0`（关闭） | mf_constrained 上 0.0/0.5/0.7 三档结果完全相同（可行全保真 1.8、可行率 0.60）→ 无实测收益，默认关闭 | `research/NEGATIVE_RESULTS.md` §10 |
| 低可行率检测 | 观测可行率 <5% 且 ≥10 条观测时提示 | step 输出 `observed_feasible_fraction` 并给出 `encode_limits_as_bounds` 建议 | `research/HARD_FEASIBILITY.md` |
| 极低可行率（<5%） | 把硬限制写进变量范围/固定因子 | 0.4% 可行盒子下：约束搜索找到可行点 1/5、写进范围后 5/5 且全部实验可行（可行全保真 11 vs 0.2） | `research/HARD_FEASIBILITY.md` |
| 两阶段可行性（约束时） | `two_phase_feasibility=true` | mf_constrained（T≤80 与 T≤65）开关结果完全相同：初始设计已命中可行点，phase A 不激活 → 保留但无实测收益 | `research/NEGATIVE_RESULTS.md` §10 |
| confirm 的条件选择 | 无可行全保真时确认“最优可行已观测条件” | mf_constrained：至少一条可行全保真的比例 0.40 → **1.00**，可行全保真条数 1.6 → 3.4 | `research/NEGATIVE_RESULTS.md` §10 |
| 全保真可行性告警 | 有全保真但全不可行时建议补测 | 同上（保留作为兜底） | `research/NEGATIVE_RESULTS.md` §10 |
| 适配器“成功”定义 | 有可用观测且完整性 OK、成本未超 | 空跑（0 条可用观测）不再算成功（`ok=false`） | `docs/EXECUTOR_ADAPTER.md` |
| 执行器并行/超时 | `max_workers=1`、`timeout=None`（按需开启） | 并行只包裹 measure，study 仍串行；超时转成失败孔位可补测 | `docs/EXECUTOR_ADAPTER.md` |
| 失败分类 | `failure_kind`（timeout / exception:<Type> / instrument_error / abandoned） | 报告 `failure_kinds` + `failure_summary()` 事后汇总，区分仪器与算法问题 | `docs/EXECUTOR_ADAPTER.md` |
| 失败重试上限 | `max_retries=2`，超过则 `cancel_pending` 放弃该建议 | 坏条件不再永久占用队列；执行日志标 `abandoned`、决策日志记 `cancel_pending`；`-1` 表示不限 | `docs/EXECUTOR_ADAPTER.md` |
| 执行器故障保护 | 连续 3 批无可用结果即停（`max_empty_batches`） | 坏仪器下不再空转到 max_steps，停止原因写入 `stopped_reason` | `docs/EXECUTOR_ADAPTER.md` |
| 无人值守循环 | `run_campaign(study, executor)` | 修复成本预算不足时的 confirm 空转：现在直接 stop（stop_category=cost_exhausted） | `docs/EXECUTOR_ADAPTER.md` |
| 无人值守 | doctor → step → doctor | 120 次跑满：ask p95 0.39 s、digest 1.1 s、无重复 id | `docs/DECISION_LAYER.md` |

## 数据质量与结论纪律

| 规则 | 说明 | 证据文件 |
|---|---|---|
| 真实数据只认实测 | ART Table 2 六条为真实测量；代理/合成明确标注 | `research/REAL_DATA_LOO.md` |
| 负结果必归档 | 9 条"试过没赢"记录，含数字与处置 | `research/NEGATIVE_RESULTS.md` |
| 描述符收益不夸大 | 合成 3/3 命中、噪声鲁棒；Tier C 仅 GP 略优；MF+MOO 中性（Round 249 重跑：momfbo 1195.3→1192.0、mfparego 1196.1→1196.6，<0.5%）；约束下可行率 +34pt | `research/DESCRIPTOR_FEATURES.md`、`research/NEGATIVE_RESULTS.md` |
