# 研究笔记索引

本目录是优化器/router 的**证据库**：每条默认值都能追溯到这里的数字。
索引由 `scripts/build_research_index.py` 生成，`tests/test_research_index.py` 防止漏登记。

## 概览

| 指标 | 数量 | 入口 |
|---|---|---|
| 研究笔记 | 96（活跃 72 / 历史归档 24） | 本目录 |
| 负结果条目 | 14 | [`NEGATIVE_RESULTS.md`](NEGATIVE_RESULTS.md) |
| 证据声明（可重跑） | 25 | [`EVIDENCE_AUDIT.md`](EVIDENCE_AUDIT.md) |
| 场景断言 | 22 | [`../docs/SCENARIOS.md`](../docs/SCENARIOS.md) |

## 核心结论

| 文档 | 结论 |
|---|---|
| [`BENCHMARK_SPEC.md`](BENCHMARK_SPEC.md) | 基准分层与指标定义（Tier A–G） |
| [`BENCHMARK_TIER_CD.md`](BENCHMARK_TIER_CD.md) | Tier C/D 高基数与混合空间基准 |
| [`MF_BENCHMARK.md`](MF_BENCHMARK.md) | 多保真基准：mfbo 调参 93.2 → 95.3 |
| [`TRANSFER_BENCHMARK.md`](TRANSFER_BENCHMARK.md) | 暖启动收益 +37%（good 源）、坏源随 top_k 恶化、prior_reliability |
| [`REAL_DATA_LOO.md`](REAL_DATA_LOO.md) | 真实 ART 六点 LOO：Spearman −1.00 → −0.83，仍无预测力（样本量） |
| [`REAL_PRIOR_TRANSFER.md`](REAL_PRIOR_TRANSFER.md) | 真实先验 → 代理战役：12 次配对运行 warm 12/12 命中，cold 0/12 |
| [`DESCRIPTOR_FEATURES.md`](DESCRIPTOR_FEATURES.md) | 描述符特征：3/3 命中、噪声鲁棒、Tier C 上只配 GP 类 |
| [`NEGATIVE_RESULTS.md`](NEGATIVE_RESULTS.md) | 负结果清单：cqnehvi / Sobol 播种 / mqehvi_nd / TPE+描述符 / … |
| [`ROUTER_CALIBRATION.md`](ROUTER_CALIBRATION.md) | 路由权重的留一校准（LOPO） |
| [`AUTONOMOUS_LAB_POSITION.md`](AUTONOMOUS_LAB_POSITION.md) | 面向 AI 全自动实验室的定位说明 |
| [`TRANSFER_ADAPTIVE.md`](TRANSFER_ADAPTIVE.md) | 分阶段迁移（已 API 化）：有偏源 +0.71 vs 固定 k3/k5 −1.54/−0.33；否决稳定 |
| [`TRANSFER_ADAPTIVE.md`](TRANSFER_ADAPTIVE.md) | 分阶段迁移：有偏源 +0.71 vs 固定 k3/k5 的 −1.54/−0.33 |
| [`EXTERNAL_BENCHMARK_SCAN.md`](EXTERNAL_BENCHMARK_SCAN.md) | 外部同类项目扫描（78 个项目）：状态机/基准协议/审计护栏四块差距 |
| [`EXTERNAL_REVIEW_RESPONSE.md`](EXTERNAL_REVIEW_RESPONSE.md) | 外部审查（2026-11-11）逐条回应：门禁环境依赖、qNEHVI/TPE/CMA-ES 命名边界、benchmark 协议待办 |
| [`NOISE_FLOOR.md`](NOISE_FLOOR.md) | 先估噪声底再声称差异：描述符 HV 差异 t=−0.704 不显著 |
| [`AUDIT_CALIBRATION.md`](AUDIT_CALIBRATION.md) | 审计阈值校准：中等偏差源分不出，低置信度建议封顶 0.6 |
| [`TRANSFER_TOP_K.md`](TRANSFER_TOP_K.md) | 修掉先验导致的平台期误判后，暖启动由 −31 变 +6~+9（top_k 扫描） |
| [`TRANSFER_COST.md`](TRANSFER_COST.md) | 成本预算下的暖启动：修掉 3 个真实缺陷；budget 24 下收益仍不明显 |
| [`MID_BUDGET.md`](MID_BUDGET.md) | 中段预算停止策略：40–60 预算优先更优但交付证据 2/5 → 1/5 |
| [`TINY_BUDGET.md`](TINY_BUDGET.md) | 极小预算：空间填充 vs 代理实测（约束场景 sobol 仅 1/4 有前沿） |
| [`TRANSFER_MOO.md`](TRANSFER_MOO.md) | 多目标暖启动：全保真 +0.5% HV/+20% 前沿；多保真中性或略负 |
| [`TRANSFER_AUDIT.md`](TRANSFER_AUDIT.md) | transfer-audit 分级：一致源 1.0、坏源 0.3–0.6、小样本降级 |
| [`ADAPTER_LONG_RUN.md`](ADAPTER_LONG_RUN.md) | 执行器适配器长跑：120 次预算，0 重复 id，失败全部补测 |
| [`HARD_FEASIBILITY.md`](HARD_FEASIBILITY.md) | 极低可行率：写进变量范围 5/5 vs 约束搜索 1/5 |
| [`EVIDENCE_AUDIT.md`](EVIDENCE_AUDIT.md) | 证据声明复核：25 条声明重跑（约 390 s） |
| [`STAGNATION_TUNING.md`](STAGNATION_TUNING.md) | 停滞后调参优于继续（−205 vs −320），patience=3 最优 |

## 全部文档

### 规格与设计

| 文档 | 标题 |
|---|---|
| [`CAPABILITIES_SPEC.md`](CAPABILITIES_SPEC.md) | CAPABILITIES_SPEC — 算法能力位、能力过滤与"先过滤后打分" |
| [`MULTIFIDELITY_SPEC.md`](MULTIFIDELITY_SPEC.md) | MULTIFIDELITY_SPEC — 多保真优化的代码级规格（successive halving + 简化 BOCA） |
| [`PAREGO_SPEC.md`](PAREGO_SPEC.md) | PAREGO_SPEC — 多目标 ParEGO 的代码级实现规格（可直接照抄） |
| [`qEHVI_SPEC.md`](qEHVI_SPEC.md) | qEHVI_SPEC — 2D EHVI/qEHVI 的代码级规格 |
| [`qNEHVI_SPEC_V2.md`](qNEHVI_SPEC_V2.md) | qNEHVI_SPEC_V2 — 带噪多目标嵌套 MC 的可实现规格（2D） |
| [`SPRINT2_SPEC.md`](SPRINT2_SPEC.md) | SPRINT2_SPEC — ARD / 局部惩罚 / TS 批量 / Sobol 2^k 的实现要点 |

### 实现与验收

| 文档 | 标题 |
|---|---|
| [`CONSTRAINT_IMPL.md`](CONSTRAINT_IMPL.md) | CONSTRAINT_IMPL — 约束优化落地方案（可直接照抄） |
| [`CONSTRAINTS_NOISE_IMPL.md`](CONSTRAINTS_NOISE_IMPL.md) | CONSTRAINTS_NOISE_IMPL — 带约束/噪声 BO 的代码级实现 |
| [`FAST_CI_ACCEPTANCE.md`](FAST_CI_ACCEPTANCE.md) | FAST_CI_ACCEPTANCE — qNEHVI #2/#4/#5 与 mf_suzuki 的快 CI 落地（含本机实测耗时） |
| [`qEHVI_IMPL.md`](qEHVI_IMPL.md) | qEHVI_IMPL — 2D EHVI/qEHVI 的逐行实现规格（可照抄） |
| [`qNEHVI_ACCEPTANCE.md`](qNEHVI_ACCEPTANCE.md) | qNEHVI_ACCEPTANCE — `moo_yield_cost` 上的 5 条 qNEHVI 验收（可直接写 pytest） |
| [`QNEHVI_APPLY_PLAN.md`](QNEHVI_APPLY_PLAN.md) | QNEHVI_APPLY_PLAN — 把 `QNEHVI_FAST_CI_FINAL` 的 diff 落地（逐文件顺序 + 回滚 + 验证） |
| [`QNEHVI_FAST_CI_PATCH.md`](QNEHVI_FAST_CI_PATCH.md) | QNEHVI_FAST_CI_PATCH — `_acquisition` 抽取的具体补丁与 #2/#4/#5 pytest |
| [`QNEHVI_MF_ACCEPTANCE.md`](QNEHVI_MF_ACCEPTANCE.md) | QNEHVI_MF_ACCEPTANCE — qNEHVI 与 mf_suzuki 剩余验收的本机实测与阈值建议 |
| [`RECORDS_V2_APPLY_PLAN.md`](RECORDS_V2_APPLY_PLAN.md) | RECORDS_V2_APPLY_PLAN — records v2 逐文件应用顺序、488 runs 与预期输出 |
| [`RECORDS_V2_IMPLEMENTATION.md`](RECORDS_V2_IMPLEMENTATION.md) | RECORDS_V2_IMPLEMENTATION — 逐文件 diff、488 runs 命令与耗时 |
| [`RECORDS_V2_PATCH.md`](RECORDS_V2_PATCH.md) | RECORDS_V2_PATCH — `collect_records` 支持真噪声/MOO/约束/MF 的最小实现补丁 |
| [`ROUTER_SCORETERM_PATCH.md`](ROUTER_SCORETERM_PATCH.md) | ROUTER_SCORETERM_PATCH — router.py 的补丁级 ScoreTerm 落地（保持 v0.6 行为逐位一致） |
| [`ROUTER_WEIGHTS_ACCEPTANCE.md`](ROUTER_WEIGHTS_ACCEPTANCE.md) | ROUTER_WEIGHTS_ACCEPTANCE — 权重机制验收与 `runs.jsonl → weights.json` 接口 |
| [`UI_P2_APPLY_PLAN.md`](UI_P2_APPLY_PLAN.md) | UI_P2_APPLY_PLAN — render.py Pareto/成本视图接线顺序与验收 |
| [`UI_P2_IMPLEMENTATION.md`](UI_P2_IMPLEMENTATION.md) | UI_P2_IMPLEMENTATION — `render.py` 上 Pareto/成本视图的最小接线 |

### 历史评审归档

| 文档 | 标题 |
|---|---|
| [`REVIEW_v01.md`](REVIEW_v01.md) | REVIEW_v01 — 对 `src/condopt` v0.1 的文献级实现审查 |
| [`REVIEW_v03.md`](REVIEW_v03.md) | REVIEW_v03 — 对 condopt v0.3 多目标实现的文献级审查 |
| [`REVIEW_v04.md`](REVIEW_v04.md) | REVIEW_v04 — 对 v0.4（Sprint2 / 约束 / ParEGO P0 修复）的文献级审查 |
| [`REVIEW_v05.md`](REVIEW_v05.md) | REVIEW_v05 — 对 v0.5（qehvi.py / ParEGO 窗口修复 / constrained 文档 / Pareto ref）的审查 |
| [`REVIEW_v06.md`](REVIEW_v06.md) | REVIEW_v06 — 对 v0.6（qEHVI P0 修复 / qNEHVI / mfbo / 复孔 alpha）的审查 |
| [`REVIEW_v07.md`](REVIEW_v07.md) | REVIEW_v07 — 对 v0.6.1（qEHVI NaN-ref / qNEHVI 告警 / mf_suzuki / router 权重）的审查 |
| [`REVIEW_v08.md`](REVIEW_v08.md) | REVIEW_v08 — 对 v0.7.1–v0.8（钉住/权重/真值HV/mfbo成本/校准 harness）的审查 |
| [`REVIEW_v09.md`](REVIEW_v09.md) | REVIEW_v09 — 对 v0.8→v0.9（LOPO / records 过滤 / cost API / 全保真 best）的审查 |
| [`REVIEW_v10.md`](REVIEW_v10.md) | REVIEW_v10 — 对 v0.9.1–v0.10 的审查（mfbo incumbent / 成本默认 / calibration tie+noise context） |
| [`REVIEW_v11.md`](REVIEW_v11.md) | REVIEW_v11 — 对 v0.10.1（全保真 cost=1.0 / CLI --cost-budget）的审查 |
| [`REVIEW_v12.md`](REVIEW_v12.md) | REVIEW_v12 — 对 v0.10.2 成本闭环的审查 |
| [`REVIEW_v13.md`](REVIEW_v13.md) | REVIEW_v13 — 对 v0.10.3（统一成本模型 / skill 文档同步）的审查 |
| [`REVIEW_v14.md`](REVIEW_v14.md) | REVIEW_v14 — 对 v0.10.4（成本语义收尾：override / 软契约 / 审计）的审查 |
| [`REVIEW_v15.md`](REVIEW_v15.md) | REVIEW_v15 — 对 v0.10.5（`fidelity_costs` 表 / 全保真计数审计）的审查 |

### 调研与参考

| 文档 | 标题 |
|---|---|
| [`ALGORITHM_SURVEY.md`](ALGORITHM_SURVEY.md) | ALGORITHM_SURVEY — 条件优化算法地图（面向实现） |
| [`ARCHITECTURE_PATTERNS.md`](ARCHITECTURE_PATTERNS.md) | ARCHITECTURE_PATTERNS — 条件优化库的程序架构模式 |
| [`GITHUB_SCAN.md`](GITHUB_SCAN.md) | GITHUB_SCAN — 开源优化框架扫描（程序架构视角） |
| [`REFERENCES.md`](REFERENCES.md) | REFERENCES — 条件优化算法与程序架构（Project6 资料库） |

### 优先级与交接

| 文档 | 标题 |
|---|---|
| [`NEXT_PRIORITIES.md`](NEXT_PRIORITIES.md) | NEXT_PRIORITIES — 下一阶段（v0.10+）优先级与理由 |
| [`NEXT_PRIORITIES_V2.md`](NEXT_PRIORITIES_V2.md) | NEXT_PRIORITIES_V2 — v0.10 之后的优先级（Round 11 版） |
| [`NEXT_PRIORITIES_V3.md`](NEXT_PRIORITIES_V3.md) | NEXT_PRIORITIES_V3 — 路线图（Round 12 版） |
| [`NEXT_ROUND_CHOICES.md`](NEXT_ROUND_CHOICES.md) | NEXT_ROUND_CHOICES — qNEHVI 快 CI / records v2 / UI 三选一 |
| [`OUTBOX.md`](OUTBOX.md) | OUTBOX — algo-research 轮次交接 |

### 历史归档（不再维护）

| 文档 | 标题 |
|---|---|
| [`FAST_CI_SNIPPETS.md`](FAST_CI_SNIPPETS.md) | FAST_CI_SNIPPETS — qNEHVI #2/#4/#5 与 mf_suzuki 的 pytest 片段（v0.9 实测耗时） |
| [`FAST_CI_SNIPPETS_V2.md`](FAST_CI_SNIPPETS_V2.md) | FAST_CI_SNIPPETS_V2 — qNEHVI #4/#5 + mf_suzuki 的可粘贴 pytest（v0.10 实测） |
| [`QNEHVI_FAST_CI_FINAL.md`](QNEHVI_FAST_CI_FINAL.md) | QNEHVI_FAST_CI_FINAL — `_acquisition` 抽取的最终 diff、行为不变验证与 #2 阈值 |
| [`RECORDS_V2_FIRST_STEP.md`](RECORDS_V2_FIRST_STEP.md) | RECORDS_V2_FIRST_STEP — 最小第一步：`BenchmarkProblem` 加 `tags/noise_std` + context 从 problem 推导 |

### 其他工作笔记

| 文档 | 标题 |
|---|---|
| [`BENCHMARK_PROTOCOL.md`](BENCHMARK_PROTOCOL.md) | 基准协议（预注册）BENCHMARK_PROTOCOL |
| [`CALIBRATION_NOISE_RUN.md`](CALIBRATION_NOISE_RUN.md) | CALIBRATION_NOISE_RUN — 带噪声 context 的核心校准实测（v0.10） |
| [`CALIBRATION_RECORDS_V2.md`](CALIBRATION_RECORDS_V2.md) | CALIBRATION_RECORDS_V2 — `collect_records` 支持真噪声 / MOO true-HV / 约束 / 多保真的最小改动 |
| [`CALIBRATION_RUN_PLAN.md`](CALIBRATION_RUN_PLAN.md) | CALIBRATION_RUN_PLAN — 正式 460-runs LOPO-CV 实跑方案 |
| [`CALIBRATION_RUN_RESULTS.md`](CALIBRATION_RUN_RESULTS.md) | CALIBRATION_RUN_RESULTS — 核心子集 LOPO 实测（v0.9） |
| [`CONFIRMATION_BUDGET.md`](CONFIRMATION_BUDGET.md) | 确认复孔批次：按预算缩放被实测否定 |
| [`CONSTRAINED_SECOND_DATASET.md`](CONSTRAINED_SECOND_DATASET.md) | 约束单目标的第二个数据集：硬可行域上 constrained 明显更好 |
| [`DELIVERY_EVIDENCE.md`](DELIVERY_EVIDENCE.md) | 交付证据：四种问题形态的自检通过率 |
| [`DIM_BOUNDARY.md`](DIM_BOUNDARY.md) | 维度边界：cmaes 规则是"维度决定"还是"地貌决定"？ |
| [`EHVI_3D.md`](EHVI_3D.md) | EHVI_3D — 3+ 目标 EHVI 的递归分区与 MC 近似 |
| [`EVIDENCE_POLICY.md`](EVIDENCE_POLICY.md) | 证据政策（Evidence Policy） |
| [`MFBO_COST_API.md`](MFBO_COST_API.md) | MFBO_COST_API — `remaining_cost()` 与 `best-only-full-fidelity` 的设计与验收 |
| [`MFBO_COST_DOWNGRADE.md`](MFBO_COST_DOWNGRADE.md) | MFBO_COST_DOWNGRADE — `remaining_cost` 传入 optimizer 与降档/减批的实现规格 |
| [`MULTI_OBJECTIVE_AND_NOISE.md`](MULTI_OBJECTIVE_AND_NOISE.md) | MULTI_OBJECTIVE_AND_NOISE — 多目标 BO 与噪声 BO 的可实现规格 |
| [`MULTIFIDELITY_MVP.md`](MULTIFIDELITY_MVP.md) | MULTIFIDELITY_MVP — 多保真最小可行实现（最小侵入） |
| [`NOISE_POLICY.md`](NOISE_POLICY.md) | 噪声处置建议：实测否掉了“默认切 UCB” |
| [`PRIOR_RELIABILITY.md`](PRIOR_RELIABILITY.md) | 先验可信度（prior_reliability）：机制修复与效果实测 |
| [`QNEHVI_ACQUISITION_EXTRACT.md`](QNEHVI_ACQUISITION_EXTRACT.md) | QNEHVI_ACQUISITION_EXTRACT — `_model_ask` → `_acquisition()` 逐行替换清单 |
| [`REPLICATES_ALPHA.md`](REPLICATES_ALPHA.md) | REPLICATES_ALPHA — 复孔/重复测量如何映射到观测与 GP alpha |
| [`ROUTER_CALIBRATION_HARNESS.md`](ROUTER_CALIBRATION_HARNESS.md) | ROUTER_CALIBRATION_HARNESS — 最小 LOPO-CV 校准 harness（可落成 `scripts/calibrate_router.py`） |
| [`ROUTER_CALIBRATION_NEW_BENCHMARKS.md`](ROUTER_CALIBRATION_NEW_BENCHMARKS.md) | Router 校准：加入新基准后的复核 |
| [`ROUTER_SCORETERM.md`](ROUTER_SCORETERM.md) | ROUTER_SCORETERM — router.py 的补丁级重构与最小校准实验 |
| [`UI_PARETO_COST.md`](UI_PARETO_COST.md) | UI_PARETO_COST — 看板 Pareto/成本视图的组件与验收 |

## Ralph 循环日志

见 [`ralph/LOG.md`](ralph/LOG.md)。
