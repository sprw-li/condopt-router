# OUTBOX — algo-research 轮次交接

## 本轮变更（v16，2026-11-05，task-26 / Round 16）

- 新增 `research/REVIEW_v15.md`（v0.10.5 审查：**0×P0 / 3×P1 / 8×P2**。P1-1：成本表用**最近 level** 匹配，缺档时静默取别档成本、无 warning；P1-2：`fidelity_costs` 只从 params 字符串读，`Objective.fidelity.costs` 仍被忽略、无优先级说明；P1-3：`min_full_fraction` 软契约的审计已补（full_requested/returned），但未为 `n_full` 预留成本（只能事后解释）。已确认：`_cost` 支持 `fidelity_costs` 表、`last_full_requested/returned` 进 `cost_trim` payload、149 tests）。
- 新增 `research/QNEHVI_ACQUISITION_EXTRACT.md`（**逐行替换清单**：qehvi 旧 119-181 → 新 `_objective_models/_acquisition/_greedy_batch/_model_ask`；qnehvi 旧 104-181 → 新 `_acquisition`（保留 warning 一次性）+ 继承贪心；每段给出"行为不变检查"5 条 + 快照脚本（`scripts/snapshot_ask.py`）与 `allclose(1e-12)` 对比 + #2/#4/#5 测试，<5 s）。
- 新增 `research/RECORDS_V2_FIRST_STEP.md`（最小第一步：`BenchmarkProblem` 只加 `tags/noise_std`、`problem_tags()` helper、`_context_from_problem`；**不动 `run_algorithm`/评估数据**，明确已知局限；3 条测试；预期输出；回滚点 R0–R3）。
- 新增 `research/NEXT_ROUND_CHOICES.md`（三选一：A qNEHVI 快 CI（0.5–1 天，风险最低）、B records v2（2–3 天，价值最高）、C UI（2–3 天，依赖 B）；**建议 A→B→C**，并给"只能选一/两周"两种方案）。
- 更新 `research/REFERENCES.md`：新增 R59 Balandat et al. 2020 BoTorch（NeurIPS 33，arXiv:1910.06403，官方页核），共 **R1–R59**。
- 未改 `src/`、`skill/`、`tests/`、`docs/`、`README.md`（只读代码）。

## 关键结论（Round 16，3 条）

1. **v0.10.5 让成本闭环"可配置、可审计"，但成本表语义仍有两处歧义**：最近 level 匹配会静默错配（P1-1）、params 表 vs `Objective.fidelity.costs` 无优先级（P1-2）。各 10–15 行可修；不影响 v0.10.5 的"149 tests"结论。
2. **qNEHVI 抽取的"逐行清单"已可直接照做**：关键仍是**先快照后改**；`allclose(1e-12)` + 打印 `max_delta`；qnehvi 的第三 GP 用 `models[0]` 等价替换；#2 阈值 Spearman ≥0.85（目标 0.90）、top-8 ≥6。
3. **下一轮建议选 A（qNEHVI 快 CI）**：0.5–1 天、风险最低、立即给核心采集函数加回归保护；B（records v2，2–3 天）是校准可信度的前提但风险更大；C（UI）强依赖 B，提前做会把"未校准/未真值"读数可视化。

## 有争议 / 未核实（下轮）

- **仍待直连核实**：R26 GECCO 2020 页码、R28/R19 ICLR 2024、R32 NeurIPS D&B 2021 的会议页；R55 书条目作者列。
- **非文献**：`PKU-DAIR/OpenBox`、`modest-py/modest` 仓库 404。
- **争议/待实测**：成本表"精确档 + 缺失 warning" vs "最近档 + 显式声明"；`fidelity_costs`（params）与 `Objective.fidelity.costs` 谁优先（建议 params > objective > 公式）；软契约是否需"预算足够时至少 1 个全保真"的事前保证；qNEHVI 快照逐位一致性（`allclose` 兜底）。
- **环境**：`cost_spent()` 仍是方法；旧 study cost=0 混合；qehvi/qnehvi 尚未应用最终 diff。

## 下轮建议（按价值排序）

1. **A：按 `QNEHVI_ACQUISITION_EXTRACT.md` 落 qNEHVI 快 CI**（先快照、后改、`allclose` 验证；<5 s）。
2. **B：按 `RECORDS_V2_FIRST_STEP.md` 做最小第一步**（tags/noise_std + context），再扩 `run_task`/四类 metric + 488 runs。
3. **成本表语义收尾**（精确匹配 + warning、params/objective 优先级、`n_full` 预留）。
4. **C：按 `UI_P2_APPLY_PLAN.md` 接线 UI**（依赖 B 的 true-HV 口径）。
5. P2 清理：`cost_trim` 审计 payload、report 成本文本、旧数据迁移、router 权重语义。

## 文件清单

`research/REFERENCES.md`（R1–R59）· `REVIEW_v01.md`…`REVIEW_v15.md` · `ALGORITHM_SURVEY.md` · `GITHUB_SCAN.md` · `ARCHITECTURE_PATTERNS.md` · `BENCHMARK_SPEC.md` · `BENCHMARK_TIER_CD.md` · `MULTI_OBJECTIVE_AND_NOISE.md` · `PAREGO_SPEC.md` · `qEHVI_SPEC.md` · `qEHVI_IMPL.md` · `qNEHVI_SPEC_V2.md` · `qNEHVI_ACCEPTANCE.md` · `QNEHVI_FAST_CI_PATCH.md` · `QNEHVI_FAST_CI_FINAL.md` · `QNEHVI_APPLY_PLAN.md` · `QNEHVI_ACQUISITION_EXTRACT.md` · `EHVI_3D.md` · `CONSTRAINTS_NOISE_IMPL.md` · `CONSTRAINT_IMPL.md` · `SPRINT2_SPEC.md` · `MULTIFIDELITY_SPEC.md` · `MULTIFIDELITY_MVP.md` · `MF_BENCHMARK.md` · `MFBO_COST_API.md` · `MFBO_COST_DOWNGRADE.md` · `REPLICATES_ALPHA.md` · `CAPABILITIES_SPEC.md` · `ROUTER_CALIBRATION.md` · `ROUTER_SCORETERM.md` · `ROUTER_SCORETERM_PATCH.md` · `ROUTER_WEIGHTS_ACCEPTANCE.md` · `ROUTER_CALIBRATION_HARNESS.md` · `CALIBRATION_RUN_PLAN.md` · `CALIBRATION_RUN_RESULTS.md` · `CALIBRATION_NOISE_RUN.md` · `CALIBRATION_RECORDS_V2.md` · `RECORDS_V2_PATCH.md` · `RECORDS_V2_IMPLEMENTATION.md` · `RECORDS_V2_APPLY_PLAN.md` · `RECORDS_V2_FIRST_STEP.md` · `QNEHVI_MF_ACCEPTANCE.md` · `FAST_CI_ACCEPTANCE.md` · `FAST_CI_SNIPPETS.md` · `FAST_CI_SNIPPETS_V2.md` · `NEXT_PRIORITIES.md` · `NEXT_PRIORITIES_V2.md` · `NEXT_PRIORITIES_V3.md` · `NEXT_ROUND_CHOICES.md` · `UI_PARETO_COST.md` · `UI_P2_IMPLEMENTATION.md` · `UI_P2_APPLY_PLAN.md` · `AUTONOMOUS_LAB_POSITION.md` · `papers/NOTES.md`
