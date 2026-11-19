# NEXT_PRIORITIES_V3 — 路线图（Round 12 版）

> 维护者：algo-research ｜ v1（2026-11-05，Round 12）
> 依据：`REVIEW_v11.md`（v0.10.1 后的 1×P0/P1 清单）、`MFBO_COST_DOWNGRADE.md`、`CALIBRATION_RECORDS_V2.md`、`EHVI_3D.md`。
> 相较 v2 的变化：成本"传参/降档"从"设计"变成"可直接照抄的规格"；records 扩展明确到 `run_task`/`metric` 级别；**UI Pareto/成本视图**与 **3D EHVI 触发条件**写入路线图。

---

## 0. 总表

| # | 事项 | 优先级 | 工作量 | 依赖 | 验收 |
|---|---|---|---|---|---|
| 1 | mfbo 成本降档/减批（`remaining_cost` 接线） | **P0** | 1 天 | 无 | `mf_suzuki` cost 超支 ≤2%；5 条 pytest（`MFBO_COST_DOWNGRADE.md`） |
| 2 | `Study.create(cost_budget=...)` 一等参数 + `report` 成本字段 | P1 | 0.5 天 | 1 | `test_study_create_cost_budget` |
| 3 | records v2：真噪声/MOO true-HV/约束/MF + context 从 problem 推导 | **P0** | 2–3 天 | 1 | `test_collect_records_covers_all_types` |
| 4 | 正式 456–488 runs LOPO + weights 入库（config/sha256/decisions/回退） | P1 | 1.5 天 | 3 | LOPO top1_strict≥0.55、top2≥0.80、regret≤0.10 |
| 5 | 快 CI 三件套（#2 先抽 `_acquisition`） | P1 | 0.5 天 | 无 | PR `pytest -m fast_ci` <15 s |
| 6 | **UI Pareto/成本视图** | P2 | 2–3 天 | 1、3、4 | 见 §3 |
| 7 | 校准口径：strict/tie 双指标、regret span、权重正则 | P1 | 0.5 天 | 3 | `noise` 组 regret∈[0,1] |
| 8 | 权重语义（负分加性、禁用从 scores 删除） | P2 | 0.5–1 天 | 4 | 负分算法权重方向符合直觉 |
| 9 | `objective_std` 字典化（多目标异方差） | P2 | 1–2 天 | 无 | qNEHVI 两目标不同 σ |
| 10 | **3D EHVI** | **P3（暂缓）** | 2–3 天 | 无（需求低） | 触发条件见 §4 |

---

## 1. P0：成本降档 + records v2（并行）

**#1 成本降档**：v0.10.1 已把"全保真 cost=1.0 + `--cost-budget` + `remaining_cost()`"补齐，但 `mfbo` 不消费 → 预算仍是装饰。`MFBO_COST_DOWNGRADE.md` 给了①传参方案（构造参数 + call override）、②`_cost/_fit_to_budget/_remaining_budget` 伪代码、③减批/`min_full_fraction` 预留、④5 条 pytest。**一天可落**。
**#3 records v2**：`CALIBRATION_RECORDS_V2.md` 给了 `BenchmarkProblem` 元数据、`run_task` 分派、`compute_metric`（true_best/true_hv_ratio/feasible_best/cost_aware_best）与 context 推导；这是正式 456-runs 的**硬前置**（否则 calibration 只能覆盖无噪单目标）。

## 2. P1：正式 456–488 runs + weights 入库

- 矩阵见 `CALIBRATION_RECORDS_V2.md` §3（488 runs ≈ 8–12 min）；
- weights 入库：`config.json["router_weights"]={version,path,sha256,metrics}` + `decisions` + 回退；
- 验收：LOPO `top1_strict ≥0.55`、`top2 ≥0.80`、`regret ≤0.10`；与默认权重不劣化；折间 min/median/max 公开。
- **注意**：v0.11 的 `top1` 若含 tie，需同时报 strict（见 `REVIEW_v11` P1-3/`REVIEW_v10` P1-4）。

## 3. P2：UI Pareto/成本视图（详细）

**依赖**：#1（成本）、#3（true-HV/约束/MF metric）、#4（weights 版本）。**在依赖完成前做 UI 会把错误口径可视化**。

**页面组件**

1. **Pareto 表**（2 目标固定；3+ 目标先表格化）：
   - 列：条件、目标均值±std、`P_nd`（概率非支配）、`E[HVC]`（期望超体积贡献）、`P(feasible)`（有约束时）；
   - `P_nd/E[HVC]` 用 `MULTI_OBJECTIVE_AND_NOISE.md` §5 的 MC（M=128）；
   - 排序默认 `E[HVC]`，可切 `P_nd`。
2. **成本视图**（多保真）：
   - x=累计成本、y=best full-fidelity（`cost_aware_best` 前缀曲线）；同图叠加 `n_evals` 口径；
   - 显示 `cost_spent/cost_budget/remaining_cost`、各 fidelity 评估次数；
   - 超支时红色标注（`cost_spent > cost_budget`）。
3. **口径提示**：
   - ref point 来源（`Objective.ref_point` 显式 vs 数据推导）与"是否可跨 run 比较"；
   - 观测值 vs 真值（合成/emulator 才有真值）；
   - 最新 weights 版本（`router_weights.version`）。
4. **导出**：Pareto 表 CSV + 成本曲线 PNG/SVG；供实验记录引用。

**验收**

- `render.py` 的 HV 与 `pareto.summarise` 同源（单测：同一 dataset 输出一致）；
- 无 `ref_point` 时显示"HV 不可跨 run 比较"，不显示数值或加 ⚠️；
- 成本视图在 `cost_budget` 未设置时显示"次数口径"并隐藏成本轴；
- 概率 Pareto 的 MC 抽样数/模型版本写入页面 footer（可复现）。

## 4. P3（暂缓）：3D EHVI 触发条件

**继续不做**，除非同时满足：
1. ≥2 个真实用户场景需要 m=3（收率/纯度/成本）；
2. ParEGO 在 3 目标上的 `HV_ratio < 0.70`（当前无 3 目标基准数据）；
3. 预算 ≥80 且 noise∈{none,low}；
4. 有 3D 参考前沿 + `EHVI_3D.md` 的 `partition_nd`/`DHV=7` 自检 + 验收脚本。

**低成本替代**（0.5 天）：3 目标继续 ParEGO，但报告 3D HV（扫描线/MC）→ 先量化"到底差多少"。

## 5. 建议节奏

| 周 | 内容 | 门禁 |
|---|---|---|
| W1 | #1 + #2 + #5 + #7（成本接线、快 CI、口径） | 141+ tests；mf 超支 ≤2%；regret∈[0,1] |
| W2 | #3 + #4（records v2、488 runs、weights 入库） | LOPO 达标；weights 可回滚 |
| W3 | #6 + #8/#9（UI、权重语义、std 字典化） | UI 口径与算法层一致 |
| 触发后 | #10（3D EHVI） | 条件满足再排期 |

---

## 6. 参考

R41（BOCA/成本）、R39/R40（预算调度）、R42（HV/真值）、R32（统计）、R54（HPO 综述）、R50（配置落盘）、`REVIEW_v11.md`、`MFBO_COST_DOWNGRADE.md`、`CALIBRATION_RECORDS_V2.md`、`EHVI_3D.md`。
