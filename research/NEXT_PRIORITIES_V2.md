# NEXT_PRIORITIES_V2 — v0.10 之后的优先级（Round 11 版）

> 维护者：algo-research ｜ v1（2026-11-05，Round 11）
> 依据：`REVIEW_v10.md`（2×P0/P1 清单）、`CALIBRATION_NOISE_RUN.md`（context-only noise 的教训）、`FAST_CI_SNIPPETS_V2.md`、`EHVI_3D.md`。
> 与 v1（`NEXT_PRIORITIES.md`）的差异：**多保真"成本预算"上升为 P0**（v0.10 只修了默认 cost，未修 budget/降档）；**校准"真噪声 records"成为正式 456-runs 的硬前置**，不再是可选。

---

## 0. 总表

| # | 事项 | 优先级 | 工作量 | 依赖 | 验收 |
|---|---|---|---|---|---|
| 1 | 成本预算闭环：`cost_budget` 入口 + `remaining_cost()` 默认值 + **全保真 cost 从 fidelity 推导** + `mfbo.ask_with_fidelity` 降档/减批 | **P0** | 1–2 天 | 无 | `test_cost_spent_includes_full_fidelity`；`mf_suzuki` 超支 ≤2% |
| 2 | records 真噪声 + MOO true-HV + 约束/MF 指标 + context 从 problem 推导 | **P0** | 2–3 天 | 1（cost 用于 MF） | 每类问题 ≥1 条 records；noise=high 不再出现 regret>1 |
| 3 | 正式 456-runs LOPO + weights 入库（config 快照/sha256/decisions/回退） | **P1** | 1 + 0.5 天 | 2 | LOPO top1_strict≥0.55、top2≥0.80、regret≤0.10；可回退 |
| 4 | 快 CI 三件套（#4/#5/mf + #2 需先抽 `_acquisition`） | **P1** | 0.5 天 | 无 | PR `pytest -m fast_ci` <15 s（实测可 ~3 s） |
| 5 | 校准口径修正：`regret` 用 span 归一、top1 strict/tie 双口径、权重向 1 正则、折间 min/median/max | **P1** | 0.5–1 天 | 无 | `noise=high` 组 regret∈[0,1]；报告含两口径 |
| 6 | UI Pareto/成本视图（固定 ref、真值/观测区分、概率前沿、成本曲线） | **P2** | 2–3 天 | 1、2 | 渲染层与 `pareto.summarise` 同源；无 ref 时标注不可比 |
| 7 | 权重语义：负分加性、禁用算法从 scores 删除、`-1e9` 移除 | **P2** | 0.5–1 天 | 3 | 负分算法权重方向符合直觉 |
| 8 | `objective_std` 字典化（多目标异方差） | **P2** | 1–2 天 | 无 | qNEHVI 两目标不同 σ 可表达 |
| 9 | 3D EHVI（m=3 递归分区/MC） | **P3（暂缓）** | 2–3 天 | 无（需求低） | 触发条件见 §4 |

---

## 1. P0 #1：成本预算闭环（为什么现在做）

v0.10 只修了"cheap run 的默认 cost"，但：
- 全保真观测 `cost=0`（`study.py:516-520`）→ `cost_spent` 系统性低估；
- `remaining_cost()` 无 `meta["cost_budget"]` 时返回 None，`Study.create`/CLI 无入口；
- `mfbo.ask_with_fidelity` 不消费剩余成本 → v0.8 实测的 25.25/24（+5.2%）超支仍会发生。

**产品影响**：多保真的核心卖点是"同成本更优"，成本口径错 = 卖点不可验证；UI 的成本视图也没有数据源。**改动小**（`study.py` + `mfbo.py` 两处，见 `MFBO_COST_API.md`）。

## 2. P0 #2：records 真噪声/MOO/约束/MF（为什么现在做）

实测证明 `collect_records(noise=...)` 只改 context 而不改数据时，同一批数据会给出相反结论（in-sample top1 0.833 vs 0.000）。因此：
- noise 必须来自 `BenchmarkProblem.noise_std`（Tier D），evaluator 加噪并保留 true 值；
- MOO 指标必须是 **true HV ratio**（`pareto.summarise(use_true=True)`）；
- 约束指标 `feasible_best`；MF 指标 `cost_aware_best`；
- context 从 problem 的 tags/元数据推导（不再用全局参数）。

**不做的后果**：正式 456-runs 的 LOPO 数字会被 context 任意操纵，校准结论不可信。

## 3. P1：正式 456-runs + weights 入库

- 矩阵见 `CALIBRATION_RUN_PLAN.md`（456 runs），预计 8 进程 ~10–20 min；
- 权重入库：写 `config.json["router_weights"]={version,path,sha256,metrics}` + `decisions`；回退 = 空 dict/旧文件；
- 验收：LOPO `top1_strict ≥ 0.55`、`top2 ≥ 0.80`、`regret ≤ 0.10`；与默认权重不劣化；折间分布公开。

## 4. P3（暂缓）：3D EHVI 的触发条件

**继续不做**，除非同时满足：
1. ≥2 个真实用户场景是 m=3（收率/纯度/成本）；
2. ParEGO 在 3 目标上的 `HV_ratio` < 0.70（当前 2 目标 qEHVI≈0.95，3 目标无数据）；
3. 预算 ≥80 且 noise∈{none,low}（高噪需要 qNEHVI-3D，成本更高）；
4. 有 3D 参考前沿与验收脚本（`EHVI_3D.md` 已给分区算法与 HV=7 自检）。

**低成本替代**（0.5 天）：3 目标用 ParEGO + 3D HV（扫描线/MC）报告，先满足"能量化前沿质量"。

## 5. P2：UI Pareto/成本视图

- 依赖 #1（cost）与 #2（true-HV）；
- 内容：Pareto 表（P_nd、E[HVC]、P(feasible)）、成本曲线、观测 vs 真值标注、ref 来源；
- 验收：`render.py` 的 HV 与 `pareto.summarise` 同源；无 ref 时显示"不可比较"。

## 6. 建议节奏

| 周 | 内容 | 门禁 |
|---|---|---|
| W1 | #1 + #4 + #5（成本、快 CI、校准口径） | 140+ tests；mf 超支 ≤2%；fast_ci <15 s；regret∈[0,1] |
| W2 | #2 + #3（真噪声 records、456-runs、weights 入库） | LOPO 达标；weights 可回滚 |
| W3 | #6 + #7/#8（UI、权重语义、异方差） | UI 口径一致 |
| 触发后 | #9（3D EHVI） | 条件满足再排期 |

---

## 7. 参考

R51/R52（折外评估/选择偏差）、R32（统计协议）、R42（HV/真值）、R41（BOCA/成本）、R12（qNEHVI）、R50（配置落盘）、`REVIEW_v10.md`、`CALIBRATION_NOISE_RUN.md`、`EHVI_3D.md`。
