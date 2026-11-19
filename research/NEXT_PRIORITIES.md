# NEXT_PRIORITIES — 下一阶段（v0.10+）优先级与理由

> 维护者：algo-research ｜ v1（2026-11-05，Round 10）
> 输入：`REVIEW_v09.md`（v0.9 的 2×P0/P1 清单）、`CALIBRATION_RUN_RESULTS.md`（核心子集实测）、`FAST_CI_SNIPPETS.md`、`EHVI_3D.md`（3D EHVI 规格）。
> 排序原则：**先修"结论会错"的口径/静默问题，再做"扩展算法"，最后做 UI**；每项给"为什么现在做/不做、依赖、验收"。

---

## 0. 优先级总表

| # | 事项 | 优先级 | 工作量 | 依赖 | 验收 |
|---|---|---|---|---|---|
| 1 | mfbo `best` 只用全保真 + cost 从 fidelity 推导 + `cost_budget` 接线 | **P0** | 1–2 天 | 无 | `mf_suzuki` 成本超支 ≤2%；`test_mfbo_best_uses_full_fidelity_only` |
| 2 | 校准 records 扩到噪声/约束/MOO/MF + MOO 用 true HV + tie 口径 | **P0** | 2–3 天 | 1 | 456-runs LOPO 跑通；每类问题 ≥1 条 records |
| 3 | 正式 456-runs 校准 + weights 入库（`config.json` 快照/sha256/decisions） | **P1** | 1 天（跑）+ 0.5 天（入库） | 2 | LOPO top1≥0.55、top2≥0.80、regret≤0.10；可回退 |
| 4 | 快 CI 三件套（qNEHVI #2 需先抽 `_acquisition`） | **P1** | 0.5 天 | 无 | PR `pytest -m fast_ci` <15 s |
| 5 | UI Pareto/成本视图（固定 ref、真值/观测区分、概率前沿） | **P2** | 2–3 天 | 1、2 | 渲染层显示 ref 来源/成本/不确定性；不重算 HV |
| 6 | 3D EHVI（m=3 递归分区/MC） | **P3（暂缓）** | 2–3 天 | 无（但需求低） | 触发条件见 §4 |
| 7 | `objective_std` 字典化（多目标异方差噪声） | **P2** | 1–2 天 | 无 | qNEHVI 两目标不同 σ 可表达 |
| 8 | 权重语义改加性 + 禁用算法从 scores 移除 | **P2** | 0.5–1 天 | 3 | 负分算法权重方向不再反直觉 |

---

## 1. P0：多保真成本/基线闭环（#1）

**为什么现在做**：v0.9 的 `mfbo` 仍 `best = max(self._y)`（混保真），`cost_spent()` 只累加 `observation.cost`（默认 0，不从 fidelity 推导），`remaining_cost()` 无默认预算且 `mfbo` 不消费 → "同成本优于全保真 BO"这一产品级结论**不可复现**（v0.8 实测 25.25/24 超支 5.2% 不会自动修好）。这是当前最"贵"的错误：样本效率声明的基础。
**改哪里**：`mfbo.py:205`（best 全保真）、`study.py:271-280/522`（cost 从 fidelity、cost_budget、report）、`mfbo.ask_with_fidelity`（按剩余成本降档/减批）。
**不做会怎样**：后续所有 mfbo 的 benchmark/验收数字都带系统偏差；UI 的成本视图也没有可信数据源。

## 2. P0：校准矩阵覆盖（#2）

**为什么现在做**：核心子集实测（60 runs）里 5/6 问题都是 bayes 最优 → LOPO 0.833 但**没有覆盖**噪声/约束/MOO/MF，而这正是 router 规则最多、权重最需要的部分；同时 MOO 的 `best_score` 是主目标而非 true HV（标签会错）。
**必须修的配套**：`collect_records` context 带 noise/smoothness/multi_fidelity；`run_algorithm` 增加 `true_hv_ratio` / `feasible_best` / `cost_aware_best` 指标；tie 检测（`|Δ|<5% span` 不进 top-1 监督）；regret 用 `(best−chosen)/(best−worst)`。
**验收**：456-runs 的 records 每类 ≥1；LOPO 报告含折间 min/median/max。

## 3. P1：正式 weights 入库（#3）

**为什么**：目前权重是进程级全局，只有 `route/init` 加载，`ask/tell` 不加载 → 同一 study 不同命令行为不一致（v0.8 P1-7 未修）。正式 weights 必须：
- 写入 `config.json["router_weights"] = {version, path, sha256, metrics}`；
- `Study.log_decision(action="router_weights", ...)`；
- 回退：删文件/空 dict 回到手写规则；保留历史文件可切回。
**验收**：`test_study_records_router_weights_snapshot`、`test_weights_rollback`。

## 4. P3（暂缓）：3D EHVI 的决策

**结论：继续暂缓（v0.10 不做）。**
- **需求**：化学场景 3 目标（收率/纯度/成本）确实存在，但可用 ParEGO 标量化 + 概率 Pareto 报告覆盖；当前 qEHVI/qNEHVI 只支持 2 目标（显式 raise，不会静默出错）。
- **成本**：3D 递归分区 box 数 `O(n²)`（n_pareto=30 → ~435 boxes），MC 每 ask 2–4 s；实现 + 验收 2–3 天，且要新增 3D 合成问题与参考前沿。
- **触发条件**（满足再启动）：①≥2 个真实用户场景需要 m=3；②ParEGO 在 3 目标上的 HV 覆盖 <70% 参考前沿；③预算 ≥80 且低噪（否则 qNEHVI-3D 更缺）。
- 备选：先做"3 目标用 ParEGO + 3D HV 报告"（0.5 天，性价比更高）。

## 5. P2：UI Pareto/成本视图（#5）

**为什么排后**：UI 依赖 ①固定 ref point 的口径（v0.9 已有 `Objective.ref_point`）、②成本 API（#1）、③概率 Pareto（`MULTI_OBJECTIVE_AND_NOISE.md` §5）。在 #1/#2 之前做 UI 会把错误口径"可视化"。
**内容**：Pareto 表（P_nd、E[HVC]、P(feasible)）、成本曲线（cost-aware best）、"观测值 vs 真值"标注、ref 来源提示。
**验收**：`render.py` 的 HV 与 `pareto.summarise` 同源；无 ref 时显示"不可比较"。

## 6. 其他（P2，可并行）

- `objective_std` 字典化（qNEHVI 异方差）；权重语义改加性/移除 `-1e9`；`_acquisition` 抽取（#4 的前置）；`calibration` 的 context schema version。

---

## 7. 建议的两周节奏

| 周 | 内容 | 验收 |
|---|---|---|
| W1 | #1（mfbo 成本/基线）+ #4（快 CI） | 137+ 测试全绿；`mf_suzuki` 超支 ≤2%；fast_ci <15 s |
| W2 | #2（records 覆盖）+ #3（456-runs + weights 入库） | LOPO top1≥0.55/top2≥0.80；weights 可回滚 |
| W3 | #5（UI）+ #7/#8（P2） | UI 口径与算法层一致 |
| 触发后 | #6（3D EHVI） | 触发条件满足再排期 |

---

## 8. 参考

R51/R52（折外评估/选择偏差）、R32（统计协议）、R42（HV/ref）、R41（BOCA/成本）、R12（qNEHVI）、`REVIEW_v09.md`、`CALIBRATION_RUN_RESULTS.md`、`EHVI_3D.md`。
