# NEXT_ROUND_CHOICES — qNEHVI 快 CI / records v2 / UI 三选一

> 维护者：algo-research ｜ v1（2026-11-05，Round 16）
> 输入：`QNEHVI_ACQUISITION_EXTRACT.md`、`RECORDS_V2_FIRST_STEP.md`、`UI_P2_APPLY_PLAN.md`、`REVIEW_v15.md`。
> 结论先行：**下一轮选 A（qNEHVI 快 CI）**；B 次之（records v2）；C 最后（UI）。理由见 §4。

---

## 1. 选项 A：qNEHVI `_acquisition` 抽取 + 快 CI

**内容**：按 `QNEHVI_ACQUISITION_EXTRACT.md` 逐行替换 qehvi/qnehvi；先快照后改；加 #2/#4/#5 测试（<5 s）。

| 维度 | 评估 |
|---|---|
| 收益 | ①给核心采集函数加"行为不变 + 收敛/批量/ref"三道契约（以后改 qNEHVI 不再靠人肉回归）；②解锁 CI 的 <5 s 快速门禁；③代码去重（qehvi/qnehvi 的贪心/模型拟合共用）；④风险最低（纯重构 + 快照证明） |
| 风险 | ①RNG 顺序/第三 GP 复用若处理不当会改变 ask 序列（已有快照对照 + 回滚点，可完全止损）；②浮点环境下快照可能非逐位一致（用 `allclose(1e-12)`） |
| 工作量 | **0.5–1 天**（diff 已有、测试已有） |
| 依赖 | 无（v0.10.5 的 149 tests 基线即可） |
| 验收 | `qehvi/qnehvi IDENTICAL`；#2/#4/#5 全过；150+ tests；PR <5 s |
| 失败回滚 | R0–R3（见应用计划），影响面仅 2 个算法文件 |

## 2. 选项 B：records v2（真噪声/MOO/约束/MF + context 推导）

**内容**：按 `RECORDS_V2_IMPLEMENTATION.md` + `RECORDS_V2_FIRST_STEP.md`，先做最小第一步（tags/noise_std + context），再扩 `run_task`/四类 metric，最后 488 runs。

| 维度 | 评估 |
|---|---|
| 收益 | ①校准终于覆盖噪声/约束/MOO/MF（router 规则最多的分支）；②LOPO 数字变成可信证据；③为正式 weights 与 UI 提供 true-HV 口径 |
| 风险 | ①改动面大（benchmarks + calibration + 488 runs），中间态多；②MOO/约束的指标实现需要仔细（signed ref、feasible 判定）；③488 runs 需要 8–12 min + 生成数据管理 |
| 工作量 | **2–3 天**（含 488 runs 与报告） |
| 依赖 | A 不需要；但 B 与 A 独立（B 不动 qehvi/qnehvi）。**建议先 A 后 B**：A 先把"改算法不被发现"的风险降下来（B 之后不会碰算法层，但仍受益于 CI 纪律） |
| 验收 | 每类 ≥1 条 records；LOPO top1_strict 0.50–0.65；noise 组 regret∈[0,1] |
| 失败回滚 | 按文件回滚（benchmarks/calibration 各一个点） |

## 3. 选项 C：UI Pareto/成本视图

**内容**：按 `UI_P2_APPLY_PLAN.md` 在 render.py 加 3 个 section + 6 条空状态 + 6 条验收。

| 维度 | 评估 |
|---|---|
| 收益 | ①用户可见的 Pareto 表/成本曲线/口径提示；②把"ref 来源/observed vs true/weights 版本"暴露出来，减少误读 |
| 风险 | ①**口径依赖**：在 B 之前上线，UI 展示的仍是"无噪/单目标"读数与 observed HV，容易把错误口径"可视化"；②render.py 改动可能影响现有报告测试 |
| 工作量 | **2–3 天**（含前端渲染与验收） |
| 依赖 | 强依赖 B（true-HV/约束/MF metric）；成本区依赖 v0.10.5（已具备） |
| 验收 | 8 条（UI_P2 规格）；不重算 HV；无 ref 时 banner |
| 失败回滚 | render.py 单文件回滚 |

---

## 4. 建议与理由

**推荐顺序：A → B → C。**

1. **A 是"低成本、高确定性"的一步**：diff 已终稿、测试已写好、回滚明确；一天内可完成并让 CI 覆盖 qNEHVI 的核心风险。当前 qnehvi 的任何改动（比如未来加 3D、改 n_base）都缺少回归保护——A 把这个洞补上。
2. **B 是"高价值、中风险"**：它是校准可信度的前提，也是 UI 的前置；但工作量最大、中间态最多，适合排在 A 之后（有 CI 保护、且不急于冒风险）。
3. **C 最后**：UI 本身不难，但**在 B 之前做等于把"未校准/未真值"的读数漂亮地展示出来**，会把口径问题固化；等 B 的 true-HV/约束/MF metric 落定后再接线。

**如果只能选一个**：选 **A**（0.5–1 天、风险最低、立即提升 CI 质量）；**如果目标是尽快让"校准"可验收**：选 **B**，但建议先花半天做 `RECORDS_V2_FIRST_STEP`（tags/noise_std + context），再扩 metric。

**两周建议**：
- 第 1 周：A（0.5–1 天）→ B 第一步（0.5 天）→ B 的 `run_task`/四类 metric（1–2 天）；
- 第 2 周：488 runs + weights 入库（1.5 天）→ C（2–3 天，若时间够）。

---

## 5. 参考

R12（qNEHVI）、R21（qEHVI）、R42（HV/真值）、R32（统计）、R41（成本/多保真）、R54（评估协议）、R57（Garnett）、R58（Frazier）、`REVIEW_v15.md`、`QNEHVI_ACQUISITION_EXTRACT.md`、`RECORDS_V2_FIRST_STEP.md`、`UI_P2_APPLY_PLAN.md`。
