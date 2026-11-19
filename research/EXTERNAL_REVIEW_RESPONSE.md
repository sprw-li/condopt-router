# 对外部审查报告的回应（2026-11-11）

审查报告原文：外部独立审查报告（非公开）。
本文逐条回复：**哪些成立并已修**、**哪些不成立（附反证）**、**哪些进入待办**。
对应的可执行断言写在 `tests/test_algorithm_naming.py`、`tests/test_release.py`、
`scripts/decision_matrix.py`。

## P0「质量门禁不全绿」——结论不成立，但暴露了一个真问题

**报告证据**：`tests/test_decision_matrix.py::test_every_decision_rule_holds` 退出码 1，
失败的两条是「低置信度审计不得建议完全信任」「平台期判定必须忽略导入的先验」。

**复核**：这两条恰好是**唯一**需要可写临时目录的规则（它们会 `Study.create` 一个一次性战役），
而报告自己也记录了当时环境存在大量 `PermissionError`。用受控实验确认：

```
# 正常环境
py -3 -c "...evaluate()..."          -> total 26 failing 0
py -3 -m pytest tests/test_decision_matrix.py   -> 4 passed
# 把 tempfile.TemporaryDirectory 打成抛 PermissionError
-> failing: ['低置信度审计不得建议完全信任', '平台期判定必须忽略导入的先验']
```

即：**失败来自沙箱不可写，不是产品缺陷**。但报告指出的问题本身有价值 ——
门禁此前**无法区分"规则失败"与"规则在此环境跑不了"**。

**已修**：`scripts/decision_matrix.py` 现在把 `PermissionError/OSError` 记为
`blocked_by_environment`，文档中渲染为 **BLOCKED（环境：需要可写临时目录）**，
与 **FAIL** 区分开；两种环境下的行为都已验证。

## P1「qNEHVI 名称与机制不等价」——成立

**复核**：`src/condopt/algorithms/qnehvi.py:87-90` 把同一个 `objective_std` 复制给两个目标；
`:150` 用 `signed + noise * randn`（带噪观测）生成 baseline 前沿，而候选才走 GP 后验。
对照 BoTorch 的 qNEHVI（`X_baseline` 后验 + noisy-EHVI mixin），确实不等价。

**已修（文档层）**：`docs/ALGORITHM_CATALOG.md` §14 列出实现边界；README 该行改写为
"baseline 由**带噪观测**生成（尚未走模型后验），`objective_std` 为单值"。

**进入待办（机制层）**：baseline 改为在 `X_baseline` 上取模型后验；`Observation` 支持逐目标 std；
加"baseline 必须调用模型"的结构测试与固定小案例数值对照。

## P1「TPE 是独立逐维 KDE」——成立

**已修**：catalog §2 标题改名「独立逐维 Parzen 估计（非 tree / 非条件密度）」，§14 记录差距，
README bullet 同步；新增断言 `tests/test_algorithm_naming.py` 确保这些限定词不被删掉。

## P1「qEHVI 是单点候选贪心；CMA-ES 无 evolution paths」——成立

**已修（命名与文档）**：catalog §5 标题改为「单点候选 EHVI + 局部惩罚批量（非联合 q 点 EHVI）」；
skill 参考里 `cmaes` 标注"简化 rank-μ，无 evolution paths"；README bullet 加限定词。
**未做**：联合 q acquisition、evolution paths —— 列入 §14 待办，不再自称等价。

## P1「主 benchmark 未达自身规格」——成立

`research/BENCHMARK_SPEC.md` 要求共享初始设计、≥20 seeds、IQM/bootstrap CI、成对检验，
而 `benchmarks.py` 各优化器自行初始化、默认 3 seeds、只报 mean/std。

**已开始的处置**：新增 `scripts/noise_floor.py`（配对 t 检验 + 噪声底估计），
首个应用见 `research/NOISE_FLOOR.md`：描述符 vs one-hot 的 MOO 可行 HV 差异
**t=−0.704 < 2.776，未超出噪声底**。
**进入待办**：共享初始设计 + ≥20 seeds + IQM/区间/效应量的协议运行器；
在满足之前，对外只称"工程 smoke / 校准"，不称学术 benchmark。

## P2「版本与文档漂移」——成立

- `README.md` 写 v0.31、`pyproject.toml` 是 0.1.0 → README 改为读包版本（现 v0.1.0），
  新增断言 `test_readme_version_matches_the_package_metadata` 防再漂；
- `docs/ARCHITECTURE.md` 说 `tell` "清空 pending"，与实现的部分/幂等回填不符 → 已按实现改写。

## 报告没有覆盖、但我自己复核出的问题

- 场景矩阵的文档校验此前只查 key 出现过，**键冲突也会通过**（新增 T/U 场景时被我发现）→ 已改为行级匹配；
- `benchmark_momfbo.py` 假设问题必有多保真函数，遇到单保真 MOO 直接崩 → 已支持。

## 不同意的地方

- 报告称"18 个注册算法不等于 18 个文献等价实现"——**完全同意**，本轮正是把这一点写进文档与断言，
  而不是反驳；
- 报告说"不能宣称全绿"：在本仓库的可写环境下 `pytest` 全绿（486→487 tests）且 18 项门禁全过，
  这一点以本次提交的 `release_check` 输出为准；同时接受"干净环境应可复现"的要求，
  已把环境依赖显式化（BLOCKED 标记）。

## 第二轮修复（backlog A1/A2，Round 302–303）

### A1 `init` 存在性守卫 —— 已修

重跑 `init` 曾静默把 `algorithm` 改回默认值并保留数据、且不写决策记录。现在：

- 默认 `--if-exists=error` → **退出码 2**，并给出两条可行路径；
- `--if-exists=keep` → `Study.reset_campaign()`：保留数据、写 `campaign_reset` 决策、置暖启动宽限期；
- `--if-exists=reset` → 丢弃配置重建。
- 验证：`tests/test_init_guard.py`（4 passed）；`skill/SKILL.md` 补了对应小节。

### A2 momfbo 的 EI incumbent 权重 —— 已修

`_propose` 用 Dirichlet 权重拟合 GP，却用**均匀权重**算 incumbent（`imp = mu - best` 不一致）。
现在 `_best_scalar(weights)` 接收同一权重向量。

- 效果：mf_moo（budget 30，3 seeds）momfbo **666.22 → 669.88（+0.55%）**；
- mfparego 670.72 仍略优（差 0.12%，在噪声内）→ 默认**不变**，但登记表 tradeoff 改为
  “不落后而非显著更好”，并注明 momfbo 曾因此被低估；
- 验证：`tests/test_momfbo_weights.py`（结构断言：incumbent 必须用本轮 Dirichlet 权重）。

## 理事会（council）复核与两条“候选缺陷”的证伪（Round 305）

`council_review` 在本 runtime 的默认模型组合连续两次“全部作答失败（阶段失败）”，
改用可用模型（bohrium/deepseek-v4.1-flash-1017）后成功返回。它**没有拿到源码附件**，
因此把清单外缺陷标注为【推断】，并给出两条候选：

| 候选 | 我的复核（可复现） | 结论 |
|---|---|---|
| `qnehvi.tell` 在“缺一个目标的 tell”后 `_Y` 与 `_noise_std` 错位 | 构造 partial + complete 两条观测：`len(_Y)=2`、`len(_noise_std)=2`，随后 `ask()` 正常 | **证伪**（`tell` 里有对齐逻辑） |
| `condopt export` 的 `con.*` 列为空 | 带约束 MOO study 导出 CSV：表头含 `con.T_max`，行值 `0.4` = 实际约束值 | **证伪**（空列是 `y`，MOO 观测本就没有标量 y） |

它的排序建议（A6 → A5 → A10 → A12 → A4 → A7/A8 → A9 → A11）与我的判断一致，
其中“先堵门禁”被采纳为下一项工作。

**教训（与 EVIDENCE_POLICY §7 一致）**：没有代码访问权的评审只能给出**推断**；
推断必须先在本地复现再决定是否动手 —— 这次两条都不是缺陷。
