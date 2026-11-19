# 条件优化 router（condopt）交付说明

**一句话**：把实验条件优化从“随手挑一个算法”变成**有理由的算法选择 + 可审计的闭环执行 + 可交付的证据**。
面向 AI 全自动实验室的前置技术：agent 用 `condopt-router` skill 决策，`condopt` 负责算法、状态与证据。

## 规模（自动统计）

| 项 | 数量 |
|---|---|
| 优化算法 | **18** |
| 发布门禁 | **19** |
| 单测 | **585** |
| 路由场景断言 | **22** |
| 决策检查（规则 + 抑制断言） | **47**（26 规则 + 21 抑制） |
| 可重跑证据声明 | **25** |
| 负结果条目 | **14** |

## 能力

| 层 | 提供什么 |
|---|---|
| **算法库** | 18 个 ask/tell 优化器（GP-BO/TPE/ParEGO/qEHVI/qNEHVI/cqEHVI/多保真家族/CMA-ES/空间填充），统一 registry 与参数 schema |
| **路由器** | 按问题特征（维度/目标数/约束/多保真/噪声/预算/地貌）选算法与参数，给出理由、备选与告警 |
| **决策层** | 开工五段建议（算法/停止/可行率/先验/交付口径）、交付槽位、确认复孔、停滞调参 |
| **执行器适配器** | 一个 measure() 方法接入真实仪器：失败补测、重试上限、并行批、超时、失败归因、进度曲线 |
| **多目标/多保真/约束** | Pareto 前沿、可交付（可行）HV、保真度通道、成本硬上限 |
| **迁移** | transfer_from（前沿选择 + 预算记账 + 元数据清洗）、transfer-audit（含置信度分级） |
| **交付自查** | report/digest 里逐条打勾：可行的最终保真观测、前沿+进展、推荐条件已复孔确认 |
| **skill** | condopt-router：问卷 → 生成配置 → 选算法 → 迭代 → 收尾，文档流程有可执行断言 |
| **证据纪律** | DEFAULTS 每条带出处、登记表（数据集/方向/tradeoff）、25 条可重跑声明、14 条负结果、测量卫生检查 |

## 怎么用（六条命令）

```powershell
# 1. 问卷 → 配置
py -3 scripts/run_condopt.py prepare --answers configs/answers.example.json --out work
# 2. 建战役
py -3 scripts/run_condopt.py init --run work/run --space work/space.json --objective work/objective.json --context work/context.json --budget 24 --batch 4
# 3. 取条件
py -3 scripts/run_condopt.py ask --run work/run --n 4
# 4. 回填实测
py -3 scripts/run_condopt.py tell --run work/run --results results.json
# 5. 看决策/收尾
py -3 scripts/run_condopt.py step --run work/run --json  # 或 digest / report
# 发布自检
py -3 scripts/release_check.py  # 全部门禁
```

## 关键实测结论（都有归档与门禁）

- **真实数据暖启动**：ART Table 2 六点做源、代理 oracle 做目标：warm 12/12 命中真实最优，cold 0/12
- **多保真 MOO 默认**：mfparego 在 3 个数据集上不落后 momfbo（674.5/641.0/27318 vs 669.5/636.3/26998）
- **高维规则**：cmaes 在 8–14 维是“高维 GP 不可靠”的兜底；窄谷型 6 维起就该切，多峰型未必
- **极小预算**：低于建模下限时屏蔽模型类；约束/MOO 保留代理 + 告警（纯 sobol 在约束下仅 1/4 有可行前沿）
- **预算优先**：40–120 预算下更优（92.9/98.8/100.6/102.2 vs 71.8/65.6/76.9/65.6），交付自查可通过
- **交付证据**：四种问题形态 12/12 通过自查；为交付最多保留 1 次评估
- **迁移可信度**：机制已修（mfbo 曾忽略 objective_std）；但结果对 reliability 不敏感，分级只作策略性约定

这些数字的出处：`docs/DEFAULTS.md`（逐条证据）、`docs/DECISIONS.md`（决策矩阵）、`research/README.md`（证据库索引）、`research/EVIDENCE_POLICY.md`（证据规则）。

## 已知限制（诚实清单）

- 真实测量只有 ART Table 2 六点，其余为合成基准或代理 oracle，结论外推需谨慎
- 多保真暖启动收益弱（+0.35%~+1%），不保证正收益
- 约束场景保证“有可行的最终保真证据”，但可行区内的最优性仍受预算/代理精度限制
- skill 的安装依赖共享目录 <agent-library>，可能被其他会话并发改动（安装器已自带逻辑）

## 发布自检

```powershell
py -3 scripts/release_check.py    # 全部门禁（含证据重跑、场景/决策断言、端到端演练、skill 流程）
```

本文件由 `scripts/build_release_notes.py` 生成（`--check` 可校验），计数直接取自代码与证据库。
