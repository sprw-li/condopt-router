# 项目状态快照（Status）

> 由 `scripts/build_status.py` 生成；计数直接取自代码与证据库（`--check` 可校验）。

## 交付形态

| 维度 | 现状 |
|---|---|
| 优化算法 | registry 注册 **18** 个（ask/tell 统一接口） |
| 路由 | 能力过滤 + 评分规则 + 可选校准权重 + 停滞重调参 |
| 决策层 | Spec/Goal、达标可行性、板效应校正、推荐判定 |
| 执行器 | `Executor.measure()` 适配器：失败补测、重试上限、并行批、超时、审计日志 |
| 多目标 | Pareto 前沿、knee 权重、可交付（可行）HV 与逐批进展曲线 |
| 多保真 | 保真作为特征、成本硬上限、可行全保真证据诊断 |
| 迁移 | `transfer_from`（前沿选择 + 预算记账）、`transfer-audit`（含置信度） |
| 服务接口 | 30 个 CLI 命令 + JSON 输出，便于 agent 驱动 |
| 交付口径 | `prepare.deliverable` 约定交付指标与必带证据；`delivery_checklist()` 在报告中打勾自查 |

## 质量门禁（`release_check` 19 项，由 release_check.py 自动读取）

| 门禁 | 内容 |
|---|---|
| `readme_test_count` | README 测试计数与实况一致 |
| `pytest` | 全量单测 |
| `audit_algorithms` | 每个算法可运行 |
| `audit_routing` | 每个上下文的路由候选满足能力要求 |
| `docs_links` | 文档链接无断链 |
| `e2e_drill` | 端到端演练（含交付自查） |
| `doc_duplication` | 规则句无重复 |
| `commands_documented` | CLI 命令都有文档 |
| `api_docs` | 核心公开 API 有文档 |
| `defaults_sync` | DEFAULTS 与代码默认值一致 |
| `scenario_matrix` | 路由场景断言 |
| `evidence_claims` | 关键数字重跑后仍成立 |
| `status_snapshot` | 本文件与代码一致 |
| `decision_matrix` | 决策规则 + 抑制断言成立且文档同步 |
| `evidence_policy` | 每条默认值都引用了存在的证据文件 |
| `release_notes` | RELEASE_NOTES.md 与代码/证据库一致 |
| `golden_numbers` | 归档数字与 GOLDEN_NUMBERS 登记一致（含“改坏必须变红”自检） |
| `eval_router` | skill 意图路由评测 |
| `skill_install` | skill 安装与静态检查 |
## 证据规模

| 类型 | 数量 |
|---|---|
| 可重跑证据声明 | **25** |
| 可执行场景断言 | **22** |
| 决策层规则（逐条断言） | **47** |
| 负结果（含已修复与仍开放） | **14** |

## 交付口径与自查（端到端）

| 阶段 | 机制 | 说明 |
|---|---|---|
| 开工前 | `condopt prepare` 的 `deliverable` | 交付指标（best_value / feasible_hypervolume / …）+ 必须包含的证据 |
| 过程中 | `next_action().diagnostics.delivery_missing` | 缺证据时给出 `complete_delivery_evidence` 建议 |
| 收尾 | `report()` / `digest` 的“交付自查” | 逐条打勾：可行的最终保真观测、前沿与进展、复孔确认 |

约束场景的交付指标一律取**可行**口径（raw HV 会高估：143.0 vs 80.9）。

证据纪律见 `research/EVIDENCE_POLICY.md`：默认值改动需 ≥2 数据集、负结果保留、生成物幂等，以及“先检验测量本身”（本项目已三次因测量/脚本缺陷撤回负面结论）。

## 已知风险与待办

| 主题 | 现状 / 待办 |
|---|---|
| 交付证据 vs 质量 | 约束 + 多保真场景已保证“有可行的全保真证据”，但可行区内的**最优性**仍受预算与代理精度限制（示例 best=34.2，理论最优约 105） |
| 小样本迁移判断 | `transfer-audit` 在可比条件 <5 时没有排序能力（真实数据 20 个切分方向反转），现以 `low_confidence` 降级；需要更鲁棒的小样本估计 |
| 多保真暖启动 | 收益极弱（k≈8 含低保真时 +0.35%），小 k 中性；真实实验室的保真偏差结构未验证 |
| 真实数据规模 | 真实证据只有 ART Table 2 六点；其余基准均为合成，结论外推需谨慎 |
| 并行执行器 | `max_workers>1` 已可用，但共享状态（如失败注入计数器）需要用户自己同步——文档已提示，未见通用守护 |
| 适配器并发语义 | 超时后线程仍在后台运行（`shutdown(wait=False)`），真实仪器需支持“放弃结果”的语义 |
| 迁移的成本口径 | 已实测（budget 24）：修复三个缺陷后 warm 臂交付自查 4/4，但 best 仍略低（33.95 vs 39.87），多保真暖启动暂无正收益；见 `research/TRANSFER_COST.md` |
| 手写循环的进展记录 | 已提供 `Study.record_progress()`，但手写循环若不调用，交付自查会缺“逐批进展”这条 |
| 外部 agent-library 漂移 | 共享目录会被并发会话改写（实测 skill 目录整体消失）；安装器已自带逻辑，路由评测对与本项目无关的失败记为外部漂移并保留明细，见 `docs/ARCHITECTURE.md` |

## 数据诚实性

- 真实测量仅 ART Table 2 六条（Br 条件，yield 71–99）；
- 其余基准、描述符表、代理 oracle 均为合成或近似，文档中逐处标注；
- 所有默认值都能在 `docs/DEFAULTS.md` 追溯到基准数字。
