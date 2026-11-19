# 外部同类项目扫描（2026-11-11，5 路子代理 + 78 个可核实项目）

目的：**借鉴而不是重复造轮子**。每行给出「来源 → 我们对应哪里 → 具体借鉴什么」。
URL 均由子代理检索/抓取获得；抓不到证据的项目已排除。对应本地文件均为只读核对的路径。

## 1. ask/tell 框架（Ax / Optuna / SMAC3 / Emukit / Trieste / Nevergrad / skopt）

| 借鉴点 | 来源 | 我们的对应 | 具体做法 |
|---|---|---|---|
| 试验状态机（RUNNING/COMPLETE/FAIL/PRUNED），失败态自动重采样而不是崩 | Optuna `study/_tell.py` <https://github.com/optuna/optuna> | `Study.tell` 只有成功路径 | 在 `observation.meta` 加 `status`，`Study` 统计失败率 |
| `pending` 作为一等公民（已推荐未回结果） | Ax `pending_observations` / BoTorch `X_pending` | `Study.pending` 已有一半 | 采集函数直接接收 pending，替代自造 fantasy |
| 算法声明能力标记（recast / one_shot / no_parallelization） | Nevergrad `optimization/base.py` | `AlgorithmSpec.capabilities` | 已类似，可补 `no_parallelization` |
| 存储后端可插拔 + `schema_version` | Optuna storage / SMAC3 RunHistory | `dataset.jsonl` 无版本字段 | 加 `schema_version` 与 per-trial 状态 |
| ask/tell 之外提供 `suggest()` 预置点 | Nevergrad / Ax `attach_trial` | 无 | 允许人工插入条件并计入审计 |

## 2. 算法/求解器自动选择（AS 竞赛 / AutoFolio / auto-sklearn / Optuna AutoSampler）

| 借鉴点 | 来源 | 我们的对应 | 具体做法 |
|---|---|---|---|
| 选择质量用 **SBS/VBS 双基线**与 normalized closed gap，而不是绝对分数 | AS 竞赛 <https://arxiv.org/abs/1805.01214> | `verify_evidence` 只验方向 | 每条默认值同时报 SBS/VBS 与 closed gap |
| portfolio 要显式优化「互补性」而非各自最优 | PoSH <https://arxiv.org/abs/2007.04074> | 路由评分是独立加权 | 引入互补性项或贪心构造 |
| 在线切换用**已完成预算阈值**并 warm-start 交接 | Optuna `AutoSampler`（切 CMA-ES 时传前 250 个 trial） | `route_round` 已按停滞切换 | 切换时保留历史（`transfer_from` 已具备） |
| 选择器本身要防过拟合：外层留出折 | AutoFolio `--outer-cv` | LOPO 折外校准已做一半 | 把折外结果写进默认值登记表 |
| 选择器要有 backup/fallback 路径 | auto-sklearn 2.0 Backup strategy | `capability_relaxed` 已类似 | 明确 worst-case 不差于基线 |

## 3. 自驱动实验室编排（AlabOS / ChemOS / Atlas / Gryffin / Bluesky / NIMO）

| 借鉴点 | 来源 | 我们的对应 | 具体做法 |
|---|---|---|---|
| 资源申请用上下文管理器，退出必释放 | AlabOS `request_resources` | `Executor` 无资源生命周期 | 适配器层加 `acquire()/release()` 钩子 |
| 失败二分：可恢复（重试/通知）vs 不可恢复（整单失败但继续跑） | AlabOS | `classify_failure` 已有 | 增加 `recoverable` 语义与失败注入测试 |
| 检查点语义显式化（`checkpoint` / `clear_checkpoint`） | Bluesky | 无 | 在 run 目录写「可安全恢复点」 |
| 人在环做成流程原语（阻塞式确认） | AlabOS notification / IvoryOS pause | `confirm` 已是流程原语 | 增加「需人工放行」的 action 类型 |
| 约束分「已知/未知」并给出加权/硬/插值三策略 | Atlas + Gryffin | `constrained` 单一策略 | 文档化三策略，按违规代价选择 |

## 4. 多目标 + 多保真基准协议（HPO-B / YAHPO / DGEMO / COCO bbob-biobj / Bayesmark）

| 借鉴点 | 来源 | 我们的对应 | 具体做法 |
|---|---|---|---|
| 固定具名种子集合 + 发布 per-seed 原始产物 | HPO-B `test0–test4`、DGEMO `--n-seed 10` | `runs/*.csv` 已发布 | 明确种子集合与 version 字段 |
| 强制报方差/噪声底，Δ 落在带内不写成提升 | DGEMO mean ± SD | `noise_floor.py` 已做 | 把它设为「声称差异」的前置 |
| 超体积参考点统一约定（归一化到 nadir） | COCO bbob-biobj | `ref_point` 由问题给出 | 在文档里统一口径 |
| 预算口径用「全保真等价成本」 | BoTorch MF 教程 | `cost_budget` 已是等价成本 | 报数时同时给次数与等价成本 |
| 跨问题汇总用 critical-difference 图 | HPO-B plotter | 无 | 后续多问题对比时采用 |
| 命名诚实：官方无 qMFES，MFParEGO 无规范实现 | BoTorch / 调研结论 | 已在 catalog §14 标注 | 保持标注 |

## 5. LLM/agent 与 skill 工程（MCP 生态 / LabMCP / bo-mcp / Agent Skills）

| 借鉴点 | 来源 | 我们的对应 | 具体做法 |
|---|---|---|---|
| 数值决策外置 + 「为什么推荐」可追问 | Optuna MCP `ask/tell`、BO-MCP | `route` 的 rationale 已具备 | 把 rationale 通过 `--json` 暴露给 agent |
| 护栏放工具侧（只读模式、参数上限、危险动作需确认） | LabMCP `--read-only` / `--limit` | `override` 需 reason | 增加只读/限值开关 |
| 全量工具调用与返回落审计日志 | LabMCP `--audit-log` | `decisions.jsonl` 只有 6 字段 | 加 `actor`/`seq`/`config_hash` |
| skill 三层渐进披露 + 触发评测 | Agent Skills 约定 | SKILL.md 单文件 + smoke | 拆 references 并按需加载（已部分） |
| skill 要有 eval 与 benchmark，不只文档 | Agent Skills evals | `verify_skill_workflow.py` 已有 | 增加 with_skill/baseline 对比 |

## 结论（写给自己）

外部同类项目的共同点是：**状态机与生命周期语义**、**基准协议纪律**、**审计与护栏放在工具侧**。
我们已经在「透明路由 + 证据治理」上有真实差异，但在状态机、schema 版本、审计字段、
基准统计纪律四块明显落后 —— 这四块同时也是外部审查点名的部分，优先级高于新增算法。
