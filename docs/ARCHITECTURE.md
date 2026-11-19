# 架构（ARCHITECTURE）

> 本文说明 condopt 的分层、核心契约、数据模型与扩展点；程序架构的文献调研见
> [`research/ARCHITECTURE_PATTERNS.md`](../research/ARCHITECTURE_PATTERNS.md)，
> 与本文的差异和采纳情况见 §7。

## 外部 agent-library 漂移

skill 的安装与路由评测依赖共享目录 `<agent-library>`，它会被其他会话并发修改
（实测：`agent-library-skill-create` 一度整体消失，`create_skill.py` 也不在原本位置）。

因此本项目：

- `scripts/install_skill.py` **自带安装逻辑**（写 SKILL.md、复制 references、补
  manifest/intents），仅在外部工具存在时才顺带调用，不再依赖其路径；
- 路由评测若失败但**没有任何 condopt 相关用例**失败，门禁按“外部漂移”记录并放行，
  输出里保留失败明细，避免把别人的改动当成自己的回归。

## 状态持久化边界（state_dict 只放计数器）

算法恢复时**由 study 数据集（`prior`）重建数据**，因此 `state_dict()` 里只允许放：
算法名、待初始化队列、CMA-ES 的均值/协方差/步长、网格游标、计数器和参考点等
**与观测条数无关**的字段。

放进去的观测级数据（X/Y/C/noise/units、约束矩阵、保真数组）会被更早写入的 state 遮蔽，
造成形状不一致甚至崩溃 —— 这一类缺陷已实测三次（`mfbo`、`momfbo`、`ehvi_nd`，
见 `research/TRANSFER_COST.md`）。`tests/test_transfer.py` 中的架构守卫测试会枚举所有
算法，禁止出现"行数 = 观测条数"的字段。

## 1. 分层

```
┌──────────────────────────────────────────────────────────────────┐
│ Agent / Skill 层（skill/SKILL.md）                                │
│  问参数（带建议范围）→ 调 route → 出建议表 → 收结果 → 迭代        │
└───────────────────────────┬──────────────────────────────────────┘
                            │ CLI / JSON（--json 全量可机读）
┌───────────────────────────▼──────────────────────────────────────┐
│ 编排层 Study（study.py）                                          │
│  run 目录状态机：ask → pending → tell → rounds → report            │
│  auto_reroute（停滞检测）、override 审计、预算/去重                 │
└───────┬───────────────────────────────────────┬──────────────────┘
        │                                       │
┌───────▼──────────────┐              ┌─────────▼──────────────────┐
│ 路由层 router.py      │              │ 算法层 algorithms/          │
│ 打分规则 + 参数策略   │──算法+参数──►│ random/sobol/lhs/grid/bayes │
│ route / route_round   │              │ ask/tell + AlgorithmSpec    │
└───────┬──────────────┘              └─────────┬──────────────────┘
        │                                       │
┌───────▼───────────────────────────────────────▼──────────────────┐
│ 数据与空间层 space.py / dataset.py / base.py / registry.py         │
│ Variable/SearchSpace（编码/采样/校验）、Observation/Dataset（JSONL）│
│ Objective、ParamSpec/AlgorithmSpec、注册表                          │
└──────────────────────────────────────────────────────────────────┘
```

**一句话**：算法只看见 `[0,1]^d` 与 `(X, y)` 历史；实验语义（单位、分类、约束、成本）都在空间/数据层；
AI 只操作 CLI/JSON，不碰算法代码。

## 2. 核心契约

### 2.1 Optimizer（算法层）

```python
class Optimizer:
    space: SearchSpace
    objective: Objective
    params: dict            # 经过 AlgorithmSpec.validate 校验
    rng: np.random.Generator  # 一切随机性的唯一来源，禁止模块级 np.random
    t: int                  # 已 tell 的观测数

    def ask(self, n: int = 1) -> list[dict]: ...   # 返回合法配置（raw values）
    def tell(self, observations: Sequence[Observation]) -> None: ...
    def state_dict(self) -> dict: ...              # 只存计数器，不存数据
    def load_state_dict(self, state: Mapping) -> None: ...
```

要点：

- **ask 返回 raw 配置**（不是编码）；编码/解码/校验由 `SearchSpace` 负责，算法不产生非法点。
- **tell 接收增量观测**；算法内部只存特征矩阵与得分，不存业务字段。
- **state 只存计数器**（Sobol 已生成数、Grid 索引）；GP 数据每次从 `Study` 数据集重建，
  避免“状态文件 vs 数据集”不一致（这是 v0.1 踩过的坑）。
- **随机性来自注入的 seed/rng**，保证同一 seed 可复现。

### 2.2 Study（编排层）

```
run/
  problem.json      space + objective + seed + budget + meta      ← 不可变
  config.json       algorithm + params + context + batch/patience ← override 时可写
  dataset.jsonl     观测（append-only）                            ← 唯一数据真源
  pending.json      当前待测建议（id/batch/algorithm/x）
  rounds.json       每轮 n/best/algorithm（停滞检测用）
  decisions.jsonl   initial_route/auto_reroute/override + reason
  opt_state.json    序列算法计数器
  suggestions.csv   最新建议表（人/机器人直接读）
```

- `ask`：若存在未 `tell` 的 pending → 抛 `PendingError`（防错配实验结果）。
- `tell`：按 `id` 匹配 pending（或直接给 `x`），校验后追加 dataset，记一轮；
  **支持部分回填与重复回填**：只有出现在 `results` 里的建议会离开 pending，已记录的 `id` 会被跳过（重传幂等）——不是“清空 pending”。
  （外部审查报告曾据本节文字判断实现与文档不符；实现见 `src/condopt/study.py::tell`。）
- `import`：导入历史数据当先验（计入预算，不记轮次）。
- `override`：必须给 `reason`，写 `decisions.jsonl`，否则报错。
- `auto_route`：`ask` 前用 `rounds` 的累计最优序列做停滞检测；连续 `patience` 轮无改进就插一批全局探索。

### 2.3 Router（决策层）

输入 `OptimizationContext`（维度/类型/预算/批量/噪声/光滑度/先验/领域…），
输出 `Recommendation`（algorithm / params / confidence / rationale / warnings / alternatives /
candidate_scores / design / plan）。所有参数都经 `AlgorithmSpec.validate`，router 不可能推荐
不存在的算法或非法参数。

## 3. 数据模型

| 对象 | 关键字段 | 不变式 |
|---|---|---|
| `Variable` | name, kind, low/high/choices, unit | 类型合法、范围有限、log>0、integer 为整数、分类 ≥2 |
| `SearchSpace` | variables（有序、唯一名） | 可编码/解码、可校验、可采样 |
| `Observation` | x, y, objectives, constraints, fidelity, cost, source | 至少一个目标值；x 必须过空间校验 |
| `Dataset` | observations | append-only；JSONL 一行一条 |
| `Objective` | name, direction, objectives, constraints | direction ∈ {max,min} |
| `ParamSpec` | name, type, default, choices, low/high | 可 `coerce`，越界报错 |
| `AlgorithmSpec` | name, family, summary, when, params, capabilities, references | 注册时校验 |
| `OptimizationContext` | 见 router.py | 枚举取值校验、`n_dims>0`、`remaining ≤ total` |
| `Recommendation` | 见上 | `validate()` 必须通过注册表 |

## 4. 可复现与审计

- 复现：`problem.json + dataset.jsonl + seed` 可重放整个优化历史；`suggestions.csv` 可对账。
- 审计：每次路由/覆写/重路由都写 `decisions.jsonl`（含 reason 与 from/to 配置）。
- 回放：给定 `config.json + rounds` 可重建任意一轮的算法选择（确定性规则、无隐藏随机）。
- 数据诚实：`Observation.source` 记录来源（manual/robot/simulation/benchmark）；仿真不冒充实验。

## 5. 扩展点

| 想加什么 | 改哪里 | 不会破坏什么 |
|---|---|---|
| 新算法 | `algorithms/` 新模块 + `@register` + `router` 规则/参数策略 + 测试 | 旧算法、Study、CLI、skill 调用方式 |
| 新空间变量类型 | `Variable` kind + `encode/decode/features` + 测试 | 现有类型与算法接口 |
| 新指标/报表 | `Study.summary/report` 或新增 `render.py` | 数据格式（append-only） |
| 新 UI | 只读 `run/` 目录，不改状态 | CLI 契约 |
| 接入机器人调度 | 读 `pending.json`、写 `results.json` 后调 `tell` | 算法层完全不用改 |

## 6. 测试策略

| 层 | 测什么 | 文件 |
|---|---|---|
| 空间 | 编码/解码 roundtrip、校验报错、采样确定性、网格覆盖 | `tests/test_space.py` |
| 算法 | 每个算法 ask/tell 契约、BO/TPE 收敛、批量不重复、Sobol 序列续跑 | `tests/test_algorithms.py` |
| 多目标 | ParEGO/qEHVI/qNEHVI 前沿与 HV、非支配排序、Study Pareto 报告 | `tests/test_multiobjective.py`、`tests/test_pareto.py`、`tests/test_qehvi.py`、`tests/test_qnehvi.py` |
| 约束多目标 | cqEHVI 两阶段、可行性概率、可行前沿 HV vs 无约束基线、router/Study 集成 | `tests/test_cqehvi.py` |
| 路由 | 典型上下文的 top-1、置信度、参数合法、停滞重路由、feature_dim 惩罚 | `tests/test_router.py` |
| 编排 | 生命周期、重载、预算、override 留痕、部分/幂等 tell、导入先验、CLI 端到端 | `tests/test_study.py` |
| 孔板/批次 | ask 打 plate/well/batch、续测同板避让已用孔位、tell 覆写、板间统计与批次效应警告、confirm 板号 | `tests/test_plates.py` |
| 自动循环 | next_action 的 tell/confirm/ask/stop 状态机、收尾前确认、预算边界、批次效应透传、CLI step | `tests/test_loop.py` |
| 参数准备 | answers.json 别名/校验、生成三件套、CLI prepare | `tests/test_prepare.py` |
| 看板 | 离线 HTML/PNG、无外部依赖、空 study 降级 | `tests/test_render.py` |
| 基准 | 最优值自检、runner/summary | `tests/test_benchmarks.py` |

当前：`py -3 -m pytest -q` → 276 passed。

## 7. 与调研架构的差异（诚实清单）

采纳：

- ask/tell + pending + 去重（Optuna/Nevergrad 范式 [R9]）；
- 空间层唯一真理源、算法只见张量；
- `AlgorithmSpec` 带参数 schema + capabilities + references；
- campaign 目录持久化、可回放（BayBE 的 campaign JSON 思路 [R17]）；
- 规则 router 先于 LLM（LLM 只在决策点，可审计）。

尚未实现 / 已知缺口（v0.2 目标）：

1. **ask 可少于 n**：当前要求返回 n；空间耗尽/预算不足时 Study 直接报错。
2. **pending 惩罚**：异步场景下，未回结果的点应进入采集惩罚/去重半径；v0.1 用 `PendingError` 阻止并发 ask
   （`tell` 已支持部分回填：只移除已回填的建议，未回填的继续 pending）。
3. **异步/多执行器**：当前一次一批、串行回填；机器人调度需要更细粒度 trial 状态机。
4. **Capabilities 能力过滤**：router 已按 `multi_objective/constraints/multi_fidelity/batch/categorical`
   做“先过滤后打分”。`parego`/`qehvi`/`qnehvi`/`cqehvi` 声明 `multi_objective`，`constrained`/`cqehvi`/
   `parego`/`mfbo`/`momfbo` 声明 `constraints`，`mfbo`/`momfbo` 声明 `multi_fidelity`，
   `momfbo` 同时声明 `multi_objective` + `multi_fidelity`；仅剩极少数无解组合会退回子集打分并 warning。
5. **约束 / 多保真 / 2 目标 qEHVI / 约束多目标 qEHVI / 多目标+多保真** 已实现（`constrained`、`mfbo`、
   `qehvi`/`qnehvi`、`cqehvi`、`momfbo`）；剩余的 3+ 目标 EHVI、暖启动/迁移见
   `docs/ALGORITHM_CATALOG.md` §11 路线图。
6. **REST/UI 服务**：见 `docs/ui/`，推荐静态看板 + 可选只读服务。

v0.3 新增：

- **多目标**：`Objective` 支持逐目标方向（`directions`），`algorithms/parego.py` 用增广 Tchebycheff +
  每点独立 Dirichlet 权重复用 GP/EI；`pareto.py` 提供非支配排序与 2D HV；`Study.report()` 输出 Pareto 前沿。
  归一化窗口在 `max(3, 2m)` 条观测后冻结，并可用 `objective_bounds` 显式固定；`Objective.ref_point` 已支持显式 HV 参考点。
- **约束多目标（v0.13）**：`algorithms/cqehvi.py` 把 qEHVI 的精确 2D 分区与 `constrained` 的两阶段可行性
  合成 `cEHVI(x) = E[DHV(f(x))·1{g(x)≤0}]`；router 在「2 目标 + 硬约束」上下文选择它，
  能力过滤要求算法同时声明 `multi_objective` + `constraints`。
- **参数准备**：`prepare.py` 把问卷答案变成三件套 JSON 并调用 router；CLI `prepare`。
- **看板**：`render.py` + CLI `render`；只读 run 目录，离线单文件 HTML + PNG。

v0.30+ 新增：

- **多目标 + 多保真**：`algorithms/momfbo.py`（随机标量化 + fidelity 维 GP + `EI/cost^α` + P(feasible)），
  router 在「多保真 + 多目标（±约束）」选择它；`parego` 亦支持硬约束（候选按 P(feasible) 择优），
  「3+ 目标 + 约束」不再退回主目标。
- **实验室交接**：`template`（生成回填 CSV）/ `tell --results`（CSV/TSV/分号自动识别）/
  `export` / `pareto` / `digest --format html` / `decisions` 审计查看。
- **诊断**：`fidelity_summary` 配对偏差 + `review_fidelity_channel` 建议；成本预算耗尽时
  `next_action` 直接 stop，多保真预算不足最便宜档位时 `ask` 返回空建议而非抛错。

已修复（来自 `research/REVIEW_v01.md` 的 P0/P1 项）：

- TS 后验采样改用 `predict(return_cov=True)`，不再退化为“后验均值 + 噪声”；
- benchmark 最优常数与 Hartmann3 公式修正，并有自检测试；
- router 用 `feature_dim`（数值 + one-hot）而不是变量个数判断 GP 可用性，初始设计点数同步；
- `override` 后 `auto_route` 不再覆盖用户选择，直到显式 `condopt auto`；
- `tell` 支持部分回填与幂等重放，不再静默清空整批 pending；
- log-EI 用渐近公式，远端候选不再全部下溢到同一分数；
- 每轮 ask 推进 RNG 流，避免重复随机序列。

## 公开 API 覆盖

`Study.current_context()` 返回一份**刷新过的上下文**（`n_observations` / `budget_remaining` /
`budget_total` / `batch_size` 按当前进度更新），重路由与 `route_round` 都基于它决策；
调用方在自定义循环里也可以用同一份上下文调用 router。

`scripts/check_api_docs.py` 会枚举 `Study` 的全部公开方法，检查它们是否在
README/docs/research/skill 中被提及；核心 API 列表缺失时非零退出。

## 证据库导航

`research/README.md` 是全部研究笔记的索引（核心结论 + 规格/实现/评审/调研分组），
由 `scripts/build_research_index.py` 生成；`tests/test_research_index.py` 保证不漏登记、不陈旧。

## 默认值治理

`docs/DEFAULTS.md` 是默认值的唯一总表：每行给出场景、默认值、实测数字与证据文件，
`tests/test_defaults_doc.py` 会校验关键行存在且引用的证据文件真实存在。

## 质量闸门

`scripts/release_check.py` 聚合全部回归门禁并在 README 计数漂移时报错：
pytest、`eval_router`、`audit_algorithms`、`audit_routing`、skill 安装检查、
README 单测数 vs `pytest --collect-only` 实际数量。
`tests/test_release.py` 固定“README 计数 == 实际收集数”，防止文档漂移。


## 多目标指标口径

`Study.pareto_summary()` 给 raw 前沿；`Study.feasible_pareto_summary()` 给可交付前沿
（同一保真口径 + 约束过滤，避免低保真偏差抬高可行 HV）。

