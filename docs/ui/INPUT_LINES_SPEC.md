# 输入线规格：「大量参数」硬化为 41 条固定输入线

- 版本：v0.1（2026-11-05）｜任务：shared task-2｜上游：`FEASIBILITY.md`
- 目标：把 condopt 的算法参数、实验参数、AI 权限参数**收敛为有限、有界、可校验、有默认值、有档位**的「输入线」，供表单/CLI/YAML/skill router 共用同一份 truth。
- 已实现事实：`AlgorithmSpec.params`（`ParamSpec`）已能自描述算法参数（`base.py`），`registry.describe_all()` 可导出 schema；`Dataset` 已用 JSONL 持久化。本文的**层 1** 线是尚未落地的 `StudySpec`，**层 2** 线直接复用 `ParamSpec`。

---

## 0. 一页摘要

| 问题 | 答案 |
|---|---|
| 参数真的很多吗？ | 是。5 个算法族共 **18 个算法参数**（bayes 11 个、random/sobol/grid 各 2、lhs 1），再加实验侧约 15 项、AI/安全侧 4 项，共 40+ |
| 硬化成几条？ | **41 条层 1 输入线** + 18 条层 2 参数线（由 schema 自动生成，不手写） |
| 必须保留可调 | **10 条**（L01/L02/L04/L06/L09/L10/L11/L12/L20/L38）——第一屏 |
| 专家可调 | **19 条**（折叠区，默认值即推荐值） |
| AI 自动填 | **8 条**（L03/L15/L18/L22/L23/L26/L28/L33），有包络、有理由、有审计 |
| 固定死 | **2 条**（L16 最大维度 30、L36 确定性 true）+ 层 2 的 6 个实现常量 |
| 预留未实现 | **2 条**（L13 线性约束、L14 禁入区，v1 收到即明确拒绝，不静默忽略） |
| 用户实际要拧的旋钮 | **4 个**：条件空间（L09–L12）、预算（L04）、批大小（L06）、算法族（L20=auto） |
| 档位 | 3 档 effort：`quick`(40 次) / `normal`(160) / `thorough`(640)；2 个 intent：`explore` / `exploit`，与 effort 正交 |
| 校验 | 22 条单体规则 + 14 条交叉规则 + 15 个错误码（E001–E015），每条错误给修复建议 |

---

## 1. 概念、分层与可调等级

### 1.1 什么是「输入线」

一条输入线 = 一个可被外部设置、可校验、可记录、可复现的参数通道。它必须同时具备：

1. **唯一 ID**（L01–L41，稳定不重编号）；
2. **字段路径**（YAML/JSON 中的位置，如 `budget.batch_size`）；
3. **类型与范围**（enum / int / float / list / dict，含上下界或枚举集）；
4. **单位**（物理单位或「无」，单位写进 UI，禁止暗含）；
5. **默认值**（必须等于「推荐值」，不写「见情况」）；
6. **档位映射**（quick/normal/thorough 的取值）；
7. **校验规则**（单体 + 交叉，见第 6 节）；
8. **可调等级**（谁能改、改了何时生效，见 1.3）；
9. **审计要求**（是否写 `decisions.jsonl`）。

### 1.2 两层结构

| 层 | 名称 | 归属 | 数量 | 生成方式 |
|---|---|---|---|---|
| 层 1 | 实验/策略输入线 | 新增 `StudySpec`（`space` 已存在，其余新增） | 41 条（L01–L41） | 手写 schema，本文定义 |
| 层 2 | 算法参数输入线 | 已有 `AlgorithmSpec.params` | 18 条（`rand2 / sobol2 / lhs1 / grid2 / bayes11`） | **由 `ParamSpec` 自动生成**；`validate_params()` 校验；未知键报错 |

> 铁律：**层 1 里不允许出现 `algorithm.params = {...任意 JSON...}` 的自由通道**。L21 只接受「该算法 `ParamSpec` 里声明过的键」，值经 `ParamSpec.coerce()` 校验；其余键直接 E010。

### 1.3 可调等级（本文核心分类）

| 等级 | 代号 | UI 表现 | 谁能改 | 是否写审计 | 数量 |
|---|---|---|---|---|---|
| 必须可调 | `MUST` | 第一屏常显 | 人；AI 只能「建议」，不可覆写 | 改动即写 | 10 |
| 专家可调 | `EXPERT` | 「专家」折叠区，非默认展开 | 人；AI 可按白名单申请覆写（见 1.4） | 改动即写 | 19 |
| AI 自动填 | `AI` | 显示「AI 自动（可锁定）」+ 锁定按钮 | AI 默认填；人可锁定成固定值 | 是（含理由） | 8 |
| 固定死 | `FIXED` | 灰显只读，hover 说明原因 | 无人（只能改代码） | 不需要 | 2 |
| 预留未实现 | `UNAVAIL` | 隐藏；若 YAML 出现则 E014 | — | — | 2 |

### 1.4 AI 覆写权限分三档（与 L38–L41 配合）

| 档 | 允许触达的线 | 条件 | 记录 |
|---|---|---|---|
| A. 建议（advisory） | 全部 41 条 | 只出现在 `decisions.jsonl` 的 `router_suggest`/`ai_advice`，不影响执行 | 只读，不阻断 |
| B. 免审覆写 | L03/L15/L18/L22/L23/L26/L28/L33（即 8 条 `AI` 线） | 必须满足 A1–A5（见第 5 节包络） | 必须写理由 + 证据 + `ai_proposal` |
| C. 审批后覆写 | L20（算法族）、L05/L07（预算结构）、L17（初始设计）、L29（批量策略） | `hitl.mode=gate` 或 `notify` 且人未否决；`hitl.mode=off` 时**一律拒绝** | 与 A 相同 + `human_approve` |

---

### 1.5 与现有实现的映射（实测 2026-11-05）

> 现状：`Study`（`src/condopt/study.py`）、`OptimizationContext`（`router.py`）、`AlgorithmSpec.params` 已覆盖 41 条线中的大部分，但**不以 L 编号暴露**。本规格的价值 = 给这些字段稳定 ID + 校验 + 档位 + 默认值，而不是另起一套实现。
>
> 计数：✅ 已实现 22 条｜🟡 部分实现 8 条｜❌ 待实现 9 条｜🚫 明确拒绝 2 条。

| 线 | 已落地字段 | 状态 | 差距 / 备注 |
|---|---|---|---|
| L01/L02 | `problem.objective.{name,direction}`（`Objective`） | ✅ | — |
| L03 | — | ❌ | 需新增 transform，并记录变换参数 |
| L04 | `problem.budget` = `context.budget_total` | ✅ | 默认 30（`runs/demo` 为 24）；预算即硬上限 |
| L05 | — | 🟡 | 无轮数上限；由 `ask` 次数与预算控制 |
| L06 | `config.batch_size`（默认 1） | ✅ | `init/route --batch` 可设 |
| L07 | — | ❌ | 墙钟未实现 |
| L08 | `Observation.cost` 仅记录 | ❌ | 无成本预算/成本感知分 |
| L09–L11 | `problem.space.variables`（`SearchSpace`） | ✅ | 6 种 kind、单位、描述齐备 |
| L12 | — | ❌ | `SearchSpace.validate_config` 拒绝额外 key；需新增 `problem.context` 并在输出建议时 merge |
| L13/L14 | `context.n_constraints`（仅计数） | 🚫 | 仅 warning，无执行；v1 明确拒绝 |
| L15 | — | ❌ | — |
| L16 | — | ❌ | router 在 dim>20 时降分+警告；建议再加 30 硬限 |
| L17 | bayes 初始设计固定 Sobol | 🟡 | 未暴露选择 |
| L18 | `params.n_initial`（router 自动算，可 override） | ✅ | 自动 = `max(3,min(dim+1,20))` 且 ≤0.4×剩余 |
| L19 | — | ❌ | — |
| L20 | `config.algorithm`（`auto` → `route()`） | ✅ | 5 族已注册 |
| L21 | `config.params`（`AlgorithmSpec.validate`） | ✅ | 未知键直接报错（E010 已有实现） |
| L22–L32 | `config.params` 同名 11 项 | ✅ | 均可经 `condopt override --params` 改 |
| L33 | `config.patience`（`route_round` 停滞检测） | 🟡 | 实现默认 3；语义=连续 N 轮无改进后插入探索批 |
| L34 | `route_round(improve_tol=1e-6)` 硬编码 | 🟡 | 绝对容差、未暴露；本规格建议改为相对值 |
| L35 | `problem.seed` | ✅ | 影响 RNG / Sobol / GP 重启 |
| L36 | 无显式开关 | ❌ | 目前只靠 seed；建议固定 true |
| L37 | — | ❌ | 审计级别未实现（`decisions.jsonl` 总是写） |
| L38 | `config.auto_route` bool | 🟡 | true≈ai_assist（AI 尚未接入），false≈hard_only；无 ai_primary |
| L39–L41 | `Study.override()` 强制要求 reason | 🟡 | 留痕 ✅；无白名单/配额/HITL |

**结论**：P1 渲染层只依赖已实现的 6 个文件；**先做 22 条已实现线的表单化，再按 P2/P3 补 9 条待实现线**，不要等 41 条全齐。

---

## 2. 输入线总表

> 列说明：类型/范围 → 单位 → 典型值 → 默认值 → 档位（quick / normal / thorough）→ 可调等级 → 生效时机。

### A. 目标与预算（L01–L08）

| ID | 字段 | 名称 | 含义 | 类型 / 范围 | 单位 | 典型值 | 默认 | 档位 q/n/t | 等级 | 生效 |
|---|---|---|---|---|---|---|---|---|---|---|
| L01 | `objective.name` | 目标名 | 观测数据里作为主目标的字段名（`Observation.y` 或 `objectives[key]`） | str，非空，≤64 字符，`^[A-Za-z_][A-Za-z0-9_.-]*$` | — | `yield` | `score` | 同 | MUST | 新 study |
| L02 | `objective.direction` | 优化方向 | 越大越好 / 越小越好 | enum `max`\|`min` | — | `max` | `max` | 同 | MUST | 新 study |
| L03 | `objective.transform` | 目标变换 | 送进模型的数值变换；`rank` 用于量纲极端/离群 | enum `raw`\|`log`\|`rank` | — | `raw` | `raw` | 同 | **AI** | 下一轮 |
| L04 | `budget.evaluations` | 总评估预算 | 本 study 允许的**实测/仿真评估总次数**（含初始设计） | int，8–20000 | 次 | 160 | 160 | **40 / 160 / 640** | MUST | 新 study |
| L05 | `budget.max_rounds` | 最大轮数 | 最多跑几轮；预算先到即停 | int，1–200 | 轮 | 18 | 18 | **8 / 18 / 38** | EXPERT | 新 study |
| L06 | `budget.batch_size` | 每轮批大小 | 一次 ask 返回的点数（并行实验通道数） | int，1–64 | 点/轮 | 8 | 8 | **4 / 8 / 16** | MUST | 下一轮 |
| L07 | `budget.wall_clock_min` | 墙钟上限 | 从启动到强制收尾的分钟数；与 L04 先到为准 | int，5–10080 | 分钟 | 180 | 180 | **30 / 180 / 720** | EXPERT | 新 study |
| L08 | `budget.cost_per_eval` | 单次评估成本 | 用于「成本感知」提示与预算报告；0=不追踪 | float，0–1e6 | ¥/次 | 0 | 0 | 同 | EXPERT | 下一轮 |

### B. 搜索空间（L09–L16）

| ID | 字段 | 名称 | 含义 | 类型 / 范围 | 单位 | 典型值 | 默认 | 档位 q/n/t | 等级 | 生效 |
|---|---|---|---|---|---|---|---|---|---|---|
| L09 | `space.variables` | 条件变量清单 | 可优化的条件变量，1–30 个；每个含 name/kind/范围/单位/描述 | list[Variable]，见 2.1 | — | 5 个 | 必填 | 同 | MUST | 新 study |
| L10 | `space.bounds` | 变量范围 | 每变量 `low/high`（数值/整数/对数）或 `choices`（分类/序数/布尔） | float/int/list，见 2.1 | 变量单位 | 见 2.1 | 必填 | 同 | MUST | 新 study |
| L11 | `space.kinds` | 变量类型 | `continuous`\|`integer`\|`log`\|`categorical`\|`ordinal`\|`bool` | enum set | — | 混合 | 必填 | 同 | MUST | 新 study |
| L12 | `context.fixed_conditions` | 固定条件 | **不优化、但每次实验必须固定的上下文**（气氛、设备、批次、操作者），随建议原样输出 | dict[str, scalar]，≤20 键，键不得与 L09 重名 | 各自单位 | `{"atmosphere":"N2","plate":"P-07"}` | `{}` | 同 | MUST | 新 study |
| L13 | `constraints.linear` | 线性约束 | v1 **未实现**；收到即 E014 拒绝（不静默忽略） | list[str] | — | — | `[]` | 同 | UNAVAIL | — |
| L14 | `constraints.forbidden` | 禁入区 | v1 **未实现**；收到即 E014 | list[dict] | — | — | `[]` | 同 | UNAVAIL | — |
| L15 | `space.dim_reduction` | 降维策略 | 维度 >16 时是否先投影到低维再建模 | enum `none`\|`pca`\|`auto` | — | `auto` | `auto` | 同 | **AI** | 下一轮 |
| L16 | `space.max_dim` | 最大维度保护 | 超过即拒绝建模（避免 GP 退化、样本需求爆炸） | int（固定 30） | 变量数 | 30 | 30 | 30 | FIXED | — |

#### 2.1 变量定义子结构（L09–L11 的展开）

| 字段 | 必填 | 校验 | 示例 |
|---|---|---|---|
| `name` | 是 | 唯一；`^[A-Za-z_][A-Za-z0-9_]{0,31}$`；不得与 L12 键重名 | `temperature` |
| `kind` | 是 | 6 种之一 | `continuous` |
| `low`/`high` | 数值/整数/对数必填 | 有限且 `low < high`；`log` 要求 `low > 0`；`integer` 边界必须整数 | `20 / 80` |
| `choices` | 分类/序数/布尔必填 | 分类 ≥2 且互异；布尔恰 2 | `["MeOH","EtOH","THF"]` |
| `unit` | 否 | ≤16 字符；**UI 必须显示** | `degC` |
| `description` | 否 | ≤120 字符 | `反应温度` |

> 固定条件的落点（实现约束）：`SearchSpace.validate_config()` 只接受变量集合内的 key，因此 L12 不能塞进 `space`；建议在 `problem.json` 新增 `context.fixed_conditions`，输出建议时执行 `suggestion.x = {**variable_values, **fixed_conditions}`，并在 `dataset.jsonl` 中保留合并后的 `x` 供审计。详见 `FEASIBILITY.md` §7。

### C. 初始设计（L17–L19）

| ID | 字段 | 名称 | 含义 | 类型 / 范围 | 单位 | 典型值 | 默认 | 档位 q/n/t | 等级 | 生效 |
|---|---|---|---|---|---|---|---|---|---|---|
| L17 | `init.method` | 初始设计方法 | 前 N 个点如何产生（`bayes` 的 `n_initial` 内部用 Sobol） | enum `sobol`\|`lhs`\|`random`\|`grid`\|`none` | — | `sobol` | `sobol` | 同 | EXPERT | 新 study |
| L18 | `init.size` | 初始设计点数 | 0 = 自动 `max(3, min(dim+1, 20))`；显式值时须满足交叉规则 X02 | int，0–200 | 点 | 16 | 16 | **8 / 16 / 32** | **AI** | 新 study |
| L19 | `init.repeats` | 每点重复次数 | 同条件重复测量以估噪；与 `noise_level` 联动（X08） | int，1–5 | 次/点 | 1 | 1 | 1 / 1 / 2 | EXPERT | 新 study |

### D. 算法选择（L20–L22）

| ID | 字段 | 名称 | 含义 | 类型 / 范围 | 单位 | 典型值 | 默认 | 档位 q/n/t | 等级 | 生效 |
|---|---|---|---|---|---|---|---|---|---|---|
| L20 | `algorithm.family` | 算法族 | `auto` = 交给 router 硬规则（推荐）；显式指定则跳过 router | enum `auto`\|`random`\|`sobol`\|`lhs`\|`grid`\|`bayes` | — | `auto` | `auto` | 同 | MUST | 新 study |
| L21 | `algorithm.params` | 算法扩展参数 | 仅允许该族 `ParamSpec` 已声明的键（见第 3 节），逐键校验合并 | dict[str, any] | — | `{}` | `{}` | 同 | EXPERT | 下一轮 |
| L22 | `algorithm.acquisition` | 采集函数 | 仅 `bayes` 有效；`auto` 时按 intent 选（explore→ucb / exploit→ei） | enum `ei`\|`logei`\|`ucb`\|`pi`\|`ts` | — | `logei` | `logei` | **ei / logei / logei** | **AI** | 下一轮 |

### E. 采集与代理（L23–L28，均仅 `bayes` 有效）

| ID | 字段 | 名称 | 含义 | 类型 / 范围 | 单位 | 典型值 | 默认 | 档位 q/n/t | 等级 | 生效 |
|---|---|---|---|---|---|---|---|---|---|---|
| L23 | `surrogate.kappa` | UCB 探索系数 | 仅 `acquisition=ucb` 用；越大越探索 | float，0–10 | — | 2.0 | 2.0 | 2.0 | **AI** | 下一轮 |
| L24 | `surrogate.xi` | EI/PI 改进阈值 | 仅 `ei/logei/pi` 用；越大越偏探索 | float，0–1 | — | 0.01 | 0.01 | 0.01 | EXPERT | 下一轮 |
| L25 | `surrogate.kernel` | GP 核 | 目标光滑度先验；`matern52` 通用 | enum `matern52`\|`matern32`\|`rbf` | — | `matern52` | `matern52` | 同 | EXPERT | 下一轮 |
| L26 | `surrogate.length_scale` | 初始长度尺度 | **单位是归一化 [0,1] 坐标**（不是原始单位！） | float，0.01–10 | 归一化坐标 | 0.3 | 0.3 | 0.3 | **AI** | 下一轮 |
| L27 | `surrogate.alpha` | 观测噪声**方差** | 注意是方差不是标准差；噪声大时 0.01–0.1 | float，1e-12–1.0 | 方差 | 1e-6 | 1e-6 | 1e-6 | EXPERT | 下一轮 |
| L28 | `observation.noise_level` | 噪声水平先验 | 语义输入；映射 alpha：none→1e-8 / low→1e-4 / medium→1e-4 / high→5e-3（若 L27 未显式设置） | enum `none`\|`low`\|`medium`\|`high` | — | `medium` | `medium` | 同 | **AI** | 下一轮 |

### F. 批量与候选池（L29–L32）

| ID | 字段 | 名称 | 含义 | 类型 / 范围 | 单位 | 典型值 | 默认 | 档位 q/n/t | 等级 | 生效 |
|---|---|---|---|---|---|---|---|---|---|---|
| L29 | `batch.strategy` | 批量选点策略 | 批量 ≥4 时用 `kriging_believer`/`constant_liar` 防扎堆；批量=1 强制 `greedy` | enum `greedy`\|`kriging_believer`\|`constant_liar` | — | `kriging_believer` | `kriging_believer` | **constant_liar / kriging_believer / kriging_believer** | EXPERT | 下一轮 |
| L30 | `batch.n_candidates` | 候选池大小 | 每轮打分候选点；越大越准越慢（实测 bayes 单轮含 GP 拟合仍在秒级） | int，64–65536 | 点 | 1024 | 1024 | **256 / 1024 / 4096** | EXPERT | 下一轮 |
| L31 | `search.local_frac` | 局部候选比例 | 候选池中围绕当前最优点的比例；越大越利用 | float，0–0.9 | — | 0.3 | 0.3 | 同 | EXPERT | 下一轮 |
| L32 | `search.local_sigma` | 局部扰动尺度 | 局部候选的单位立方体扰动标准差 | float，0.01–1.0 | 归一化坐标 | 0.15 | 0.15 | 同 | EXPERT | 下一轮 |

### G. 停止·复现·审计（L33–L37）

| ID | 字段 | 名称 | 含义 | 类型 / 范围 | 单位 | 典型值 | 默认 | 档位 q/n/t | 等级 | 生效 |
|---|---|---|---|---|---|---|---|---|---|---|
| L33 | `stopping.patience` | 早停耐心 | 连续多少轮最优值无改进即停；0 = 关闭 | int，0–50 | 轮 | 5 | 5 | **3 / 5 / 8** | **AI** | 下一轮 |
| L34 | `stopping.min_improve` | 最小改进阈值 | 相对改进率 < 该值视为「无改进」（实现里 `route_round` 用绝对 1e-6，本规格改成相对值） | float，0–0.1 | 相对值 | 1e-3 | 1e-3 | 同 | EXPERT | 下一轮 |
| L35 | `repro.seed` | 随机种子 | 采样/GP 重启/候选池共用同一 seed 派生流 | int，0–2147483646 | — | 0 | 0 | 同 | EXPERT | 新 study |
| L36 | `repro.determinism` | 确定性模式 | 固定迭代顺序与 RNG；跨机器数值可复现（同类 BLAS 下） | bool（固定 true） | — | true | true | true | FIXED | — |
| L37 | `audit.level` | 审计级别 | `basic` 只记 router；`full` 记 router+AI+人审+哈希链 | enum `basic`\|`full` | — | `full` | `full` | 同 | EXPERT | 下一轮 |

### H. AI 与安全（L38–L41）

| ID | 字段 | 名称 | 含义 | 类型 / 范围 | 单位 | 典型值 | 默认 | 档位 q/n/t | 等级 | 生效 |
|---|---|---|---|---|---|---|---|---|---|---|
| L38 | `router.mode` | 路由模式 | `hard_only` = 只用硬规则；`ai_assist` = 硬规则先选、AI 可解释/覆写；`ai_primary` = AI 先选、硬规则兜底（实验性） | enum `hard_only`\|`ai_assist`\|`ai_primary` | — | `ai_assist` | `ai_assist` | 同 | MUST | 下一轮 |
| L39 | `ai.override_whitelist` | AI 免审覆写白名单 | 只允许 `AI` 线；子集，默认全 8 条 | list[enum]，元素 ∈ 8 条 AI 线 | — | 8 条 | 全 8 条 | 同 | EXPERT | 下一轮 |
| L40 | `ai.override_max` | 覆写次数上限 | 每个 study 最多覆写几次；超出即回退硬规则 | int，0–10 | 次/study | 3 | 3 | 同 | EXPERT | 下一轮 |
| L41 | `hitl.mode` | 人审模式 | `off` 不审（需显式授权）；`notify` 事后通知；`gate` 结构性变更先审 | enum `off`\|`notify`\|`gate` | — | `gate` | `gate` | 同 | EXPERT | 下一轮 |

---

## 3. 层 2：18 条算法参数线（schema 自动生成，不手写 UI）

来源：`registry.describe_all()`；`AlgorithmSpec.validate()` 会拒绝未声明键（E010）。**表单由 `ParamSpec` 渲染**：`type=int/float` → 数字输入；`choices` → 下拉；`bool` → 开关；`low/high` → 前端范围提示 + 后端二次校验；`description` → 帮助文本；`unit` → 后缀。

### 3.1 参数清单与暴露策略

| 算法族 | 参数 | 类型/默认 | 前端暴露 | 对应层 1 线 |
|---|---|---|---|---|
| bayes | `kernel` | choice `matern52` | 专家下拉 | L25 |
| bayes | `length_scale` | float 0.3（0.01–10） | AI 显示 | L26 |
| bayes | `alpha` | float 1e-6（1e-12–1） | 专家输入 | L27 |
| bayes | `n_initial` | int 0（0–200） | 隐藏，由 L18 解析 | L18 |
| bayes | `n_candidates` | int 1024（64–65536） | 专家输入 | L30 |
| bayes | `acquisition` | choice `ei` | AI 下拉 | L22 |
| bayes | `kappa` | float 2.0（0–10） | AI 输入 | L23 |
| bayes | `xi` | float 0.01（0–1） | 专家输入 | L24 |
| bayes | `batch_strategy` | choice `kriging_believer` | 专家下拉，按 L29 | L29 |
| bayes | `local_frac` | float 0.3（0–0.9） | 专家输入 | L31 |
| bayes | `local_sigma` | float 0.15（0.01–1） | 专家输入 | L32 |
| random | `dedupe` | bool `True` | **固定**（实现常量） | — |
| random | `max_attempts_factor` | int 20（1–1000） | **固定** | — |
| sobol | `scramble` | bool `True` | **固定** | — |
| sobol | `dedupe` | bool `True` | **固定** | — |
| lhs | `optimization` | choice `random-cd`（`none`/`random-cd`/`lloyd`） | 专家下拉 | L21 |
| grid | `max_points` | int 200（2–100000） | 专家输入 | L21 |
| grid | `levels` | int 5（2–20） | 专家输入 | L21 |

### 3.2 层 2 固定死清单（6 个实现常量，不进表单）

| 常量 | 值 | 出处 | 为什么固定 |
|---|---|---|---|
| GP `normalize_y` | `True` | `algorithms/bayes.py` `_fit()` | 目标量纲差异大时几乎总是更好；暴露只会制造坏配置 |
| GP `n_restarts_optimizer` | `1`（≥6 点时）/ `0` | 同上 | 与 `n_candidates` 组合影响耗时，收益低 |
| GP 抖动 `_JITTER` | `1e-9` | 同上 | 数值稳定性常量 |
| `random.dedupe` / `sobol.dedupe` | `True` | `algorithms/sampling.py` | 去重几乎总是对的；极端小离散空间有兜底 |
| `sobol.scramble` | `True` | 同上 | 不打乱会与默认初始设计重复 |
| 候选池构成 | 50% uniform + 50% Sobol + `local_frac` 局部 | `bayes._candidate_pool()` | 平衡全局覆盖与局部精修，调它需要消融实验 |

### 3.3 换算表（语义线 → 数值线）

| 语义输入 | 数值落点 | 规则 |
|---|---|---|
| L28 `noise_level` | L27 `alpha` | none→1e-8，low→1e-4，medium→1e-4，high→5e-3（与 `router._bayes_params` 一致）；L27 显式设置时以 L27 为准，并提示不一致 |
| L02 `direction` | `Objective.sign`（已有） | max→+1，min→−1 |
| L03 `transform=log` | 观测进模型前取 `log1p(y - min+1)`（y 可负时拒绝并建议 rank） | 记录变换参数到 `metrics.json` |
| L18 `init.size=0` | bayes `n_initial` | 自动 = `max(3, min(dim+1, 20))` |
| L29 批量=1 | 强制 `greedy` | 忽略用户设置并写一条 warning |
| L15 `dim_reduction=auto` | dim>16 且 budget≥64 → 开 PCA（保留 90% 方差）；否则 none | 记录实际投影维度 |

---

## 4. 档位预设（3 档 × 2 意图，正交）

### 4.1 effort 三档（其余未列出的线三档相同）

| 线 | quick | normal | thorough | 依据 |
|---|---|---|---|---|
| L04 预算 | 40 | 160 | 640 | 初始设计 + 轮数×批大小恰好用完预算 |
| L06 批大小 | 4 | 8 | 16 | 实验室并行通道数常见 4–16 |
| L05 轮数 | 8 | 18 | 38 | 40=8+8×4；160=16+18×8；640=32+38×16 |
| L07 墙钟 | 30 | 180 | 720 | 分钟；与轮数×单轮耗时上限匹配 |
| L18 初始点 | 8 | 16 | 32 | ≥2×dim 且 ≤0.4×预算 |
| L19 重复 | 1 | 1 | 2 | thorough 才建议估噪 |
| L22 采集 | ei | logei | logei | logei 数值稳定，适合长期累积 |
| L29 批量策略 | constant_liar | kriging_believer | kriging_believer | 小批量 cl 更防扎堆 |
| L30 候选池 | 256 | 1024 | 4096 | 与预算/耗时成正比 |
| L33 早停 | 3 | 5 | 8 | 探索期短，生产期长 |
| L03/L15/L23/L24/L25/L26/L27/L28/L31/L32/L34/L35/L36/L37/L38–L41 | 同 normal | 同左 | 同左 | 与 effort 无关 |

### 4.2 intent 两个（叠加在 effort 之上，只改 7 条线）

| 线 | explore（探索） | exploit（利用） | 说明 |
|---|---|---|---|
| L22 acquisition | `ucb` | `ei` | — |
| L23 kappa | 2.5 | 1.0 | explore 更大探索 |
| L24 xi | 0.05 | 0.001 | exploit 更敏感于小改进 |
| L26 length_scale | 0.5 | 0.2 | explore 更平滑/全局 |
| L31 local_frac | 0.1 | 0.6 | exploit 更依赖局部 |
| L32 local_sigma | 0.25 | 0.10 | exploit 局部更窄 |
| L33 patience | 12 | 4 | 探索期多给耐心 |

> 组合方式：`preset = effort + intent`，例如 `normal/explore`；默认 `normal/exploit`（生产）与 `normal/explore`（研究）。UI 上用两个分段控件，不用 6 个平铺选项。

### 4.3 三档的预期行为（写进 UI 说明）

| 档位 | 适用 | 预期评估数 | 单轮耗时（bayes，本机实测 gp 拟合+1024 候选约 0.2–1s；不含实际实验） | 风险 |
|---|---|---|---|---|
| quick | 快速探路 / 目标未知 / 实验昂贵 | 40 | <1s | 空间覆盖不足，结论只能当筛选 |
| normal | 常规优化 | 160 | 1–2s | 默认，速度与质量平衡 |
| thorough | 生产实验 / 需要稳定最优 | 640 | 2–5s | 实验成本高；建议先 quick 验证流程 |

---

## 5. AI 自动填的 8 条线（包络 + 证据 + 审计）

### 5.1 允许范围（硬包络，越界即拒绝）

| 线 | AI 允许值 | 必须先读的证据 |
|---|---|---|
| L03 目标变换 | `raw`/`log`/`rank` | 观测 y 的偏度/量级（`max/min` 比值、最值差 >1e3 或可负） |
| L15 降维 | `none`/`pca`/`auto` | 维度、预算、变量相关性（PCA 解释方差） |
| L18 初始点数 | `[max(3, 2*dim), min(200, floor(0.4*budget))]` | 维度、预算、已有观测数 |
| L22 采集函数 | `ei`/`logei`/`ucb`/`pi`/`ts` | 预算剩余、噪声水平、当前覆盖 |
| L23 kappa | `[1.0, 3.0]` | 只在 `acquisition=ucb` 时允许 |
| L26 length_scale | `[0.1, 1.0]` | 观测点密度、目标变化尺度（归一化坐标） |
| L28 噪声水平 | 4 档枚举 `none/low/medium/high` | 重复测量方差或用户声明；无证据只能 `medium` |
| L33 早停耐心 | `[3, 15]` | 轮数、历史改进曲线 |

### 5.2 自动填的硬性要求

1. 每条 AI 填值必须带 `reason_code`（枚举，见 `AI_HARD_INTEGRATION.md` §4）与 `evidence`，且 `evidence` 至少含 1 个具体数值；
2. 值必须落在包络内，否则 `ai_proposal` 直接被 schema 拒绝并写 `ai_reject`；
3. AI 填的值**必须在 UI 上标「AI」**，人可一键锁定（锁定后 AI 不再改，直到解锁）；
4. 每个 study 最多 `L40=3` 次覆写；超过即回退硬规则；
5. AI 离线/超时/返回非法时，使用默认值（fail-closed：不影响执行，只损失优化）。

---

## 6. 校验规则

### 6.1 单体规则（22 条）

| 规则 | 线 | 条件 | 失败动作 |
|---|---|---|---|
| R01 | 全部 | 类型/枚举匹配 | E001 拒绝 |
| R02 | L04 | 8 ≤ x ≤ 20000 | E002 拒绝并给范围 |
| R03 | L05 | 1 ≤ x ≤ 200 | E002 |
| R04 | L06 | 1 ≤ x ≤ 64 | E002 |
| R05 | L07 | 5 ≤ x ≤ 10080 | E002 |
| R06 | L08 | 0 ≤ x ≤ 1e6 | E002 |
| R07 | L09 | 1 ≤ len ≤ 30；name 唯一；kind 合法 | E003 定位到第几个变量 |
| R08 | L10 | 数值 low<high 有限；log 要求 low>0；integer 边界为整数 | E003 |
| R09 | L11 | 分类 ≥2、互异；bool 恰 2 | E003 |
| R10 | L12 | dict，≤20 键，值 scalar，键不与 L09 重名 | E004 |
| R11 | L13/L14 | 非空即拒绝（v1 未实现） | E014 明确「v1 不支持，请移除或用 grid 粗筛」 |
| R12 | L15 | 枚举内 | E001 |
| R13 | L17 | 枚举内；与 L20=`random/sobol/lhs/grid` 时不得为 `grid`（除非 L20=`grid`） | E005 |
| R14 | L18 | 0 或 [2, 200] | E002 |
| R15 | L19 | 1–5 | E002 |
| R16 | L20 | 枚举内；`auto` 正常 | E001 |
| R17 | L21 | 键 ∈ 该族 ParamSpec；值经 `coerce()` | E010/E011 |
| R18 | L22–L32 | 仅当 L20=`bayes`（或 `auto` 解析为 bayes）时生效；否则忽略并 warning | W001 |
| R19 | L23 | `acquisition=ucb` 时才校验 0–10 | W002 |
| R20 | L35 | 0 ≤ x ≤ 2^31−2 | E002 |
| R21 | L39 | 非空子集 ∈ 8 条 AI 线 | E012 |
| R22 | L40/L41 | 0–10；枚举；`hitl=off` 且 `override_max>0` 需环境变量授权 | E013 |

### 6.2 交叉规则（14 条）

| 规则 | 公式 | 失败动作 |
|---|---|---|
| X01 | `L18 + L05*L06 ≤ L04` | 若仅轻微超出：自动 `clamp(L05, floor((L04-L18)/L06))` 并写 warning；`clamp 后 L05 < 3` → E005 建议 quick |
| X02 | `L18=0` → 解析为 `max(3,min(dim+1,20))`；解析值须 ≤ `0.4*L04` | 超则 clamp 到 `0.4*L04`，写 warning |
| X03 | `L06 > 1` 且 `L29=greedy` | W003：提示批量下 greedy 易扎堆，自动改 `kriging_believer` |
| X04 | `L20 ∈ {random,sobol,lhs,grid}` 且 L21 含 bayes 参数 | E010：未知参数 |
| X05 | 分类变量占比 > 0.5 且 dim > 10 | W004：建议 random/sobol 或降维；不改用户设置 |
| X06 | L12 键 ∩ L09 名称 ≠ ∅ | E004 |
| X07 | dim > L16(30) | E006：请删除变量或降维（L15） |
| X08 | L28=`high` 且 L27 未显式设置 | 自动 alpha=1e-2，并写 warning |
| X09 | L28=`high` 且 L19=1 | W005：重复测量或调高 alpha |
| X10 | L40 > 0 且 L41=`off` 且未设 `CONDOPT_ALLOW_UNREVIEWED=1` | E013 |
| X11 | L39 ⊄ 8 条 AI 线 | E012 |
| X12 | 预算 < `2*(dim+1)`（BO 样本不足） | W006：推荐 sobol/random；`auto` 时 router 强制 model-free |
| X13 | `L04 < L07*单实验成本`（若 L08>0） | W007：预算可能先于墙钟耗尽 |
| X14 | `L20=auto` 且 `L38=hard_only` 且 AI 覆写白名单非空 | W008：白名单无效（无 AI 参与） |

### 6.3 错误码与修复建议（UI 直接展示）

| 码 | 消息（中文，≤60 字） | 修复建议 |
|---|---|---|
| E001 | 「{线名} 的值 {值} 不在允许集合 {集合} 内」 | 下拉选择有效值 |
| E002 | 「{线名} 必须在 {lo}–{hi} {单位} 之间，当前 {值}」 | 输入范围内数值 |
| E003 | 「条件变量 {name} 定义非法：{原因}」 | 按 2.1 修正范围/类型 |
| E004 | 「固定条件与可优化变量重名：{键}」 | 改名或从固定条件移除 |
| E005 | 「预算/轮数组合不成立：初始 {n0} + {轮}×{批} > 总预算 {N}」 | 用 quick 档或降轮数/批大小 |
| E006 | 「变量数 {dim} 超过上限 30」 | 删除变量或开启降维（L15） |
| E007 | 「算法 {name} 无参数 {key}」 | 只填该算法的合法参数 |
| E008 | 「{参数} 的值 {值} 超出 {lo}–{hi}」 | 按 schema 修正 |
| E009 | 「采集函数 {acq} 不适用于 {算法}」 | bayes 才设置采集函数 |
| E010 | 「算法族 {族} 不接受参数 {key}」 | 清空该参数 |
| E011 | 「算法参数类型错误：{key} 需要 {type}」 | 按类型输入 |
| E012 | 「AI 白名单含非法线：{id}」 | 只允许 L03/L15/L18/L22/L23/L26/L28/L33 |
| E013 | 「关闭人审时不允许 AI 自动覆写（L40>0）」 | 设 hitl=notify/gate，或 override_max=0 |
| E014 | 「{线名} 在 v1 不支持」 | 移除该项；用固定条件/网格粗筛替代 |
| E015 | 「配置哈希不一致：文件 {a} ≠ 解析后 {b}」 | 重新 `condopt config resolve` |

---

## 7. 优先级、解析与记录

### 7.1 解析优先级（高 → 低）

```
CLI 显式参数
  > 环境变量（仅 CONDOPT_AI / CONDOPT_ALLOW_UNREVIEWED 等白名单）
  > study.yaml 的显式字段
  > 档位预设（effort × intent）
  > 线默认值
  > AI 自动填（仅 8 条 AI 线，且必须是人未锁定、未显式设置）
```

规则：

- 人显式设置 > AI：AI 不得覆盖任何**人显式写进 YAML 的线**（写审计 `override_rejected: human_pinned`）；
- AI 填的线与默认值线**必须可区分**：记录 `source: "human" | "preset" | "default" | "ai"`，每线一份（建议放 `problem.json.meta.resolved_lines` 或 `config.json.resolved` 块）；
- 解析结果写入 `problem.json`（或 `config.json` 的 `resolved` 块，与现有 `Study` 持久化兼容），并算 `config_hash = sha256(canonical_json(resolved_lines))`；执行入口必须校验「解析结果哈希 == 运行时配置哈希」，否则 E015。

### 7.2 一处定义、四处复用

| 使用方 | 用法 |
|---|---|
| CLI | 现：`condopt init --run <dir> --space <file> --budget 160 --batch 8 [--noise medium ...]`；建议扩展 `--preset normal/explore`（P3） |
| 表单 | 按组渲染，AI 线有锁定按钮，固定线灰显 |
| skill router | 读 `OptimizationContext`（由 space + 配置派生）做打分选择，**不读原始 YAML、不解析自然语言** |
| 审计 | 每轮把 `config_hash` 写进 `decisions.jsonl`，复现用同一 hash |

### 7.3 解析后的 YAML 示例（normal/exploit，可直接改成配置）

```yaml
# study.yaml —— 41 条线解析后（normal/exploit）；省略行取默认
schema: condopt.study/v1
preset: {effort: normal, intent: exploit}
objective: {name: yield, direction: max, transform: raw}     # L01–L03
budget:                                                       # L04–L08
  evaluations: 160
  max_rounds: 18
  batch_size: 8
  wall_clock_min: 180
  cost_per_eval: 0.0
space:
  variables:                                                  # L09–L11
    - {name: temperature, kind: continuous, low: 20, high: 80, unit: degC}
    - {name: time_h, kind: continuous, low: 0.5, high: 8, unit: h}
    - {name: equiv, kind: integer, low: 1, high: 5, unit: eq}
    - {name: conc, kind: log, low: 0.01, high: 1.0, unit: M}
    - {name: solvent, kind: categorical, choices: [MeOH, EtOH, THF]}
  dim_reduction: auto                                         # L15
  max_dim: 30                                                 # L16 (FIXED)
context:
  fixed_conditions: {atmosphere: N2, plate: P-07}             # L12
init: {method: sobol, size: 16, repeats: 1}                   # L17–L19
algorithm:
  family: auto                                                # L20
  acquisition: ei                                             # L22 (intent=exploit)
  params: {}                                                  # L21
surrogate:                                                    # L23–L28
  kernel: matern52
  length_scale: 0.2
  alpha: 1.0e-6
  kappa: 1.0
  xi: 0.001
  noise_level: medium
batch: {strategy: kriging_believer, n_candidates: 1024}       # L29–L30
search: {local_frac: 0.6, local_sigma: 0.1}                   # L31–L32
stopping: {patience: 4, min_improve: 1.0e-3}                  # L33–L34
repro: {seed: 0, determinism: true}                           # L35–L36 (L36 FIXED)
audit: {level: full}                                          # L37
router: {mode: ai_assist}                                     # L38
ai: {override_whitelist: [L03, L15, L18, L22, L23, L26, L28, L33], override_max: 3}  # L39–L40
hitl: {mode: gate}                                            # L41
```

> 注意：示例里的 5 个变量维度 `dim=5`，初始 16 点、18 轮×8=144，总计 160 = 预算，满足 X01。

---

## 8. 表单/配置落地映射

### 8.1 布局

| 区域 | 内容 | 默认展开 |
|---|---|---|
| 第一屏 | 档位选择（3 段 + 2 段）、目标名/方向、预算、批大小、条件变量表格、固定条件 | 是 |
| AI 区 | 8 条线，显示当前值 + `AI` 徽章 + 锁定开关；可展开「为什么这么填」 | 是（只读态） |
| 专家区 | 19 条，按 E/F/G/H 分组折叠 | 否 |
| 固定/预留 | L16、L36 灰显加 tooltip；L13、L14 隐藏但 YAML 出现即报 E014 | 否 |

### 8.2 控件映射

| 线类型 | 控件 | 联动/灰显规则 |
|---|---|---|
| int 有界 | `<input type=number>` + 范围提示 | L05 上限随 L04/L06 联动（X01 实时显示剩余预算） |
| float 有界 | number + step；长度尺度显示「归一化坐标」 | L23 仅 `acquisition=ucb` 可编辑；L27 仅 noise_level 未显式时由 AI 填 |
| enum | `<select>` | L22/L23–L32 仅 `algorithm=auto→bayes` 或 `bayes` 时可用，否则禁用并提示 |
| 枚举多选 | checkbox 组（8 条 AI 线） | L39 与 L38=`hard_only` 时提示 W008 |
| 条件变量 | 可增删行的小表格（name/kind/low/high/choices/unit/desc） | kind 切换时 low/high↔choices 互斥显示 |
| 固定条件 | key-value 编辑器 | 与变量重名实时 E004 |
| 档位 | 分段控件，切换后预览「哪些线变了」diff | 人工改过的线不被档位覆盖（显示「已手改」标记） |

### 8.3 前端必须复刻的校验（与后端同源）

前端只做**提示**，后端为准：E001–E006、E010–E014 全部在 `condopt config validate` 里实现；前端通过 `GET /api/schema`（本地服务时）或内联 schema 显示范围。**禁止只在前端校验**（CLI 也能被 YAML 直接调用）。

---

## 9. 验收用例（可直接做成 pytest/手工清单）

| # | 输入 | 期望 |
|---|---|---|
| T01 | `preset=quick` 不填任何线 | L04=40、L06=4、L05=8、L18=8，`config resolve` 输出 41 行全部有值 |
| T02 | 同时写 `budget.evaluations=40`、`max_rounds=20`、`batch_size=8` | X01：轮数被 clamp 到 4（(40−16)/8=3 → <3 报 E005）——按示例 L18=16 时 clamp 到 3 并告警 |
| T03 | `space.variables` 32 个 | E006（>30） |
| T04 | `fixed_conditions.atmosphere` 与变量 `atmosphere` 重名 | E004 |
| T05 | `algorithm.family=random` 且 `algorithm.params={kernel: rbf}` | E010 |
| T06 | `algorithm.family=bayes`、`params.kappa=99` | E008（0–10） |
| T07 | `constraints.linear: ["x1+x2<=1"]` | E014（v1 不支持，明确拒绝） |
| T08 | `hitl.mode=off`、`ai.override_max=3`、无环境变量 | E013 |
| T09 | `ai.override_whitelist: [L04]` | E012（L04 不是 AI 线） |
| T10 | `log` 变量 `low=0` | E003 |
| T11 | 同一 YAML 解析两次 | `config_hash` 完全相同（canonical JSON 排序键） |
| T12 | 人显式写 `surrogate.length_scale=0.05` 后 AI 提案改 0.5 | `override_rejected: human_pinned`，值保持 0.05 |
| T13 | 显式 `preset=thorough` 但 `wall_clock_min=30` | W007/W006 提示但不阻断（预算是硬约束，时间是软约束） |
| T14 | 8 条 AI 线全部锁定 | AI 只剩「建议」权限；执行完全按解析值 |
| T15 | `algorithm.family=grid` + `batch_size=8` | grid 正常 ask 8 点（grid 支持批量），L22–L32 禁用 |

---

## 10. 与 skill router 的接口（给实现者的最小契约）

```python
# 建议落在 src/condopt/study.py + src/condopt/router.py
@dataclass(frozen=True)
class ResolvedStudy:
    lines: dict[str, Any]          # L01–L41 的解析值（含每线 source）
    lines_source: dict[str, str]   # "human" | "preset" | "default" | "ai"
    preset: tuple[str, str]        # (effort, intent)
    config_hash: str               # sha256(canonical_json(lines))
    space: SearchSpace
    context: dict                  # L12
    objective: Objective

class Router:                      # 硬规则；AI 只能提案（见 AI_HARD_INTEGRATION.md）
    def suggest(self, study: ResolvedStudy, history: Dataset) -> RouterDecision: ...
```

- router **只读 `ResolvedStudy` + `Dataset` 统计特征**，不读原始 YAML、不读 UI 状态；
- router 输出 `RouterDecision`（含 `algorithm`、`params`、`rule_id`、`confidence`、`why`）；
- 所有线最终注入算法时**再走一次** `AlgorithmSpec.validate()`（双保险）。

---

## 11. 变更与不做

| 项 | 处理 |
|---|---|
| 新增档位（如 `custom`） | 只加 effort 枚举，不动线 ID |
| 新增算法 | 只需 `AlgorithmSpec`（schema 自动进表单），层 1 线大概率不变 |
| 删除线 | 不允许；只能标 `deprecated`，保留解析兼容 |
| 多目标 | 预留：`objective.objectives: list[str]`；v1 遇到 >1 个即 E014，不静默取第一个（`Objective.n_objectives` 已可判断） |
| 线性约束 | v2 再做；v1 拒绝，避免「以为约束生效其实没有」 |
| 参数恢复默认 | 每个线有「恢复默认」；批量操作「整组恢复」 |
