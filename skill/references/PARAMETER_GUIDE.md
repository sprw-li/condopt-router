- 复孔：默认 `confirm_n=3`（`meta.confirm_n` 可覆盖）；预算可探索次数 = 预算 − 复孔占位。

## 开工前检查：可行率与停止口径

`condopt prepare` 会返回两段建议：

- `feasibility`：可行率预警。可用 `feasible_fraction_estimate` 直接给数；
  没给时按“约束条数 / 维度”推测（≥3 条约束且 ≤6 维 → 高风险）。
  `risk=high` 时先把硬限制写进变量范围/固定因子（0.4% 可行域实测：约束搜索 1/5 vs 写进范围 5/5）。
- `transfer-audit` 建议值分级：**≥0.8 用；0.5–0.8 视为可疑（top_k 压到 1–2）；<0.5 不要用大 top_k**（4 类源实测：一致 1.0，shifted/shuffled/flipped 0.3–0.6；`research/TRANSFER_AUDIT.md`）。
- 多目标暖启动：`transfer_from` 会按非支配前沿选点；实测可行 HV +0.4–0.5%、前沿点数 +20%（`research/TRANSFER_MOO.md`）。
- `prior`：有先验数据时的暖启动建议（先 `condopt transfer-audit` 判可信度，再设 `--prior-reliability`；坏源 top_k 1→8 从 −369 恶化到 −1190）。
- `stopping`：停止口径建议。预算 ≥60 建议 `budget_first`（`--budget-first`），
  否则用默认停滞优先。**预算里要留确认复孔**：`prepare` 输出的
  `effective_exploration_budget` 才是净探索次数（例：预算 12 → 保留 3、可探索 9）。
  选 `budget_first` 时注意 `delivery_tradeoff` 提示。

# 参数指南：怎么问、怎么填、建议范围

本文件是 `condopt-router` skill 的参数参考。目标：把用户的自然语言回答，无损地变成
`space.json` + `context.json` + 算法参数，并知道每个数的合理范围。

## 1. 搜索空间变量（`space.json`）

| `kind` | 含义 | 编码方式 | 建议 | 例子 |
|---|---|---|---|---|
| `continuous` | 连续实数 | 线性归一化到 [0,1] | 给出物理上下限；范围不要跨太多数量级 | 温度 40–120 °C |
| `integer` | 整数（计数、份数） | 线性归一化，解码取整 | 上下限必须是整数；范围窄 | 配体当量 1–4 |
| `log` | 正数、跨数量级 | 对数归一化 | 浓度/速率这类跨 10–1000 倍必须用 log | 浓度 0.02–1.0 M |
| `categorical` | 无序分类 | 单位立方体等距编码 + 特征 one-hot | 每类有意义、≤6 类；类别多时先做分组或分层 | 溶剂 MeCN/EtOH/water/THF |
| `ordinal` | 有序分类 | 同 categorical，但顺序有意义 | 顺序确实单调时才用 | 阶段 early/mid/late |
| `bool` | 布尔 | 2 类 | 开/关、有/无 | 惰性气氛 |

建议：

- **变量数 3–15**：≤15 维 BO 可靠；>20 维 router 会退回空间填充采样。
- 每个变量都要有 `unit`（可空但强烈建议填），否则实验人员会填错数量级。
- 连续变量的上下限应是**安全可达**范围；不要在优化中动态改。
- 分类变量不要超过 6 类；超过时考虑拆成两个变量或先做分组筛选。
- 所有变量名用英文短名（`T`、`time`、`conc`、`solvent`），中文写进 `description`。

## 2. 问题上下文（`context.json`）

| 字段 | 取值 | 建议范围 | 怎么从用户回答推断 |
|---|---|---|---|
| `budget_total` | 正整数 | 8–200；**低于建模下限时 router 会屏蔽模型类并给 underpowered 说明**，20–100 最适合 BO | 用户说“还能做 30 次实验” |
| `batch_size` | 正整数 | 1–16（孔板 4–24）；≥4 时 BO 用批量策略 | 每轮能并行几个反应器/孔位 |
| `noise` | `none/low/medium/high` | 复孔 CV：<1% 低、1–5% 中、>5% 高 | 用户不知道 → `medium` |
| `smoothness` | `smooth/mixed/rough` | 能二次/指数近似=smooth；有阈值/相变=rough | 用户不知道 → `mixed` |
| `n_objectives` | ≥1 | 1 = 单目标；2 = 走 qehvi/qnehvi/ParEGO；2 + 约束 = cqehvi | 收率 vs 纯度 = 2 |
| `n_constraints` | ≥0 | 有安全/可行性边界就填 | 温度上限、放热、溶解度 |
| `multi_fidelity` | bool | 有仿真/低精度筛选时为 true | 用户说“先算再试” |
| `compute` | `low/medium/high` | 决定候选池大小 | 无 GPU/算力紧张 → low |
| `domain` | `chemistry/materials/process/generic` | 化学默认 chemistry | 任务所属领域 |
| `n_observations` | ≥0 | ≥6 可暖启动 | 已有历史数据条数 |
| `notes` | 字符串 | 自由备注 | 任何“边界条件/经验”都写进来 |

`budget_remaining` 不用手填：`condopt` 会按“总预算 − 已做 − 待测”自动算。

## 3. 算法参数建议值

### 3.1 `bayes`（GP 贝叶斯优化）

| 参数 | 建议 | 何时调 |
|---|---|---|
| `kernel` | `matern52` 默认；`matern32` 目标粗糙；`rbf` 很光滑且维度 ≤8 | 预测与实测差很多、曲线不平滑时 |
| `length_scale` | 0.3（单位立方体） | 维度大时更小；数据密集时更小 |
| `alpha` | 噪声方差：none 1e-8、low 1e-6、medium 1e-4、high 5e-3 | 噪声大就调大；过大会欠拟合 |
| `n_initial` | `min(max(3, dim+1), 0.4×预算)`，router 自动给 | 预算紧时不超过 40% |
| `n_candidates` | 1024；算力低 512；维度 ≤8 且算力足 2048 | 建议质量 vs 速度 |
| `acquisition` | 低噪声 `ei`/`logei`；高噪声 `ucb`；批量大 `logei`；想更随机 `ts` | 太早收敛→UCB；太发散→EI |
| `kappa` | 2.0（高噪声 2.5） | UCB 的探索强度 |
| `xi` | 0.01 | EI/PI 的改进阈值 |
| `batch_strategy` | 批量=1 `greedy`；2–3 `kriging_believer`；≥4 `constant_liar` | 批量大时防扎堆 |
| `local_frac` | 0.3（≤6 维 0.4） | 局部搜索强度 |
| `local_sigma` | 0.15 | 局部扰动幅度 |

### 3.2 `sobol` / `lhs` / `random` / `grid`

| 算法 | 关键参数 | 建议 |
|---|---|---|
| `sobol` | `scramble`、`dedupe` | 默认即可；用于初始设计/批量探索 |
| `lhs` | `optimization` | `random-cd` 降相关；预算小用 `none` 求快 |
| `random` | `dedupe`、`max_attempts_factor` | 默认即可；高维/极小预算基线 |
| `grid` | `max_points`、`levels` | `max_points` ≈ 预算；`levels` 5；仅低维小离散空间 |

### 3.3 多目标 / 约束（`parego` / `qehvi` / `qnehvi` / `constrained` / `cqehvi`）

| 参数 | 建议 | 说明 |
|---|---|---|
| `mc_draws` | 128（算力足 256） | qehvi/qnehvi/constrained/cqehvi 的 MC 精度 |
| `constraint_margin` | 0.0；安全裕量给 1–5% 的量程 | 约束按 `value − margin` 判可行，宁可保守 |
| `feasibility_threshold` | 0.02 | PoF 过低的候选不参与采集，防止追问不可行区 |
| `ref_point` | 显式给出（原始尺度，按目标顺序） | 固定 qehvi/cqehvi 的 HV 口径；缺失时只能同 run 内比较 |
| `n_base` / `n_y` | 32–64 / 64–128 | qnehvi 的嵌套 MC 精度，噪声大时调大 |

> 2 目标 + 硬约束必须走 `cqehvi`；回填结果时每个点都要带 `constraints`，缺值按不可行处理。

## 4. 常见映射

| 用户说法 | 结论 |
|---|---|
| “浓度从 0.01 到 1 M” | `kind=log`，`low=0.01`，`high=1.0` |
| “只能做 6 次实验” | 单目标：`budget_total=6` → router 选 `sobol`/`lhs`；约束/多目标：保留代理类但会显式告警（没有便宜的可用算法），可显式 `algorithm=sobol` |
| “每轮 12 个孔位” | `batch_size=12`；BO 用 `constant_liar` |
| “复孔偏差 10%” | `noise=high`；`alpha` 调大；建议加权/复孔 |
| “温度超过 110 会分解” | 既是安全边界：空间上限 110 + `constraints` 里写 `T_max <= 110`（走 constrained/cqehvi） |
| “之前做过 10 批” | `n_observations=10`，BO 暖启动；数据导入 `dataset.jsonl` |
| “要同时看收率和纯度” | `n_objectives=2`，router 选 qehvi/qnehvi/ParEGO；报告 Pareto 前沿而不是单一最优 |
| “有仿真可以先筛” | `multi_fidelity=true`，router 选 `mfbo`（fidelity 作 GP 输入） |

## 5. 校验与边界

- 变量范围：`low < high`；`log` 要求 `low > 0`；`integer` 上下限为整数。
- 分类变量取值必须来自 `choices`；布尔只接受 true/false。
- `budget_total ≥ 1`；`batch_size ≥ 1`；`budget_remaining ≤ budget_total`。
- 算法参数必须出现在 `condopt spec --algorithm X` 的 schema 里，越界/未知会直接报错。

> 任何“模型推荐值”都不是物理规律，只影响采样策略；实验安全边界必须写死在空间或约束里。
多保真多目标（momfbo）：router 默认 `min_full_fraction=0.4`；实测 0.25/0.4/0.8 的可行 HV 为 636.4/639.7/640.7（`mf_moo_constrained` budget 40, 5 seeds），0.4 的全保真可行点最多。
