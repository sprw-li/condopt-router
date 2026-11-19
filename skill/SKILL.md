---
name: condopt-router
description: >
  条件优化算法路由器 + 实验闭环驱动。用户要优化实验/反应/工艺条件（温度、时间、浓度、溶剂、
  配比、气氛等）、要设计下一批实验、或说“用贝叶斯优化/随机取点找最优条件”时使用。
  运行后先问必要参数并同时给出建议范围；再调用 condopt 路由器选算法与参数；生成待测条件表；
  回收真实实测结果；迭代到预算耗尽或目标达成。skill 本身是 router：算法不写死，按问题特征选。
  不用于：一次性随机抽样即可的任务、纯理论计算、没有实验评价手段的场合。
compatibility: Requires Python 3.10+ and numpy / scipy / scikit-learn
metadata:
  version: "0.2.0"
  project: <repo>
---

# condopt-router：条件优化路由器

这是「AI 全自动实验室」的前置 skill：把实验条件优化从“随手挑一个算法”变成
**有理由的算法选择 + 可审计的闭环执行**。

三层结构，缺一不可：

1. **硬算法层**（`src/condopt/`）：统一 ask/tell 优化器、混合空间、GP-BO、TPE、ParEGO/qEHVI/qNEHVI
   （多目标）、cqEHVI（2 目标 + 硬约束）、空间填充采样。
   算法行为是确定性的，AI 不能改代码。
2. **路由器层**（`condopt route`）：把问题上下文映射成“算法 + 参数 + 理由 + 备选”。
   规则透明可解释，不靠拍脑袋。
3. **Agent 层**（本 skill）：问参数 → 调 router → 出条件表 → 收结果 → 迭代。
   AI 可以覆写路由结果，但**必须给理由并留痕**（`decisions.jsonl`）。

## 何时用 / 何时不用

**用**：有明确的实验目标值（收率、纯度、转化率、选择性、成本……），有若干可调条件，
实验昂贵或带噪声，想用更少的实验找到更好的条件。

**不用**：只要随机抽样凑数；没有真实评价通道；目标无法量化；空间无法参数化（先做实验设计）。

## 第 0 步：先问必要参数（一次问完，不要挤牙膏）

把下面 9 个问题**放进一条消息**问用户，每题都给出**建议范围**和默认值；用户说“不确定”
就用默认值，并在后续输出里标注“低置信，建议先补充”。最多追问一轮，不要为了问卷停下来。

| # | 问题 | 建议范围 / 默认 | 为什么问 |
|---|---|---|---|
| 1 | 优化目标：名称、单位、**最大化还是最小化**；是否还有第二目标（如收率 vs 成本/纯度） | 1 个目标；2–3 个目标时给出各自方向（max/min），会走 ParEGO | 决定 `objective`（含 `objectives` / `directions`） |
| 2 | 可调条件清单：每个变量的**名称 / 类型 / 范围 / 单位** | 3–15 个变量最合适；连续变量给物理上下限；浓度这类跨数量级的用 **log**；分类变量 ≤6 类 | 决定搜索空间与特征编码 |
| 3 | 总预算：还能做多少次实验 | 8–200 次；<8 用采样，20–100 最适合 BO，>60 BO 收益明显 | 决定算法族与初始设计占比 |
| 4 | 批量：每轮能并行做几个 | 1–16；孔板常见 4–24；批量 ≥4 时 BO 要用批量策略 | 决定批量采集策略 |
| 5 | 噪声：重复实验的相对波动 | `<1%` = 低；`1–5%` = 中；`>5%` = 高；未知按中 | 决定采集函数与 GP 噪声参数 |
| 6 | 光滑度：目标对条件的响应是否平滑 | 可用二次/指数近似 = smooth；有阈值/相变/平台 = rough；不确定 = mixed | 决定核函数与是否该用 BO |
| 7 | 先验数据：已有多少条历史实验、在哪个文件 | 0 条也正常；≥6 条可暖启动 | 决定初始设计点数与是否暖启动 |
| 8 | 约束与安全边界：温度上限、放热、溶解度、成本等 | 有就逐条写清（阈值 + `<=`/`>=`）；没有写“无” | 决定是否走 `constrained`（单目标）或 `cqehvi`（2 目标 + 硬约束） |
| 9 | 领域与节奏：化学/材料/工艺；每次实验耗时；是否可先用仿真/短时试跑粗筛 | 默认 chemistry、按实际填；有粗筛通道会走 `mfbo` | 决定先验偏好与多保真路线 |

如果用户已经给了其中一些信息，**只问缺的**，并在回复开头用一行复述你理解到的参数。

## 第 1 步：把回答写成一个 answers.json，再生成三个文件

**推荐**：把回答按 `answers.json` 写好，用 `prepare` 一次完成校验 + 生成 + 路由（模板见
`py -3 scripts/run_condopt.py prepare --template`）：

```json
{
  "objective": {"name": "yield", "direction": "max"},
  "variables": [
    {"name": "T", "kind": "continuous", "low": 40, "high": 120, "unit": "degC"},
    {"name": "time", "kind": "continuous", "low": 5, "high": 120, "unit": "min"},
    {"name": "conc", "kind": "log", "low": 0.02, "high": 1.0, "unit": "M"},
    {"name": "solvent", "kind": "categorical", "choices": ["MeCN", "EtOH", "water"]},
    {"name": "equiv", "kind": "integer", "low": 1, "high": 4},
    {"name": "inert", "kind": "bool"}
  ],
  "budget": 24, "batch": 4, "noise": "medium", "smoothness": "smooth",
  "domain": "chemistry", "prior": 0, "notes": "自由基反应，温度上限 110"
}
```

```powershell
py -3 scripts/run_condopt.py prepare --answers <workspace>\answers.json --out <workspace> --json
```

会写出 `space.json / context.json / objective.json` 并返回 router 推荐；多目标时 `objective` 写成：
`{"name":"yield","objectives":["yield","cost"],"directions":["max","min"]}`。

> 也可以手写 `space.json` / `context.json` / `objective.json`（字段见 `references/JSON_SCHEMAS.md`），
> 或只传 `--space` 并用 `--budget/--batch/--noise/...` 覆盖 context。

## 第 2 步：让 router 选算法（不要自己挑）

```powershell
cd <repo>
py -3 scripts/run_condopt.py route --space <workspace>\space.json --context <workspace>\context.json --json
```

把返回的 `algorithm / params / confidence / rationale / alternatives / warnings / design / plan`
**翻译成人话**展示给用户，至少包括：

- 选了什么算法，为什么（列 2–3 条理由）；
- 关键参数（初始设计点数、采集函数、批量策略）；
- 有什么风险/警告；
- 备选算法是什么，什么情况下换它。

- 用户/AI 想换算法或改参数时：**必须写一句理由**，用 `override` 记录（见第 3 步）；
  不允许无理由改。覆写后自动重路由会暂停（`manual_override=true`），要交回 router 就运行
  `py -3 scripts/run_condopt.py auto --run <workspace>\run`。
  若用户要的算法不在注册表里，说明当前支持哪些（`condopt spec --json`），不要硬编。

## 第 3 步：建 study（注意 run 目录已存在时）

```powershell
py -3 scripts/run_condopt.py init --run <workspace>\run --space ... --objective ... --budget 40 --batch 4
```

- **run 目录已存在战役时，`init` 默认拒绝（退出码 2）**，不会像以前那样静默把算法改回默认值。
  想保留已测数据、只换算法/参数并留痕：

  ```powershell
  py -3 scripts/run_condopt.py init --run <workspace>\run --space ... --objective ... \
      --algorithm bayes --if-exists keep      # 已测数据保留，写 decisions.jsonl，并置暖启动宽限期
  ```

  想丢弃配置重新开始用 `--if-exists reset`；不确定就先看 `py -3 scripts/run_condopt.py doctor --run <run>`。
- 续测**同一块**战役不要重新 `init`：直接 `ask`/`tell` 即可（`pending.json` 会告诉你还差哪些结果）。

## 第 3 步：建 study 并生成第一批待测条件

```powershell
py -3 scripts/run_condopt.py init --run <workspace>\run --space <workspace>\space.json `
    --objective yield:max --budget 24 --batch 4 --json
# 多目标：--objective 直接传 prepare 生成的 objective.json
py -3 scripts/run_condopt.py init --run <workspace>\run --space <workspace>\space.json `
    --objective <workspace>\objective.json --budget 24 --batch 4 --json
# 有历史数据时：作为先验导入（计入预算，便于暖启动）
py -3 scripts/run_condopt.py import --run <workspace>\run --results <workspace>\prior.json --json
# --objective 也可传 JSON 文件；Windows PowerShell 里不要把内联 JSON 直接放命令行（引号会被吞）
# 如需覆写：
py -3 scripts/run_condopt.py override --run <workspace>\run --algorithm bayes `
    --params '{"acquisition":"ucb","alpha":0.005}' --reason "用户要求更偏探索"

py -3 scripts/run_condopt.py ask --run <workspace>\run --json
# 续测同一块孔板（默认每轮自动新开 P1、P2…；同一块板会自动跳过已用孔位）
py -3 scripts/run_condopt.py ask --run <workspace>\run --n 4 --plate P1 --json
```

把 `suggestions.csv` 的表格展示给用户（条件值 + 单位 + 建议 id + `plate/well`），并明确：

> 这些是**待测建议**，请按 id 回填**真实实测值**。没有做实验就不要回填，也不要让我编造。
> `plate`/`well` 是执行位置标记：请按 A1、A2… 加样；实际换了板或重排孔位时，在结果里覆写 `plate`/`well`。

## 第 4 步：回收实测结果

把用户给的结果写成 `results.json`（也支持 JSONL）：

```json
[{"id": "r1-1", "y": 78.3}, {"id": "r1-2", "y": 64.1}]
```

多目标时回填目标向量（有约束时**每个点都必须带** `constraints`，`constrained`/`cqehvi` 依赖它判可行）：

```json
[{"id": "r1-1", "objectives": {"yield": 78.3, "cost": 11.2}, "constraints": {"T_max": 76.0}}]
```

然后：

```powershell
py -3 scripts/run_condopt.py tell --run <workspace>\run --results <workspace>\results.json --json
py -3 scripts/run_condopt.py report --run <workspace>\run
```

- 结果必须来自真实测量；`source` 可标 `manual / robot / simulation`，但仿真不能当实验结论。
- 缺失/异常结果要追问，不要猜。
- 实际执行位置与建议不同时，在结果里加 `"plate": "P2", "well": "C3"`（也可带 `batch`）覆写；不写则沿用建议的板/孔。
- 每轮结束后报告：当前最优条件、本轮最优、累计最优、剩余预算。
- `report` 会按板汇总（n/均值/极差）并做**批次效应检查**：出现“检测到批次/孔板效应”警告时，
  要向用户指出板间差异（附 p 值/相对差），建议同批内比较、下一轮混板或把 plate 记录为协变量；
  **不要自动用板均值校正数据**（那会改变口径）。

## 第 4.5 步：有历史数据时用「分阶段迁移」

不要一次性把源战役的很多点搬过来：先搬 1 条，跑两批本地观测，再让审计决定是否放大：

```powershell
py -3 scripts/run_condopt.py transfer-staged --run <run> --from-run <source_run> --start-top-k 1 --top-k 5
```

- 第一次调用搬 1 条（`stage=first_look`），跑满两批后再调一次；
- 第二次会做 `transfer-audit`：建议值低于否决线（默认 0.6，实测安全区间 0.4–0.6）就 **不放大**
  （`stage=vetoed`），否则放大到 `--top-k`；
- 实测（`research/TRANSFER_ADAPTIVE.md`）：有偏源上这样做 +0.71，而固定 top_k=3/5 分别是 −1.54/−0.33；
- 否决线 <0.4 起不到保护作用，≥0.8 会连好源一起否决（机制退化为“永不放大”）；
- `--force` 可跳过否决线，但要在理由里留痕。

## 第 5 步：迭代、自动循环与重路由

每轮 `tell` 之后，**先问引擎下一步做什么**，不要自己猜：

```powershell
py -3 scripts/run_condopt.py step --run <workspace>\run --json
```

返回的 `action` 是机器可读的循环决策（`recommend`/`confirm`/`should_stop` 已合并进来）：

| action | 含义 | 你的动作 |
|---|---|---|
| `tell` | 当前批次还有建议没回填 | 拿到实测值后 `tell`；不要重复 `ask` |
| `ask` | 可以继续优化 | `ask --n <batch> [--plate P]`（默认自动新开板） |
| `confirm` | 收尾前需要确认最优条件（目标/平台期已到，或预算只剩 ≤ `confirm_n` 次） | `confirm --run R --n <confirm_n>`，测完 `tell` |
| `stop` | 预算/目标/平台期已满足，且（如适用）最优条件已确认 | 输出推荐与收尾报告，不再 ask |

- `action=confirm` 时按 `confirm_n` 做复孔；`confirm` 复制的就是当前最优条件，测完回填后
  `step` 会给出 `stop`，此时 `recommendation.score` 用复孔均值（避免“幸运的一次测量”当结论）。
- 返回里还有 `should_stop / stop_reason / recommendation`（条件、规格符合性、确认统计）与
  `batch_effect`（批次效应警告，`detected=true` 时向用户指出）；`next_command` 是可直接执行的命令。
- 想跳过确认直接停：建 study 时在 `problem.json` 的 `meta` 里设 `{"auto_confirm": false}`；
  复孔次数用 `meta.confirm_n`（默认 3）。
- 循环终止条件：`action=stop`、用户叫停、或数据异常需要人工判断。
- Study 默认开启 `auto_route`：连续 `patience` 轮（默认 3）最优值没有改善时，router 会自动插入
  一批全局探索（sobol/random），并在 `decisions.jsonl` 记录原因；之后自然回到原算法。
- 若出现以下信号，主动建议换算法（用 override + 理由）：
  - 目标方差极大、复孔差异 >10% → 增大 `alpha`，或减少批量、加复孔；
  - 连续多轮预测与实测差很多 → 检查空间边界/隐藏变量，考虑 `matern32` 或回到采样；
  - 分类变量占主导且组合数少 → 切 `grid` 穷举更直接。
- 不要每轮都换算法：重路由要有证据（停滞、异常、用户要求）。
- **多目标（ParEGO/qEHVI/qNEHVI）**：`report` 会给出 Pareto 前沿与 HV；不要只报“单一最优”。把前沿交给用户，
  让他按业务偏好选折中点；需要指定偏好时可把目标权重交给用户后再切单目标 `bayes`（新建 study）。
- **2 目标 + 硬约束（cqEHVI）**：`report` 的前沿与 HV 只基于**可行**观测；不可行点即使目标值更好也不进前沿。
  回填时缺 `constraints` 会被当作不可行，务必逐点补齐。

## 第 5.5 步：决策层口径（用默认值，也要知道为什么）

这些口径都有实测支撑（`docs/DECISIONS.md` 逐条断言，`research/EVIDENCE_POLICY.md` 说明证据规则）：

- **停止口径**：`prepare` 的 `stopping_advice` 会给出建议——预算 ≥60 时建议
  `init --budget-first`（跑满预算；实测最优值更高），预算小时保持默认（停滞即停）。若选 budget_first
  且交付证据是硬要求，注意 `delivery_tradeoff` 提示。
- **交付槽位**：预算里有一部分被保留给确认复孔，`effective_exploration_budget` 才是净探索次数
  （例：预算 12 → 保留 3、可探索 9）。开工时就把这个数告诉用户，别把交付成本当探索成本。
- **交付自查**：`report` / `digest` 里的「交付自查」逐条打勾（可行的最终保真观测、前沿+进展、
  推荐条件已复孔确认）。有 `[缺 ]` 就先补证据再交付，不要用缺证据的结果下结论。
- **极小预算**：低于建模下限时 router 会屏蔽模型类并给出说明；约束/多目标上下文没有便宜算法时
  保留代理 + 告警（可用 `algorithm=sobol` 显式覆盖）。不要在小预算上承诺“找到最优”。
- **迁移可信度**：`transfer-audit` 的建议值要用 `prepare` 的 `reliability_grade` 解读
  （≥0.8 trust / 0.5–0.8 suspect / <0.8 需谨慎；低置信度一律不作为结论）。实测中“调低 reliability”
  并不能救坏源，坏源就该少迁或不迁（`top_k` 1–3）。

## 第 6 步：收尾
- 先用 `step --json` 确认 `action=stop`，再用 `recommend --json` 输出最终推荐（条件 + 规格符合性 + 确认统计）；
  若 `action=confirm`，先按 `confirm_n` 做复孔并回填，再收尾；
- 单目标：输出最终最优条件（含单位）、目标值、总实验数、样本效率（找到最优用了多少次实验）；
- 多目标：输出 Pareto 前沿表（每行一组条件 + 各目标值 + HV），标注“端点解”与“折中解”；
- 给出 2–3 组“可选的验证条件”（最优/前沿附近 + 一组空间填充对照）；
- 归档：`run/` 目录保留 `dataset.jsonl / decisions.jsonl / suggestions.csv`，可复现；
- 需要看板时运行 `py -3 scripts/render.py --run <workspace>\run`（或 `py -3 scripts/run_condopt.py render --run ...`），
  生成离线 `dashboard.html` 与 `convergence.png`；
- 如用户需要，把本轮算法选择与效果写回 `docs/ALGORITHM_CATALOG.md` 或研究记录。

## 硬性规则

1. **不伪造实验数据**。没有实测值就不调用 `tell`。
2. **不越权**：只能从注册表选算法、只能改 schema 内参数；任何覆写必须有理由并留痕。
3. **不超预算**：ask 的点数不能超过 `--budget`；超了先和用户确认。
4. **不改空间**：一次 study 的 `space.json` 不可变；要改就新建 study，旧数据当先验导入。
5. **可解释**：每次算法选择都引用 router 的 `rationale`，不要只说“我觉得”。
6. **失败要报告**：命令报错、结果缺失、预算冲突，直接说清楚，不要静默重试或编造。

## 命令速查

| 目的 | 命令 |
|---|---|
| 答案模板 | `py -3 scripts/run_condopt.py prepare --template` |
| 生成问题文件 | `py -3 scripts/run_condopt.py prepare --answers answers.json --out workspace --json` |
| 算法目录/参数 schema | `py -3 scripts/run_condopt.py spec [--algorithm bayes] --json` |
| 路由推荐 | `py -3 scripts/run_condopt.py route --space S.json [--context C.json] --json` |
| 建 study | `py -3 scripts/run_condopt.py init --run R --space S.json --objective yield:max --budget N --batch B` |
| 导入历史数据 | `py -3 scripts/run_condopt.py import --run R --results prior.json --json` |
| 出建议 | `py -3 scripts/run_condopt.py ask --run R [--n B] [--plate P] --json` |
| 回结果 | `py -3 scripts/run_condopt.py tell --run R --results results.json --json` |
| 看进度 | `py -3 scripts/run_condopt.py report --run R [--json]` |
| 自动循环下一步 | `py -3 scripts/run_condopt.py step --run R --json` |
| 覆写（须理由） | `py -3 scripts/run_condopt.py override --run R --algorithm A --params P --reason "..."` |
| 恢复自动路由 | `py -3 scripts/run_condopt.py auto --run R` |
| 看板 | `py -3 scripts/run_condopt.py render --run R [--out DIR] [--no-png]` |

> Windows 中文输出乱码时，先设置 `$env:PYTHONUTF8 = "1"`。
> PowerShell 下 `--params '{"json":...}'` 的引号可能被吞：把参数写进 `params.json` 再传文件路径更稳。

## 自检（改技能前后跑一次）

```powershell
py -3 scripts/verify_skill_workflow.py
```

它按本文件描述的流程在临时目录真跑一遍（prepare → init → step → ask → tell → digest，
外加 spec 窗口分支），任何产物/输出缺失都会报 problem。发布门禁 `skill_install` 已包含它。

## 参考

- `references/PARAMETER_GUIDE.md`：每个参数的物理含义、建议范围、怎么从用户回答里推断。
- `references/ALGORITHM_CHOICE.md`：算法速查（适用/不适用/关键参数/失效模式）。
- `references/JSON_SCHEMAS.md`：space / context / results / problem 的字段说明与校验。
- 项目文档：`docs/ALGORITHM_CATALOG.md`、`docs/ROUTING_POLICY.md`、`docs/ui/`。
- 决策与状态：`docs/DECISIONS.md`（决策层规则矩阵，逐条断言）、`docs/STATUS.md`（当前交付形态与已知风险）、
  `research/EVIDENCE_POLICY.md`（默认值的证据规则：两数据集、负结果保留、测量卫生）。

## 目标窗口与决策层（v0.12）

真实实验常常不是“把某个指标最大化”，而是“命中若干规格窗口 + 满足硬约束”。

- 如果用户说的是“某指标要在 X 附近 / 落在 A–B 之间 / 不超过 C”，不要强行转成 max/min；
  在 `answers.json` 里用 `spec`：
  ```json
  {
    "variables": [...],
    "spec": {
      "goals": [
        {"metric": "yield", "mode": "target", "target": 80, "tolerance": 2, "scale": 5},
        {"metric": "purity", "mode": "range", "low": 98, "high": 100}
      ],
      "constraints": {"temp": {"op": "<=", "value": 110}}
    },
    "budget": 40, "batch": 4
  }
  ```
- `prepare` 会额外写出 `spec.json`，`init --spec` 会用它生成单目标 `spec_score`（加权和），
  约束进入 constrained/可行性处理。
- 每轮 `report` 后，用 `recommend --run ... --json` 拿“可执行推荐”：条件、逐项规格符合性、
  确认复孔统计、是否该停；自动循环直接用 `step --json`（把 recommend/confirm/should_stop 合并成一个 action）。
- 当推荐里 `should_stop=true` 或用户要收尾时：
  ```powershell
  py -3 scripts/run_condopt.py step --run <run> --json      # 先看 action（多半是 confirm）
  py -3 scripts/run_condopt.py recommend --run <run> --json
  py -3 scripts/run_condopt.py confirm --run <run> --n 3 --json
  ```
- 需要给人看的单页摘要时：`py -3 scripts/run_condopt.py digest --run <run>`（写 run/digest.md；`--format html` 输出 digest.html）。
- 只需推荐依据（不读整份 report）时：`py -3 scripts/run_condopt.py decision --run <run> --json`；
  需要理由链时加 `--explain`。
- 多目标 Pareto 收尾时，如需偏向某个目标：
  `py -3 scripts/run_condopt.py knee-weights --run <run> --weights yield=10,cost=1 --json`
  （写入 meta + 审计；只影响 Pareto knee 的 utopia 距离权重）。
  - 停滞阈值：默认 patience=3（实测最优）；patience<2 会被夹紧到 2（过早停止有害）。
- 停滞时优先调参：`py -3 scripts/run_condopt.py retune --run <run> --json`（换 acquisition/抬 kappa；噪声下更优，硬约束下可行点更多，强噪声+约束时持平）。
`py -3 scripts/run_condopt.py retune --run <run> --json`（换 acquisition/抬 kappa，实测优于原样继续与探索重启）。
- 停止语义：`step.json` 的 `diagnostics.stop_category`（budget/cost/target/max_rounds/stagnation）；停滞但仍有预算时看 `consider_exploration_restart` 建议。
- 无人值守模板：`py -3 scripts/unattended_campaign.py --problem <p> --budget N`
  （doctor → ask/tell → confirm → stop，可直接替换 oracle 为真实执行器）。
- 真实先验迁移示例：`py -3 examples/paper_campaigns/run_real_prior_transfer.py`
  （真实 ART 历史 → 代理战役；先测几条 → transfer-audit → transfer_from）。
- 迁移影响演练：`py -3 scripts/run_condopt.py transfer-impact --run <t> --from-run <s> --json`（副本上预览推荐变化，不改 live run）。
- 迁移可信度评估：`py -3 scripts/run_condopt.py transfer-audit --run <target> --from-run <source> --json`（目标需已有 >=3 条观测）→ suggested_reliability。
- 多保真成本：pending 项带 `fidelity`，回写 `cost`；成本预算是硬上限（ask/confirm 都会裁剪）。
- 执行器对接：`pending --run <dir> --json` 轮询待执行建议；`tell --results <file>` 回填；失败孔位跳过即可（详见 docs/EXECUTOR_INTEGRATION.md）。
- 研究证据索引：`research/README.md`（默认值来源与全部笔记）。
- `condopt prepare` 的 `feasibility` 字段给出开工前可行率预警（可用 `feasible_fraction_estimate` 显式给数）；risk=high 时先考虑把限制写进变量范围。
- 可行率很低（<5%）时：把硬限制写进变量范围或固定因子，不要靠约束采集硬找；`step` 的 `observed_feasible_fraction` 与建议 `encode_limits_as_bounds` 会自动提醒（实测 1/5 vs 5/5，见 `research/HARD_FEASIBILITY.md`）。
- 证据复核（发布前）：`py -3 scripts/verify_evidence.py --quick`（重跑关键基准，确认默认值依据没漂）。
- 开工前先看 `docs/PREPARE_BRIEF.md`（三段建议的实际样例与判据）。
- `condopt prepare` 的 `stopping` 字段会给出停止口径建议（预算 ≥60 建议 budget_first，理由与开启方式一并给出）；照着回填 `condopt init` 的 `--budget-first` / `--patience`。
- 需要跑满预算：`Study.create(..., stop_on_stagnation=False)`（默认停滞即停；实测 69 vs 120 次观测，见 `docs/DECISION_LAYER.md`）。
- 端到端流程：`condopt prepare`（含 deliverable 交付约定）→ `condopt init`
  → `condopt step`（缺证据时给 complete_delivery_evidence）→ `condopt digest`
  （含交付自查，等价于 `Study.delivery_checklist()`）。
- 真实实验接入：实现 `condopt.executor.Executor.measure`，用 `run_campaign` 驱动（见 `docs/EXECUTOR_ADAPTER.md`）。
- 默认值依据：`docs/DEFAULTS.md`（每个默认值对应哪次实测）。
- 负结果参考：`research/NEGATIVE_RESULTS.md`（为什么某些算法不是默认）。
- 发布前自检：`py -3 scripts/release_check.py`（单测/评估/审计/skill/计数 6 项）。
- run 目录自检（无人值守前/后）：`py -3 scripts/run_condopt.py doctor --run <run> --json`。
- 预测新条件（不花实验）：`py -3 scripts/run_condopt.py predict --run <run> --config '{"T": 80}' --json` → mean/std/P(feasible)/spec。
- 全算法体检：`py -3 scripts/audit_algorithms.py`（能力匹配基准上逐个跑 ask/tell）。
- 审计 AI/人工改动：`py -3 scripts/run_condopt.py decisions --run <run> --action override`。
- 多保真 study 的 `template` 会带 `fidelity` 列：实验室按该档位做，回填时原样保留。
- 导出 Pareto 前沿（多目标，含 x/obj/con 列）：`condopt pareto --run <run>` → run/pareto_front.csv。
- 导出全部观测（交接/存档）：`py -3 scripts/run_condopt.py export --run <run>` → run/dataset_export.csv。
- 生成实测回填模板：`py -3 scripts/run_condopt.py template --run <run>` → run/results_template.csv，
  实验室填完直接 `tell --results`。
- 实测结果也支持 CSV：列名用 `id` / `y` / `fidelity` / `plate` / `well`，
  条件列用 - 高基数分类（>8 类）建议在 space 的变量里配置 `descriptors`（每个 choice 的数值描述符，可只覆盖部分 choice），GP 就不再依赖 one-hot。
`x.<name>`，目标列 `obj.<name>`，约束列 `con.<name>`（无前缀时按条件变量解析）。
`confirm` 会复制当前最优条件做 3 次复孔；回填后 `recommend` 的分数优先用复孔均值，
  避免把“幸运的一次测量”当结论。`confirm` 也会占用 `budget`，预算不足时 `step` 会拒绝并直接 `stop`。
- 多目标真 Pareto 已由 `parego`/`qehvi`/`qnehvi` 提供，**2 目标 + 硬约束**由 `cqehvi` 提供
  （可行域内 EHVI，先 PoF 再优化）；`spec` 仍是“加权标量化 + 约束”，两者不要混用：
  要真 Pareto 就用 `objectives`，要命中规格窗口就用 `spec`。
- `spec.aggregation` 可选 `weighted_sum`（默认）或 `worst_case`（短板主导）。
- 有历史 study（同一空间/目标）时先 warm start：
  `py -3 scripts/run_condopt.py import --run <target> --from-run <source> --top-k 10 --reason ...`
  会严格校验空间/目标一致，观测计入预算，并写 `transfer_import` 决策。
- 初始 `tolerance`/`scale` 不确定时：`py -3 scripts/run_condopt.py calibrate-spec --run <run> --json`
  用历史观测给出稳健建议（1.4826×MAD；target 默认 2σ 带）；确认后 `init --spec` 开新 study，
  不会自动改当前 study。也可在 `prepare` 的 answers 里加 `"spec_calibrate_from": "<old-run>"`，
  直接生成校准后的 spec.json + spec_calibration.json。
  有孔板信息时用 `calibrate-spec --by-plate`：按板内尺度校准（忽略板间偏移）。
  若 `step` 建议 `review_spec`：用 `suggest-spec --run <run>` 生成可达 spec 建议
  （target 越界会移到观测边界，并给 changed 列表），确认后再开新 study。
  也可一条命令：`condopt fork --run <old> --new-run <new> --spec-mode suggested --top-k 10`，
  自动继承 space/context/budget 并 warm start 旧观测；`--spec-file <spec.json>` 可直接用外部建议 spec；`--by-plate` 按板内尺度校准。

