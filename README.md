# condopt —— 条件优化算法库与算法路由器

> **定位：AI 全自动实验室（self-driving lab）的前置必要技术。**
>
> 自动实验平台的瓶颈不是「能不能做实验」，而是「下一批做什么条件」。condopt 提供一套内置、参数可灵活调节的条件优化算法，并用一个可解释的 router 把「问题上下文」映射到「算法 + 参数」，最终以 Agent Skill 的形态让 AI 稳定地驱动 ask→实验→tell 闭环。

## 1. 为什么需要它

真实的实验条件空间是混合的：温度、时间、浓度（跨数量级）、溶剂（分类）、配比（整数）、惰性气氛（布尔）。目标函数昂贵、带噪声、常常只能并行做 4–24 个孔位。常见现状是：

- 算法又多又杂（贝叶斯优化、随机取点、网格、遗传算法……），没有统一接口，换个算法就要重写数据管道；
- 每个算法参数一堆（核函数、采集函数、批量策略、初始化点数），实验人员/AI 不知道怎么选；
- AI 直接「凭感觉」选算法，无法解释、无法审计、无法复现；
- 实验数据散落在 Excel 里，无法形成闭环。

condopt 把这三件事拆成三层：

| 层 | 内容 | 谁负责 |
|---|---|---|
| **硬算法层** | 统一的 `ask/tell` 优化器、混合搜索空间、GP-BO、空间填充采样、注册表与参数 schema | 代码，确定性行为 |
| **路由器层** | 把「维度/类型/预算/批量/噪声/光滑度/先验数据」映射为「算法 + 参数 + 理由 + 备选」，并支持中途重路由 | 一组透明规则，可解释 |
| **Agent/Skill 层** | 先问必要参数并给建议范围；调用 router；生成待测条件表；回收实测结果；异常时覆写并留痕 | AI + 人（HITL） |

核心原则：**算法是硬的，参数是活的，AI 的覆写必须留痕。** AI 只能从注册表里选算法、只能在 schema 里改参数；每一次路由和覆写都写入 `decisions.jsonl`。

### 1.1 在自主实验室中的位置（L3 决策内核）

自主实验室可以拆成四层，condopt 只做其中**唯一与领域无关、但决定样本效率的一层**：

```
L4 语义/规划层   LLM agent：理解目标、拆任务、调工具、解释结果        → ChemCrow / LLAMBO
L3 条件优化层   「下一次做什么实验」的决策内核                      → condopt（本项目）
L2 编排/执行层   实验请求 → 机器人/人工执行 → 测量 → 结果回写        → ChemOS / A-Lab / 自驱动实验室
L1 数据/记录层   实验记录、谱图、元数据、审计日志
```

- **样本效率是唯一有意义的指标**：A-Lab 在 17 天内自主合成 58 个目标化合物，靠的是“先验 + 主动学习规划下一批 + 机器人执行”的闭环，而不是更复杂的采集函数（Szymanski et al., *Nature* 624, 86–91, 2023，[DOI](https://doi.org/10.1038/s41586-023-06734-w)）。condopt 因此以“每目标所需实验数”验收，而不是算法跑分。
- **planner / executor 解耦**：ChemOS 用 RPC 把优化器与硬件执行解耦（Roch et al., *Science Robotics* 3, eaat5559, 2018，[DOI](https://doi.org/10.1126/scirobotics.aat5559)）；condopt 只暴露 `ask/tell` 与 JSON 请求，任何执行器都能接。
- **LLM 负责语义，数值决策交给算法**：ChemCrow 让 LLM 规划工具调用、由工具执行具体参数（Bran et al., *Nature Machine Intelligence* 6, 525–535, 2024，[DOI](https://doi.org/10.1038/s42256-024-00832-8)）；condopt 的 router 可以让 LLM 选算法/解释结果，但**不允许** LLM 直接决定数值采样点。
- **边界**：condopt 不碰硬件、不做安全判断、不做语义规划；它回答“下一批条件是什么、为什么、不确定度多大”，并把每次决策写成可审计记录。

## 2. 架构

```
                  回答问卷（带建议范围）               HITL 确认/覆写(必须给理由)
  实验人员 / AI ───────────────────────► skill ───────────────────────────┐
                                                │                          │
                                                ▼                          ▼
                                        condopt route ──► Recommendation（算法+参数+理由+备选）
                                                │
                        ┌───────────────────────┴────────────────────────┐
                        ▼                                                ▼
                 Study（run 目录）                                 decisions.jsonl
   problem.json / config.json / dataset.jsonl / rounds.json / pending.json / suggestions.csv
                        │
          ask(n) ───────┴──────────────────────────────────────────────► tell(results)
                        │                                                │
                        ▼                                                ▼
                 Optimizer (ask/tell)  ◄──── 实测结果（人工/机器人/仿真，禁止伪造）
                 random / sobol / lhs / grid / bayes(GP+EI/UCB/PI/TS) / tpe
                        │
                        └── 连续 N 轮无改进 → router 自动插入全局探索批次（auto_route）
```

## 3. 已实现（v0.1.0）

- **搜索空间** `condopt.space`：连续 / 整数 / 对数（跨数量级浓度）/ 分类 / 有序 / 布尔；单位立方体编码 + 特征编码（数值归一化 + 分类 one-hot，可用 `descriptors` 换成物理描述符，如 MW/logP/介电常数；描述符按维度 min-max 归一化后再进 GP；允许只覆盖部分 choice，缺失项用零向量 + 一维 unknown 指示位区分）。
- **观测与数据集** `condopt.dataset`：单目标/多目标字段、约束字段、保真度、成本、来源；JSONL 追加式持久化。
- **优化器接口** `condopt.base`：统一 `ask(n)` / `tell(observations)`；每个算法声明 `AlgorithmSpec`（参数 schema + 能力 + 参考文献）。
- **算法**：
  - `random` 随机搜索（高维/极小预算基线）
  - `sobol` 低差异序列（空间填充、批量初始设计）
  - `lhs` 拉丁超立方（一次性批量分层抽样）
  - `grid` 网格/析因设计（少量离散条件穷举）
  - `bayes` GP 贝叶斯优化：Matérn/RBF 核、EI/log-EI/UCB/PI/Thompson 采样、Kriging-believer / constant-liar 批量策略、混合空间特征
  - `cmaes` 简化 rank-μ ES（**无 evolution paths**，非完整 CMA-ES）：单位立方体上的协方差自适应策略，中高维连续光滑问题替代 GP（实现边界见 `docs/ALGORITHM_CATALOG.md` §14）
  - `tpe` **独立逐维** Parzen 估计（l/g 核密度比；**不含 tree/条件密度**）：支持分类/混合/高特征维空间，小样本更稳（命名边界见 `docs/ALGORITHM_CATALOG.md` §14）
  - `qehvi_nd` 3–4 目标 MC-EHVI：逐目标 GP 后验采样 + 递归切片 HV（可选，默认仍 parego）
  - `parego` 多目标贝叶斯优化：每轮随机 Dirichlet 权重 + 增广 Tchebycheff 标量化，复用 GP+EI，产出 Pareto 前沿
  - `constrained` 约束贝叶斯优化：先最大化可行性概率找到安全点，再做带可行性的期望改进（MC 估计 EI_c）
  - `qehvi` 2 目标 **单点候选 EHVI + 局部惩罚批量**（非联合 q 点 EHVI）：分区公式算超体积改进（MC 估计），预算充足/低噪声时前沿覆盖优于 ParEGO
  - `cqnehvi` 2 目标 + 约束 + 带噪：嵌套 MC EHVI × P(feasible)，中高噪声下的约束多目标
  - `qnehvi` 2 目标带噪 EHVI：对 Pareto 前沿做嵌套 MC，但 baseline 由**带噪观测**生成（尚未走模型后验），`objective_std` 为单值；噪声 medium/high 时的默认选择（边界见 `docs/ALGORITHM_CATALOG.md` §14）
  - `cqehvi` 2 目标 + 硬约束：可行域内 EHVI（`E[DHV·1{feasible}]`），无可行点时先最大化 PoF 找安全条件；条件必须全部回填 `constraints`
  - `mfbo` 多保真 BO：fidelity 作为 GP 输入 + `EI/cost^α`，先低成本粗筛再全保真确认；含硬约束时乘 P(feasible)
  - `momfbo` 多目标 + 多保真：随机标量化（增广 Tchebycheff）+ fidelity 维 GP + `EI/cost^α`，支持硬约束
- **噪声/复孔**：`Observation` 支持 `objective_std` + `n_replicates`，GP 的 `alpha` 按 `(σ²/k)/y_var` 逐点估计；`results.json` 可用 `std`/`k` 回填。
- **路由器** `condopt.router`：透明打分规则 + 能力过滤；输出算法、参数、理由、备选、风险、初始设计切分与阶段计划；支持停滞检测与重路由。
- **Study** `condopt.study`：一个 run 目录即一次实验战役，跨进程可恢复；所有建议、结果、决策可审计；支持导入历史数据暖启动。
- **孔板/批次**：`ask` 给每条建议自动打 `plate`（默认 P1、P2…，可 `--plate` 续测同一块板）+ `well`（A1…H12）+ `batch`，写入 `pending.json`/`suggestions.csv` 并在 `tell` 时落到观测 `meta`；结果里可用 `plate/well/batch` 覆写（实际换板/重排孔位）。`summary`/`report` 给出每块板的 n/均值/极差，并用单因素 ANOVA 做**批次效应检查**（板间差异显著且相对 ≥2% 时警告，提示同批比较或把 plate 当随机效应/特征）。
- **命令行**：`spec / route / prepare / init / ask / tell / pending / import / transfer-staged / template / export / pareto / decisions / retune / doctor / predict / transfer-audit / transfer-impact / report / step / recommend / confirm / calibrate-spec / suggest-spec / knee-weights / fork / digest / override / auto / render`，支持 `--json`；`doctor` 自检 run 目录一致性（ID 唯一、pending/dataset 不重叠、opt_state 与算法一致、JSON 可解析）。
- **实测数据交接**：`template` 生成 `results_template.csv`（id/plate/well/x.*/y|obj.*/con.*），实验室填完 `tell --results` 回填；CSV/TSV/分号表自动嗅探，`export` 导出 `dataset_export.csv`；`digest --format html` 输出单页 HTML 摘要。
- **多目标**：`Objective` 支持逐目标方向（max/min）+ `ParEGO`/`qEHVI`/`qNEHVI`/`cqEHVI` 优化器；`Study.report` 输出非支配前沿与 2–4 目标超体积（HV，3+ 用递归切片）；2 目标 + 硬约束走 `cqEHVI`（可行域内前沿）。
- **参数准备** `condopt.prepare`：把 skill 问卷的一份 `answers.json` 校验并生成 `space.json/context.json/objective.json`，同时返回 router 推荐。
- **算法细节（Sprint 2）**：GP 支持 ARD（每特征 length_scale + 维度缩放先验，小样本冻结超参）；批量采集默认 local penalization（不伪造数据）；TS 批量每点独立后验抽样；初始设计用平衡的 2^k Sobol 设计。
- **可视化** `condopt.render` + `scripts/render.py`：只读 run 目录生成单文件离线 `dashboard.html`（内联 SVG：收敛曲线、空间覆盖投影、建议批次表、router 决策/覆写理由、预算进度；P2 起含 Pareto 前沿与参数重要性）+ `convergence.png`；无 CDN、无预测造假。`scripts/serve.py` 提供 127.0.0.1 只读本地服务。
- **Skill 入库**：`condopt-router` 已安装进 `<agent-library>/catalog`（`限本机`），
  `agentlib-router` 可把“帮我用贝叶斯优化找最优反应温度”这类话路由到它；`scripts/install_skill.py` 可幂等刷新。
- **基准** `condopt.benchmarks` + `scripts/benchmark.py`：Tier A–F 问题（Branin/Hartmann/6 维混合反应/Suzuki 风格/高基数溶剂/离散催化剂/2 目标收率-成本/2 目标+双硬约束 `moo_constrained`/3 目标+约束 `moo3_constrained`/多保真+约束 `mf_constrained`/多目标+多保真 `mf_moo`），多次种子对比算法与基线；`benchmark_mf_constrained.py`、`benchmark_momfbo.py` 给出成本感知对比。
- **测试**：585 个单测覆盖编码、采样、每类优化器（含 momfbo）、BO/TPE/ParEGO/约束 BO/2D EHVI/qNEHVI/cqEHVI（2 目标 + 硬约束）/多保真收敛、Pareto/HV、路由规则与校准权重（含 LOPO 折外验证、能力过滤、tie 口径、context 覆盖）、Study 生命周期（含部分 tell / 幂等 tell / 覆写持久 / 显式算法钉住 / 成本记账 / 剩余成本 / 成本预算降档与一等参数 / 全保真 best / 约束 / 复孔 / fidelity / 孔板-批次-板间效应检验 / 自动循环 action 决策）、benchmark 自检（含 mf_suzuki 成本感知、moo_constrained 可行 HV 与约束向量）、参数准备、校准 harness、看板/只读服务与编码卫生。

## 4. 快速开始

无需安装，直接用源码启动（Windows）：

```powershell
cd <repo>
$env:PYTHONUTF8 = "1"

# 1) 看算法目录与参数 schema
py -3 scripts/run_condopt.py spec --algorithm bayes

# 2) 写一个搜索空间 space.json 后，问 router 该用哪个算法
py -3 scripts/run_condopt.py route --space configs\space.example.json --budget 24 --batch 4 --smoothness smooth --json

# 2b) 或者：把 skill 问卷答案写成 answers.json，一次生成 space/context/objective 并得到路由
py -3 scripts/run_condopt.py prepare --answers answers.json --out workspace --json

# 3) 建 study（algorithm=auto 即交给 router）
py -3 scripts/run_condopt.py init --run runs\demo --space configs\space.example.json `
    --objective yield:max --budget 24 --batch 4

# 3b) 有历史实验数据时，先作为先验导入（计入预算、可暖启动）
py -3 scripts/run_condopt.py import --run runs\demo --results prior.json

# 4) 生成待测条件（写入 runs\demo\suggestions.csv；自动带 plate/well/batch）
py -3 scripts/run_condopt.py ask --run runs\demo
#    续测同一块孔板时显式指定板号（默认每轮新开一块板）
py -3 scripts/run_condopt.py ask --run runs\demo --plate P1

# 5) 做完实验后回收结果（results.json: [{"id":"r1-1","y":78.3}, ...]）
py -3 scripts/run_condopt.py tell --run runs\demo --results results.json
#    或走实验室交接闭环：先生成回填模板，填完 CSV 再 tell（CSV/TSV 自动识别）
py -3 scripts/run_condopt.py template --run runs\demo
#    py -3 scripts/run_condopt.py tell --run runs\demo --results results_template.csv

# 6) 查看进度与下一轮建议
py -3 scripts/run_condopt.py report --run runs\demo

# 6b) 自动循环：让引擎给出下一步（tell / confirm / ask / stop，含 next_command）
py -3 scripts/run_condopt.py step --run runs\demo --json

# 6c) 导出/审计：Pareto 前沿 CSV、全部观测 CSV、单页 HTML 摘要、决策审计
py -3 scripts/run_condopt.py pareto --run runs\demo --json
py -3 scripts/run_condopt.py export --run runs\demo
py -3 scripts/run_condopt.py digest --run runs\demo --format html
py -3 scripts/run_condopt.py decisions --run runs\demo --limit 10

# 7) 生成离线看板（dashboard.html + convergence.png）
py -3 scripts/render.py --run runs\demo
# 等价：py -3 scripts/run_condopt.py render --run runs\demo
```

覆写与恢复自动路由（AI/人必须留痕）：

```powershell
py -3 scripts/run_condopt.py override --run runs\demo --algorithm tpe --reason "分类变量主导，改用 TPE"
py -3 scripts/run_condopt.py auto --run runs\demo
```

合成数据端到端演示（6 维混合空间、24 次实验、批量 4）：

```powershell
$env:PYTHONUTF8 = "1"
py -3 scripts/demo.py
```

多保真 + 成本回写示例：`py -3 examples/multifidelity_executor.py`。
长跑稳定性演练：`py -3 scripts/long_run_drill.py --budget 120 --force-budget`（停滞停止后可选择继续探索）。

执行器接入指南见 [`docs/EXECUTOR_INTEGRATION.md`](docs/EXECUTOR_INTEGRATION.md)（pending → 执行 → tell 三个契约点 + 最小 Python 模板）；
想直接跑起来用适配器模板 [`docs/EXECUTOR_ADAPTER.md`](docs/EXECUTOR_ADAPTER.md)：实现 `Executor.measure` 后调用 `run_campaign(study, executor)`，
失败孔位、成本记账与 `execution_log.jsonl` 都由它处理（示例：`py -3 examples/executor_adapter.py`）。
分阶段迁移（先小 top_k，跑两批后按审计决定是否放大）：

```powershell
py -3 scripts/run_condopt.py transfer-staged --run work/run --from-run work/source_run --top-k 5
```

实测在有偏源上这样做 +0.71，而固定 top_k=3/5 分别是 −1.54/−0.33（`research/TRANSFER_ADAPTIVE.md`）。

多目标 + 约束的适配器示例：`py -3 examples/multiobjective_executor.py`（cqehvi，23 次观测 / 18 条可行 / Pareto 前沿 4 点；HV 143.03，可交付 HV 80.94）。

无人值守循环模板（doctor → step 状态机 → confirm → stop → doctor；支持断点续跑）：

```powershell
py -3 scripts/unattended_campaign.py --problem sphere2d --budget 16 --batch 4
# 任意时刻中断后重跑同一命令即可继续：pending 会先被回填，dataset 不会重复
```

一键质量闸门（单测 + 路由评估 + 算法/路由审计 + skill 安装 + README 计数一致性）：

```powershell
py -3 scripts/release_check.py        # 全部门禁通过才发布（项数与明细见 docs/STATUS.md）
```

算法基准（多种子、多个合成问题）：

```powershell
py -3 scripts/benchmark.py --budget 20 --batch 2
```

约束多目标基准（`moo_constrained`：可行率 + 可行前沿 HV，cqehvi vs 无约束基线）：

```powershell
py -3 scripts/benchmark_cqehvi.py --budget 40 --batch 4 --seeds 0 1 2
```

论文战役复现示例（2026 自动化实验室论文 → 复杂空间取点）：[examples/paper_campaigns/README.md](examples/paper_campaigns/README.md)

文献扫描管线（可复用的“论文 → router 经验”）：[docs/LITERATURE_PIPELINE.md](docs/LITERATURE_PIPELINE.md)

决策层（目标窗口/规格、停止规则、确认复孔、可执行推荐）：[docs/DECISION_LAYER.md](docs/DECISION_LAYER.md)



路由权重校准（小矩阵 MVP；输出 `runs/calibration/weights.json`，可用 `route --weights` 回放）：

```powershell
py -3 scripts/calibrate_router.py --budget 8 --iterations 60

校准输出会先打印 `metric coverage`（`true_best` / `true_hv` / `feasible_best`），再给 in-sample 与 **LOPO 折外**指标；只有折外指标可以作为“校准有效”的证据。
混合类型矩阵已能产出四类 metric；扩到 **180 runs（12 问题×8 算法×4 seeds）** 后 **LOPO top1 0.75 vs 默认 0.167**、top2 0.833、regret 0.0015 vs 8.15 → 候选权重存于 `configs/router_weights_provisional.json`（**不设为默认**，仍需 488-runs 复核并在 decisions 记录来源）。

```

测试：

```powershell
py -3 -m pytest -q
```

## 5. 路由示例

| 问题特征 | 推荐 | 主要理由 |
|---|---|---|
| 5 维连续、光滑、预算 40 | `bayes` (EI) | 样本效率最高，GP 在该维度可靠 |
| 25 维、预算 40 | `sobol` / `random` | GP 代理失效，均匀覆盖更稳 |
| 连续光滑、≥8 维、预算 ≥40 | **`cmaes`** | 协方差自适应比 GP 代理更可靠（非光滑/离散不适用） |
| 10 维 Rosenbrock/Ackley（budget 80） | **`cmaes`** | 实测 -14.4 / -2.13 vs bayes -215.0 / -5.04（sigma0=0.1, c_mu=0.5） |
| 预算 5 次 | `sobol` / `lhs` | 任何模型都来不及学习 |
| 全离散、组合数 12、预算 24 | `grid` | 可完整枚举，比建模更可靠 |
| 4 维、批量 8、噪声高 | `bayes` (UCB, constant-liar) | 大噪声需要代理复用信息，批量需防扎堆 |
| 2–3 个目标（max 收率 / min 成本） | `parego` | 输出 Pareto 前沿与 HV，不再压成单一目标 |
| 有安全/可行性约束（温度上限、溶解度） | `constrained` | 先找可行点，再做 EI_c；不要事后过滤 |
| 2 目标 + 硬约束 + 中高噪声 | `cqehvi` | 实测带噪嵌套 MC（cqnehvi）未超过 cqehvi；cqnehvi 可手动指定 |
| 2 目标 + 硬约束（收率 vs 成本 + 温度/时间上限） | `cqehvi` | 可行域内 EHVI：先 PoF 找可行条件，再只覆盖可行前沿；`objective.constraints` 必须回填 |
| 2 目标、预算 ≥40、低噪声 | `qEHVI` | 直接优化超体积，前沿覆盖更好 |
| 2 目标、预算 ≥40、噪声中高 | `qnehvi` | 对前沿不确定性做嵌套 MC，避免追噪声 |
| 2 目标、预算 <40 | `parego` | 标量化在小预算下更稳 |
| 有低成本粗筛（短时试跑/低精度仿真） | `mfbo` | 直接全保真跑会浪费预算 |
| 多保真 + 硬约束 | `mfbo` | `EI × P(feasible) / cost^α` |
| 多保真 + 2+ 目标（±约束） | `momfbo` | 随机标量化 + fidelity 维 GP |
| 3+ 目标 + 硬约束 | `parego` | cqehvi/qehvi 仅支持 2 目标 |
| 3 目标、预算 ≥40、低/中噪声 | `qehvi_nd` | MC-EHVI + PI 预筛；实测 HV 5797 vs parego 5578 |
| 4 目标、预算 ≥40、低/中噪声 | `qehvi_nd` | 实测可行 HV 91244 vs parego 90134，单次约 20 s（增量 HV 贡献） |
| 连续 3 轮无改进 | 插入 `sobol` 探索批次 | 跳出代理模型的错误吸引域，再回到 BO |

## 6. 与可视化 / 自动化平台的集成

- 所有状态都在 `run/` 目录、所有命令支持 `--json`，因此 UI 或机器人调度只需读 `pending.json`、写 `results.json`。
- **已落地看板**：`py -3 scripts/run_condopt.py render --run <run>` 生成离线 `dashboard.html`（收敛曲线 / 空间覆盖 / 建议批次表 / router 决策与覆写理由 / 预算进度）+ `convergence.png`，无 CDN、无预测造假。
- 可视化可行性、41 条固定输入线、AI 机动性与硬算法整合的设计见 [`docs/ui/`](docs/ui/)；后续可加只读本地服务与 Pareto 视图。

## 7. 路线图

- **算法补全**：`cqEHVI`（2 目标 + 硬约束）、`mfbo`（多保真 + 约束）、`momfbo`（多目标 + 多保真）已内置；剩余 3+ 目标 EHVI 递归分区、CMA-ES/GA、跨底物迁移。
- **路由器升级**：按 `research/ROUTER_CALIBRATION.md` 把规则重构为 ScoreTerm 并用 LOPO-CV 校准（Tier C 暴露了“小预算 + 高基数分类时随机基线仍强”的问题）。
- **已知 P0（来自 `research/REVIEW_v03.md`）**：ParEGO 归一化窗口已冻结、HV 参考点已支持显式 `ref_point`、缺目标观测已容错；`qEHVI`/`cqEHVI` 的 `ref_point` 仍建议在 `objective.json` 显式给出（跨 run 可比的 HV 口径）。
- **基准套件**：Tier A–F 全量进 CI，接入 Olympus 风格噪声 emulator 与公开反应数据。
- **Skill 化与入库**：已完成 `condopt-router` 入库 + 幂等刷新；补 eval 用例与多目标/约束话术。
- **UI**：P0/P1/P2 已落地（静态看板 + Pareto/重要性 + 只读服务）。

## 8. 目录

```
src/condopt/          核心库（space / dataset / base / registry / router / study / cli / prepare / pareto / render / algorithms / benchmarks）
skill/                Agent Skill：condopt-router（先问参数、再路由、驱动 ask/tell 闭环）+ 参考
docs/                 ARCHITECTURE.md / ALGORITHM_CATALOG.md / ROUTING_POLICY.md
docs/ui/              可视化可行性、41 条固定输入线、AI×硬算法整合、原型与已实现看板
research/             持续调研：REFERENCES.md、算法地图、Benchmark 规格、ParEGO/约束噪声规格、GitHub 扫描、程序架构、自主实验室定位
scripts/              免安装启动器、端到端 demo、基准、看板、skill 安装
tests/                单测（585）
configs/              示例搜索空间与上下文
```

文档索引：

- [架构（ARCHITECTURE）](docs/ARCHITECTURE.md)
- [算法目录（ALGORITHM_CATALOG）](docs/ALGORITHM_CATALOG.md)
- [路由策略（ROUTING_POLICY）](docs/ROUTING_POLICY.md)
- [可视化可行性（docs/ui）](docs/ui/FEASIBILITY.md)
- [参考文献（research/REFERENCES）](research/REFERENCES.md)

## 9. 参考文献与研究交付

`research/REFERENCES.md` 维护编号参考文献（[R1]–[R38]，论文 + 开源项目），`docs/ALGORITHM_CATALOG.md` 给出算法与参数的工程化目录。关键来源：

- Jones, Schonlau & Welch (1998), *Efficient Global Optimization of Expensive Black-Box Functions* —— EI/EGO。
- Srinivas et al. (2010), *Gaussian Process Optimization in the Bandit Setting* —— GP-UCB。
- Bergstra & Bengio (2012), *Random Search for Hyper-Parameter Optimization* —— 随机搜索是强基线。
- Knowles (2006), *ParEGO* —— 多目标随机标量化（本项目 `parego` 的实现依据）。
- Daulton, Balandat & Bakshy (2021), *qNEHVI*, NeurIPS —— 噪声多目标/批量采集函数（本项目 `qnehvi`）。
- Gardner et al. (2014), *Bayesian Optimization with Inequality Constraints* —— EI-CDF（本项目 `constrained` / `cqehvi` 的可行性概率）。
- Letham et al. (2019), *Constrained Bayesian Optimization with Noisy Experiments* —— 带噪约束 EI（本项目 `constrained` / `cqehvi` 的 MC 形式）。
- Szymanski et al. (2023), *A-Lab*, Nature —— 自主实验室闭环（本项目的定位来源）。
- Balandat et al. (2020), *BoTorch* —— 批量/蒙特卡洛采集函数的工程架构。
- Akiba et al. (2019), *Optuna* —— TPE 与 ask/tell 式研究管理。

按主题的研究交付（均可直接落地）：

| 主题 | 文件 |
|---|---|
| 参考文献（R1–R38，逐条核实） | [`research/REFERENCES.md`](research/REFERENCES.md) |
| 算法地图（BO/TPE/CMA-ES/多目标/批量/约束/多保真） | [`research/ALGORITHM_SURVEY.md`](research/ALGORITHM_SURVEY.md) |
| 实现级 Review（6×P0 / 12×P1 / 12×P2） | [`research/REVIEW_v01.md`](research/REVIEW_v01.md) |
| 基准规格（Tier A–F + router 校准方法） | [`research/BENCHMARK_SPEC.md`](research/BENCHMARK_SPEC.md) |
| Tier C/D 代码级定义与最优值核验 | [`research/BENCHMARK_TIER_CD.md`](research/BENCHMARK_TIER_CD.md) |
| ParEGO / qEHVI / qNEHVI / noisy EI 公式 | [`research/MULTI_OBJECTIVE_AND_NOISE.md`](research/MULTI_OBJECTIVE_AND_NOISE.md)、[`research/PAREGO_SPEC.md`](research/PAREGO_SPEC.md) |
| 约束/噪声实现（Letham 式、复孔 alpha） | [`research/CONSTRAINTS_NOISE_IMPL.md`](research/CONSTRAINTS_NOISE_IMPL.md)、[`research/CONSTRAINT_IMPL.md`](research/CONSTRAINT_IMPL.md) |
| 多目标/多保真/校准设计 | [`research/qEHVI_SPEC.md`](research/qEHVI_SPEC.md)、[`research/qEHVI_IMPL.md`](research/qEHVI_IMPL.md)、[`research/qNEHVI_SPEC_V2.md`](research/qNEHVI_SPEC_V2.md)、[`research/EHVI_3D.md`](research/EHVI_3D.md)、[`research/MULTIFIDELITY_SPEC.md`](research/MULTIFIDELITY_SPEC.md)、[`research/MULTIFIDELITY_MVP.md`](research/MULTIFIDELITY_MVP.md)、[`research/MF_BENCHMARK.md`](research/MF_BENCHMARK.md)、[`research/ROUTER_CALIBRATION.md`](research/ROUTER_CALIBRATION.md)、[`research/ROUTER_SCORETERM.md`](research/ROUTER_SCORETERM.md)、[`research/ROUTER_CALIBRATION_HARNESS.md`](research/ROUTER_CALIBRATION_HARNESS.md) |
| 复孔/噪声 alpha 映射 | [`research/REPLICATES_ALPHA.md`](research/REPLICATES_ALPHA.md) |
| 实现 review（P0/P1/P2） | [`research/REVIEW_v03.md`](research/REVIEW_v03.md)、[`research/REVIEW_v04.md`](research/REVIEW_v04.md)、[`research/REVIEW_v05.md`](research/REVIEW_v05.md) |
| Sprint 2（ARD/local penalization/TS 批量/Sobol 2^k） | [`research/SPRINT2_SPEC.md`](research/SPRINT2_SPEC.md) |
| 程序架构模式（ask/tell、注册表、持久化） | [`research/ARCHITECTURE_PATTERNS.md`](research/ARCHITECTURE_PATTERNS.md) |
| GitHub 开源项目扫描（18 个仓库，license 核实） | [`research/GITHUB_SCAN.md`](research/GITHUB_SCAN.md) |
| 自主实验室定位（README 可引用结论） | [`research/AUTONOMOUS_LAB_POSITION.md`](research/AUTONOMOUS_LAB_POSITION.md) |
| 真实测量数据留一验证（ART 6 点，含负结果） | [`research/REAL_DATA_LOO.md`](research/REAL_DATA_LOO.md) |
| 分类描述符特征验证（60 类别合成受控） | [`research/DESCRIPTOR_FEATURES.md`](research/DESCRIPTOR_FEATURES.md) |
| 描述符质量敏感性（含垃圾描述符对照） | [`research/DESCRIPTOR_FEATURES.md`](research/DESCRIPTOR_FEATURES.md) |
| 适配器长跑演练（120 次预算，3 组对照） | [`research/ADAPTER_LONG_RUN.md`](research/ADAPTER_LONG_RUN.md) |
| 证据声明复核（重跑基准，6 条约 50 s，已入门禁） | [`research/EVIDENCE_AUDIT.md`](research/EVIDENCE_AUDIT.md) |
| 真实场景 → 默认算法（19 个可执行场景） | [`docs/SCENARIOS.md`](docs/SCENARIOS.md) |
| 多保真多目标（含约束）对比 | [`docs/ALGORITHM_CATALOG.md`](docs/ALGORITHM_CATALOG.md) |
| 暖启动收益（transfer_from，等预算） | [`research/TRANSFER_BENCHMARK.md`](research/TRANSFER_BENCHMARK.md) |
| Router 校准复核（新基准 + LOPO） | [`research/ROUTER_CALIBRATION_NEW_BENCHMARKS.md`](research/ROUTER_CALIBRATION_NEW_BENCHMARKS.md) |
| 默认值来源总表（每个默认值的实测依据） | [`docs/DEFAULTS.md`](docs/DEFAULTS.md) |
| 开工简报（route/stopping/feasibility 三段建议，自动生成） | [`docs/PREPARE_BRIEF.md`](docs/PREPARE_BRIEF.md) |
| 项目状态快照（交付形态 / 门禁 / 已知风险，自动生成） | [`docs/STATUS.md`](docs/STATUS.md) |
| 研究笔记索引（全部证据文档） | [`research/README.md`](research/README.md) |
| 负结果清单（试过但没赢的方案） | [`research/NEGATIVE_RESULTS.md`](research/NEGATIVE_RESULTS.md) |
| 真实先验 → 代理战役（ART 六条真实点） | [`research/REAL_PRIOR_TRANSFER.md`](research/REAL_PRIOR_TRANSFER.md) |
| 能力（Capabilities）规格 | [`research/CAPABILITIES_SPEC.md`](research/CAPABILITIES_SPEC.md) |

## 9.5 已知缺陷与改进待办

外部独立审查（2026-11-11）与同类项目扫描的结果、已修缺陷与待办清单：
[`docs/IMPROVEMENT_BACKLOG.md`](docs/IMPROVEMENT_BACKLOG.md)、[`research/EXTERNAL_REVIEW_RESPONSE.md`](research/EXTERNAL_REVIEW_RESPONSE.md)、[`research/EXTERNAL_BENCHMARK_SCAN.md`](research/EXTERNAL_BENCHMARK_SCAN.md)。
已修：路由初始设计吃掉全部预算、目标判据方向、热启动宽限期不持久、**cmaes 在 Study 路径不自适应**、命名夸大、版本漂移。

## 10. 声明

本项目中的示例目标函数是合成函数，仅用于验证闭环；任何真实实验结论必须来自真实测量。AI 不会、也不允许伪造实验数据。
