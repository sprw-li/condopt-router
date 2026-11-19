# GITHUB_SCAN — 开源优化框架扫描（程序架构视角）

> 维护者：algo-research ｜ 扫描日期：2026-11-05
> 方法：GitHub REST API（`/repos/{owner}/{repo}`、`/repos/.../license`、`contents` 读 LICENSE 文件头）逐条核对 license 与元数据；⭐/最后 push 为扫描当日快照，会变化，不作为选型依据。
> 关注点不是"哪个库最强"，而是**哪些抽象值得抄进 Project6 的算法库**。

## 0. 总表（18 个，全部 API 核对）

| # | 项目 | 仓库 | License（核对方式） | ⭐ | 最后 push | 定位 |
|---|---|---|---|---|---|---|
| 1 | Ax | `meta/Ax` | MIT（API） | 2838 | 2026-11-01 | 自适应实验平台（生产级） |
| 2 | BoTorch | `meta-pytorch/botorch` | MIT（API） | 3645 | 2026-10-28 | PyTorch 版 BO 内核 |
| 3 | Optuna | `optuna/optuna` | MIT（API） | 15008 | 2026-11-04 | 通用 HPO 框架，define-by-run |
| 4 | Emukit | `EmuKit/emukit` | Apache-2.0（API） | 680 | 2026-08-30 | 决策/实验设计工具箱（模块化教学级） |
| 5 | scikit-optimize | `scikit-optimize/scikit-optimize` | BSD-3-Clause（API） | 2831 | 2024-02-23 | sklearn 风格 SMBO（维护停滞） |
| 6 | BayBE | `emdgroup/baybe` | Apache-2.0（API） | 527 | 2026-11-04 | 化学实验计划器（低数据 regime） |
| 7 | Olympus | `the-matter-lab/olympus` | MIT（API） | 108 | 2024-11-24 | 化学噪声优化基准 |
| 8 | Dragonfly | `dragonfly/dragonfly` | MIT（API） | 896 | 2023-06-19 | 可扩展 BO（多保真/多目标） |
| 9 | GPyOpt | `SheffieldML/GPyOpt` | BSD-3-Clause（API） | 943 | 2023-01-17 | 经典 GP-BO 包（维护停滞） |
| 10 | pyGPGO | `josejimenezluna/pyGPGO` | MIT（API） | 245 | 2022-03-02 | 轻量 BO 教学实现 |
| 11 | SMAC3 | `automl/SMAC3` | BSD-3-Clause（LICENSE 头；GitHub 显示 NOASSERTION） | 1268 | 2026-11-04 | 森林代理 + 预算感知 HPO |
| 12 | Nevergrad | `facebookresearch/nevergrad` | MIT（API） | 4225 | 2026-11-02 | 大规模无梯度优化器注册表 |
| 13 | ProcessOptimizer | `novonordisk-research/ProcessOptimizer` | New BSD / BSD-3（LICENSE 头；API NOASSERTION） | 133 | 2026-08-01 | skopt 的工业工艺优化分叉 |
| 14 | ChemOS | `the-matter-lab/ChemOS` | Apache-2.0（LICENSE 头） | 66 | 2022-06-22 | 自动化实验编排（已停更） |
| 15 | HEBO | `huawei-noah/HEBO` | **无 LICENSE 文件（2026-11-05 实测 license=null，根目录无 LICENSE/COPYING）** | 2818 | 2026-10-12 | MESMO/分区数据的高维 BO |
| 16 | Trieste | `secondmind-labs/trieste` | Apache-2.0（API） | 255 | 2026-10-15 | TensorFlow 版模块化 BO |
| 17 | hyperopt | `hyperopt/hyperopt` | BSD-3（LICENSE 头 "Copyright 2013 James Bergstra"；API NOASSERTION） | 7600 | 2026-11-02 | 老牌 TPE 实现（含分布式 MongoTrials） |
| 18 | pymoo | `the-dual-space/pymoo` | Apache-2.0（API） | 2989 | 2026-07-07 | 多目标进化算法全家桶 |

**未能核实（不要写进依赖清单）**：
- `PKU-DAIR/OpenBox`：仓库路径 404，GitHub 搜索未命中，本轮不下结论（见 `OUTBOX.md` 待核项）。
- A-Lab（Nature 2023 的自主实验室系统）：未找到官方开源仓库/可核实 DOI，本轮不引用。
- `modest-py/modest`（稀疏 GP 混合空间 BO）：路径 404，未核实。

**License 风险提示**
- **HEBO 无 LICENSE 文件** → 默认"保留所有权利"，不能直接复制代码进商业库；只能用论文思路重新实现。（这是本轮扫描的实质发现，建议 Lead 记录。）
- ChemOS 为 Apache-2.0 但已停更 4 年，只能作为架构参考，不能作为依赖。
- scikit-optimize / GPyOpt / pyGPGO / Dragonfly 均已 1–4 年无实质更新；**抄架构可以，当依赖不行**（Python 版本/依赖腐烂）。

---

## 1. 生产级框架

### 1.1 Ax（`meta/Ax`, MIT）

- **定位**：Meta 的"自适应实验平台"，把 BO 包装成**实验管理 + 生成策略 + 执行器**的完整闭环，支持批量、多目标、约束、多保真、早停。
- **核心抽象**：
  - `Client`：实验入口（`configure_experiment` 已弃用，现为 `Client` + `GenerationStrategy`）。
  - `SearchSpace` + `ParameterConstraint`：参数、变换（`Log`、`IntToFloat`、`UnitX`）、约束（线性和/或约束）。
  - `GenerationStrategy`（原 `Scheduler`）：把"生成一步"拆成**图（GenerationNode）**——节点输入由 `NodeInputConstructors`（`Sobol` 初始节点、`BoTorch` 节点、`EmpiricalBayes` 节点等）提供，节点间可用变换串联。
  - `Runner`：把 trial 参数喂给目标函数并回传指标（`attach_trial` / `complete_trial` 手动模式用于外部实验）。
  - `Metrics` + `Adapters`：多目标/约束/多保真通过 "metric + 变换" 表达。
- **值得抄**：
  1. **"生成策略是图"**：初始设计节点 → BO 节点 → 后续节点，每个节点声明"输入是什么、变换是什么"。这让"前 10 个点用 Sobol，之后用 qNEI，多目标时用 qNEHVI"变成配置而非 if-else。
  2. **数据与算法解耦**：`Client` 只管 trial 生命周期，算法只从"数据 + 当前状态"生成候选——正是我们要的 ask/tell。
  3. **变换层**：`Log`/`Standardize`/`IntToFloat` 等 transform 组合，比在每个算法里手写预处理干净。
- **坑**：API 演进快（`Experiment` → `Client/GenerationStrategy` 破坏性变更）；代码量大、依赖重（BoTorch + PyTorch + SQLAlchemy），不适合直接嵌进轻量库；学概念即可。

### 1.2 BoTorch（`meta-pytorch/botorch`, MIT）

- **定位**：BO 的"数值内核"，提供 GP 模型、采集函数、采集优化器；不是端到端框架。
- **核心抽象**：
  - 模型：`SingleTaskGP`、`ModelListGP`、`MultiTaskGP`（多任务/迁移）、`HeteroskedasticSingleTaskGP`（异方差噪声）、`SaasFullyBayesianSingleTaskGP`（高维稀疏先验）。
  - 输入/输出变换：`Normalize`、`Standardize`、`Log`、`AppendFeatures`（把 fidelity 拼进输入）、`ChainedInputTransform`。
  - 采集函数：`LogExpectedImprovement`、`qLogExpectedImprovement`、`qLogNoisyExpectedImprovement`、`qLogNoisyExpectedHypervolumeImprovement`（qNEHVI）、`qUpperConfidenceBound`、`qProbabilityOfImprovement`、`qKnowledgeGradient`。
  - 采样器：`SobolQMCNormalSampler`（准 MC，方差更低）、`IIDNormalSampler`。
  - 采集优化：`optimize_acqf`（多起点 L-BFGS）、`optimize_acqf_cyclic`（批量循环）、`optimize_acqf_discrete`（离散候选池）。
  - 拟合：`fit_gpytorch_mll(ExactMarginalLogLikelihood)`（PyTorch 优化超参）。
- **值得抄**：
  1. **采集函数 = 对后验的可微函数**：`acq(X)` 支持 batch × q × d 张量，梯度由自动微分给——这是"任何空间都能优化"的关键。
  2. **log 空间实现**（logEI）——我们应直接采用，避免下溢。
  3. **`X_baseline` 概念**（qNEI 的 baseline 集）+ `prune_baseline`：噪声场景的标准做法。
  4. **`optimize_acqf` 的三段式**：`raw_samples` 采样 → 取 top-k → L-BFGS-B 多起点；我们可照搬并换成 numpy/scipy 实现。
- **坑**：PyTorch 依赖重（CPU 版可接受）；版本迭代快（`qEI` 与其 log 版并存，`SobolQMCNormalSampler` 已弃用 → `SobolQMCSampler`）；API 名称每半年变一次，抄概念别抄 API。

### 1.3 Optuna（`optuna/optuna`, MIT）

- **定位**：最流行的 HPO 框架；架构上最有价值的是**"空间定义与算法解耦"**。
- **核心抽象**：
  - `Study` / `Trial`：一次研究=一个 campaign；`Trial` 持有参数建议历史与状态。
  - `Sampler`：`TPESampler`、`RandomSampler`、`CmaEsSampler`（把 CMA-ES 当采样器）、`GridSampler`、`NSGAIISampler`（多目标）、`BoTorchSampler`（官方集成 `optuna-integration`）。
  - **define-by-run**：空间在 `trial.suggest_float/int/categorical` 调用处定义，天然支持条件/动态空间。
  - **ask/tell**：`study.ask()` → `trial.suggest_*` → `study.tell(trial, value)`（支持异步、外部执行器）。
  - `Storage`（`RDBStorage` SQLite/Postgres、`InMemoryStorage`）：多进程/分布式共享 trial。
  - `Pruner`：`MedianPruner`、`HyperbandPruner`、`SuccessiveHalvingPruner`、`PatientPruner`（早停）。
- **值得抄**：
  1. **`ask/tell` 一等公民 + `Trial` 生命周期**（RUNNING/COMPLETE/PRUNED/FAIL）：外部实验（机器人/人工）场景的必备语义。
  2. **RDB storage**：并发安全的历史存储 + trial 状态机；比 pickle 文件可靠得多。
  3. **Sampler 可插拔**：`study.sampler` + `study.pruner` 组合，正是 router 需要的扩展点。
  4. **`enqueue_trial`**（人工指定下一个点/暖启动）——化学家给先验点非常有用。
- **坑**：为通用 HPO 设计，**不支持批量建议**（`ask` 一次一个，批量要 `optuna-integration`/并行 study）；不支持约束（除 `NSGAIISampler` 的约束参数）、多保真较弱；`define-by-run` 在需要"空间可序列化"（如跨进程重放）时不如显式 Space 对象方便。

### 1.4 Emukit（`EmuKit/emukit`, Apache-2.0）

- **定位**：学术圈最"教科书式"的模块化决策工具箱；代码可读性高，适合作为**架构模板**。
- **核心抽象**：
  - `ParameterSpace`：空间抽象（含编码/解码）。
  - `ModelWrapper`：把任意模型（GPy/GPflow 等）适配成 `predict` 接口。
  - `Acquisition`：采集函数对象（`ExpectedImprovement`、`ProbabilityOfImprovement`、`NegativeLowerConfidenceBound`、`EntropySearch`）。
  - **`OuterLoop`**（`BayesianOptimizationLoop`）：组合 `ModelUpdater`（`IModelUpdater`）+ `CandidatePointCalculator` + `AcquisitionOptimizer`（`GradientAcquisitionOptimizer`），`run_loop(num_iterations)` 驱动。
  - `multi_fidelity` 子模块：`IMultiFidelityAcquisition`、KNOWLEDGE GRADIENT 等。
  - 其他：`experimental_design`（空间填充）、`sensitivity_analysis`、`bayesian_quadrature`。
- **值得抄**：
  1. **"循环 = 更新器 + 候选计算器 + 采集优化器"** 的组件化——我们的 ask/tell 可以照此拆成三个可替换件。
  2. **`ParameterSpace` 自带编码/解码**，算法只处理连续向量。
  3. 命名与 docstring 水准（适合作为新人上手资料）。
- **坑**：维护力度一般（2026 年仍有小更新）；依赖 GPy/GPflow 版本敏感；多保真/批量的实现不如 BoTorch 完整；不支持 categorical（只处理连续/离散整数）。

### 1.5 SMAC3（`automl/SMAC3`, BSD-3-Clause）

- **定位**：**预算感知 + 混合空间**的生产级实现，架构分层最清晰。
- **核心抽象**（六层）：
  - `Scenario`：搜索空间（`ConfigSpace`）、目标、预算（`budget_limit`）、实例（instances）、确定性/随机、并行度。
  - `Facade`：场景化入口——`HyperparameterOptimizationFacade`、`MultiFidelityFacade`、`MultiObjectiveFacade`、`AlgorithmConfigurationFacade`、`HyperbandFacade`。
  - `Intensifier`：`SuccessiveHalving`、`Hyperband`、`Racing`（同配置在多个 instance/budget 上竞争）。
  - `Surrogate`：随机森林（分类/回归）。
  - `AcquisitionMaximizer`：`LocalSearch`、`RandomSearch`、`CMAES`、`Scipy`。
  - `InitialDesign`：`Sobol`、`LHD`、`Default`（用默认配置）。
  - `RunHistory`（SQLite/JSON）：配置 × 实例 × 预算 × 种子 的完整记录 + 状态机。
  - `TargetFunction`：`__call__(config, seed, budget, instance)`——把"评估"抽象成带上下文的函数。
- **值得抄**：
  1. **Facade 模式**：用户只描述"场景 + 预算"，内部自动拼装 算法/代理/采集优化器——比暴露 20 个参数友好。
  2. **RunHistory 数据模型**：`(config, instance, seed, budget) → cost, status, timestamps`；这是可复现与 benchmark 的基础。
  3. **Intensifier 抽象**：把"预算/实例分配"从算法里抽出来，化学实验的"多批次/多条件重复"正好对应。
  4. **随机森林代理 + 条件空间**：categorical/条件占比高时的正确选择。
- **坑**：依赖 `ConfigSpace`（自己一套空间语言，与 numpy 参数空间不通用）；森林代理在连续低维上不如 GP；代码较"企业化"（层次多），抄概念要有取舍。

### 1.6 Nevergrad（`facebookresearch/nevergrad`, MIT）

- **定位**：**大规模无梯度优化器注册表**，上百种算法统一在 ask/tell 下，Facebook 内部用于调参与架构搜索。
- **核心抽象**：
  - `parametrization`：`Array`、`Log`（对数尺度）、`Integer`、`Choice`（categorical）、`TransitionChoice`（有序离散）、`Tuple`、`Instrumented`（带噪声/多目标）、`Folder`。
  - `Optimizer`：`ask()` / `tell(x, value)`；`optimize()` 便捷封装；`minimize`/`maximize`。
  - `ng.optimizers` 注册表：`NgIohTuned`（自动选算法）、`CMandAS2`、`BO`、`TwoPointsDE`、`Portfolio`、`NGOpt`、`PCEDA` 等。
  - `BudgetManager` / `Distributed`：分布式预算分配。
  - `MultiobjectiveFunction`：多目标通过"标量化 + 多目标注册表"。
- **值得抄**：
  1. **参数化对象负责"空间 ↔ 连续向量"的映射**（`Log`、`Integer`、`Choice` 都是 parametrization）——与我们前面说的 encoder/decoder 同构，且实现干净。
  2. **算法注册表 + 统一的 "建议/告知" 协议**：router 可以直接按名字+参数实例化。
  3. **`Portfolio` 元算法**：同时跑多个优化器、按表现切预算——router 的"多算法并行竞争"可以直接抄。
- **坑**：面向"评估便宜、次数多"的场景（内部大量用 1e4–1e6 次评估），对"昂贵实验、样本极少"不友好；`Choice` 用 one-hot 类似机制，高基数分类仍弱；文档偏内部风格。

### 1.7 Trieste（`secondmind-labs/trieste`, Apache-2.0）

- **定位**：TensorFlow 生态的模块化 BO，工程分层与 Emukit 类似但更现代（支持批量、多目标、多保真、异步）。
- **核心抽象**：
  - `Observer` + `AcquisitionRule`（`EfficientGlobalOptimization`、`qExpectedHypervolumeImprovement`、`AsynchronousGreedy`、`UpcomingMinValues`）。
  - `ProbabilisticModel`（GPflow/GPflux 封装）。
  - `BayesianOptimizer`（`optimize(num_steps)`）、`AskTellOptimizer`（异步/外部实验）。
  - `AcquisitionFunction`（`ExpectedImprovement`、`ProbabilityOfImprovement`、`NegativeLowerConfidenceBound`、`GreedyAcquisition`）。
  - `LocalPenalization`（批量）、`multi_fidelity` 模块、`Objective`/`Observer` 支持约束（`Observer` 可挂多个 `Constraint`）。
- **值得抄**：`AskTellOptimizer` 的"pending 点管理"是异步实验的标准实现；`AcquisitionRule` 把"用哪些模型/哪些采集函数/如何组合"封装成可组合对象。
- **坑**：TensorFlow 依赖在 2026 年已非主流（团队用 numpy/scipy/torch），仅作概念参考。

---

## 2. 化学 / 实验专用

### 2.1 BayBE（`emdgroup/baybe`, Apache-2.0）

- **定位**：Merck KGaA 的实验计划器，**与本项目最同构**（化学条件优化 + 低数据 + 可序列化 campaign）。
- **核心抽象**（名称以官方文档为准）：
  - `Campaign`：一次实验活动，`from_config`/`to_config` JSON 往返（**可持久化、可版本化**）。
  - `SearchSpace`：由 `Parameter` 组合；`NumericalDiscreteParameter`、`CategoricalParameter`、`SubstanceParameter`（化学物质）、`TaskParameter`（任务/上下文变量）。
  - `Constraint`：离散空间上的约束（通过 `SubspaceDiscretizer` 枚举可行子空间）。
  - `Recommender`：`RandomRecommender`、`SequentialGreedyRecommender`（单点/批量）、`BotorchRecommender`、`TwoPhaseMetaRecommender`（先空间填充后 BO，可配置切换点）、`FPSRecommender`（最远点采样）。
  - `Measurement`：`campaign.add_measurements(df)`；`campaign.recommend(n)`。
  - `Telemetry`/回调：记录推荐时间、模型状态等。
- **值得抄**：
  1. **campaign 即配置**（JSON 可存、可 diff、可复现）——上层 agent/skill 传一个 JSON 就能继续跑。
  2. **Recommender 的"两阶段/可切换"设计**：低数据随机/LHS → 有数据后 BO；且切换点可配（`switch_after`）。
  3. **`SubstanceParameter`**：把"化学物质"作为一等参数类型（带描述符/数据库查询），而不是字符串。
  4. 离散化搜索空间（`SearchSpace` 内部把所有参数离散化）→ 让"候选池枚举"成为默认实现，天然支持混合空间与约束。
- **坑**：文档/API 变化快（2024–2026 多次重构）；依赖较重（pandas/pydantic/botorch）；只做"计划"，不含实验编排。

### 2.2 Olympus（`the-matter-lab/olympus`, MIT）

- **定位**：化学优化的**基准集**（比算法库更有价值）：提供带真实噪声的 emulator 与统一 `Planner` 接口。
- **核心抽象**：
  - `Dataset`/emulator：`HPLC`、`Fullerenes`、`Suzuki`、`SnAr`、`Photocatalysis` 等（来自真实实验数据训练的代理模型，带噪声模型）。
  - `Planner` 基类：`Botorch`、`Gryffin`、`Random`、`Entmoot`、`Phoenics`（不同算法的统一封装，`suggest_experiments` / `add_results`）。
  - `benchmark`：`run_benchmark`、`Benchmark`（多算法 × 多数据集 × 多 seed 的批量跑分 + 统计）。
- **值得抄**：**"Planner 统一接口 + emulator 数据集"** 两层结构——我们的 benchmark 模块可以直接照此设计（算法只需实现 `suggest`/`add`，评估器可替换为 emulator/真实实验）。
- **坑**：停更于 2024；算法实现较旧（BoTorch 版本老）；数据集/依赖需手动对齐。

### 2.3 ChemOS（`the-matter-lab/ChemOS`, Apache-2.0，停更）

- **定位**：**自动化实验编排**（orchestration）而非优化算法：把优化器（Gryffin/Botorch）与实验硬件（Chemspeed）连起来。
- **核心抽象**：`ChemOS` 主控、`Experiment`（参数/结果/状态）、`Planner`（优化器适配）、RPC/客户端-服务端分离、底层 `cmd` 指令层（硬件协议）。
- **值得抄**：
  1. **优化器与执行器分离**（planner 只产出"实验请求"，执行器负责安全/硬件/回传）——这正是本项目"算法库 ↔ agent skill ↔ 实验室"的边界。
  2. **实验请求/结果的 schema 化** + 状态机（pending/running/done/failed）。
  3. **异步 + 失败重试**的编排循环。
- **坑**：停更 4 年、Python 版本老、强绑定特定硬件；**只抄架构不抄代码**。

---

## 3. 经典 / 教学参考实现

### 3.1 scikit-optimize（BSD-3-Clause，停更）

- **核心抽象**：`Space`/`Real`/`Integer`/`Categorical`；`Optimizer(base_estimator, n_initial_points, acq_func, acq_optimizer, initial_point_generator)`；`ask()/tell()`；`dump()/load()`（joblib）；`gp_minimize`/`forest_minimize`/`gbrt_minimize` 便捷函数；`Space.from_yaml`（空间可配置化）。
- **值得抄**：**空间的可序列化定义（YAML/JSON）+ `Optimizer` 的 ask/tell**——这是最"小而全"的实现，适合作为第一版骨架。
- **坑**：停更（2024 最后 push，实际停更多年）、GP 实现基于 sklearn（速度一般）、不支持批量/约束/多目标、categorical 处理粗糙（只 one-hot）。

### 3.2 GPyOpt（BSD-3-Clause，停更）

- **核心抽象**：`BayesianOptimization(objective, domain, constraints, acquisition_type, initial_design_type)`；`modular` 子包把 `Model` / `Acquisition` / `AcquisitionOptimizer` / `Normalization` / `InitialDesign` 拆开；`core.task`（`SingleTask`、`ContextTask` 支持 contextual BO）；`evaluator`。
- **值得抄**：`modular/` 的组件拆分（尤其 `ContextTask` 的"上下文变量"思想，对应化学里的"批次/设备/原料批次"）。
- **坑**：依赖 GPy（Python 3.10+ 兼容性差）；停更 3 年+。

### 3.3 pyGPGO（MIT，停更）

- **核心抽象**：`GPGO(f, bounds, init_evals, n_iter, acquisition='EI', optimizer='scipy')` + `.run()`；`priors` 模块（MCMC 采样超参）；`acquisition` 模块（EI/PI/UCB）；`gp` 模块（GPy/sklearn 封装）。
- **值得抄**：**边界字典 + 一行式 API** 的极简封装（`bounds = {'x': ('continuous', 0, 1)}`），适合做"用户友好层"的参考。
- **坑**：2022 后停更；MCMC 超参在昂贵场景过慢；不支持批量/多目标。

### 3.4 hyperopt（BSD-3，仍维护）

- **核心抽象**：`hp.uniform`/`hp.loguniform`/`hp.choice`/`hp.quniform` 等空间描述；`fmin(fn, space, algo=tpe.suggest, max_evals, trials)`；`Trials`（含 `MongoTrials` 分布式、`SparkTrials`）；`algo` 可换 `tpe.suggest`/`rand.suggest`/`atpe.suggest`；early stop 回调。
- **值得抄**：**TPE 参考实现**（Parzen 估计、`n_EI_candidates`、constant liar 并行）+ `Trials` 存储抽象。
- **坑**：API 古怪（`hp.choice` 返回索引）、无类型系统、并行支持弱（`SparkTrials` 依赖 Spark）；仅作 TPE 算法参考。

### 3.5 Dragonfly（MIT，停更）

- **核心抽象**：`maximize(func, domain, max_capital, ...)`；`Domain`/`CartesianProduct` + `option`（Discrete/Continuous/Integer）；`exp.init`（初始化配置）；`gp`（GP 模型选择/超参）；`acquisition`（EI/UCB/TS）；`multiobjective`（ParEGO/EHVI）；`evolution`（CMA-ES 等进化算法）。
- **值得抄**：**"优化器配置对象"**（一次 `maximize` 调用前用 `exp.init` 指定 init/acquisition/gp），以及多目标/多保真的统一入口。
- **坑**：停更 3 年；代码风格老、依赖重（需 numpy/scipy/sklearn，部分功能需额外包）。

### 3.6 pymoo（Apache-2.0）

- **核心抽象**：`Problem`（`_evaluate` 返回 `(F, G)` 目标与约束）、`minimize(problem, algorithm, termination, seed)`、`Algorithm`（`NSGA2`、`NSGA3`、`MOEAD`、`CMAES`、`DE`）、`Sampling`（`LHS`、`Sobol`）、`Result`、`Operator`（交叉/变异）。
- **值得抄**：**"问题对象 + 算法对象 + 终止条件"** 三段式；约束用 `(F, G)` 约定（G≤0 可行），干净通用；多目标结果对象（Pareto 集 + 历史）。
- **坑**：面向"评估便宜"的进化计算，不做代理模型（昂贵场景需自己嵌 GP）；API 有一定学习曲线。

---

## 4. 横向对比（选型/借鉴时看这张表）

| 维度 | Ax/BoTorch | Optuna | SMAC3 | BayBE | Emukit/Trieste | Nevergrad | skopt/GPyOpt |
|---|---|---|---|---|---|---|---|
| ask/tell | ✅（Client/Trial） | ✅（study.ask/tell） | ✅（Facade） | ✅（recommend/add） | ✅（loop/AskTell） | ✅（原生） | ✅（Optimizer） |
| 批量 | ✅ 强（qEI/qNEHVI） | ⚠️ 弱（需并行/集成） | ✅（intensifier/并行） | ✅（recommend(n)） | ⚠️ | ⚠️ | ❌ |
| 混合/分类空间 | ✅（transform） | ✅（define-by-run） | ✅ 强（ConfigSpace） | ✅ 强（离散化+Substance） | ⚠️ 弱 | ✅（parametrization） | ⚠️ 弱 |
| 条件空间 | ✅ | ✅ | ✅ 强 | ⚠️ | ❌ | ⚠️ | ❌ |
| 约束 | ✅ | ⚠️（NSGAII） | ⚠️ | ✅（离散） | ⚠️ | ⚠️ | ❌ |
| 多目标 | ✅ qNEHVI | ✅（NSGAII/MOTPE） | ✅ | ⚠️ | ✅ | ⚠️ | ❌ |
| 多保真 | ✅ | ⚠️（pruner） | ✅ 强（Hyperband） | ⚠️ | ✅ | ❌ | ❌ |
| 持久化 | ✅（SQL） | ✅ 强（RDB） | ✅（RunHistory） | ✅ 强（JSON） | ❌ | ⚠️ | ✅（joblib dump） |
| 轻量可嵌入 | ❌ | ⚠️ | ❌ | ❌ | ⚠️ | ✅ | ✅ |
| 化学友好 | ⚠️ | ⚠️ | ⚠️ | ✅ 强 | ❌ | ❌ | ⚠️ |

---

## 5. 给 Project6 的借鉴清单（**核心输出**）

### 5.1 推荐组合（按层）

| 层 | 抄谁 | 怎么抄 |
|---|---|---|
| 空间抽象 | **BayBE**（Parameter/SearchSpace/Constraint）+ **Nevergrad**（parametrization） | 统一 `encode/decode/validate/project`；离散化后支持枚举；chem 参数类型（Substance）单独扩 |
| ask/tell 主循环 | **Optuna**（Trial 状态机 + storage）+ **Emukit**（loop 组件化） | `ask(n) → 执行 → tell(results)`；pending 队列、去重、状态机 |
| 算法注册表 | **Nevergrad**（按名字+config 实例化）+ **SMAC3 Facade** | `registry: name → (class, param_schema, defaults, capabilities)` |
| 采样器实现 | **BoTorch 概念 + 自研 numpy/scipy**（GP、logEI、qNEI、qNEHVI 概念） | 不能依赖 torch 时用 numpy 写 GP + scipy L-BFGS；或 torch(cpu) 只用于 GP |
| 混合/条件空间 | **SMAC3**（森林代理）+ **TPE（hyperopt 语义）** | 条件空间默认走 TPE/森林，GP 只在连续为主时用 |
| 实验计划 | **BayBE**（两阶段 recommender、JSON campaign） | 低数据 → 空间填充；有数据 → BO；可配切换 |
| benchmark | **Olympus**（emulator + Planner 接口） | 同步一套 planner 接口，接入 Olympus 数据集 |
| 编排边界 | **ChemOS**（planner/executor 分离 + 请求 schema） | 算法库只产"实验请求"，执行/安全由上层负责 |
| 多目标 | **pymoo**（问题对象约定）+ **BoTorch qNEHVI 概念** | 先 ParEGO 标量化，再上 qNEHVI |
| 预算/早停 | **SMAC3 Intensifier** + **Optuna Pruner** | 把"预算分配"独立成组件，算法无感知 |

### 5.2 明确**不要**做的

- ❌ 不要直接依赖 Ax（依赖爆炸、API 不稳）——只借"生成策略=图 / 变换层"的思想。
- ❌ 不要把 Optuna 当唯一后端（它不支持批量、约束弱）——它适合做"统一存储 + 编排"的上层，算法自己实现。
- ❌ 不要依赖停更库（skopt/GPyOpt/pyGPGO/Dragonfly/ChemOS）作为运行时依赖。
- ❌ 不要抄 HEBO 代码（**无 LICENSE**）；如果要 MESMO/分区思想，按论文重写。
- ❌ 不要一开始就上 LLM 决策——先把数值算法与 benchmark 打牢，LLM 只做 router/诊断。

### 5.3 本轮"值得单独记录"的三个架构点

1. **"候选池 + 枚举打分" 应当是默认执行路径**（BayBE/Olympus 都这么干）：混合空间、约束、批量、可复现全部依赖它；梯度优化作为连续子域的加速分支。
2. **campaign 配置即 JSON**（BayBE）：agent/skill 传递的是配置对象而非代码，天然支持 resume/审计/复现。
3. **算法能力声明（capabilities）**：每个算法登记 `{batch: bool, constraints: bool, multi_objective: bool, multi_fidelity: bool, categorical: bool, max_dim_soft: int}`，router 选算法时先做能力过滤，再做打分——比"LLM 凭感觉选"可靠一个量级。

---

## 6. 待核实 / 下轮补充

- `PKU-DAIR/OpenBox` 真实仓库路径（404）与 license。
- `modest-py/modest`（稀疏轴对齐 GP，适合混合空间）真实路径与 license。
- A-Lab 官方代码是否开源（Nature 2023）。
- 补充扫描：`facebook/Ax` 的 `Analysis`/`Adapters` 新模块、`optuna-integration` 的 BoTorch 采样器、`Enthought/ame`? 、`autofolio`? 、`HpBandSter`（BOHB 原实现）、`SALib`（敏感度分析，做维度筛选）、`scipy.stats.qmc` 的 Sobol 维度上限实测。
