# Router experience from literature labels

Generated: 2026-11-05T21:53:44+08:00
Labels: 356 (from 34 subagent files)

## Space type

- continuous_heavy: 165
- mixed: 99
- categorical_heavy: 67
- unknown: 25

## Literature method

- BO: 122
- other: 106
- DoE: 53
- none: 23
- evolutionary: 23
- RL: 12
- unknown: 12
- TPE: 5

## Router recommendation

- bayes: 103
- parego: 46
- grid: 37
- lhs: 34
- tpe: 33
- none: 29
- qehvi: 28
- constrained: 23
- mfbo: 9
- sobol: 6
- qnehvi: 5
- random: 3

## Method agreement: 0.474 (n=215)

## Candidate rules

### categorical_heavy -> prefer tpe (n=38, medium)

- evidence: tpe=10, bayes=9, grid=7, parego=4, none=3, sobol=2, qnehvi=1, qehvi=1, constrained=1
- action: keep tpe preference for categorical-heavy spaces

### continuous_heavy -> prefer bayes (n=123, medium)

- evidence: bayes=47, lhs=17, parego=14, qehvi=14, constrained=10, none=7, grid=5, mfbo=4, qnehvi=2, sobol=2, tpe=1
- action: keep bayes/GP preference for continuous spaces

### constraints -> prefer constrained (n=54, medium)

- evidence: constrained=17, qehvi=11, none=7, parego=5, bayes=4, tpe=4, grid=3, lhs=1, qnehvi=1, mfbo=1
- action: keep constrained routing (PoF/EI_c) / check if BO with penalty dominates

### multi_objective -> prefer qehvi (n=83, medium)

- evidence: qehvi=25, parego=24, constrained=7, bayes=6, none=5, qnehvi=4, grid=4, lhs=3, mfbo=3, tpe=1, sobol=1
- action: align multi-objective router (parego/qehvi/qnehvi) with literature practice

## Cross-tabs (router recommendation)

### by space_type

- continuous_heavy: bayes=62, lhs=23, parego=19, qehvi=14, constrained=13, none=11, grid=11, mfbo=6, qnehvi=2, sobol=2, random=1, tpe=1
- unknown: bayes=7, none=6, parego=5, qehvi=4, constrained=3
- mixed: bayes=23, parego=14, grid=14, tpe=13, lhs=10, qehvi=9, none=6, constrained=5, mfbo=3, sobol=1, qnehvi=1
- categorical_heavy: tpe=19, grid=12, bayes=11, parego=8, none=6, sobol=3, qnehvi=2, random=2, constrained=2, qehvi=1, lhs=1

### by problem_type

- process: bayes=25, lhs=18, none=9, grid=9, parego=8, constrained=7, qehvi=5, mfbo=4, sobol=1, qnehvi=1
- other: tpe=14, none=9, constrained=9, parego=8, bayes=8, grid=4, qehvi=3, mfbo=2, lhs=2, random=2, sobol=1
- formulation: bayes=18, parego=11, grid=10, qehvi=8, lhs=6, tpe=3, none=3, qnehvi=2, constrained=2
- materials: bayes=28, parego=13, qehvi=8, none=7, tpe=7, grid=6, lhs=4, constrained=3, mfbo=3, qnehvi=2, sobol=1, random=1
- reaction: bayes=24, tpe=9, grid=8, parego=6, qehvi=4, lhs=4, sobol=3, constrained=2, none=1

### by objectives

- single: bayes=86, tpe=28, grid=27, lhs=22, none=19, constrained=14, sobol=5, mfbo=4, random=2
- multi: parego=46, qehvi=28, bayes=10, grid=9, lhs=8, constrained=7, none=7, qnehvi=5, mfbo=5, tpe=4, sobol=1, random=1
- unknown: bayes=7, lhs=4, none=3, constrained=2, tpe=1, grid=1

### by constraints

- false: bayes=80, grid=29, parego=24, lhs=22, tpe=20, qehvi=15, none=15, mfbo=6, sobol=5, qnehvi=2, random=2
- unknown: bayes=16, lhs=10, parego=9, tpe=7, none=6, grid=4, sobol=1, qnehvi=1, qehvi=1, random=1
- true: constrained=23, parego=13, qehvi=12, none=8, bayes=7, tpe=6, grid=4, mfbo=3, lhs=2, qnehvi=2

### by automation

- self-driving lab: bayes=12, qehvi=7, constrained=4, none=3, parego=3, grid=1
- active learning: bayes=10, tpe=4, qehvi=4, parego=2, qnehvi=2, none=1, constrained=1, mfbo=1
- flow: bayes=11, grid=5, parego=3, lhs=3, qehvi=1
- unknown: bayes=4, parego=3, none=2, lhs=1, tpe=1
- none: bayes=60, parego=31, grid=24, lhs=24, tpe=20, none=20, constrained=18, qehvi=13, mfbo=8, qnehvi=3, sobol=2, random=2
- HTE: tpe=6, lhs=4, grid=4, sobol=4, bayes=4, qehvi=2, parego=2, none=2, random=1
- robotic: grid=3, parego=2, bayes=2, lhs=2, tpe=2, none=1, qehvi=1

## Subagent lessons

- (lesson_00.json) 遇到 1–5 个连续工艺参数、单一可测目标、已有自动化闭环时直接选 bayes，不必先跑 DoE 筛选。
- (lesson_00.json) 多目标且带工艺约束时选 qehvi/qnehvi；若同时存在多保真层级（仿真/低成本实验），先评估 mfbo 以压缩高保真实验次数。
- (lesson_00.json) 摘要把目标写成多个 CQA 或多项性能时，即使作者只提 BO，也应按 multi 标注并按多目标路由。
- (lesson_00.json) 先识别控制类与非标量目标：连续闭环调节用 RL，图像/专家偏好用主动学习，这两类不要路由到 bayes。
- (lesson_01.json) 类别/离散变量占主导且无自动化平台时，先用 DoE（正交、田口）筛因子是主流做法；只有需要多轮序贯补点时再切 TPE（对类别维度做 native split），不要默认高维连续 BO。
- (lesson_01.json) 离散组合数少于约 50 且单次评估便宜（如 7×6 组传感器/分类器消融）时直接用 grid 穷举，BO 的代理模型在这个预算内不划算，预算应转投复孔降噪。
- (lesson_01.json) 2-3 个连续参数 + 双目标 + 硬几何/工艺约束时选 ParEGO（或批量场景 qEHVI），并把约束作为可行性过滤显式处理，避免降级为 constrained 单目标 EI。
- (lesson_01.json) 本 chunk 中真正可判读优化问题的仅 6/40 篇：摘要若只有“制备某配方/调节某工艺并与基线对比”，应标 literature_method=none、confidence=low，宁可少标也不要臆造 BO 证据污染 router 校准。
- (lesson_02.json) 本chunk召回噪声极高：查询词 self-driving lab / autonomous laboratory 大量命中 'autonomous cortisol secretion'、'autonomous vehicles'、'autonomous region' 与纯医学病例，40篇中仅3篇可见优化问题；只按'变量/目标/算法名'可见性打标，其余一律跳过，勿为凑数编造标签。
- (lesson_02.json) 判定 noisy 要抓摘要里的变异证据词：'noise floor'、'AE fluctuations'、'across drivers/participants'、个体差异描述；一旦 noisy=true，路由器应同时输出复孔/重复测量建议，而不能只给算法名。
- (lesson_02.json) 算法与自动化平台解耦：self-driving lab / robotic 标签本身不构成选 BO 的理由（10.1039/d6ra06905c 用阈值标定即够）；真正决定 router 的是维度、变量类型与预算。
- (lesson_02.json) 多目标判断以摘要明确列出的结局指标数为准（如主观KSS+Alpha/Theta+SDLP=multi），但若空间小到可穷举，仍优先 grid/全因子而非 ParEGO/qEHVI。
- (lesson_03.json) This chunk is dominated by off-domain literature (medical case reports, taxonomy, corrections, editorials): only 3 of 40 records describe a comparable condition-optimisation setup, so any router calibration from chunk_03 must be weighted very lightly (3 labels out of 40).
- (lesson_03.json) All three labelable studies are small enumerable categorical/recipe grids solved by full enumeration (grid) or factorial comparison with replicates; treat 'categorical_heavy + cheap assay + <= 20 candidates' as a grid/DoE trigger and reserve TPE/BO for larger or costlier categorical spaces.
- (lesson_03.json) Two-stage practice recurs: coarse screen over all categorical levels first, then a small factorial in the survivors - the router should allow a plan with a screening phase whose output (surviving levels) becomes the search space of the next phase.
- (lesson_03.json) Multi-criteria outcomes (oil uptake vs porosity/hydrophobicity; LC50 potency vs field efficacy at reduced rate) were resolved by reporting Pareto-like trade-off tables rather than a MOO acquisition function, so for <= 10 evaluated candidates a Pareto table from a grid is a valid substitute for ParEGO/qEHVI.
- (lesson_04.json) 混合空间不要一律 one-hot:类别因素少(<=2 个)时用分层/分组核的 BO 并显式建模批次(孔板)约束,只有高维纯离散或数据不平整时才切 TPE。
- (lesson_04.json) 多目标先看并行能力:能一次测一批(HTE/孔板/多通道)→ qEHVI;只能顺序做且预算小 → ParEGO;不要在几十次实验的预算下用 NSGA-II 追 Pareto 前沿。
- (lesson_04.json) 把'失败/不可行'当数据用:显式可行域约束(或可行性分类)比在扩展空间里盲搜省一个数量级的实验(23 vs 512);噪声大时(复孔 σ 达 75 mV)必须用噪声感知 GP 传播不确定性。
- (lesson_04.json) 连续大空间分两段走:先用 LHS/梯度 DoE 做空间填充拿到覆盖性数据,再交 BO 做局部精化;不要把 BO 冷启动直接扔进高维连续组分空间。
- (lesson_05.json) 类别型空间不要 one-hot 硬上 GP:数十个离散成分/序列候选 + 单目标 → tpe;双目标小批量 → parego;候选维度再高、数据不平整时 TPE 的样本效率优势最明显。
- (lesson_05.json) 连续 + 高维 + flow/自驱动实验时默认 bayes,并可在摘要层面直接引用预算对照(BO 比 DoE 少约 3 倍条件,GP-AL 比网格少约 80%);DoE/LHS 退居初始化或 <=3 因子摸底。
- (lesson_05.json) 多目标按目标数分流:2 目标且能批量 → qnehvi/qehvi(本 chunk 有明确 qNEHVI 实例);3 目标以上或带可持续性约束 → qehvi,避免 ParEGO 标量化;只有顺序单点小预算才用 ParEGO。
- (lesson_05.json) 把噪声/保真度当一等信息:多保真或跨尺度数据 → mfbo(并加同族体系迁移先验,可再省约 50% 采样);只有 2-3 个离散水平的高噪声生物学终点 → 复孔+全因子枚举,不要用 BO。
- (lesson_06.json) 类别变量少（<=10，如 3 个候选溶剂）就别上 BO：先验打分 + 全枚举（grid）即可，且结果可解释；类别变量上千（scaffold/序列库）改用 tpe，GP+one-hot 会失真且核矩阵失去意义。
- (lesson_06.json) 反应/工艺优化先问有没有历史数据：有则用多任务/迁移的 bayes 解决冷启动（这是 chunk 内唯一被点名的高置信做法）；没有历史数据且维度 <=3 时，DoE/网格 + 复孔比 BO 更省事。
- (lesson_06.json) 多目标按目标数分档：<=3 目标且有并行批次用 qehvi，>=4 目标用 parego 标量化；只有硬约束没有第二目标时不要选 qehvi，选 constrained 或 bayes。
- (lesson_06.json) HTE 平台 + 1-3 因子是网格的主场（96 孔一次铺满、复孔抗噪）；BO 的收益门槛是维度 >=4 或单次实验昂贵，不要把 HTE 平台上的小设计硬套 BO。
- (lesson_07.json) 多目标按并行能力分流,不按目标数分流:能一批测几十个(HTE/自动化配方站)→ qEHVI;只能顺序单点、预算几十次 → ParEGO 标量化;本 chunk 无一篇在几十次预算下用 NSGA-II 追 Pareto 前沿。
- (lesson_07.json) 平台决定第一跳:flow/扫描流动池/自驱动实验室配 GP-BO 或 Bayesian active learning,HTE 板式配批量 BO;没有自动化平台的低维问题(<=4 连续因子)先用 LHS/因子设计铺域,拿到覆盖数据再上 BO,别冷启动直接 BO。
- (lesson_07.json) 改问题形式比换算法更有效:二元成败(胶凝/失败)做成可行性约束或分类而不是回归,噪声(复孔 ±0.01~0.15%、生物变异)写进 GP 的重复观测/异方差,安全与成本类硬约束(无溶剂、避免 NaH、合成成本上限)在空间定义阶段剪枝后再用 constrained BO,而不是当惩罚项。
- (lesson_07.json) 10^6 量级的离散库(序列库、组分枚举)不属于 GP-BO 的适用范围:用代理(DFT/生成模型)先筛 + 批量主动学习,或直接策略梯度(RL);router 应保留 tpe/RL 这类非 GP 分支,并在离散枚举域上改用「代理筛选 + AL」而不是普通 EI。
- (lesson_08.json) 预算少（<~50 次实验）且变量 <=5 个、以连续参数为主时，直接走 bayes（GP-BO，可用 RF/PCA 简化代理）；本 chunk 中 3 篇这类工作全部用 BO 且都收敛很快。
- (lesson_08.json) 如果实验平台能一次跑满 96/384 孔板，第一轮用 sobol/lhs 空间填充（约为全因子 25% 的预算）拿趋势，第二轮再切 BO 并换成真实目标（如 cost-penalized yield），比一开始就全因子更省。
- (lesson_08.json) 多目标选择按目标数分档：2-3 个目标用 qehvi（批量、闭环 HTE 场景下比 Sobol 基线高一个数量级），>=4 个目标用 parego；一旦存在硬可行性阈值（选择性、可成药性、配方 in-spec）则改用 constrained。
- (lesson_08.json) 类别型 HTE/离散空间常用 one-hot + GP-BO 或 LLM/枚举 + 空间填充；纯库式定向进化或几千孔位密集枚举场景不要强推 BO（router 应给 none/evolutionary/sobol），评估成本分层明显时才值得上 mfbo。
- (lesson_09.json) 类别变量占主导(配体/碱/溶剂/底物枚举)时默认 GP-BO+类别核+批量采集;只有在单点成本极低且空间可全枚举时,才用全因子网格/HTE 板做起点。
- (lesson_09.json) 2 目标用 ParEGO/qEHVI 出 Pareto 前沿,>=3 目标或强冲突目标用批量 qEHVI;不要把多目标压成单一加权标量后只报一个最优点。
- (lesson_09.json) 摘要出现 active/soft constraint、物理可行性边界、操作安全时选 constrained BO 并标 constraints=true;出现复孔/批次/重复测量时标 noisy=true 并配噪声感知 GP。
- (lesson_09.json) 若论文先用实验数据辨识并验证了机理/动力学模型,推荐 none(在模型内做优化);只有模型缺失、必须边实验边学时 BO 才是正解。
- (lesson_10.json) 连续单目标无约束的条件优化先给 GP-BO：8-12 个 LHS/Sobol 初始点 + GP/EI；窄边界或高维用 TuRBO；DoE/RSM 只作基线（BO 在 10.1002/anie.5778118、10.3390/mi17050619、10.1039/d6ra02906j 均优于 RSM/人工试错，单次收敛常在 12 次迭代内）。
- (lesson_10.json) 只要条件空间存在设备损伤/失败区，router 必须返回 constrained 而不是 bayes，并同时输出'先用仿真或代理学失败边界→软约束/可行性分类器'的动作建议；安全不能靠事后过滤不可行点。
- (lesson_10.json) 成本结构出现两档（廉价代理 vs 昂贵真值/高保真）时选 mfbo：粗阶段用代理缩边界、细阶段只在高保真引擎精修；本chunk 的 DL+MC 粗到细 TuRBO 即此范式，单保真 BO 会浪费预算。
- (lesson_10.json) 类别变量先数基数再选算法：≤ ~20 个离散候选可用 one-hot GP-BO 配 LHS 覆盖；离散取值更多或树结构改 tpe；目标 ≥2 时昂贵评估用 parego/qehvi、廉价仿真用 NSGA-II，并同时给出复孔/稳健统计建议（noisy=true 样本约占 1/12）。
- (lesson_11.json) 先按维度与变量类型分诊：≤4 个连续因子且单目标直接 DoE/网格（2^k 全因子）；5-8 维连续或含 1-2 个离散类别用 GP-BO（bayes/qEHVI）；上千至上万组合的高维离散空间改 TPE 或分阶段多样化采样——anie.4418883 明确显示维度越高普通 BO 越吃亏。
- (lesson_11.json) 多目标按目标数分流：2 目标且连续主导用 qEHVI（可批量取点，配 flow/半自动平台）；≥3 目标、强混合或带约束用 ParEGO 标量化，或先 LHS 2000+ 样本建代理再跑 NSGA-II，不要在高目标维硬套 qEHVI。
- (lesson_11.json) 约束、噪声、多目标三者同时出现时才升级到 qNEHVI，把可行性与噪声都建模；只有单一约束用 constrained、只有噪声用噪声感知 GP、都没有用 bayes，避免过度升级采集函数。
- (lesson_11.json) 摘要中 BO 只出现在 hyperparameter tuning/fine-tune 时不计为条件优化证据（本 chunk 有 3 篇此类），标 problem_type=other 并给 low/medium confidence；遇到动态闭环控制则 router 返回 none 并提示 RL。
- (lesson_12.json) 类别型候选库(催化剂/配体/反溶剂)默认走 bayes,关键是编码方式而非算法:分子描述符或几何编码比 one-hot 更能让 GP 在几十个点内收敛;本 chunk 没有 TPE 证据,不要把类别多当成切 TPE 的理由。
- (lesson_12.json) 连续 3-4 因子且没有闭环反馈时不要上 BO:先用 LHS/Box-Behnken 类设计铺满响应面,再用 GP/GBR 拟合找最优点;BO 的收益依赖'按轮次看结果再选点'。
- (lesson_12.json) 多目标先看能不能把指标合成一个评分:能合成(holistic metric)→ 单目标 bayes 更省样本;不能合成且实验可并行 → qehvi,只能顺序做且预算<100 → parego。
- (lesson_12.json) 自动化平台决定 batch 语义:自驱动/HTE 每轮产生一批点,必须用 batch-aware 采集(或先 lhs 铺库再 BO);纯人工单点实验则优先 DoE,BO 仅在预算很紧时使用。
- (lesson_13.json) chunk_13 以综述/机理/表征类摘要为主(40 篇里 30+ 篇无变量/目标/算法名), 只对 8 篇打标;遇到只说 AI-driven design、high-throughput screening 而不给变量与算法的综述, 一律视为信息不足跳过, 不要给 router 派任务。
- (lesson_13.json) 目标数 >=3 且设计空间是离散成分枚举(高熵合金等)时优先 parego 而不是 qehvi: 前者在组合空间用标量化 Pareto 搜索 + 批量多样性选点即可, 后者依赖连续域与大批量候选。
- (lesson_13.json) 摘要出现 limited power supply / predefined constraints 等资源或操作上限时选 constrained;而出现 Sobol/敏感性分析 + 少数参数主导时选 sobol 先筛后优化;出现 factorial effect + ANOVA 的低因子研究则选 grid 全因子 DoE。
- (lesson_13.json) 自动化平台与算法的配对在本 chunk 中主要由 flow 平台体现: 批次+连续流混合的小样本数据先用 GP 代理并以后验不确定性排序(可升级为 bayes);当代理评估成本近零、数据已离线积累时改用 tpe/evolutionary, 不要继续跑序贯 BO。
- (lesson_14.json) 类别变量占主导时不要裸 one-hot 硬塞 GP:先做物理/化学性质编码(亲核性、Hammett 等),或直接换 TPE;离散候选 >10 类时先投影到连续潜空间(VAE)再 BO,避免组合爆炸。
- (lesson_14.json) 多目标按目标数分档:2 目标且评估昂贵用 qEHVI/ParEGO 批量采集;>=3 目标用 ParEGO 标量化;只有纯仿真且能承受大种群评估时才用 NSGA-II。
- (lesson_14.json) 实测含噪声与工艺窗口时用渐进收紧约束的 constrained BO,先 2-3 组复孔估 sigma 再收紧;样本 <30 时 GP 代理 + BO 明显优于全因子/RSM。
- (lesson_14.json) 平台与算法配对:flow/self-driving 闭环(FT-IR/UHPLC 在线反馈)适合小批量 BO 与 RL 序列决策;HTE 微反应器适合 GP 代理多目标;无闭环的离线批量数据才用 LHS/正交表 DoE + ANN。
- (lesson_15.json) 识别'伪优化'语料：本 chunk 40 篇里 9 篇最终 router=none 属负样本（超参调优、参数标定 ABC/MCMC、RL 控制、纯代理+SHAP 人工选点）。router 训练/校准时需保留这类负例，避免见到 'Bayesian/TPE/GA' 就触发实验 BO。
- (lesson_15.json) 按变量类型分派：连续型少因子（≤5）多目标→parego（顺序小预算）或 qehvi（可并行）；离散候选库占主导→先估基数，<10^5 且打分廉价用 grid 穷举，否则用类别核 BO/active learning；混合空间用混合核而非简化为连续。
- (lesson_15.json) 多响应（≥3 个指标）不要用多次标量化或 NSGA-II 跑大规模种群，改批量 qEHVI + TOPSIS 式后验选点交付实验；初值统一用 LHS/Taguchi 铺 10–27 点，性价比高于全因子 DoE。
- (lesson_15.json) 有硬约束（能级/相似度/组分和/工艺窗口）或噪声（复孔、仪器漂移）时，路由到 constrained 或噪声感知 GP-BO，并在摘要证据里核对作者是否真的报了复孔/σ，不要把'未提及'当成 noisy=true。
- (lesson_16.json) 变量类型先分两段处理：类别型候选（材料种类、工艺路线、壳层化学）用 RSM/单因素/DoE 筛到少数几项，再对连续配比或工艺参数上 GP-BO；直接对 10+ 类别做 BO 或全 grid 都不划算。
- (lesson_16.json) 按目标数路由：2 目标选 parego，3 目标选 qehvi，4 目标或数据噪声大选 qnehvi；摘要里出现 activity/stability/cost 这类并列指标时，即使元数据写 multi_objective=false 也按 multi 处理。
- (lesson_16.json) 先判断优化问题是否可写成连续向量：离散组合 + 硬约束（QUBO、子集选择）应返回 none；带漂移的时变目标要用 constrained/TVBO 而非静态 qehvi；纯超参搜索用 tpe。
- (lesson_16.json) BO 与 GA/CMA-ES 在低维连续工艺参数上常收敛到同一最优，BO 的优势体现在实验次数而非代理模型内层优化速度；当代理评估几乎免费时，不要用 BO 去替代进化算法做内层搜索。
- (lesson_17.json) 小样本连续配方空间(几十个点、十几个因子)不要直接高维 GP:先 PCA(基线)→ 必要时 t-SNE/UMAP 降维,再在低维空间跑 GP-BO,并把配方反向映射回组成空间;降维是 BO 能否可用的前置步骤而不是替代方案。
- (lesson_17.json) 多目标采集函数按目标数选:2-3 目标用 qEHVI/qPMHI 类批量超体积改进(可加性分解、支持大 batch);目标数 ≥4 用 qNEHVI;离散分子池用 generate-then-optimize(先大批生成候选再排序选批),比端到端连续隐空间优化更稳、更容易扩展。
- (lesson_17.json) 有廉价保真源就上多保真:文献汇总数据 + 代理模型预测构成低保真层,MFBO 能把高保真实验压到十几次(降 75% 以上);只有在没有低保真来源时才用单保真 BO。
- (lesson_17.json) 区分'预测建模'与'序贯优化':只为训练代理模型而扫参数(如 918 组 TCAD 工艺条件)用 LHS/空间填充 DoE 即可,套 BO 属过度设计;而噪声大、标签稀疏的离散筛选应先做成对比较/概率排序再批量选点,TPE 类密度模型比点式 GP 更耐受噪声与类别变量。
- (lesson_18.json) 类别变量按基数分档处理：1-2 个二值类别（配体 MPA/Cys、还原方式）直接用 mixed kernel GP-BO；10-20 级高基数类别（树脂池、溶剂池）需分组核或物理化学描述符编码，one-hot 下 GP 退化时切 TPE 或先投影到潜空间；只有 1 个离散主因子且水平 <=5 时才用穷举 grid，不要为它上代理模型。
- (lesson_18.json) 多响应小样本优先 Pareto 方法：RSM 的 desirability 加权实质就是随机权重标量化，与 ParEGO 同构；2 目标用 qEHVI 直接算超体积，>=3 目标或需多次重跑用 ParEGO。摘要一旦报告 lack of fit 或 predicted R2 高但实测偏差大，最优点必须降级为待验证点，并改为逐响应 GP 建模。
- (lesson_18.json) 用因子数决定 DoE 还是 BO：3-4 个连续因子的 CCD/BBD/Doehlert（15-30 组）可接受，但只要有流动/自动化/机械臂执行器就应切 GP-BO，通常能省一半实验；因子数 >=6（如反应/诱导/发酵共 8 因素）继续单因素+RSM 是预算浪费，应上高分批坐标式 BO 或先做敏感性筛因子。
- (lesson_18.json) 保真度与决策类型走两条独立分支：直接/间接测量或仿真-实验并存时用 co-kriging 多保真（mfbo）先融合再选点；目标是策略/时序（电位脉冲、补料曲线）时用 RL，但静态参数仍先用 BO 建基线；无闭环执行器、只能一次性批量做实验时退回 LHS 或 Doehlert 一轮空间填充再局部细化。
- (lesson_19.json) 本 chunk 的 40 篇里约三分之二是机器人硬件、医学影像、综述与生物传感/检测类文献(BO 相关证据稀缺),算法名几乎不出现在摘要;打标签时只看是否真的存在'条件空间+目标',不要因 automation_tags 含 robotic 就假定用了 BO。
- (lesson_19.json) 区分测绘(mapping)与寻优(optimization):近 1000 点全网格适合重建条件-产物网络(Robowski);若目标是单一产物选择性,同预算应换批量 BO(qNEI/qEHVI),把网格当'默认安全选项'会浪费 10 倍以上实验。
- (lesson_19.json) 变量少(<=4)且测量廉价、响应平滑时,CCD/LHS 一次性 DoE 与 BO 效果相当,优先可解释的 DoE;只有单次评估昂贵(器件寿命、昂贵表征)或需混合离散空间时,才优先 TPE/GP-BO。
- (lesson_19.json) 执行层约束与设计层约束要分开:SDL 的工序先后/工位/时间同步由调度器解决,router 的 constrained 选择只应对设计空间的可行性约束;噪声与系统偏差先靠复孔+分块控制,再选噪声感知采集函数。
- (lesson_20.json) 本chunk以医学影像/临床AI/生物工程综述为主（PRISMA/叙述性综述、评论、影像分割），与实验条件优化无关，40篇仅7篇可标；本chunk统计量不应赋予与其他chunk相同的权重。
- (lesson_20.json) 小离散空间（2-3个类别因子、水平少、单目标）在真实化学文献里仍以全因子/OFAT 为主：给这类问题配 BO 会显得过度工程化，router 应能对 categorical_count<=3 且预算<20 直接回 grid。
- (lesson_20.json) 自动化平台与算法选择要解耦：flow/HTE 只决定'每轮能测多少'，不决定用不用模型；真正的判据是变量类型（条件化离散→tpe；纯/低维连续→bayes）、评估成本与是否已有廉价代理打分。
- (lesson_20.json) router 目前缺 evolutionary/DE 与纯枚举两条路径：DE 用于廉价可并行的高维连续搜索（如网络融合层权重），超大规模廉价枚举用于片段/库筛选；这两类应显式返回 none 并说明'不属于序贯实验优化'，而不是硬塞 bayes。
- (lesson_21.json) Single-continuous-variable problem with a non-monotonic response (cellulose 0-20 wt%, optimum 15 wt%): route to grid; do not spend BO budget on a 1-D sweep.
- (lesson_21.json) 3-4 continuous factors with several responses and no automation: run Sobol/LHS screening first, then ParEGO/qEHVI; the literature's one-factor sweeps are a baseline, not a target.
- (lesson_21.json) When the abstract only states a post-hoc optimum (e.g. '1:1, 300 W, 10 min' or '0.75 wt% loading'), infer variable counts from the stated factors and mark confidence low; never present it as evidence for a named optimizer.
- (lesson_21.json) Treat 'closed-loop', 'microfluidic', and 'robotic' as hardware descriptors unless the abstract shows an algorithm selecting the next experiment; keep automation=none and avoid training automation-to-algorithm pairs on such text.
- (lesson_22.json) 配方类问题先问'实验贵不贵、因子几个':2-4 个连续因子且测量便宜时,LHS/正交表 + 多响应加权评分比 BO 更省事;只有单点昂贵或需 20+ 轮闭环时才切 BO。
- (lesson_22.json) 类别维度是 router 的分水岭:<=3 个类别因素、总因子 <=5 → 混合设计覆盖(one-hot + LHS/D-optimal);>=15 维且噪声大 → TPE/密度估计而非 GP-BO,并允许 LLM/专家先验生成候选。
- (lesson_22.json) 多目标别默认 qEHVI:本 chunk 的双/多响应问题全是顺序小预算,ParEGO 标量化(或加权评分)才匹配;qEHVI 的前提是能并行测一批。
- (lesson_22.json) 把规格内/可行性当硬约束喂给 constrained BO;维度高时先 Sobol 敏感度筛掉不敏感因子再优化(压片线 6 h 达 in-spec、省 65% API 是可复用的基准数字)。
- (lesson_23.json) 单因子工艺扫描(晶化温度、DES 摩尔比)在文献里就是 grid/一维穷举,而且常常几十个点就够;router 遇到 continuous_count<=2 的廉价实验不要冷启动 BO,先用 grid 把趋势和最优点位置拿到。
- (lesson_23.json) 混合空间(1~2 个类别因素 + 1~2 个连续配比)在配方/聚合物文献里以人工系列相图为主:先 lhs/正交铺点建立趋势,再针对单一标量目标切 bayes,才能真正体现 BO 的样本效率优势。
- (lesson_23.json) 多目标 + 约束 + 噪声的配方问题(水凝胶、缓释载体)应走 qehvi(批量、混合核)或 parego(顺序小预算),并把生物相容性/工艺窗口作为可行性约束建模,而不是加权成单目标。
- (lesson_23.json) 本段 40 篇里真正含优化方法证据的只有 6 篇:crystallization 类查询召回的多是综述与机理论文;router 校准时该段应主要提供'grid/lhs 起步、none 占多数'的先验,不要被少量 BO/ML 综述抬高三者对 bayes 的默认权重。
- (lesson_24.json) 把 chunk_24 当'负样本'用:36/40 篇没有算法证据,说明本领域大量'条件优化'只是叙述性 trade-off;router 不要因为标题里有 battery/electrocatalyst/optimized 就切到 bayes。
- (lesson_24.json) 见到 self-driving lab + 电化学读出(尤其 CV/电沉积)可直接给 bayes 低置信先验;一旦同时出现孔板/多通道并行批次,升级 qehvi,而不是继续单点 BO。
- (lesson_24.json) 可枚举的离散设计库(点阵单元胞、候选清单)优先 grid/枚举筛选,代理模型只用于筛选后的连续微调;这是 categorical_heavy 且候选少的默认路径。
- (lesson_24.json) 材料类多目标(能量密度 vs 寿命、容量 vs 倍率)在摘要层面只能读到目标冲突、读不到方法:标注一律 low confidence,且 constraints/noisy 用 null,避免把'论文没写'误判成'无噪声/无约束'。
- (lesson_25.json) If the abstract shows only 1-2 continuous variables plus a reported optimum (e.g. resistance + NaCl concentration, or a Br:Bi ratio series), recommend grid/one-factor DoE; BO is over-engineering at that dimensionality.
- (lesson_25.json) For mixed categorical+continuous spaces with fewer than ~100 runs, route to TPE (tree/RF surrogate) rather than GP-BO; the only labelled mixed case in this chunk used RF/HistGradientBoosting and then extracted optimal conditions plus catalyst ranking.
- (lesson_25.json) When the automation tag is HTE/plate-based screening without a closed loop, start with space-filling LHS and only switch to BO after the initial label set exists; sequential single-point acquisition wastes plate capacity.
- (lesson_25.json) Do not trust the source algorithm_terms / noisy / constraints fields as ground truth: 'TPE' appears in this chunk on an ECL-imaging paper and a review (false positives), and most constraint/noise flags are contradicted by the abstracts.
- (lesson_26.json) chunk_26 的 40 篇里只有 11 篇能读出优化问题,且没有一篇点名 BO/TPE/EHVI:光催化与交叉偶联领域的算法命名率极低,router 不能靠 'optimized' 字样触发 bayes,必须用变量类型/预算/平台特征做决策。
- (lesson_26.json) 类别型(配体/碱/溶剂/底物)与连续型(当量/温度/时间)并存的混合空间,文献主流是分阶段 DoE(HTE 全因子筛查→显著因子 CCD→稳健性验证);router 对应 lhs/grid 起步,只在实验昂贵且需序贯收窄时才切 bayes/tpe。
- (lesson_26.json) HTE+ML 的论文要区分'预测'与'优化':没有 acquisition function/闭环描述就记 literature_method=other;反之 self-driving/闭环平台才给 BO 先验。离散配体或底物类别空间默认 tpe,不要用 GP 连续核硬套 one-hot。
- (lesson_26.json) noisy=true 与 multi=true 是两个独立路由信号:高噪声 HTE 组分数据(离群点即最优)先给噪声感知 bayes 并保留下侧极值;双目标且可批量并行给 qehvi,顺序单点且目标数>2 给 parego。
- (lesson_27.json) 本 chunk 最强的信号是"DoE 仍主导连续工艺优化":5 篇独立论文(3-5 个连续因子、单响应)全部用 Box-Behnken/CCD 而非 BO。router 遇到 continuous_heavy + 单目标 + 无闭环平台时,默认应先给 lhs/DoE 而不是 bayes,只有维度低或样本极贵时才改用 GP-BO。
- (lesson_27.json) 类别型条件选择的两条现实路径:有万级历史数据 → 训练分类/排序模型直接推荐试剂类别(tpe 式离散搜索);只有小候选库 → 穷举筛选再进 DoE。判断依据是"候选是否可枚举 + 是否有历史数据",不是分子大小或领域。
- (lesson_27.json) 目标数与响应数要分开读:摘要写 yield + selectivity 或 yield + BET + 粒径时,响应数常 >=2 而元数据 n_objectives 仍为 2 甚至 false。router 应按摘要里的响应个数选 parego(2)/qehvi(2+批量)/qnehvi(>=3),不要照抄元数据。
- (lesson_27.json) 把"评估成本"作为 bayes vs random/evolutionary 的第一判据:廉价 ML 代理 + 1e19 空间用 GA/PSO/随机,昂贵湿实验 + 低维连续才用 GP-BO;自动化平台(automated synthesis/active learning)是升级到闭环 BO 的加分项,但文献里多数仍停在"离线代理+验证实验"。
- (lesson_28.json) 摘要出现 Box-Behnken/RSM/中心复合设计时，说明该领域默认 DoE 而非 BO：router 用 lhs 起步，只有因子多或单次评估昂贵时才升级 bayes。
- (lesson_28.json) 标 objectives=multi 之前先看是否被 GRA/加权/单指标替代；作者做了标量化时仍路由 parego，但 confidence 降一档，避免高估其多目标能力。
- (lesson_28.json) space_type 按变量类型而非领域判定：整数阶数、片段替换、采伐选择等离散决策即使数量少也记 categorical_heavy，路由 tpe/constrained；连续温压参数才用 GP-BO。
- (lesson_28.json) flow/HTE 只说明平台自动化，不等于优化算法先进；'平台强、优化器弱（OFAT）'的论文恰是 BO 收益最大的证据来源，本 chunk 无 self-driving/HTE/robotic 样本，平台-算法配对不可外推。
- (lesson_29.json) 自动化平台→算法配对在本 chunk 是干净的一对一:robotic/active learning 闭环 ⇒ bayes(批量则 qehvi)。看到 'liquid handling robot'、'seed library'、'iterations'、'active learning' 就应直接走贝叶斯路线,不要先跑 sobol 冷启动烧样本。
- (lesson_29.json) 先按'能否穷举'分流离散空间:候选库可枚举(14.8 万组成、候选清单)→ grid 枚举;组合爆炸且单点昂贵(掺杂位点、离散溶剂)→ tpe 或离散代理 BO 并分块并行。纯离散空间不要用连续核 GP-BO 的默认先验。
- (lesson_29.json) 因子少(≤4)、响应多(≥3)、样本便宜的配方/混料问题,文献主流仍是 DoE(Taguchi 筛选→CCD/混料设计→desirability→确认运行);router 此时给 grid 比给 bayes 更贴合实践,并把'区间/规格约束'计入 constraints。
- (lesson_29.json) 多目标路由按约束与并行度分叉:有硬约束或并行批次→qehvi,2 目标顺序小数据→parego;摘要只出现 balancing/trade-off 而没有优化器名时不要据此调整多目标先验,confidence 保持 low、noisy/constraints 用 null。
- (lesson_30.json) 自驱动/并联反应器 + 噪声 + 连续与类别变量混杂时直接选 bayes（BED/BO）；这是本 chunk 唯一 high confidence 的自动化—算法配对，可作为 SDL→bayes 的默认规则。
- (lesson_30.json) 先把任务分成“实验条件优化”与“模型超参调优”：后者（Optuna/XGBoost 调参）走 tpe，且文献证据是 BO > grid/random；可用于校准 tpe，但不要并入条件优化的统计。
- (lesson_30.json) 选 router 前先估单次评估成本：候选数达 1e9 而代理打分近乎免费时用 grid 枚举，单次真实评估昂贵时才用 bayes——序贯 BO 并非默认更优。
- (lesson_30.json) 摘要出现 2–3 个并列性能指标时标 multi 并路由 parego/qehvi，不要压成单目标；若同时点出兼容性等硬约束则用 constrained。
- (lesson_31.json) 自动化平台→算法配对在本 chunk 是干净的一对一:flow + 在线实时读出、GP-BO 主动学习循环、并行混合空间 BO 一律对应 bayes;看到 'fully automated'、'active learning loop'、'within N iterations'、'pool of N candidates' 就直接走贝叶斯路线,不要用 sobol 冷启动烧样本(有并行批次时再升 qehvi)。
- (lesson_31.json) 离散空间先判'能否穷举':候选池 3700/36万但评估昂贵→池内迭代采样(bo/discrete);组合爆炸且只有 60-100 次实验预算→tpe 或少样本代理先验筛;候选 ≤1e5 且评估便宜→grid 全枚举。纯组合空间不要套默认连续核 GP。
- (lesson_31.json) 连续参数 d=4-20 且单目标时,文献主流是代理模型顺序优化(SMAC/BO+Hyperband、PHBO、流式闭环 BO);只有在作者明显只做了单因素试验、且因子数 ≤4 时才给 lhs/DoE 类初始设计,并把'已报告最优值'当作 BO 的强先验而不是新证据。
- (lesson_31.json) 多目标不要只按'目标数'定方法:先看维度与约束——低维 2-3 目标要交付 Pareto 前沿给 qehvi,高维(≥10)或顺序小预算给 parego,有硬可行性约束先走 constrained/可行性筛选;本 chunk 多目标论文都没点名 ParEGO/qEHVI,相关先验只能记 low。
- (lesson_32.json) 本 chunk 是“无方法证据”样本：摘要只给 optimal/optimized 单点时仍可打标签（problem_type 与 space_type 可从变量语义可靠判断），但 literature_method 一律 unknown、confidence=low，绝不能把作者的单点最优当作 DoE 结论。
- (lesson_32.json) 变量计数缺失（space 全为 null）时按语义估空间类型而不硬猜个数：级联反应→连续（浓度/载量/pH/温度/时间）；配方→mixed（种类离散+配比连续）；categorical_count/continuous_count 填 null。
- (lesson_32.json) 自动化平台在本 chunk 近乎缺失（仅 1 篇 robotic 标签且与优化无关），因此 “HTE/self-driving → BO” 配对在本 chunk 无证据，不得用它加强先验；也不要把 TPE 字样直接当作优化器（10.3390/polym18172133 的 TPE 为误抽取）。
- (lesson_32.json) 多目标只在“权衡”措辞里出现：默认 parego 标量化，等出现并行批次/硬约束再升 qehvi；本 chunk 无任何 Pareto 前沿数据，无法验证 qehvi/qnehvi 的收益。
- (lesson_33.json) 看到 multi-fidelity 字样(5 vs 800 epochs、解析/CFD/实测三级、2500 LF vs 500 HF)就直接给 mfbo:本 chunk 三篇多保真论文无一例外用低保真探索+高保真精修,单保真 BO 会被迫在高保真上多花一个量级的预算。
- (lesson_33.json) 类别/离散主导不等于一律 BO:12 个二元变量的风险策略选择用 MILP 精确求解;离散实体组合的微结构用图代理+主动学习;含类别与条件依赖超参的调参用 TPE+剪枝。router 在 categorical_heavy 时应先问'候选规模'——小到可枚举就给 none/精确法。
- (lesson_33.json) 多目标先看执行方式再选算法:可批量并行(HPC FEA/CFD、多通道)→ qehvi 逼近 Pareto;只能单点顺序或指标已融合成评分 → parego/标量化 bayes;摘要里出现 Pareto-optimal + HPC 是 qehvi 的强信号。
- (lesson_33.json) 约束要么进可行域要么进目标,别做无约束搜索:有硬边界(±0.25 D、隐私预算、制造可行性)时用 constrained/可行性分类;只有软惩罚时才退回标量化 bayes,并保留噪声感知(复孔/仿真随机性)。

## Caveat

These labels come from titles/abstracts and LLM judgement, not from
re-running experiments. They are router *priors*, to be tested on
benchmarks/proxy campaigns before changing defaults.

Reproduce:

```powershell
py -3 scripts/literature_experience_report.py --input research/literature_scan
```