# AUTONOMOUS_LAB_POSITION — 自主实验室架构与"条件优化层"的边界

> 维护者：algo-research ｜ v1（2026-11-05，Round 3）
> 用途：给 `README.md` / 愿景文档/申报材料提供**可直接引用的结论**（含 DOI/URL），以及"condopt 在自主实验室里到底负责什么"的边界说明。
> 主要依据：R33（A-Lab）、R27（ChemOS）、R37（MacLeod 自驱动实验室）、R38（ChemCrow）、R20（FunSearch）、R19（LLAMBO）、R17（BayBE）、R18（Olympus）、R8（SMAC3）、R10（Letham）。

---

## 1. 一个自主实验室的四层结构

```
┌─────────────────────────────────────────────────────────────┐
│ L4 语义/规划层   LLM agent：理解目标、拆任务、调工具、解释结果      │  ChemCrow [R38], LLAMBO [R19]
├─────────────────────────────────────────────────────────────┤
│ L3 条件优化层   「下一次做什么实验」的决策内核                     │  ← condopt 在这里
│   搜索空间/代理模型/采集函数/批量/约束/多目标/多保真/历史回放        │
├─────────────────────────────────────────────────────────────┤
│ L2 编排/执行层   实验请求→机器人/人工执行→测量→结果回写、安全/重试   │  ChemOS [R27], A-Lab [R33], MacLeod [R37]
├─────────────────────────────────────────────────────────────┤
│ L1 数据/记录层   实验记录、谱图、元数据、审计日志                   │  ChemOS/PLOS ONE, BayBE campaign JSON [R17]
└─────────────────────────────────────────────────────────────┘
```

**condopt 的定位**：L3 的"决策内核"——只回答"下一批实验条件是什么、为什么、不确定度多大"；**不碰硬件、不做安全判断、不做语义规划**。

---

## 2. 可直接引用的 8 条结论（README/愿景用）

### 结论 1：自主实验室的闭环里，优化算法只是其中一层，价值在于"每目标所需的实验数"
A-Lab 在 17 天内对 58 个目标化合物完成自主合成，其核心是"文献/数据库先验 + 主动学习规划下一批实验 + 机器人执行"的闭环，而不是某个更复杂的采集函数 [R33]。这决定了 condopt 的产品指标应该是**样本效率（每目标实验数）**，而不是"算法跑分"。
> Szymanski et al., *Nature* 624, 86–91 (2023). https://doi.org/10.1038/s41586-023-06734-w

### 结论 2：优化器必须与执行器解耦（planner / executor 分离）
ChemOS 把"优化器产出实验请求"与"硬件执行/回传"用 RPC 解耦，使同一个 planner 可以接不同仪器 [R27]；MacLeod 等人的自驱动薄膜实验室同样把"BO 建议"与"机器人合成/表征"分层 [R37]。condopt 因此只暴露 `ask/tell` 与 JSON 请求，不依赖任何执行器实现。
> Roch et al., *Science Robotics* 3(24), eaat5559 (2018). https://doi.org/10.1126/scirobotics.aat5559
> MacLeod et al., *Science Advances* 6(20), eaaz8867 (2020). https://doi.org/10.1126/sciadv.aaz8867

### 结论 3：LLM 负责语义与工具调用，数值决策必须留给优化算法
ChemCrow 让 LLM 规划"用哪些化学工具、按什么顺序"，但具体合成参数的计算与执行由工具完成 [R38]；LLAMBO 也只把 LLM 放在 BO 的冷启动/候选生成等环节，采集函数仍由 GP 计算 [R19]。因此 condopt 的 router 可以让 LLM 选算法/解释结果，但**不能**让 LLM 直接决定数值采样点。
> Bran et al., *Nature Machine Intelligence* 6, 525–535 (2024). https://doi.org/10.1038/s42256-024-00832-8
> Liu et al., "Large Language Models to Enhance Bayesian Optimization", *ICLR* 2024, arXiv:2402.03921

### 结论 4：当"候选"是程序/流程而不是数值向量时，LLM 演化是 BO 之外的第二范式
FunSearch 用"LLM 生成程序 + 自动 evaluator 打分 + 程序数据库"在数学构造上取得新结果 [R20]。对应到化学：配方/流程/规则类搜索空间（而非连续条件）适合"LLM 提议 + 数值评估"，与 condopt 的数值 BO 互补而非替代。
> Romera-Paredes et al., *Nature* 625, 468–475 (2024). https://doi.org/10.1038/s41586-023-06924-6

### 结论 5：实验预算昂贵且噪声大，样本效率与不确定性报告是硬需求
自主实验室的单次实验成本高（合成、表征、清洗），且测量噪声不可忽略；A-Lab/ChemOS 都以"少量实验内改进"为核心指标 [R33][R27]。condopt 的推荐必须同时给出**预测均值、不确定度、可行性概率**，而不是一个点。
> 参见 R33、R27；噪声实验基准见 Häse et al., *Mach. Learn.: Sci. Technol.* 2, 035021 (2021). https://doi.org/10.1088/2632-2153/abedc8

### 结论 6：可复现/可审计是自主实验室的合规底线
ChemOS 记录每次实验请求与结果并可重放 [R27]；BayBE 把整个 campaign（空间、约束、推荐器、测量）序列化为 JSON [R17]。condopt 的 `Study` 目录（problem/config/dataset/pending/rounds/decisions）就是为审计与断点续跑设计的。
> Roch et al., *Science Robotics* (2018) 同上；Fitzner et al., *Digital Discovery* (2025). https://doi.org/10.1039/D5DD00050E

### 结论 7：失败/不可行实验也要回写进优化循环
SMAC3 的 RunHistory 用状态机记录 SUCCESS/FAILED/CRASHED，并把失败配置纳入历史 [R8]；带噪约束 BO 必须把"测出来不可行"当作信息而不是丢弃 [R10]。condopt 的 `tell` 必须支持部分结果与失败记录（见 `REVIEW_v01.md` P0-5）。
> Lindauer et al., *JMLR* 23(54), 1–9 (2022). https://jmlr.org/papers/v23/21-0888.html
> Letham et al., *Bayesian Analysis* 14(2), 495–519 (2019). https://doi.org/10.1214/18-BA1110

### 结论 8：基准必须来自真实实验噪声的 emulator，而不是纯合成函数
Olympus 提供带真实噪声的化学 emulator 与统一 Planner 接口，用来避免"合成函数上很强、真实实验上失效" [R18]。condopt 的 benchmark 采用"合成 + 化学 emulator + 真实历史数据"三层（见 `BENCHMARK_SPEC.md`）。
> Häse et al., *Mach. Learn.: Sci. Technol.* 2, 035021 (2021). https://doi.org/10.1088/2632-2153/abedc8

---

## 3. 边界：condopt 负责什么、不负责什么

| 关注点 | condopt（L3） | 不应由 condopt 承担 |
|---|---|---|
| 搜索空间 | 变量类型、编码/解码、条件空间、约束投影 | 仪器能力/安全边界（执行器层） |
| 代理模型/采集 | GP/森林/TPE、EI/qNEI/qEHVI、批量/约束/多目标 | 谱图解析、物性计算（工具层） |
| 实验请求 | 生成待测条件 + 预测 + 不确定度 + 理由 | 硬件指令、加料顺序执行（执行器层） |
| 历史 | 观测记录、pending、轮次、审计、可复现 | 原始数据存证/合规（L1 数据层） |
| 决策 | 算法选择（router）、参数策略、reroute | 项目级目标拆解、文献检索（LLM/规划层） |
| 失败 | 记录 failed/expired、部分 tell 语义 | 报警/停机/人工复核流程（编排层） |

**接口原则**（写进 README）：
1. condopt 只依赖"实验请求/观测"两个 JSON 契约（`ARCHITECTURE_PATTERNS.md` §7.2）；
2. 任何 LLM/agent 只能**在 condopt 给出的可行算法集合内**选择（`CAPABILITIES_SPEC.md`）；
3. 每个推荐必须可解释（理由 + 预测 + 不确定度），并写入 `decisions.jsonl`。

---

## 4. README 可用段落（可直接粘贴）

> **Why a dedicated condition-optimization core?**
> 自主实验室（A-Lab、ChemOS、自驱动薄膜实验室）证明了"机器人 + 主动学习"闭环的可行性 [R33][R27][R37]。但这类系统的公开实现通常把优化算法与编排、硬件耦合在一起，难以复用、难以审计。condopt 把其中**唯一与领域无关、但决定样本效率的一层**——条件优化——抽成独立库：混合搜索空间、GP/TPE/进化采样器、批量/约束/多目标、可持久化的 ask/tell 历史，以及一个可解释的算法 router。上层 agent 只负责语义与工具调用，下层执行器只负责把条件变成实验。

> **What it is / is not**
> condopt 是"下一次做什么实验"的决策内核；它不是实验编排器（ChemOS 的角色）、不是 LLM agent（ChemCrow 的角色）、也不是数据管理系统。它通过一个稳定的 `ask/tell` 契约与它们协作。

---

## 5. 一句话定位（供标题/摘要）

**condopt = 自主实验室的"条件优化内核"**：用样本高效的贝叶斯/进化/TPE 采样与可审计的 ask/tell 历史，把"下一批实验做什么"从经验判断变成可复现、可解释的决策；上层交给 agent，下层交给机器人。

---

## 6. 参考链接汇总

- A-Lab：https://doi.org/10.1038/s41586-023-06734-w
- ChemOS：https://doi.org/10.1126/scirobotics.aat5559 （详细版 https://doi.org/10.1371/journal.pone.0229862）
- MacLeod 自驱动实验室：https://doi.org/10.1126/sciadv.aaz8867
- ChemCrow：https://doi.org/10.1038/s42256-024-00832-8
- FunSearch：https://doi.org/10.1038/s41586-023-06924-6
- LLAMBO：https://arxiv.org/abs/2402.03921
- Olympus：https://doi.org/10.1088/2632-2153/abedc8
- BayBE：https://doi.org/10.1039/D5DD00050E
- SMAC3：https://jmlr.org/papers/v23/21-0888.html
- Letham 带噪约束 BO：https://doi.org/10.1214/18-BA1110
