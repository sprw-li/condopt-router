# PAPERS NOTES — 关键论文要点笔记（供实现时回查）

> 维护者：algo-research ｜ 2026-11-05
> 说明：每条先给**已核实的元数据**，再给"实现要点"（基于论文方法的整理，不是逐字摘要；逐字摘要仅 R4 TPE 那条是从 NeurIPS 官方 abstract 抓的）。
> 编号对应 `REFERENCES.md`。

---

## R4 TPE (Bergstra et al., NeurIPS 2011)

**元数据**：NeurIPS 2011，4 作者（Bergstra, Bardenet, Bengio, Kégl），官方页 https://proceedings.neurips.cc/paper_files/paper/2011/hash/86e8f7ab32cfd12577bc2619bc635690-Abstract.html

**官方 abstract 关键句（抓取）**："...we optimize hyper-parameters using random search and two new greedy sequential methods based on the expected improvement criterion... This work contributes novel techniques for making response surface models P(y|x) in which many elements of hyper-parameter assignment (x) are known to be irrelevant given particular values of other elements."

**实现要点**：
- 双 Parzen 密度：`l(x)=p(x|y<y*)`、`g(x)=p(x|y≥y*)`，`γ=p(y<y*)`；`EI ∝ (γ+(1-γ)·l/g)^{-1}` → 最大化 `l/g`（log 空间）。
- 每维独立 KDE + 先验混合；数值维可用自适应带宽（hyperopt 的 adaptive Parzen）。
- 候选不是均匀采样，而是**从 `l` 抽 `n_EI_candidates`（默认 24）**再打分。
- 关键洞见："很多参数在给定其他参数时无关" → 树状/条件空间的天然支持（TPE 的最强卖点）。

---

## R12 qNEHVI (Daulton et al., NeurIPS 2021)

**元数据**：arXiv:2105.08195；作者 Samuel Daulton, Maximilian Balandat, Eytan Bakshy；2021。

**实现要点**：
- 目标：`E[HV(P ∪ {f(X)}) - HV(P)]`，`P` 由 baseline 观测决定；噪声下 `P` 也随机 → 对 baseline 做嵌套 MC 积分。
- 与 qEHVI（2020, arXiv:2006.05078）的差别：qEHVI 假设 baseline 已知/无噪；qNEHVI 把 baseline 的观测噪声积分掉，**不会因幸运观测而过早收敛**。
- BoTorch 接口：`qNoisyExpectedHypervolumeImprovement(model, ref_point, X_baseline, sampler=...)`；`X_baseline` 取已评估点（n 大时抽样或 `prune_baseline=True`）。
- 实践：2–3 目标直接上；ref_point 必须按业务定义（不是默认 nadir）；q 大时用 cyclic 优化。

---

## R17 BayBE (Fitzner et al., Digital Discovery 2025)

**元数据**：DOI `10.1039/D5DD00050E`；Digital Discovery 2025；GitHub `emdgroup/baybe`（Apache-2.0）。

**实现要点**：
- `Campaign` ↔ JSON：整个实验活动（空间、约束、推荐器、已有测量）可序列化/反序列化 → **agent 传配置、断点续跑**。
- `SearchSpace` 内部**离散化**（连续维切档、categorical 枚举、substance 用描述符）→ 候选池枚举是默认执行路径。
- `TwoPhaseMetaRecommender`：低数据阶段用随机/LHS/FPS，足够数据后切 BO；切换点可配。
- `SubstanceParameter`：化学物质作为一等参数（SMILES/名称 + 描述符），是通用 HPO 库没有的东西。

---

## R19 LLAMBO (Liu et al., ICLR 2024)

**元数据**：arXiv:2402.03921；作者 Tennison Liu, Nicolás Astorga, Nabeel Seedat, Mihaela van der Schaar。

**实现要点**：
- LLM 增强 BO 的三个位置：①warm-start 初始点（少样本时增益最大）；②候选提议（替代 Sobol 随机候选，进采集函数优化）；③（可选）经验性的代理/似然估计。
- 结论：低预算 + 空间语义可描述（变量名/单位/领域知识）时最有效；**GP + 数值采集函数保留兜底**。
- 工程：LLM 输出要结构化成"点列表 + 理由"，过 schema 校验；prompt 里放历史表格 + 变量说明 + 约束。

---

## R14 TuRBO (Eriksson et al., NeurIPS 2019)

**元数据**：arXiv:1910.01739；作者 Eriksson, Pearce, Gardner, Turner, Poloczek。

**实现要点**：
- 在超矩形 trust region 内拟合 GP 并做标准 EI；`L`（边长）按连续成功次数加倍、失败则减半；多个 trust region 并行（batch）。
- 高维（10–200）显著优于全局 GP；实现简单，**router 在 d>20 时可直接切换**。
- 参数：初始 `L=0.8·range`、`L_min=2^-7`、`L_max=1.6`；成功阈值 `success_tol`。

---

## R15 Vanilla BO in high dims (Hvarfner et al., ICML 2024)

**元数据**：arXiv:2402.02229；作者 Carl Hvarfner, Erik Orm Hellsten, Luigi Nardi。

**实现要点**：
- 两个小改动让普通 GP-BO 在高维打平 TuRBO：①lengthscale 先验按维度缩放 `LogNormal(√2+0.5·ln d, √3)`；②更好的 `get_max`（最大化后验均值）候选策略。
- 结论：**先试 vanilla + 好先验，再考虑复杂局部化**；TuRBO 的增益在噪声/非平稳时被高估。
- 对实现的意义：默认配置里就把 lengthscale 先验设成维度相关的，而不是 Gamma(3,6)（后者在高维会过度收缩）。

---

## R13 Garrido-Merchán & Hernández-Lobato, Neurocomputing 2020

**元数据**：DOI `10.1016/j.neucom.2019.11.004`；Neurocomputing 380:20–35。

**实现要点**：
- categorical：one-hot 编码 → 在连续松弛 `[0,1]^k` 上优化采集函数 → **投影到最近合法 one-hot** 再评估。
- integer：连续松弛 + 取整；取整造成的重复用"邻域扰动"再采样。
- 意义：让"只支持连续空间"的 BO 库能跑混合空间；但高基数 categorical 仍会维度爆炸（改用描述符/TPE）。

---

## R20 FunSearch (Romera-Paredes et al., Nature 2024)

**元数据**：DOI `10.1038/s41586-023-06924-6`；Nature 625:468–475。

**实现要点**：
- 搜索对象是**程序**：LLM 基于"程序数据库中的高分程序 + evaluator 打分"做变异/交叉；岛屿模型维持多样性。
- 与 BO 的分工：BO 搜数值向量；FunSearch 搜"构造/规则/配方"。二者可用同一个 evaluator 闭环。
- 工程：程序数据库 + 去重 + 打分 + prompt 模板是核心；LLM 只做生成，正确性/打分由执行器保证。

---

## 待补笔记（下轮）

qEHVI（arXiv:2006.05078）分区实现、Gryffin（10.1063/5.0048164）categorical 描述符核、BOCA/UAA 多保真公式、Letham 2019 截断后验推导、OPRO（arXiv:2309.03409）prompt 轨迹设计。
