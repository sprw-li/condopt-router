# 论文战役复现（2026 自动化实验室论文 → condopt 取点）

> 用户要求「从这一个月的自动化实验室论文里找复杂实际样例跑一下」。
> 本机常见目录（本机若干常见工作目录）没有找到该月的自动化实验战役记录，
> 因此按用户要求改为**网上找论文**，并用两篇 **2026 年论文** 重建复杂战役。
> 这里只演示“验证空间 → 路由 → ask/tell 闭环 → 报告”，**不涉及真实实验**。

## 案例 1：ART 金属光氧化还原 HTE（Angew 2026）

- 论文：*Combining High-Throughput Experimentation and Automated Flow for Bayesian
  Optimization of a Metallaphotoredox Reaction*, Angew. Chem. Int. Ed. 2026, 65, e3129364
  （2026-05 上线，不算“本月”，但属于 2026 年自动化/BO 实验论文）
- 链接：<https://pmc.ncbi.nlm.nih.gov/articles/PMC13360563/>
- **真实部分**：5 个类别变量与各自水平（Table 1），组合数 = 12×10×5×10×5 = **30,000**；
  Br 底物的部分实测最优条件与收率（Table 2：99 / 95 / 91 / 78 / 77 / 71%）。
- **代理部分**：文章正文没有给出 30,000 点的完整收率表，所以迭代循环里的收率是
  **按论文趋势手工参数化的代理模型**（DMF/NBP、4CzIPN、Ni(PPh3)2Cl2、dimethoxy-bpy、
  morpholine 得分最高），不是实测数据。

运行：
```powershell
py -3 examples/paper_campaigns/run_paper_campaigns.py
```

结果（同一个空间与种子，只改变路由/算法）：

| 运行 | 路由选择 | 100 次提议后最优代理收率 | 是否命中论文 99% 条件 |
|---|---|---|---|
| `auto`（路由器） | `tpe` | 91.0% | 否 |
| 强制 `bayes` | `bayes`（GP + EI，one-hot 37 维） | **99.0%** | **是**，精确命中 DMF/4CzIPN/Ni(PPh3)2Cl2/dimethoxy-bpy/morpholine |

**发现**：路由器当前“高基数类别 → TPE”的规则在这个 30,000 组合的空间上**不如 GP-BO**。
这不是说 TPE 一般更差，而是说明这条规则需要校准——正好对应项目里的
`configs/router_weights_provisional.json` 与 488-runs 校准计划。

## 案例 2：热固性丙烯酸酯配方（Chem. Mater. 2026）

- 论文：*Scalable Thermoset Informatics Integrated with Autonomous Experimentation
  for On-Demand Discovery*, Chem. Mater. 2026, 38, 10562–10575（2026-10-09 上线）
- 链接：<https://pmc.ncbi.nlm.nih.gov/articles/PMC13718556/>
- **真实部分**：5 种单体 + 4 种交联剂的配方库；单配方 ≤3 组分、交联剂 ≤50 wt%；
  工艺约束固化时间 ≤10 s、固化温升 ≤70 °C（论文 2.1 节）。
- **合成部分**：论文可抓取文本里没有目标性能表，因此目标窗口与性能 oracle 是**合成的**
  （E≈4.5 MPa、UTS≈7.5 MPa、伸长率≈75%，并模拟论文提到的“相区/孤岛”非光滑响应）。

结果：路由器选择 `constrained`（PoF → EI_c）；32 次提议后得到可行最优
`IDA / PEGDA-250 / 30.99 wt% 交联剂 / 2.5 wt% 第二单体`，代理综合得分 **0.951**，
预测固化时间 6.48 s、温升 42.4 °C——两个约束均满足。

## 产物

```
examples/paper_campaigns/
├─ run_paper_campaigns.py     # 复现脚本（可重复运行；会重建 runs/）
├─ results.json               # 两次战役 + bayes 对照的机器可读结果
└─ runs/
   ├─ art_hte_angew2026/            # auto 路由（tpe）
   ├─ art_hte_angew2026_bayes/      # 强制 bayes 对照
   └─ thermoset_psdl_chemmater2026/ # 约束配方战役
```

每个 run 目录都有 `dashboard.html`（离线看板）、`convergence.png`、`dataset.jsonl`、
`decisions.jsonl`（含路由/重路由记录）、`problem.json`、`suggestions.csv`、`rounds.json`。

## 局限（必须明确）

1. **所有目标值都是代理**，不是真实测量；案例 1 的代理是按论文趋势手工标定的，
   案例 2 的 oracle 是合成的。因此**不能用这些数字宣称“算法优于论文”**。
2. 案例 1 只用了 Br 底物的部分 Table 2 条件作种子（4 条，保留 99% 条件作为“未知最优”），
   没有复现论文的 24 点 LHS 初始设计。
3. 案例 2 是简化的 5 变量编码（2 个单体槽 + 交联剂 + 两个配比），没有完整复现论文的
   7 维组成单纯形与多目标 regret 目标。
4. 这些示例的用途是：**验证 skill 在复杂空间上的流程、路由、约束/类别处理、审计产物**，
   以及暴露“路由规则是否需要校准”；正式结论仍需真实数据。

## 决策层两阶段验收（`run_spec_decision.py`）

`run_spec_decision.py` 跑两遍同一热固性体系：

1. **原始规格**：E 4.5±0.5、UTS 7.5±0.7、elongation 75±8；
   合成代理下三个目标不能同时命中，20 次评价后 `in_spec=0`，
   decision layer 给出 `review_spec` + `spec_review`（每个目标观测范围）。
2. **校准/建议后规格**：`suggest_spec` 先按历史校准 tolerance/scale（本数据下 target 都在观测范围内，
   因此 target 不动，tolerance 变宽），再跑一遍同规模 campaign；
   结果 `in_spec=true`，推荐 E=4.86、UTS=6.74、elong=76.3，并完成 3 次确认复孔。

结果汇总在 `thermoset_spec_decision_results.json`（base / suggestion / suggested_campaign）。

## 组合场景验收（`run_combined_decision.py`）

真实热固性组件/约束 + 合成 oracle 与合成板偏差，四者叠加：

- 2 目标（E、elongation）+ 硬约束（cure ≤10 s、ΔT ≤70 °C）；
- 板效应（P1 E 偏高/elong 偏低，P2 相反，偏差放大到可检测）；
- spec（elongation 75±8）；
- 算法 cqehvi，next_action 自动循环 + 确认复孔。

本次运行结果：`selection_basis=pareto_spec_plate_adjusted`，检测到板效应，
Pareto 前沿 7 个点，完成 2 次确认复孔；因规格窗口在该约束下不可达，诚实返回 in_spec=false。
结果在 `thermoset_combined_decision_results.json`。
## 真实测量数据验收

`art_table2_real.csv` 是从 Angew. Chem. Int. Ed. 2026（ART metallaphotoredox HTE）
Table 2 Br 战役转录的 6 条真实测量条件/产率（无代理函数）。
`tests/test_real_data.py` 通过 `condopt import --results` 直接导入，
断言最优条件 == 实测 99% 那条，并跑通 summary/recommend/digest。
