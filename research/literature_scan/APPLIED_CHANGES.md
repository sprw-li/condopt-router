# 文献经验 → router 改动记录

本轮做了两件事：**批量拉文献并抽取路由经验**，以及**把可验证的经验落到 router**。

## 1. 语料规模

- 检索：Europe PMC，40 条查询，2024–2026 开放获取
- 去重后：**1357 篇**（`corpus.jsonl`，摘要覆盖率 88%）
- 子代理：34 个并行 analyst，每块 40 篇，逐篇打标签
  - 产出 **356 条标签**（`labels/labels_*.jsonl`）
  - 34 份经验文件（`lessons/lesson_*.json`）
- 高/中置信标签：**216 条**（用于权重校准，低置信不进训练）

### 标签分布（356 条）

| 空间类型 | 数量 | 文献里最常见推荐 |
|---|---|---|
| continuous_heavy | 165 | bayes (47) > lhs (17) ≈ parego/qehvi (14) |
| mixed | 99 | bayes（混合空间首选 GP-BO） |
| categorical_heavy | 67 | tpe (10) ≈ bayes (9) > grid (7) |
| unknown | 25 | — |

多目标标签中 **qehvi 25 / parego 24**（几乎持平）；约束标签中 **constrained 17**，其次 qehvi 11（多目标+约束）。

## 2. 校准出的候选权重（不设为默认）

`scripts/router_literature_calibrate.py` 把标签转成近似 context，用随机搜索学习乘性权重：

| 指标 | 默认规则 | 候选权重 |
|---|---|---|
| 标签一致率（训练） | 0.352 | **0.497** |
| 标签一致率（按论文 hash 留出 20%） | — | **0.4595** |
| 180-run benchmark LOPO top1 | 0.167 | 0.75（固定权重） |

输出：`configs/router_weights_literature.json`（`provisional` 性质，需 `route --weights` 显式启用）。

## 3. 实际落地到 router 的两处改动

### 3.1 小维纯类别空间：从“默认 TPE”改为“GP-BO 优先”

**证据**：
- 文献标签 categorical_heavy 中 tpe/bayes 基本持平（10 vs 9），grid 只用于很小可枚举空间；
- 多个子代理的 lessons 指出“类别数不大时 one-hot GP 能建模交互，独立密度 TPE 会忽略交互；只有高基数/高维离散才该 TPE”；
- 论文战役复现：同一 30,000 组合 ART 空间，TPE 100 次停在 91%，GP-BO 精确命中论文 99% 条件。

**规则**（`router._score_bayes` / `_score_tpe`）：

```
all_discrete
and n_dims <= 8
and feature_dim <= 60
and budget_total >= 50
=> bayes +35（并降低 one-hot 维度惩罚）
=> tpe -15（警告：独立密度忽略变量交互）
```

其他情况（高维/one-hot 宽、类别数多、预算小）保持原来的 TPE 倾向。

### 3.2 自动探索重启：修复“每轮都切 sobol、不回主算法”

**问题**：`Study.ask` 的 auto_reroute 只改局部变量，不更新 study 的 algorithm；
`route_round` 在停滞时每轮都建议 sobol，于是之后**每一轮都插一批 sobol**，一半预算浪费在探索，模型永远回不到主算法。

**修复**：
- 探索批次记录 `_exploration_main`，下一轮 `resume_route` 回到主算法；
- 增加 `_reroute_cooldown = patience`，主算法至少连续跑 `patience` 批才允许下一次探索重启；
- 普通（非探索）重路由会持久化到 study config。

**效果**（ART 代理战役，相同预算 120 / 批量 4 / 种子）：
- 修复前：auto → bayes 但被 sobol 反复打断，best **91%**
- 修复后：auto → bayes，best **99%**，精确命中论文条件
- 纯手工钉住 bayes 同样 99%（对照）

## 4. 验证

- `pytest`：**165 passed**（新增 literature 6 项 + 路由/重路由回归 2 项）
- `agentlib-router` eval：**84/84 保持 100%**
- 论文战役：`examples/paper_campaigns/results.json`（auto 99 / bayes 99 / thermoset 0.951）
- 文献管线可复跑：`py -3 scripts/literature_scan.py all --config configs/literature_scan.json`

## 5. 明确边界

- 文献标签来自标题/摘要 + LLM 判断，空间/变量数常缺失；只能作为 **router 先验**，不能替代基准或真实实验。
- 论文战役的目标值是代理（趋势标定/合成 oracle），不是实测；数字用于比较路由/流程，不用于宣称化学结论。
- 候选权重 `configs/router_weights_literature.json` 未设为默认；默认规则改动只有上面 3.1/3.2 两处，且都有回归测试。
```
