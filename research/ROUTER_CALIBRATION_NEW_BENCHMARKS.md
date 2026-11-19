# Router 校准：加入新基准后的复核

`py -3 scripts/calibrate_router.py --problems branin sphere2d mixed_reaction discrete_catalyst
suzuki_like high_cardinality_solvent constraint_easy constraint_hard moo_yield_cost
solvent_constrained mf_moo mf_moo3 --algorithms random sobol lhs bayes tpe constrained parego
qehvi cqehvi mfbo momfbo mfparego --budget 12 --batch 3 --seeds 0 1 --iterations 80`

## 结果

- 默认（现有 router 权重）：top1 0.667、top2 0.917、mean_regret 0.00438
- **LOPO（留一，实际可用指标）：top1 0.75、top2 1.00、mean_regret 0.00115**
- 标签（每个问题的最佳算法）：
  `branin/sphere2d/mixed_reaction/suzuki_like → bayes`、
  `constraint_easy/constraint_hard/solvent_constrained → constrained`、
  `moo_yield_cost → qehvi`、
  **`mf_moo` 与 `mf_moo3` → mfparego**、
  `discrete_catalyst/high_cardinality_solvent → random`（预算很小，随机基线够用）

## 由校准促成的默认值变更

四处独立证据都指向 `mfparego` 略优于 `momfbo`：

| 设置 | mfparego | momfbo |
|---|---|---|
| `mf_moo` 预算 30 | 672.3 | 666.2 |
| `mf_moo` 预算 40 | 676.1 | 670.3 |
| `mf_moo_constrained` 预算 40 | 642.5 | 635.6 |
| `mf_moo3` 预算 40 | 27399 | 26869（全保真点 35.3 vs 26.0） |
| 校准标签（预算 12） | 胜 | — |

因此 router 现在把 **多保真 + 多目标（含约束、3+ 目标）默认改为 `mfparego`**，
`momfbo` 保留为候选（其 `min_full_fraction=0.4` 的调参结论仍用于手动选择时）。

校准权重已写入 `runs/calibration_new_benchmarks/weights.json`，可用
`condopt route --weights <file>` 对比实验；本轮未替换默认权重（LOPO 已 0.75/1.00，
且替换需重新跑一遍 89 条路由评估）。
## A/B：默认规则 vs 校准权重（都用 LOPO）

`scripts/ab_router_weights.py`（默认侧无需训练，逐问题留一；校准侧取 `weights.json` 的 LOPO）：

| 指标 | 默认规则 | 校准权重 |
|---|---|---|
| top1 | 0.750 | 0.750 |
| **top2** | 0.917 | **1.000** |
| mean_regret | 0.001152 | 0.001152 |

校准权重把 **top2 从 0.917 提升到 1.000**（候选列表中一定含最佳算法），
top1 与 regret 不变。

**决策：暂不把校准权重设为运行时默认**——它是在 budget 12 的 12 个问题上拟合的，
只带来二选一排序收益；默认规则更易解释、且与新基准的 LOPO top1 持平。
需要时可显式加载：`condopt route --weights runs/calibration_new_benchmarks/weights.json`。
## 加入 10 维问题后的复校（cmaes 评分修正之后）

`--problems ... rosenbrock10 ackley10 ... --budget 30 --batch 5 --seeds 0 --iterations 60`

- 标签：`rosenbrock10`/`ackley10` → **cmaes**（与修正后的路由一致）；
  `discrete_catalyst` → sobol；`high_cardinality_solvent` → bayes。
- 默认规则 LOPO：top1 0.714 / top2 0.786；校准权重 LOPO：top1 0.643 / top2 0.786
  → **默认规则胜出**，继续不启用校准权重。

### 单 seed 标签不可信（重要教训）

该次校准只跑了 1 个 seed，`sphere2d` 的标签被偶然判成 `cmaes`。用 3 seeds 复核：

| 问题 | bayes | cmaes |
|---|---|---|
| sphere2d | −0.0001 | −0.0002（都到最优，噪声内） |
| branin | **−0.4115** | −5.1178 |
| suzuki_like | **99.741** | 98.206 |

→ 低维问题仍应选 GP（bayes），单 seed 标签会误导。
`calibrate_router.py` 现在在 `--seeds` 只有 1 个时打印警告。

