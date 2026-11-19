# 决策层规则矩阵（Decision Matrix）

> 由 `scripts/decision_matrix.py` 生成；`docs/SCENARIOS.md` 管“选哪个算法”，
> 本文件管“决策层怎么判”：停止口径、交付槽位、可行率告警、迁移可信度分级。
> 每条规则都带证据出处，改动规则就会改动这里的数字。

| 规则 | 期望 | 实测 | 证据 |
|---|---|---|---|
| 预算 12 的交付槽位 | 3（= default_confirm_n(12)） | OK `3` | research/CONFIRMATION_BUDGET.md |
| 默认复孔批次为 3 且可覆盖 | 3（两种情况一致，可用 meta.confirm_n 覆盖） | OK `3` | research/CONFIRMATION_BUDGET.md（两个问题方向不一致，保持默认） |
| 预算 12 的净探索预算 | 12 − 3 = 9 | OK `9` | Round 231（prepare 与运行时同源） |
| 大预算建议跑满 | budget_first（>=60） | OK `budget_first` | research/ADAPTER_LONG_RUN.md（69/120 vs 120/120） |
| 可信源的迁移建议必须是分阶段的 | 先 top_k 1 | OK `可信度 0.90：按原值暖启动（不要打折），但**审计分高 ≠ 收益为正**（实` | research/TRANSFER_ADAPTIVE.md（有偏源 +0.71 vs 固定 k3/k5 −1.54/−0.33） |
| 描述符编码不得改变约束场景的算法选择 | True | OK `True` | docs/SCENARIOS.md T/U（描述符版仍选 constrained / cqehvi） |
| 描述符建议必须与约束感知算法成对出现 | True | OK `True` | research/DESCRIPTOR_FEATURES.md（只换编码不换算法会更差） |
| 审计定位为一票否决器而非分级器 | 审计分高 | OK `可信度 0.90：按原值暖启动（不要打折），但**审计分高 ≠ 收益为正**（实` | research/AUDIT_CALIBRATION.md（同一 seed 内审计分与偏差无关） |
| 低置信度审计不得建议完全信任 | True | OK `True` | research/AUDIT_CALIBRATION.md（有偏源 0.65 却 −0.54） |
| 可信源的先验不应默认打折 | 不要打折 | OK `可信度 0.90：按原值暖启动（不要打折），但*` | research/PRIOR_RELIABILITY.md（r=1.0 +44.9 -> r=0.4 +39.3） |
| 平台期判定必须忽略导入的先验 | True | OK `True` | research/TRANSFER_TOP_K.md（先验曾导致暖启动 16-19/30 次就停） |
| 高噪声建议必须包含复孔与基线臂 | 3 | OK `3` | research/NOISE_POLICY.md（复孔与基线臂是站得住的两条） |
| 预算优先必须带交付权衡说明 | 不是 None（必须给出权衡说明） | OK `True` | research/MID_BUDGET.md（2/5 → 1/5 可交付证据） |
| 小预算建议早停 | stagnation_first（<60） | OK `stagnation_first` | research/STAGNATION_TUNING.md |
| 无约束时不做可行率告警 | none | OK `none` | research/HARD_FEASIBILITY.md |
| 窄可行域告警 | high（<=5%） | OK `high` | research/HARD_FEASIBILITY.md（1/5 vs 5/5） |
| 约束场景交付指标取可行口径 | feasible_best_value | OK `feasible_best_value` | research/NEGATIVE_RESULTS.md §10（143.0 vs 80.9） |
| 多目标+约束交付指标取可行 HV | feasible_hypervolume | OK `feasible_hypervolume` | research/NEGATIVE_RESULTS.md §10 |
| 多保真交付必须带全保真证据 | 含“全保真档位的交付证据” | OK `全保真档位的交付证据，而不是只有低保真推测；最优条件的复孔确认（避免单次幸运值）` | research/NEGATIVE_RESULTS.md §10 |
| 高可信源 -> trust | trust（>=0.8） | OK `trust` | research/TRANSFER_AUDIT.md |
| 中可信源 -> suspect | suspect（0.5–0.8） | OK `suspect` | research/TRANSFER_AUDIT.md |
| 低可信源 -> avoid | avoid（<0.5，置信度非 low 时按数值判定） | OK `avoid` | research/TRANSFER_AUDIT.md |
| 低可信 + 低置信度同样降级 | low_confidence（数值不可信时不下结论） | OK `low_confidence` | research/NEGATIVE_RESULTS.md §11–§12 |
| 低置信度一律降级 | low_confidence / suspect（不得为 trust） | OK `low_confidence` | research/NEGATIVE_RESULTS.md §11–§12（真实 20 切分） |
| 多保真多目标走 mfparego | mfparego | OK `mfparego` | docs/SCENARIOS.md 场景 F |
| 约束单目标走 constrained | constrained | OK `constrained` | docs/SCENARIOS.md 场景 H |

## 规则抑制（这些情形下对应规则**不该**触发）

> 路由层的能力过滤同时被 `scripts/audit_routing.py` **穷举**校验（96 个上下文，
> 候选与推荐项都必须满足上下文能力）；下表是抽样断言。

| 情形 | 期望 | 实测 | 证据 |
|---|---|---|---|
| 6 维不得触发 cmaes 规则 | 不是 cmaes | OK `bayes` | docs/SCENARIOS.md 场景 B2 |
| 预算 30 不得走 budget_first | 不是 budget_first | OK `stagnation_first` | research/ADAPTER_LONG_RUN.md |
| 无约束不得告警可行率 | none | OK `none` | research/HARD_FEASIBILITY.md |
| 低置信度不得给 trust | 不是 trust | OK `low_confidence` | research/NEGATIVE_RESULTS.md §11–§12 |
| auto_confirm=False 不得确认 | 不是 confirm | OK `ask` | docs/DECISION_LAYER.md |
| 单目标不得要求逐批进展 | False | OK `False` | docs/EXECUTOR_ADAPTER.md |
| top_k=0 不得导入观测 | 0 | OK `0` | docs/DECISION_LAYER.md |
| 无约束不得用可行 HV 口径 | hypervolume | OK `hypervolume` | research/NEGATIVE_RESULTS.md §10 |
| 多保真上下文不得出现非多保真算法 | 空列表 | OK `[]` | docs/SCENARIOS.md 场景 F |
| 约束上下文不得出现不支持约束的算法 | 空列表 | OK `[]` | docs/SCENARIOS.md 场景 H |
| 单目标上下文不得出现多目标算法 | 空列表 | OK `[]` | audit_routing.py |
| 多目标上下文不得出现单目标算法 | 空列表 | OK `[]` | docs/SCENARIOS.md 场景 E/G |
| 10 维不得选 bayes | 不是 bayes | OK `cmaes` | docs/SCENARIOS.md 场景 B |
| 预算 4 单目标退回空间填充 | 算法在 sobol/lhs/random/grid 内且给出 underpowered 说明 | OK `['sobol', True]` | docs/ROUTING_POLICY.md |
| 12 级分类 + 预算 30 不得把 TPE 判为数据不足 | False | OK `False` | docs/SCENARIOS.md 场景 D（TPE 不拟合 one-hot GP） |
| 预算 60 不得判为数据不足 | False | OK `False` | docs/ROUTING_POLICY.md |
| 极小预算 + 多目标保留模型类 | 算法在 parego/qehvi/qnehvi/cqehvi/cqnehvi/qehvi_nd内 | OK `parego` | research/TINY_BUDGET.md（budget 4：parego 503.6 HV / 3-4 次有前沿 vs sobol 1/4） |
| 极小预算 + 多目标不得违反能力契约 | False | OK `False` | research/TINY_BUDGET.md |
| 极小预算 + 多目标必须显式告警 | True | OK `True` | research/TINY_BUDGET.md |
| 充足预算不得告警建模舒适区 | False | OK `False` | docs/ROUTING_POLICY.md |
| 极小预算不得出现 EHVI 家族 | 空列表 | OK `[]` | docs/ROUTING_POLICY.md |

共 26 条规则（通过 26）与 21 条抑制断言（通过 21）。
