# 真实场景 → 默认算法（可执行矩阵）

共 22 个场景；每行都是一个真实实验场景的上下文；`scripts/scenario_matrix.py` 会实际调用 router 校验，
不匹配即非零退出。默认值依据见 [`DEFAULTS.md`](DEFAULTS.md)。

| 场景 | 默认算法 | 实测依据 |
|---|---|---|
| A. 少量连续变量单目标（3 维） | `bayes` | 2–6 维平滑，GP 代理最稳 |
| B. 高维连续（10 维，光滑） | `cmaes` | budget 60、3 seeds：−18.1 vs bayes −155.4（rosenbrock10）、−2.20 vs −5.01（ackley10） |
| B2. 中维连续（6 维） | `bayes` | 低于 8 维 GP 仍占优（cmaes 规则不触发） |
| C. 混合离散（连续+分类+整数） | `bayes` | 12 维 one-hot；预算极小时 router 会退到 `tpe`/`random` |
| D. 高基数分类（12 级溶剂） | `tpe` | one-hot 维度高；加物性描述符后降到 3 维会自动转向 GP 类 |
| E. 单目标 + 硬约束（可行域大） | `constrained` | 溶剂约束基准 90.39 vs bayes 84.7 / tpe 85.7 |
| F. 2 目标 | `qehvi` | 2 目标 EHVI 与 parego 接近 |
| G. 2 目标 + 硬约束（低噪声） | `cqehvi` | 2 目标 + 约束的默认 |
| H. 2 目标 + 硬约束 + 噪声 | `cqehvi` | cqnehvi 未超过（638/635 vs 631/626） |
| I. 3–4 目标 | `qehvi_nd` | HV 5797 vs parego 5578（3 目标）、91244 vs 90134（4 目标） |
| J. 单目标 + 多保真 | `mfbo` | acquisition=ei：93.2 → 95.3 |
| K. 2+ 目标 + 多保真 | `mfparego` | 四处基准一致略优于 momfbo，全保真点更多 |
| L. 多目标 + 多保真 + 约束 | `mfparego` | 642.5 vs 635.6；另有可行全保真诊断兜底 |
| M. 高噪声 + 复孔 | `bayes` | GP alpha 使用逐点噪声（objective_std / n_replicates） |
| N. 4 目标 | `qehvi_nd` | 4 目标 HV 型采集（91244 vs parego 90134） |
| O. 4 目标 + 多保真 | `mfparego` | mf_moo4：515205 vs momfbo 507932 |
| P. 单目标 + 多保真 + 约束 | `mfbo` | mfbo 声明 constraints + multi_fidelity |
| Q. 小预算（8 次） | `sobol` | 预算不足时空间填充优于代理建模 |
| R. 大批量（batch=16） | `bayes` | GP 仍占优，批量由 batch_strategy 处理 |
| S. 2 目标 + 硬约束 + 高基数分类（描述符配对） | `cqehvi` | 新增基准 `moo_solvent_constrained`：可行 HV 1060.3（one-hot）vs 1045.1（描述符，噪声内），但描述符版可行点 +25%（29.7→37.0）→ 描述符须与约束感知算法成对 |
| T. 高基数分类 + 硬约束（描述符编码） | `constrained` | 描述符把 13 列降到 3 列，但路由仍选约束感知算法（可行率 0.550→0.875） |
| U. 2 目标 + 硬约束 + 高基数分类（描述符编码） | `cqehvi` | 编码换掉后仍选 cqehvi（可行点 +25%，HV −1.4% 噪声内） |

## 指标口径（约束场景必读）

| 指标 | 含义 | 陷阱 |
|---|---|---|
| raw HV | 全体观测的非支配前沿体积 | 含不可行点，**会高估交付价值**（实测 143.0 vs 80.9） |
| `hypervolume_feasible` | 只按可行观测算的前沿体积 | 约束场景应以它为准 |
| `feasible_fraction` | 逐批可行观测占比 | 用来判断是否该收窄变量范围 |

`condopt report` / `digest` 会同时给出“Pareto 前沿”与“可交付前沿”两行。

## 反例（不该用优化器硬找的场景）

| 场景 | 正确做法 | 依据 |
|---|---|---|
| 可行率 <5% 的窄盒子 | 把硬限制写进变量范围/固定因子 | 约束搜索 1/5 命中，写进范围 5/5（`research/HARD_FEASIBILITY.md`） |
| 停滞且预算有余 | `retune_for_exploration()`（不要直接停） | −205.4 vs 停止 −373.7（`research/STAGNATION_TUNING.md`） |

## 复现

```powershell
py -3 scripts/scenario_matrix.py
```

## 暖启动决策（transfer）

| 情形 | 建议 | 依据 |
|---|---|---|
| 无先验数据 | 不用暖启动 | — |
| 全保真 + 同问题/相关源 | 开，`top_k` 3–5，先 `transfer-audit` | 可行 HV +0.4–0.5%，前沿点数 +20% |
| 多保真 + 多目标 | 小 `top_k` 中性；`top_k≈8` 且含低保真时略正（+0.35%） | 626.92 → 629.09；`r=0.5` 反而 −0.95% |
| 迁移点会不会占预算 | **会**，`transfer_from` 返回 `budget_charged` | 各臂总观测都被压到 ~20 |

`condopt transfer-audit` 的建议值按下面三级处理：

| 建议值 | 评级 | 动作 |
|---|---|---|
| ≥0.8 | `trust` | 按该值暖启动，`top_k` 3–5 |
| 0.5–0.8 | `suspect` | `top_k` 压到 1–2，或先补目标观测再评估 |
| <0.5 | `avoid` | 不要用大 `top_k`；宁可从零开始 |

证据：`research/TRANSFER_MOO.md`、`research/TRANSFER_AUDIT.md`。

## 证据脚本映射

| 场景 | 证据脚本 | 基准问题 |
|---|---|---|
| A. 少量连续变量单目标 | `scripts/benchmark.py` | `branin` |
| B. 高维连续（10 维） | `scripts/benchmark_highdim.py` | `rosenbrock10` |
| B2. 中维连续（6 维） | `scripts/benchmark_highdim.py` | `ackley10` |
| C. 混合离散（连续+分类+整数） | `scripts/benchmark.py` | `discrete_catalyst` |
| D. 高基数分类（12 级溶剂） | `scripts/benchmark_tier_c.py` | `high_cardinality_solvent` |
| E. 单目标 + 硬约束（可行域大） | `scripts/benchmark_solvent_constrained.py` | `solvent_constrained` |
| F. 2 目标 | `scripts/benchmark_3obj.py` | `moo_yield_cost` |
| G. 2 目标 + 硬约束（低噪声） | `scripts/benchmark_3obj.py` | `moo_constrained` |
| H. 2 目标 + 硬约束 + 噪声 | `scripts/benchmark_cqehvi.py` | `moo_constrained` |
| I. 3–4 目标 | `scripts/benchmark_3obj.py` | `moo3_constrained` |
| J. 单目标 + 多保真 | `scripts/benchmark_mf_constrained.py` | `mf_constrained` |
| K. 2+ 目标 + 多保真 | `scripts/benchmark_momfbo.py` | `mf_moo` |
| L. 多目标 + 多保真 + 约束 | `scripts/benchmark_momfbo.py` | `mf_moo_constrained` |
| N. 4 目标 | `scripts/benchmark_3obj.py` | `moo4_constrained` |
| O. 4 目标 + 多保真 | `scripts/benchmark_momfbo.py` | `mf_moo4` |
| P. 单目标 + 多保真 + 约束 | `examples/constrained_mf_executor.py` | `mf_constrained` |
| Q. 小预算（8 次）单目标 | `scripts/benchmark.py` | `branin` |
| R. 大批量（batch=16） | `scripts/benchmark.py` | `mixed_reaction` |
| M. 高噪声 + 复孔 | `scripts/benchmark_noise.py` | `sphere2d` |

> 由 `scripts/scenario_matrix.py` 的 SCENARIOS 生成；`tests/test_scenarios.py` 校验一致性。
