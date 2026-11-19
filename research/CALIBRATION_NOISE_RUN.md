# CALIBRATION_NOISE_RUN — 带噪声 context 的核心校准实测（v0.10）

> 维护者：algo-research ｜ v1（2026-11-05，Round 11）
> 目的：验证 `collect_records(noise=..., smoothness=...)` 在**同一批无噪数据**上只改 context 时，in-sample vs LOPO 会怎样变化，为正式 456-runs（含真噪声 records）定口径。
> 环境：Windows/CPU、v0.10、`calibration.py`；采集 6 问题 × 5 算法 × budget10 × batch2 × 2 seeds = 60 runs/组。

---

## 1. 执行

```python
from condopt.calibration import (collect_records, evaluate_weights, lopo_validate,
                                 random_search_weights, best_algorithm_per_problem)
PROBLEMS = ["branin", "sphere2d", "mixed_reaction", "suzuki_like",
            "discrete_catalyst_v2", "high_cardinality_solvent"]
ALGORITHMS = ["random", "sobol", "lhs", "bayes", "tpe"]

for noise, smooth in (("none", "smooth"), ("high", "rough")):
    records = collect_records(PROBLEMS, ALGORITHMS, budget=10, batch_size=2,
                              seeds=(0, 1), noise=noise, smoothness=smooth)
    labels = best_algorithm_per_problem(records)
    default = evaluate_weights(records, None)
    tuned = random_search_weights(records, iterations=60, seed=0)
    lopo = lopo_validate(records, iterations=60, seed=0)
```

> **关键局限（写在结果解读前）**：`collect_records(noise=...)` 只把噪声写进 **router context**；`run_algorithm` 仍对 `problem.fn` 求值（**无噪标量**），因此两条 context 的 **labels/score 完全相同**，变化的只是 router 打分。这不是"噪声场景校准"，只是"噪声标签下的路由灵敏度测试"。

---

## 2. 结果

### 2.1 两组 context 的总览

| context | 耗时 | labels | in-sample default | in-sample tuned | LOPO |
|---|---|---|---|---|---|
| `noise="none", smooth="smooth"` | 3.83 s | bayes×5, lhs×1 | top1 **0.833** / top2 0.833 / regret 0.00544 | 0.833 / 0.833 / 0.00190 | top1 **0.667** / top2 0.833 / regret **0.08495** |
| `noise="high", smooth="rough"` | 3.75 s | 同上 | top1 **0.000** / top2 0.833 / regret **1.28256** | 0.833 / 1.000 / 0.00544 | top1 **0.833** / top2 0.833 / regret 0.00190 |

### 2.2 权重（in-sample 搜索；**不可入库**）

| context | weights |
|---|---|
| none/smooth | `{bayes 1.444, lhs 1.358, random 1.021, sobol 0.859, tpe 1.111}` |
| high/rough | `{bayes 1.318, lhs 0.938, random 1.221, sobol 0.641, tpe 0.764}` |

两组权重差异大（`sobol` 0.859 vs 0.641、`lhs` 1.358 vs 0.938），说明 context 对校准结果**强影响**；在真实 records 下应把 context 当作"问题特征"（由 problem 元数据推导），而不是全局旋钮。

### 2.3 LOPO 逐折（noise=high/smooth=rough）

| held-out | top1 | top2 | regret |
|---|---|---|---|
| branin | 1.0 | 1.0 | 0.0 |
| discrete_catalyst_v2 | 1.0 | 1.0 | 0.0 |
| high_cardinality_solvent | **0.0** | 0.0 | 0.0114 |
| mixed_reaction | 1.0 | 1.0 | 0.0 |
| sphere2d | 1.0 | 1.0 | 0.0 |
| suzuki_like | 1.0 | 1.0 | 0.0 |

`high_cardinality_solvent` 折失败（标签 bayes，但 held-out 路由选了别的算法/或分数未达 tie 阈值）；该问题 12 级 one-hot 特征维 24，router 在 noise=high 下给 bayes 降分（`dim` 相关项 + 噪声项），是 LOPO 能抓到的真实"泛化失败"。

### 2.4 与 v0.9 核心子集（noise 默认 medium）对比

| 组 | in-sample default | LOPO top1 |
|---|---|---|
| v0.9 默认（medium/mixed） | 0.667 | 0.833 |
| v0.10 none/smooth | 0.833 | 0.667 |
| v0.10 high/rough | 0.000 | 0.833 |

同一个无噪数据集，只是 context 不同，in-sample 从 0.667 跳到 0.000——**这证明"context 不是校准数据"**：校准必须用**带真实噪声的观测**（evaluator 加噪）并把 context 从问题元数据推导，否则 LOPO 指标本身可被 context 任意操纵。

---

## 3. 结论与建议

1. **不要把 context 噪声当作校准变量**：正式 456-runs 的 noise 必须来自 `BenchmarkProblem.noise_std`（Tier D 真噪声），`collect_records` 从 problem 推导，而不是外部参数。
2. **LOPO top1 在 6 问题下波动大**（0.667–0.833）：正式矩阵需 ≥10 问题，并报告折间 min/median/max。
3. **regret 必须改归一化**（`noise=high` 组 1.28 已越界）；否则 in-sample 指标会被单个"near-zero best"问题主导。
4. **tie 口径需要拆 strict/tie**：本组 `discrete_catalyst_v2` 折在 tie 计入后 top1 由 0→1，掩盖了"标签算法未被选中"的事实。

**下一步（正式 456-runs 前置）**：修 `REVIEW_v10` 的 P0-1/P0-2（cost_budget/全保真 cost）与 P1-1（真噪声 records）；之后按 `CALIBRATION_RUN_PLAN.md` 跑，报告 `top1_strict/top1_tie/top2/regret + 折间分布`。

---

## 4. 参考

R51（Cawley & Talbot：折外评估）、R52（Varma & Simon：选择偏差）、R32（多 seed/统计）、R42（真值口径）、`CALIBRATION_RUN_PLAN.md`、`REVIEW_v10.md`。
