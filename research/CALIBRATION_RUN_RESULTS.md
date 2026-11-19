# CALIBRATION_RUN_RESULTS — 核心子集 LOPO 实测（v0.9）

> 维护者：algo-research ｜ v1（2026-11-05，Round 10）
> 目的：先跑**核心子集**验证 harness 的 in-sample vs LOPO 差距、耗时与 weights JSON 形态；正式 456-runs 按 `CALIBRATION_RUN_PLAN.md` 执行。
> 环境：Windows/CPU、Python 3.14、v0.9、`calibration.py`（LOPO 已合入）。命令见 §1，原始数字见 §2，可复现脚本见 §5。

---

## 1. 执行清单（本机实测已跑）

```python
from condopt.calibration import (collect_records, random_search_weights,
                                 evaluate_weights, lopo_validate,
                                 best_algorithm_per_problem)
problems = ["branin", "sphere2d", "mixed_reaction", "suzuki_like",
            "discrete_catalyst_v2", "high_cardinality_solvent"]
algorithms = ["random", "sobol", "lhs", "bayes", "tpe"]
records = collect_records(problems, algorithms, budget=10, batch_size=2, seeds=(0, 1))
labels = best_algorithm_per_problem(records)
default = evaluate_weights(records, None)                    # 手写规则（w=1）
tuned = random_search_weights(records, iterations=60, seed=0)  # in-sample
lopo = lopo_validate(records, iterations=60, seed=0)          # 折外
```

- 规模：6 问题 × 5 算法 × budget 10 × batch 2 × seeds {0,1} = **60 runs**
- 采集耗时：**3.1 s**（单进程）
- in-sample 搜索 + LOPO：**0.1 s**（纯查表，可忽略）

> 若要按 `scripts/calibrate_router.py` 跑：`py -3 scripts/calibrate_router.py --budget 10 --batch 2 --seeds 0 1 --iterations 60`（默认 problems/algorithms 与上面一致）。

---

## 2. 结果

### 2.1 标签（每个问题的最优算法，按 mean best_score）

```json
{"branin": "bayes", "sphere2d": "bayes", "mixed_reaction": "bayes",
 "suzuki_like": "bayes", "discrete_catalyst_v2": "lhs",
 "high_cardinality_solvent": "bayes"}
```

### 2.2 in-sample vs LOPO

| 指标 | 默认规则（w=1） | in-sample 调优 | **LOPO（折外）** |
|---|---|---|---|
| top1 | 0.667 | 0.833 | **0.833** |
| top2 | 0.833 | 1.000 | **0.833** |
| mean_regret | 0.00837 | 0.00190 | **0.00190** |
| 备注 | — | 不可作证据（样本内） | 6 折；1 折失败 |

**LOPO 逐折**

| held-out | 折内 weights | top1 | top2 | regret | 说明 |
|---|---|---|---|---|---|
| branin | 全局权重组 | 1.0 | 1.0 | 0.0 | bayes 正确 |
| sphere2d | 全局 | 1.0 | 1.0 | 0.0 | bayes 正确 |
| mixed_reaction | 全局 | 1.0 | 1.0 | 0.0 | bayes 正确 |
| suzuki_like | 全局 | 1.0 | 1.0 | 0.0 | bayes 正确 |
| high_cardinality_solvent | 全局 | 1.0 | 1.0 | 0.0114 | bayes 正确；regret 非零 |
| discrete_catalyst_v2 | `{bayes 1.444, lhs 1.358, random 1.021, sobol 0.859, tpe 1.111}` | **0.0** | **0.0** | **0.0** | 标签 lhs，但分数并列 → regret=0 与 top1=0 矛盾（tie 口径问题） |

### 2.3 weights JSON（in-sample 搜索结果；**不能直接入库**）

```json
{
  "version": "calibration-probe-2026-11-05",
  "weights": {"bayes": 1.401, "lhs": 1.537, "random": 0.623, "sobol": 1.026, "tpe": 0.702},
  "trained_on": {"problems": 6, "algorithms": 5, "budget": 10, "batch": 2, "seeds": [0, 1],
                 "n_runs": 60},
  "metrics_in_sample": {"top1": 0.833, "top2": 1.0, "mean_regret": 0.0019},
  "metrics_lopo": {"top1": 0.833, "top2": 0.833, "mean_regret": 0.0019, "n_folds": 6},
  "status": "probe_only"
}
```

**入库条件**（缺一不可）：①只有 LOPO 指标；②覆盖 ≥4 类问题（单目标/噪声/约束/MOO/MF 至少各 1）；③`n_folds ≥ 10`（问题数 ≥10）或明确标注"小矩阵，仅供参考"；④同输入可复现（seed/iterations/range 落盘）。

---

## 3. 解读与结论

1. **默认规则在这个小矩阵上已经很强**：6 个问题里 5 个 bayes 最优（小 budget=10 下 GP 的优势明显），所以 `top1` 的提升空间只有 `discrete_catalyst_v2` 一折；LOPO 0.833 与 in-sample 0.833 相同，**不能证明校准有效**，只能说明 harness 跑通。
2. **失败折暴露 tie 口径问题**：`discrete_catalyst_v2`（3×4×4×81 全离散，budget 10 太小）多算法拿到相同 best_score；标签按 mean argmax 取 `lhs`，但 `bayes` 分数并列 → `top1=0` 而 `regret=0`。正式矩阵应加 tie 检测（|Δ| <5% span → 记 tie，不进 top-1 监督），并提高 budget（该问题组合数 3888，budget 10 远不足）。
3. **折间权重不稳定**：`discrete_catalyst_v2` 折的权重与全局差异大（`lhs` 从 1.537→1.358、`sobol` 1.026→0.859）。6 个问题做 LOPO 时每折训练只有 5 个问题，权重方差大 → 正式矩阵需 ≥10 问题 + 权重向 1 正则。
4. **覆盖缺口**：本子集没有噪声/约束/MOO/MF 问题，因此校准结果**只对单目标采样器组有效**；多目标/约束/多保真分支的权重仍需按 `CALIBRATION_RUN_PLAN.md` 扩矩阵。

---

## 4. 正式 456-runs 执行清单（下一步）

```bash
# 0) 先修 REVIEW_v09 的 P0/P1（mfbo best 全保真、cost 从 fidelity 推导、context 元数据、MOO 指标、tie 口径）
# 1) 核心（384 runs）
py -3 scripts/calibrate_router.py --problems branin sphere2d mixed_reaction suzuki_like \
    discrete_catalyst_v2 high_cardinality_solvent --algorithms random sobol bayes tpe \
    --budget 16 32 --batch 4 --seeds 0 1 2 3 4 5 6 7 --iterations 300 \
    --out runs/calib/r10_core
# 2) extras（噪声 48 + 约束 32 + MOO 24 + MF 16）——需要 v0.10 的 runner 支持噪声/HV/cost metric
# 3) 合并 + LOPO + 权重入库（CALIBRATION_RUN_PLAN §6）
```

预计耗时：核心采集 8 进程 ~1.5–3 min；extras 中 MOO 占大头（~2.5 min）；总 ~10 min。预期（LOPO）：top1 0.55–0.70（问题数增多后会更低更真实）、top2 0.80–0.90、regret 0.05–0.10。

---

## 5. 可复现脚本（可直接存 `scripts/probe_calibration.py`）

```python
"""Probe calibration on the core subset (fast, ~4 s)."""
from condopt.calibration import (best_algorithm_per_problem, collect_records,
                                 evaluate_weights, lopo_validate, random_search_weights)
PROBLEMS = ["branin", "sphere2d", "mixed_reaction", "suzuki_like",
            "discrete_catalyst_v2", "high_cardinality_solvent"]
ALGORITHMS = ["random", "sobol", "lhs", "bayes", "tpe"]

def main():
    records = collect_records(PROBLEMS, ALGORITHMS, budget=10, batch_size=2, seeds=(0, 1))
    print("labels:", best_algorithm_per_problem(records))
    print("default:", evaluate_weights(records, None))
    tuned = random_search_weights(records, iterations=60, seed=0)
    print("in-sample:", tuned["metrics"], tuned["weights"])
    print("lopo:", {k: v for k, v in lopo_validate(records, iterations=60, seed=0).items()
                    if k != "folds"})

if __name__ == "__main__":
    main()
```

---

## 6. 参考

R32（多 seed/IQM/统计）、R51（折外评估）、R52（选择偏差）、R24（权重可解释）、`CALIBRATION_RUN_PLAN.md`、`REVIEW_v09.md`（P1-3 tie、P1-5 稳定性、P1-1 context）。
