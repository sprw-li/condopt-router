# CALIBRATION_RUN_PLAN — 正式 460-runs LOPO-CV 实跑方案

> 维护者：algo-research ｜ v1（2026-11-05，Round 9）
> 对应：`calibration.py`（v0.8 MVP，**样本内**）、`scripts/calibrate_router.py`、`ROUTER_CALIBRATION_HARNESS.md`、`ROUTER_WEIGHTS_ACCEPTANCE.md`。
> 文献：R32（多 seed/IQM/LOPO 协议）、R51（Cawley & Talbot：嵌套评估避免乐观偏差）、R52（Varma & Simon：CV 选择偏差）、R42（HV/真值口径）、R50（权重应随 study 落盘）。
> **前置**：先修 `REVIEW_v08` P0-1（LOPO）与 P0-2（多类型 records）；否则本计划只能跑"样本内 smoke"，结果不可用于验收。

---

## 1. 矩阵（目标 ≈460 runs）

固定 `batch=4`（接近并行实验器），`seeds=0..7`（8 个）。

| 组 | 问题 | 算法（按能力过滤后） | 预算 | runs |
|---|---|---|---|---|
| 单目标核心 | `branin`, `sphere2d`, `mixed_reaction`, `suzuki_like`, `discrete_catalyst_v2`, `high_cardinality_solvent` | `random, sobol, lhs, bayes, tpe`（5） | 16、32 | 6×5×2×8 = **480** |
| 噪声 | `noisy_sphere`（σ=0.05/0.15） | `bayes, tpe, random` | 24 | 2σ×3×8 = **48** |
| 约束 | `constraint_easy`, `constraint_hard` | `constrained, bayes` | 24 | 2×2×8 = **32** |
| 多目标 | `moo_yield_cost` | `parego, qehvi, qnehvi` | 32 | 1×3×8 = **24** |
| 多保真 | `mf_suzuki` | `mfbo, bayes`（cost 预算 24） | — | 1×2×8 = **16** |

合计 ≈ **600 runs**（若只跑核心 480 + 抽查组 72 = 552；按 Lead 的 460 口径，可把单目标核心的预算减到只跑 16 或算法减到 4 → 6×4×2×8=384 + 72 = **456 ≈ 460**）。**建议**：核心用 4 算法 `random, sobol, bayes, tpe`（`lhs` 与 `sobol` 高度重叠，可去掉），得到 456。

---

## 2. 每类问题的记录字段（`runs.jsonl`，一行一 run）

```json
{"problem":"moo_yield_cost","algorithm":"qehvi","budget":32,"batch":4,"seed":3,
 "status":"ok",
 "metric":"true_hv_ratio","metric_value":0.94,
 "best_scalar":89.9, "n_feasible":null, "total_cost":32.0,
 "context":{"n_dims":2,"n_objectives":2,"noise":"none","constraints":0,"multi_fidelity":false},
 "runtime_s":38.4,"run_dir":"runs/calib/moo_yield_cost/qehvi/b32_s3"}
```

- 单目标：`metric="signed_best"`（越大越好；已按方向取符号）；
- 多目标：`metric="true_hv_ratio"`（**必须真值**：`pareto.summarise(use_true=True)` 或合成问题重算，`REVIEW_v07` P0-2）；
- 约束：`metric="feasible_best"`（可行观测中的最优；无可行点记录 `status="no_feasible"`）；
- 多保真：`metric="cost_aware_best"`（在固定成本预算内的最优全保真值）；
- 噪声：`metric="signed_best"`（用重复测量均值）或 `true_value`（合成问题）。

**聚合标签**：每个 `(problem, budget, batch)` 内，算法指标取 `IQM`（去掉各 25%）；`best = argmax`；top2 = 前二；`|IQM(best) − IQM(second)| < 5%·span` 记 tie（不进 top-1 监督）。

---

## 3. 实跑步骤

```bash
# 0) 前置修复后（LOPO + 多类型 records）先做自检
py -3 -m pytest tests/test_calibration.py -q

# 1) 采集（8 进程；每 run 独立目录，便于复查）
py -3 scripts/calibrate_router.py --collect --matrix full \
  --problems branin sphere2d mixed_reaction suzuki_like discrete_catalyst_v2 high_cardinality_solvent \
  --algorithms random sobol bayes tpe --budgets 16 32 --batch 4 --seeds 0 1 2 3 4 5 6 7 \
  --workers 8 --out runs/calib/r9_core.jsonl
py -3 scripts/calibrate_router.py --collect --matrix extras \
  --problems noisy_sphere constraint_easy constraint_hard moo_yield_cost mf_suzuki \
  --workers 8 --out runs/calib/r9_extras.jsonl

# 2) 合并 + LOPO 拟合
cat runs/calib/r9_core.jsonl runs/calib/r9_extras.jsonl > runs/calib/r9_all.jsonl
py -3 scripts/calibrate_router.py --fit runs/calib/r9_all.jsonl \
  --folds lopo --min-seeds 5 --iterations 300 --beta 0.8 --factor-range 0.6 1.6 \
  --out research/router-weights/r9-2026-11-08.json \
  --report research/router-weights/r9-2026-11-08.md

# 3) 路由端验收（同一批 held-out 问题）
py -3 scripts/calibrate_router.py --evaluate runs/calib/r9_all.jsonl \
  --weights research/router-weights/r9-2026-11-08.json --folds lopo
```

---

## 4. 耗时估算（本机 CPU，8 进程）

| 组 | 单 run 估算 | runs | 串行 | 8 进程 |
|---|---|---|---|---|
| 单目标核心（16/32 budget） | 0.2 s（采样器）–3 s（bayes） | 384 | ~8 min | **~1.5 min** |
| 噪声（24 budget，tpe/bayes） | 2–4 s | 48 | ~2.5 min | ~0.5 min |
| 约束（24 budget，constrained 2 GP + MC） | 3–5 s | 32 | ~2 min | ~0.5 min |
| 多目标（32 budget，qehvi/qnehvi） | 30–60 s（`n_y=128`/`n_base=64`，8 asks） | 24 | ~15 min | **~2.5 min** |
| 多保真（cost 24，mfbo） | 2–3 s | 16 | ~0.8 min | ~0.2 min |
| 拟合 + LOPO（10 折 × 300 iter） | — | — | ~2–4 min | — |
| **合计** | — | ~504 | ~30 min | **≈8–12 min**（含 I/O/启动开销，留 20 min 预算） |

**降本选项**：qehvi/qnehvi 的 `n_y` 降到 64、`n_base` 降到 32 → MOO 组降到 ~2 min；但验收前需确认 HV 差异 <1%。

---

## 5. 预期指标（基于 v0.8 小矩阵与实测）

| 指标 | v0.8 样本内（不可用于验收） | 正式 LOPO 预期 | 门槛 |
|---|---|---|---|
| top-1 | 0.50（小矩阵） | **0.55–0.65** | ≥0.55 |
| top-2 | 1.00（小矩阵，过乐观） | **0.80–0.90** | ≥0.80 |
| mean regret | 0.038（样本内） | **0.05–0.10** | ≤0.10 |
| 默认权重（w=1）对照 | — | top1 0.45–0.55 | 拟合后不劣化 |

> 样本内 1.00 的 top-2 说明小矩阵太小、标签可分性过强；正式矩阵必须按 §1 扩到 456+ runs，且只用 held-out 折报告。

---

## 6. weights JSON 入库与回退

### 6.1 目录与版本

```
research/router-weights/
  r9-2026-11-08.json          # 本次产物
  r9-2026-11-08.md            # 报告（LOPO 每折、失败案例）
  r8-...json                  # 历史，保留可回滚
```

**schema**（在 `ROUTER_CALIBRATION.md` §2.2 基础上加字段）：

```json
{
  "version": "router-weights-2026-11-08",
  "weights": {"tpe": 1.12, "sobol": 0.92, "qehvi": 1.20},
  "trained_on": {"matrix": "r9-full", "n_contexts": 22, "min_seeds": 5,
                 "problems": ["..."], "n_runs": 456, "beta": 0.8,
                 "factor_range": [0.6, 1.6], "search": "random+cd", "search_seed": 0},
  "validation": {"top1": 0.60, "top2": 0.85, "regret_gap": 0.06, "n_folds": 11},
  "in_sample": {"top1": 0.0, "top2": 0.0},
  "sha256": "<file sha256>"
}
```

### 6.2 接入与审计

1. `init --weights FILE`：加载→`Study.create` 时把**生效权重**写进 `config.json["router_weights"]`（版本 + 文件路径 + sha256），这样 `ask/tell/report` 都能从 config 读，不依赖进程全局（`REVIEW_v08` P1-7）；
2. `Study.log_decision(action="router_weights", payload={"version":..., "sha256":..., "metrics":...})` —— 每次加载/更换权重都留痕；
3. `report --json` 输出 `router_weights: {version, source, metrics}`。

### 6.3 回退

- 删除/清空 `weights` → 自动回到 v0.8 手写规则（空 dict = 默认）；
- 回滚到旧版本：`init --weights research/router-weights/r8-....json`；
- **禁止**把权重写死在代码里；文件缺失/损坏 → warning + 默认（已有回退逻辑，`router.set_algorithm_weights`）。

---

## 7. 风险与检查清单

| 风险 | 检查 |
|---|---|
| 样本内过拟合 | 只用 LOPO 折外指标报告（P0-1 修复后） |
| MOO 指标用 observed HV | 强制 `true_hv_ratio`（P0-2） |
| context 元数据丢失（noise/constraint/mf） | `test_collect_records_covers_moo_noisy_constrained` |
| 能力过滤崩溃（qehvi 用于单目标） | 采集前按 `_required_capabilities` 过滤 |
| 权重语义（正/负分） | 只对正分算法 >1 的 factor 做 >1；负分算法 factor 限制在 [0.8,1.2]（或等加性语义） |
| 记录不可复查 | 每 run 写 `run_dir` + `context` + `seed`；`runs.jsonl` 入 git-lfs/归档 |
| 3D EHVI | 仍暂缓（m=3 用 ParEGO，见 `EHVI_3D.md`） |

---

## 8. 参考

R32（统计协议）、R51（折外评估）、R52（CV 选择偏差）、R42（HV/真值）、R12（qNEHVI）、R50（权重/配置随 study 落盘）、`ROUTER_CALIBRATION_HARNESS.md`、`ROUTER_WEIGHTS_ACCEPTANCE.md`、`REVIEW_v08.md`。
