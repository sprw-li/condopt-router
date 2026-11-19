# RECORDS_V2_APPLY_PLAN — records v2 逐文件应用顺序、488 runs 与预期输出

> 维护者：algo-research ｜ v1（2026-11-05，Round 15）
> 对应：`RECORDS_V2_IMPLEMENTATION.md`（逐文件 diff 正文）、`RECORDS_V2_PATCH.md`（设计）、`CALIBRATION_RUN_PLAN.md`。
> 前置：v0.10.4、148 tests 全绿、工作树干净；`_cost`/`cost_remaining` 已统一（records 的 MF 路径依赖它）。

---

## 0. 改动前基线

```bash
py -3 -m pytest -q                          # 期望 148 passed
git rev-parse HEAD                          # 回滚点 R0
git status --porcelain                      # 期望空
```

---

## 1. `benchmarks.py`（先改，calibration 依赖它）

按以下顺序，**每步后跑一次 `python -c "import condopt.benchmarks"`**：

1. **dataclass 字段**（7 个带默认值）：`tags/noise_std/n_replicates/fidelity_levels/fidelity_costs/reference_front`；
2. **`builtin_problems()` 标签**：`noisy_sphere`（noise=0.05/reps=3）、`constraint_easy|hard`（constrained）、`moo_yield_cost`（multi_objective + `reference_front=moo_reference_front()`）、`mf_suzuki`（multi_fidelity + levels/costs）；
3. **`_evaluate`**（真值 + 噪声 + constraints + `meta["true"]`）；
4. **`_metric`**（true_hv_ratio / feasible_best / cost_aware_best / true_best）；
5. **`_context`**（从 tags 推导 noise/smoothness/multi_fidelity/domain/n_constraints）；
6. **`run_task`**（ask/tell 循环；MF 分支用 `opt.cost_remaining = cost_budget` + `ask_with_fidelity`，并把 fidelity/cost 写回 Observation）。

**检查点 C1**：
```bash
py -3 -m pytest -q tests/test_benchmarks.py    # 期望 passed（现有基准测试不能回归）
py -3 - <<'PY'
from condopt.benchmarks import builtin_problems, run_task
for name, algo in [("branin","bayes"), ("noisy_sphere","bayes"),
                   ("constraint_easy","constrained"), ("moo_yield_cost","qehvi"),
                   ("mf_suzuki","mfbo")]:
    r = run_task(builtin_problems()[name], algo, budget=8, batch_size=2,
                 seed=0, cost_budget=8.0)
    print(name, r["metric"]["name"], round(r["metric"]["value"], 4))
PY
```
**预期输出**（示例）：
```
branin true_best 0.39xx
noisy_sphere true_best -0.xx
constraint_easy feasible_best xx
moo_yield_cost true_hv_ratio 0.6–0.9
mf_suzuki cost_aware_best 60–90
```
若 `moo_yield_cost` 抛 `qehvi` 的 ref 错误 → 检查 `reference_front` 是否命名为 signed（应与 `moo_reference_front()` 一致）。

**回滚点 R1**：`git checkout -- src/condopt/benchmarks.py`。

---

## 2. `calibration.py`

顺序：
1. `RunRecord` 加 `metric_name: str = "true_best"`；
2. `collect_records` 签名去掉 `noise/smoothness/multi_fidelity`，加 `cost_budget=None`；
3. `collect_records` 内 `run_algorithm(...)` → `run_task(...)`，`score/metric_name/context` 从返回值取；
4. 能力过滤改为 `required ⊆ caps`（`required` 从 `problem.tags` 推导），跳过时记 `RunRecord(..., metric_name="skipped_capability")`；
5. 标签口径（`best_algorithm_per_problem`/`evaluate_weights`）：忽略 NaN/skipped；tie 检测（|best−second| < 5%·span → 记 tie）；regret 用 span 归一。

**检查点 C2**：
```bash
py -3 -m pytest -q tests/test_calibration.py    # 期望 passed
py -3 - <<'PY'
from condopt.calibration import collect_records, lopo_validate
recs = collect_records(["branin","noisy_sphere","constraint_easy","moo_yield_cost","mf_suzuki"],
                       ["bayes","constrained","qehvi","mfbo","parego"],
                       budget=8, batch_size=2, seeds=(0,), cost_budget=8.0)
print("records:", len(recs))
print("metrics:", sorted({r.metric_name for r in recs}))
print("lopo:", lopo_validate(recs, iterations=30, seed=0)["top1"])
PY
```
**预期**：`metrics` 含 `true_hv_ratio/feasible_best/cost_aware_best/true_best`（`skipped_capability` 可能出现）；`lopo` 为 0.0–1.0 的浮点。

**回滚点 R2**：`git checkout -- src/condopt/calibration.py`。

---

## 3. 488 runs 执行清单

```bash
# 3.1 单目标核心（384）：6 问题 × 4 算法 × budget{16,32} × 8 seeds
py -3 - <<'PY'
import json
from condopt.calibration import collect_records
P = ["branin","sphere2d","mixed_reaction","suzuki_like",
     "discrete_catalyst_v2","high_cardinality_solvent"]
A = ["random","sobol","bayes","tpe"]
recs = []
for budget in (16, 32):
    recs += collect_records(P, A, budget=budget, batch_size=4, seeds=range(8))
json.dump([r.__dict__ for r in recs], open("runs/calib/r15_core.jsonl","w"), ensure_ascii=False)
print(len(recs))
PY
# 3.2 噪声（32）、约束（32）、多目标（24）、多保真（16）——同上，Problem/Algo 换成各组的
# 3.3 合并 → LOPO
py -3 - <<'PY'
from condopt.calibration import lopo_validate
from condopt.calibration import RunRecord
import json
recs = [RunRecord(**row) for row in map(json.loads, open("runs/calib/r15_all.jsonl"))]
print(lopo_validate(recs, iterations=300, seed=0))
PY
```

**预期输出**：总 records **488**（每类数量：384/32/32/24/16）；LOPO 指标：
`top1_strict ≈0.50–0.65`、`top2 ≈0.75–0.90`、`regret ≈0.05–0.12`（与 `CALIBRATION_RUN_PLAN.md` 的预期区间一致；若 top1 >0.75 需查是否泄漏/tie 计入）。

**耗时（8 进程）**：core ≈2 min；noise/constraint ≈1 min；MOO ≈2.5 min；MF ≈0.5 min；拟合 ≈2 min → **8–12 min**。

---

## 4. 验收与提交

```bash
py -3 -m pytest -q                           # 期望 148 + 4（records 覆盖/边界/context/skip）
py -3 -m pytest -q -m fast_ci --durations=10
git diff --stat                              # 期望仅 benchmarks.py/calibration.py/tests
```
**提交信息**：
```
feat(calibration): records v2 — true noise/MOO/constraint/MF + problem-derived context

- BenchmarkProblem gains tags/noise_std/n_replicates/fidelity/reference_front
- run_task dispatches per type; metrics: true_best/true_hv_ratio/feasible_best/cost_aware_best
- collect_records uses run_task; context from problem metadata; capability skip recorded
- 488-run LOPO matrix (core/noise/constraint/MOO/MF)
```

**风险点**：① `moo_reference_front()` 返回 signed 坐标，`_metric` 里 ref 变换不要重复取负；② `mf_suzuki` 的 ask_with_fidelity 需要 `opt.cost_remaining`（v0.10.4 已支持 call override，建议直接传参）；③ 噪声组的 seed 决定观测噪声（同 seed 可复现）；④ `skipped_capability` 的 `score=NaN` 必须在标签/统计时过滤。

---

## 5. 参考

R32（指标口径）、R42（HV/真值）、R12（噪声多目标）、R41（成本/多保真）、R18（budget/cost）、R54（评估协议）、`RECORDS_V2_IMPLEMENTATION.md`、`CALIBRATION_RUN_PLAN.md`、`REVIEW_v14.md`。
