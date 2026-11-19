# RECORDS_V2_IMPLEMENTATION — 逐文件 diff、488 runs 命令与耗时

> 维护者：algo-research ｜ v1（2026-11-05，Round 14）
> 对应：`RECORDS_V2_PATCH.md`（Round 13 设计）、`CALIBRATION_RECORDS_V2.md`（Round 12 设计）。
> 目标：把设计落成**逐文件 diff**（`benchmarks.py` + `calibration.py`），并给出 488 runs 的执行命令、字段口径与预期耗时。

---

## 1. `src/condopt/benchmarks.py`（+约 90 行）

### 1.1 dataclass 字段（默认值，向后兼容）

```diff
 @dataclass
 class BenchmarkProblem:
     name: str
     space: SearchSpace
     objective: Objective
     fn: Callable[[dict], float]
     ...
+    tags: tuple[str, ...] = ()
+    noise_std: float = 0.0
+    n_replicates: int = 1
+    fidelity_levels: tuple[float, ...] = ()
+    fidelity_costs: tuple[float, ...] = ()
+    reference_front: np.ndarray | None = None
```

### 1.2 `builtin_problems()` 每个问题的标签（一行/问题）

```diff
 "noisy_sphere": BenchmarkProblem(
-    name="noisy_sphere", ...),
+    name="noisy_sphere", ..., tags=("noisy",), noise_std=0.05, n_replicates=3),
 "constraint_easy": BenchmarkProblem(..., tags=("constrained",)),
 "constraint_hard": BenchmarkProblem(..., tags=("constrained",)),
 "moo_yield_cost": BenchmarkProblem(..., tags=("multi_objective",),
+                                   reference_front=moo_reference_front()),
 "mf_suzuki": BenchmarkProblem(..., tags=("multi_fidelity",),
+                             fidelity_levels=(0.25, 0.5, 1.0),
+                             fidelity_costs=(0.25, 0.5, 1.0)),
```

### 1.3 `run_task` + `_evaluate` + `_metric` + `_context`（新函数，约 70 行）

```python
def run_task(problem, algorithm, *, budget, batch_size, seed,
             cost_budget: float | None = None) -> dict:
    rng = np.random.default_rng(seed)
    opt = registry.build(algorithm, problem.space, problem.objective, seed=seed)
    obs: list[Observation] = []
    done = 0
    while done < budget:
        n = min(batch_size, budget - done)
        if getattr(opt, "supports_fidelity", False) and problem.fidelity_levels:
            opt.cost_remaining = cost_budget
            pairs = opt.ask_with_fidelity(n)
            new = []
            for cfg, fid in pairs:
                ob = _evaluate(problem, cfg, rng); ob.fidelity = fid; ob.cost = fid
                new.append(ob)
        else:
            new = [_evaluate(problem, cfg, rng) for cfg in opt.ask(n)]
        opt.tell(new); obs.extend(new); done += len(new)
    return {"metric": _metric(problem, obs, cost_budget),
            "observations": obs,
            "context": _context(problem, budget, batch_size)}

def _evaluate(problem, config, rng):
    true = problem.objectives_fn(config) if problem.objectives_fn else {"y": problem.fn(config)}
    noisy = {}
    for name, value in true.items():
        reps = [value + rng.normal(0.0, problem.noise_std) for _ in range(problem.n_replicates)]
        noisy[name] = float(np.mean(reps))
    return Observation(x=config, objectives=noisy,
                       objective_std=float(np.std(reps, ddof=1)) if len(reps) > 1 else 0.0,
                       n_replicates=problem.n_replicates,
                       constraints=problem.constraints_fn(config) if problem.constraints_fn else {},
                       meta={"true": true})

def _metric(problem, obs, cost_budget):
    tags = set(problem.tags)
    if "multi_objective" in tags:
        signed = np.vstack([signed_scores(list(o.meta["true"].values()), problem.objective) for o in obs])
        ref = np.asarray(problem.objective.ref_point) * np.asarray(
            [1.0 if d == "max" else -1.0 for d in problem.objective.directions])
        hv = hypervolume_2d(signed[non_dominated_mask(signed)], ref)
        return {"name": "true_hv_ratio",
                "value": hv / max(hypervolume_2d(problem.reference_front, ref), 1e-12)}
    if "constrained" in tags:
        feas = [o for o in obs if all(g <= 0 for g in o.constraints.values())]
        return {"name": "feasible_best",
                "value": max((o.meta["true"]["y"] for o in feas), default=-float("inf"))}
    if "multi_fidelity" in tags:
        info = cost_aware_best(obs, budget=cost_budget)
        return {"name": "cost_aware_best", "value": info["best_full_fidelity"]}
    return {"name": "true_best", "value": max(o.meta["true"]["y"] for o in obs)}

def _context(problem, budget, batch_size):
    tags = set(problem.tags)
    return OptimizationContext.from_space(
        problem.space, budget_total=budget, batch_size=batch_size,
        n_objectives=problem.objective.n_objectives,
        n_constraints=len(problem.objective.constraints),
        noise="high" if problem.noise_std >= 0.1 else "medium" if problem.noise_std > 0 else "none",
        smoothness="rough" if "rough" in tags else "smooth" if "smooth" in tags else "mixed",
        multi_fidelity=bool(problem.fidelity_levels),
        domain="chemistry" if "chemistry" in tags else "generic")
```

---

## 2. `src/condopt/calibration.py`（约 -10/+15 行）

```diff
 @dataclass
 class RunRecord:
     problem: str
     algorithm: str
     score: float
+    metric_name: str = "true_best"
     context: dict = field(default_factory=dict)

 def collect_records(problems, algorithms, *, budget, batch_size, seeds,
-                    noise="medium", smoothness="mixed", multi_fidelity=False):
+                    cost_budget=None):
     records = []
+    probs = builtin_problems()
     for name in problems:
-        problem = builtin_problems()[name]
-        context = OptimizationContext.from_space(problem.space, budget_total=budget,
-            batch_size=batch_size, n_objectives=problem.objective.n_objectives)
+        problem = probs[name]
+        required = set(problem.tags) & {"multi_objective", "constraints", "multi_fidelity"}
         for algorithm in algorithms:
-            capabilities = set(registry.spec(algorithm).capabilities)
-            if required and not required <= capabilities: continue
-            special = capabilities & {"multi_objective","constraints","multi_fidelity"}
-            if special and not special <= required: continue
+            caps = set(registry.spec(algorithm).capabilities)
+            if required and not required <= caps:
+                records.append(RunRecord(name, algorithm, float("nan"), "skipped_capability", {}))
+                continue
             for seed in seeds:
-                result = run_algorithm(problem, algorithm, budget=budget,
-                                       batch_size=batch_size, seed=seed)
-                records.append(RunRecord(problem=name, algorithm=algorithm,
-                    score=float(result.best_score), context=context.to_dict()))
+                result = run_task(problem, algorithm, budget=budget,
+                                  batch_size=batch_size, seed=seed, cost_budget=cost_budget)
+                records.append(RunRecord(problem=name, algorithm=algorithm,
+                    score=float(result["metric"]["value"]),
+                    metric_name=result["metric"]["name"],
+                    context=result["context"].to_dict()))
     return records
```

- **删除** `noise/smoothness/multi_fidelity` 参数（context 从 problem 推导）；
- `lopo_validate`/`evaluate_weights` 无需改（只吃 `RunRecord`）；标签口径改为"higher-better + tie 检测"（`score` 的 NaN 行跳过）。

---

## 3. 488 runs 执行清单

```bash
# 1) 单目标核心（384）
py -3 - <<'PY'
from condopt.calibration import collect_records, lopo_validate
P = ["branin","sphere2d","mixed_reaction","suzuki_like",
     "discrete_catalyst_v2","high_cardinality_solvent"]
A = ["random","sobol","bayes","tpe"]
recs = collect_records(P, A, budget=16, batch_size=4, seeds=range(8))
recs += collect_records(P, A, budget=32, batch_size=4, seeds=range(8))
import json; print(json.dumps([r.__dict__ for r in recs]))
PY
# 2) 噪声（32）：noisy_sphere σ=0.05/0.15 × (bayes,tpe) × 8 seeds
# 3) 约束（32）：constraint_easy/hard × (constrained,bayes) × 8
# 4) 多目标（24）：moo_yield_cost × (parego,qehvi,qnehvi) × 8
# 5) 多保真（16）：mf_suzuki × (mfbo,bayes) × 8（cost_budget=24）
# 合并 → records.jsonl → lopo_validate(records, iterations=300, seed=0)
```

**耗时（8 进程）**：core ≈2 min；noise/constraint ≈1 min；MOO ≈2.5 min（qehvi/qnehvi 的 `n_y` 默认 128）；MF ≈0.5 min；拟合+LOPO ≈2 min → **总计 8–12 min**（单机 CPU）。

---

## 4. 验收

```python
def test_records_cover_all_metric_types():
    recs = collect_records(["branin","noisy_sphere","constraint_easy",
                            "moo_yield_cost","mf_suzuki"],
                           ["bayes","constrained","qehvi","mfbo","parego"],
                           budget=8, batch_size=2, seeds=(0,))
    assert {"true_best","true_hv_ratio","feasible_best","cost_aware_best"} <= {r.metric_name for r in recs}

def test_moo_metric_is_true_hv_bounded():
    recs = collect_records(["moo_yield_cost"], ["qehvi"], budget=8, batch_size=2, seeds=(0,))
    assert 0.0 <= recs[0].score <= 1.0 + 1e-9

def test_context_from_problem():
    recs = collect_records(["constraint_easy"], ["constrained"], budget=4, batch_size=2, seeds=(0,))
    assert recs[0].context["n_constraints"] == 1

def test_capability_skip_recorded():
    recs = collect_records(["branin"], ["qehvi"], budget=4, batch_size=2, seeds=(0,))
    assert recs and recs[0].metric_name == "skipped_capability"
```

**报告**：LOPO 给 `top1_strict`/`top1_tie`/`top2`/`regret(span 归一)` + 每折明细；不再用 context 外部参数。

---

## 5. 参考

R32（指标口径/统计）、R42（HV/真值）、R12（噪声多目标）、R41（成本/多保真）、R18（budget/cost）、R54（评估协议）、`RECORDS_V2_PATCH.md`、`CALIBRATION_RUN_PLAN.md`、`REVIEW_v13.md`。
