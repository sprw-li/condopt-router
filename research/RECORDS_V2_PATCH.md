# RECORDS_V2_PATCH — `collect_records` 支持真噪声/MOO/约束/MF 的最小实现补丁

> 维护者：algo-research ｜ v1（2026-11-05，Round 13）
> 对应：`CALIBRATION_RECORDS_V2.md`（Round 12 设计）、`REVIEW_v12.md`（records 仍未扩展）。
> 目标：**最小 diff**（`benchmarks.py` + `calibration.py` 两个文件）让 records 覆盖四类问题，并给出 456–488 runs 清单与验收。

---

## 1. `benchmarks.py`：`BenchmarkProblem` 增加元数据（默认值，向后兼容）

```python
@dataclass
class BenchmarkProblem:
    ...  # 现有字段不变
    # NEW
    tags: tuple[str, ...] = ()
    noise_std: float = 0.0
    n_replicates: int = 1
    fidelity_levels: tuple[float, ...] = ()
    fidelity_costs: tuple[float, ...] = ()
    reference_front: "np.ndarray | None" = None
```

**各问题只需加一行**（示例）：

```python
noisy_sphere = BenchmarkProblem(..., tags=("noisy",), noise_std=0.05, n_replicates=3)
constraint_easy = BenchmarkProblem(..., tags=("constrained",))
moo_yield_cost = BenchmarkProblem(..., tags=("multi_objective",),
                                  reference_front=moo_reference_front())
mf_suzuki = BenchmarkProblem(..., tags=("multi_fidelity",),
                             fidelity_levels=(0.25, 0.5, 1.0),
                             fidelity_costs=(0.25, 0.5, 1.0))
```

---

## 2. `benchmarks.py`：新增 `run_task`（按类型分派）

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
            opt.cost_remaining = cost_budget       # 与 Study 相同的注入方式
            pairs = opt.ask_with_fidelity(n)
            new = []
            for cfg, fid in pairs:
                ob = _evaluate(problem, cfg, rng); ob.fidelity = fid; ob.cost = fid
                new.append(ob)
        else:
            new = [_evaluate(problem, cfg, rng) for cfg in opt.ask(n)]
        opt.tell(new); obs.extend(new); done += len(new)
    return {"metric": _metric(problem, obs, cost_budget), "observations": obs,
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

### `_metric`（四口径）

```python
def _metric(problem, obs, cost_budget):
    tags = set(problem.tags)
    if "multi_objective" in tags:
        signed = np.vstack([signed_scores(list(o.meta["true"].values()), problem.objective) for o in obs])
        ref = np.asarray(problem.objective.ref_point) * np.asarray(
            [1.0 if d == "max" else -1.0 for d in problem.objective.directions])
        hv = hypervolume_2d(signed[non_dominated_mask(signed)], ref)
        return {"name": "true_hv_ratio", "value": hv / max(hypervolume_2d(problem.reference_front, ref), 1e-12)}
    if "constrained" in tags:
        feas = [o for o in obs if all(g <= 0 for g in o.constraints.values())]
        return {"name": "feasible_best", "value": max((o.meta["true"]["y"] for o in feas), default=-float("inf"))}
    if "multi_fidelity" in tags:
        info = cost_aware_best(obs, budget=cost_budget)
        return {"name": "cost_aware_best", "value": info["best_full_fidelity"]}
    values = [o.meta["true"]["y"] for o in obs]
    return {"name": "true_best", "value": max(values)}
```

---

## 3. `calibration.py`：`collect_records` 改为调用 `run_task`

```diff
-                result = run_algorithm(problem, algorithm, budget=budget, batch_size=batch_size, seed=seed)
+                result = run_task(problem, algorithm, budget=budget,
+                                  batch_size=batch_size, seed=seed, cost_budget=cost_budget)
                 records.append(
                     RunRecord(problem=name, algorithm=algorithm,
-                              score=float(result.best_score),
-                              context=context.to_dict()))
+                              score=float(result["metric"]["value"]),
+                              metric_name=result["metric"]["name"],
+                              context=result["context"].to_dict()))
```

`RunRecord` 加 `metric_name: str = "true_best"`；`collect_records` 的 `noise/smoothness/multi_fidelity` 外部参数**删除**（context 从 problem 推导）；保留 `budget/batch/seeds/cost_budget`。

**能力过滤**：`required` 从 `problem.tags` 推导（`multi_objective`/`constraints`/`multi_fidelity`），过滤逻辑改为"required ⊆ capabilities"，并保留"算法声明了 context 不需要的互斥能力"显式规则（qehvi/qnehvi 需 MOO）。

---

## 4. 456–488 runs 清单

| 组 | 问题 | 算法 | 预算 | runs |
|---|---|---|---|---|
| 单目标核心 | branin, sphere2d, mixed_reaction, suzuki_like, discrete_catalyst_v2, high_cardinality_solvent | random, sobol, bayes, tpe | 16/32 | 6×4×2×8 = **384** |
| 噪声 | noisy_sphere（σ=0.05/0.15, reps 1/3） | bayes, tpe | 24 | 2×2×8 = **32** |
| 约束 | constraint_easy/hard | constrained, bayes | 24 | 2×2×8 = **32** |
| 多目标 | moo_yield_cost | parego, qehvi, qnehvi | 32 | 1×3×8 = **24** |
| 多保真 | mf_suzuki | mfbo, bayes | cost 24 | 1×2×8 = **16** |
| **合计** | | | | **488** |

预计耗时：core ~2 min（8 进程）、MOO ~2.5 min、其余 ~2 min、拟合 ~2 min → **8–12 min**。

---

## 5. 验收（可直接写 pytest）

```python
def test_records_cover_all_types():
    recs = collect_records(["branin", "noisy_sphere", "constraint_easy",
                            "moo_yield_cost", "mf_suzuki"],
                           ["bayes", "constrained", "qehvi", "mfbo", "parego"],
                           budget=8, batch_size=2, seeds=(0,))
    names = {r.metric_name for r in recs}
    assert {"true_best", "true_hv_ratio", "feasible_best", "cost_aware_best"} <= names

def test_noise_changes_records():
    a = collect_records(["noisy_sphere"], ["bayes"], budget=8, batch_size=2, seeds=(0,))
    b = collect_records(["noisy_sphere"], ["bayes"], budget=8, batch_size=2, seeds=(0,))
    assert a[0].score != b[0].score          # 噪声下同 seed 也可能因 context/重复数不同而不同
    # 更好：比较 true_best 的 seed 方差（replicates>1 应更小）

def test_moo_metric_true_hv_bounded():
    recs = collect_records(["moo_yield_cost"], ["qehvi"], budget=8, batch_size=2, seeds=(0,))
    assert 0.0 <= recs[0].score <= 1.0 + 1e-9

def test_context_from_problem_metadata():
    recs = collect_records(["constraint_easy"], ["constrained"], budget=4, batch_size=2, seeds=(0,))
    assert recs[0].context["n_constraints"] == 1

def test_capability_skip_is_recorded():
    # qehvi 用于单目标 → status="skipped_capability"，不抛异常
```

**报告口径**：LOPO 用 `top1_strict` + `top1_tie` 双列；regret = `(best − chosen)/(best − worst + eps)`；每折明细。

---

## 6. 参考

R32（指标口径）、R42（HV/真值）、R12（噪声多目标）、R41（成本/多保真）、R18（budget/cost）、R54（HPO 评估协议）、`CALIBRATION_RECORDS_V2.md`、`CALIBRATION_RUN_PLAN.md`、`REVIEW_v12.md`。
