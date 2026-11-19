# CALIBRATION_RECORDS_V2 — `collect_records` 支持真噪声 / MOO true-HV / 约束 / 多保真的最小改动

> 维护者：algo-research ｜ v1（2026-11-05，Round 12）
> 对应：`REVIEW_v11.md` P1-2、`CALIBRATION_NOISE_RUN.md`（context-only noise 的教训）、`CALIBRATION_RUN_PLAN.md`（456-runs）。
> 文献：R32（指标口径按任务类型）、R42（HV/固定 ref/真值）、R12（噪声多目标）、R41（成本/多保真）、R18（budget/cost 协议）、R54（HPO 评估协议综述）。
> 目标：**最小改动**让 `collect_records` 产出四类 records（单目标/噪声/多目标/约束/多保真），且标签与 context 都来自 problem 元数据，而不是调用方参数。

---

## 1. `BenchmarkProblem` 增加元数据（`benchmarks.py`）

```python
@dataclass
class BenchmarkProblem:
    ...
    # --- 已有：space/objective/fn/optimum/best_config/objectives_fn/constraints_fn ---
    # --- 新增 ---
    tags: tuple[str, ...] = ()                 # ("continuous","mixed","discrete","noisy","constrained","multi_objective","multi_fidelity","chemistry")
    noise_std: float = 0.0                     # 观测噪声 σ（绝对，目标单位）
    n_replicates: int = 1                      # 每次评估重复次数
    fidelity_levels: tuple[float, ...] = ()    # 空 = 单保真
    fidelity_costs: tuple[float, ...] = ()     # 成本（默认 = levels）
    reference_front: np.ndarray | None = None  # MOO 的 signed 参考前沿（moo_yield_cost 已有 moo_reference_front()）
```

**各问题的填充（示例）**

| 问题 | tags | noise_std | n_rep | fidelity |
|---|---|---|---|---|
| `branin`/`sphere2d`/`mixed_reaction`/`suzuki_like` | continuous/mixed | 0 | 1 | — |
| `discrete_catalyst_v2`/`high_cardinality_solvent` | discrete | 0 | 1 | — |
| `noisy_sphere` | noisy | 0.05 / 0.15 | 1 / 3 | — |
| `constraint_easy/hard` | constrained | 0 | 1 | — |
| `moo_yield_cost` | multi_objective | 0（可配 1.5） | 1 | — |
| `mf_suzuki` | multi_fidelity | 0 | 1 | (0.25,0.5,1.0) |

---

## 2. `run_task`（新函数，按类型分派）

```python
def run_task(problem, algorithm, *, budget, batch_size, seed,
             cost_budget: float | None = None) -> dict:
    """单目标/噪声/约束/MOO/多保真统一入口；返回 metric + 轨迹。"""
    rng = np.random.default_rng(seed)
    opt = registry.build(algorithm, problem.space, problem.objective, seed=seed)
    obs = []

    def evaluate(config) -> Observation:
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

    done = 0
    while done < budget:
        n = min(batch_size, budget - done)
        if getattr(opt, "supports_fidelity", False):
            pairs = opt.ask_with_fidelity(n, cost_budget=cost_budget)
            new = [evaluate(cfg) for cfg, _ in pairs]
            for (cfg, fid), ob in zip(pairs, new):
                ob.fidelity = fid; ob.cost = problem_cost(problem, fid)
        else:
            new = [evaluate(cfg) for cfg in opt.ask(n)]
        opt.tell(new); obs.extend(new); done += len(new)

    # ---- metric（higher is better）----
    metric = compute_metric(problem, obs, cost_budget)
    return {"metric_name": metric["name"], "metric_value": metric["value"],
            "context": context_from_problem(problem, budget, batch_size),
            "trajectory": metric["trajectory"], "n_obs": len(obs)}
```

### 2.1 `compute_metric`（标签口径）

```python
def compute_metric(problem, obs, cost_budget):
    tag = set(problem.tags)
    if "multi_objective" in tag:
        # 真值 HV ratio（固定 ref，来自 Objective.ref_point）
        scores = np.vstack([signed_scores(list(o.meta["true"].values()), problem.objective) for o in obs])
        ref = np.asarray(problem.objective.ref_point) * np.asarray(
            [1.0 if d == "max" else -1.0 for d in problem.objective.directions])
        hv = hypervolume_2d(scores[non_dominated_mask(scores)], ref)
        hv_ref = hypervolume_2d(problem.reference_front, ref)
        return {"name": "true_hv_ratio", "value": hv / max(hv_ref, 1e-12),
                "trajectory": prefix_hv(obs, ref, hv_ref)}
    if "constrained" in tag:
        feas = [o for o in obs if all(g <= 0 for g in o.constraints.values())]
        best = max((o.meta["true"]["y"] for o in feas), default=-inf)
        return {"name": "feasible_best", "value": best,
                "trajectory": prefix_feasible_best(obs)}
    if "multi_fidelity" in tag:
        info = cost_aware_best(obs, budget=cost_budget)
        return {"name": "cost_aware_best", "value": info["best_full_fidelity"],
                "trajectory": info["curve"]}
    # 单目标（含噪声：用真值或重复均值）
    values = [o.meta["true"]["y"] for o in obs]
    return {"name": "true_best", "value": max(values), "trajectory": np.maximum.accumulate(values)}
```

### 2.2 context 从 problem 推导（不再传参）

```python
def context_from_problem(problem, budget, batch_size):
    tag = set(problem.tags)
    return OptimizationContext.from_space(
        problem.space, budget_total=budget, batch_size=batch_size,
        n_objectives=problem.objective.n_objectives,
        n_constraints=len(problem.objective.constraints),
        noise=("high" if problem.noise_std >= 0.1 else "medium" if problem.noise_std > 0 else "none"),
        smoothness=("rough" if "rough" in tag else "smooth" if "smooth" in tag else "mixed"),
        multi_fidelity=bool(problem.fidelity_levels),
        domain=("chemistry" if "chemistry" in tag else "generic"),
    )
```
- **删除** `collect_records` 的 `noise=`/`smoothness=`/`multi_fidelity=` 外部参数（或降级为 debug override，并在报告里标注）。

### 2.3 `RunRecord` 扩展

```python
@dataclass
class RunRecord:
    problem: str
    algorithm: str
    score: float              # = metric_value（higher is better）
    metric_name: str          # true_best / true_hv_ratio / feasible_best / cost_aware_best
    context: dict
    status: str = "ok"        # ok / failed / skipped_capability
```

---

## 3. 正式 456-runs 的更新清单

| 组 | 问题 | 算法（能力过滤后） | 预算 | runs |
|---|---|---|---|---|
| 单目标核心 | branin, sphere2d, mixed_reaction, suzuki_like, discrete_catalyst_v2, high_cardinality_solvent | random, sobol, bayes, tpe | 16/32 | 6×4×2×8 = 384 |
| 噪声 | noisy_sphere（σ=0.05/0.15；replicates 1/3） | bayes, tpe | 24 | 2σ×2×8 = 32 |
| 约束 | constraint_easy/hard | constrained, bayes | 24 | 2×2×8 = 32 |
| 多目标 | moo_yield_cost | parego, qehvi, qnehvi | 32 | 1×3×8 = 24 |
| 多保真 | mf_suzuki | mfbo, bayes（cost 24） | — | 1×2×8 = 16 |
| **合计** | | | | **488（≈460–500）** |

- 单目标 core 的 `lhs` 可去掉（与 sobol 高度重叠）→ 上面 384 已是 4 算法；
- 噪声组只 2 算法（bayes/tpe）足够验证 router 噪声规则；
- 预计耗时：core ~2 min（8 进程）、MOO ~2.5 min、其余 ~2 min、拟合 ~2 min → **~8–12 min**。

---

## 4. 验收

```python
def test_collect_records_covers_all_types():
    recs = collect_records(["branin", "noisy_sphere", "constraint_easy",
                            "moo_yield_cost", "mf_suzuki"],
                           ["bayes", "constrained", "qehvi", "mfbo", "parego"],
                           budget=8, batch_size=2, seeds=(0,))
    metrics = {r.metric_name for r in recs}
    assert {"true_best", "true_hv_ratio", "feasible_best", "cost_aware_best"} <= metrics
def test_noise_affects_records():
    # 同一 problem，noise_std=0 vs 0.15（replicates=3）→ true_best 的 seed 方差显著更大
def test_moo_metric_uses_true_hv():     # value ≤ 1 + 1e-9；与 observed HV 区分
def test_capability_filter_skips_singleobjective_algorithms_for_moo():
    # 把 qehvi 传给单目标 problem → status="skipped_capability"，不抛异常
def test_context_matches_problem_metadata():
    # constraint_easy → ctx.n_constraints==1；mf_suzuki → ctx.multi_fidelity True
```

**报告口径**：LOPO 指标用 `top1_strict`（chosen == label）与 `top1_tie`（并列最优）双列；regret 用 `(best − chosen)/(best − worst + eps)`；每折给明细。

---

## 5. 参考

R32（指标口径）、R42（HV/真值）、R12（噪声多目标）、R41（成本/多保真）、R18（budget/cost）、R54（HPO 评估协议）、`CALIBRATION_NOISE_RUN.md`、`CALIBRATION_RUN_PLAN.md`、`REVIEW_v11.md`。
