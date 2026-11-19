# RECORDS_V2_FIRST_STEP — 最小第一步：`BenchmarkProblem` 加 `tags/noise_std` + context 从 problem 推导

> 维护者：algo-research ｜ v1（2026-11-05，Round 16）
> 对应：`RECORDS_V2_IMPLEMENTATION.md`（完整版）、`RECORDS_V2_APPLY_PLAN.md`（完整流程）。
> 目标：**最小风险的一步**——只加元数据字段、只改 context 来源，**不动 `run_algorithm`/评估数据**。这样 `collect_records` 的 score/labels 保持不变，只有 router context 变真实；为下一步（真噪声/MOO/约束/MF）铺路，且可单独回滚。

---

## 1. `src/condopt/benchmarks.py`（+约 20 行）

### 1.1 dataclass 字段（只加 2 个）

```diff
 @dataclass
 class BenchmarkProblem:
     ...
+    tags: tuple[str, ...] = ()
+    noise_std: float = 0.0
```

### 1.2 `builtin_problems()` 加标签（逐问题一行）

```diff
 "branin": BenchmarkProblem(..., tags=("continuous", "smooth")),
 "sphere2d": BenchmarkProblem(..., tags=("continuous", "smooth")),
 "mixed_reaction": BenchmarkProblem(..., tags=("mixed", "chemistry")),
 "suzuki_like": BenchmarkProblem(..., tags=("mixed", "chemistry")),
 "discrete_catalyst_v2": BenchmarkProblem(..., tags=("discrete",)),
 "high_cardinality_solvent": BenchmarkProblem(..., tags=("discrete", "high_cardinality")),
 "noisy_sphere": BenchmarkProblem(..., tags=("noisy",), noise_std=0.05),
 "constraint_easy": BenchmarkProblem(..., tags=("constrained",)),
 "constraint_hard": BenchmarkProblem(..., tags=("constrained",)),
 "moo_yield_cost": BenchmarkProblem(..., tags=("multi_objective",)),
 "mf_suzuki": BenchmarkProblem(..., tags=("multi_fidelity", "chemistry")),
```

### 1.3 一个只读 helper（供 calibration 用，避免 import 循环）

```python
def problem_tags(problem: BenchmarkProblem) -> dict:
    """Metadata used to derive the router context (no heavy imports)."""
    tags = set(problem.tags)
    return {
        "noise": "high" if problem.noise_std >= 0.1 else "medium" if problem.noise_std > 0 else "none",
        "smoothness": "rough" if "rough" in tags else "smooth" if "smooth" in tags else "mixed",
        "domain": "chemistry" if "chemistry" in tags else "generic",
        "multi_fidelity": bool(problem.objective.fidelity),
        "n_constraints": len(problem.objective.constraints),
        "n_objectives": problem.objective.n_objectives,
    }
```

**注意**：`multi_fidelity` 从 `objective.fidelity` 推导（`mf_suzuki` 已配置）；这样不必依赖 tags。

---

## 2. `src/condopt/calibration.py`（+约 15 行）

### 2.1 新增 context 构造函数

```diff
+def _context_from_problem(problem, budget: int, batch_size: int) -> OptimizationContext:
+    meta = problem_tags(problem)
+    return OptimizationContext.from_space(
+        problem.space, budget_total=budget, batch_size=batch_size,
+        n_objectives=meta["n_objectives"], n_constraints=meta["n_constraints"],
+        noise=meta["noise"], smoothness=meta["smoothness"],
+        multi_fidelity=meta["multi_fidelity"], domain=meta["domain"],
+    )
```

### 2.2 `collect_records` 里替换 context 来源

```diff
 def collect_records(problems, algorithms, *, budget, batch_size, seeds,
-                    noise="medium", smoothness="mixed", multi_fidelity=False):
+                    cost_budget=None):
     records = []
     for name in problems:
         problem = builtin_problems()[name]
-        context = OptimizationContext.from_space(
-            problem.space, budget_total=budget, batch_size=batch_size,
-            n_objectives=problem.objective.n_objectives,
-            n_constraints=len(problem.objective.constraints),
-            noise=noise, smoothness=smoothness,
-            multi_fidelity=multi_fidelity,
-        )
+        context = _context_from_problem(problem, budget, batch_size)
```

**保留** `run_algorithm` 不变（score 仍是"无噪标量"）——这是本步的**已知局限**，在 docstring 写明：

```python
"""Collect calibration records.

Step 1: contexts are derived from problem metadata (tags/noise_std); the
scores still come from the noiseless scalar runner, so noisy/MOO/constrained
problems are NOT yet represented faithfully (see RECORDS_V2_IMPLEMENTATION).
"""
```

### 2.3 兼容旧调用

`collect_records(..., noise=..., smoothness=...)` 的旧调用点（`scripts/calibrate_router.py` 等）需删除这些参数；建议在签名里保留 `**legacy` 并 `warnings.warn("noise= is ignored; context derives from problem metadata")`，避免外部脚本直接崩。

---

## 3. 测试

```python
def test_context_from_problem_metadata():
    from condopt.benchmarks import builtin_problems, problem_tags
    from condopt.calibration import _context_from_problem
    probs = builtin_problems()
    ctx = _context_from_problem(probs["constraint_easy"], 24, 2)
    assert ctx.n_constraints == 1
    ctx2 = _context_from_problem(probs["mf_suzuki"], 24, 2)
    assert ctx2.multi_fidelity is True
    ctx3 = _context_from_problem(probs["noisy_sphere"], 24, 2)
    assert ctx3.noise in ("medium", "high")
    ctx4 = _context_from_problem(probs["moo_yield_cost"], 32, 4)
    assert ctx4.n_objectives == 2

def test_collect_records_still_runs_on_core_subset():
    recs = collect_records(["branin", "sphere2d"], ["random", "bayes"],
                           budget=6, batch_size=2, seeds=(0,))
    assert len(recs) == 4
    assert all(r.score == r.score for r in recs)   # 非 NaN

def test_legacy_args_warn():
    with pytest.warns(UserWarning, match="ignored"):
        collect_records(["branin"], ["random"], budget=4, batch_size=2,
                        seeds=(0,), noise="high")
```

---

## 4. 验证与预期输出

```bash
py -3 -m pytest -q                          # 期望 149 passed（+3 新测试）
py -3 - <<'PY'
from condopt.calibration import collect_records, evaluate_weights
recs = collect_records(["branin","noisy_sphere","constraint_easy","mf_suzuki"],
                       ["bayes","constrained","mfbo"], budget=8, batch_size=2, seeds=(0,))
print("n=", len(recs))
print({r.problem: r.context["noise"] for r in recs})
PY
```
**预期**：`noisy_sphere` 的 `noise` 为 `medium`（σ=0.05 <0.1）；`mf_suzuki` 的 context `multi_fidelity=true`；`constraint_easy` 的 `n_constraints=1`；`branin` 的 `noise=none`。

**注意**：本步会改变 `evaluate_weights`/LOPO 的路由结果（context 变了、数据没变），与 Round 11 实测的"context-only 会反转证据"一致——所以**本步只用于让 context 正确**，LOPO 数字仍不能当噪声校准的证据；下一 step 才引入真噪声/MOO 指标。

---

## 5. 回滚点

| 点 | 操作 |
|---|---|
| R0 | `git rev-parse HEAD` 记录 |
| R1 | benchmarks 改完未动 calibration：`git checkout -- src/condopt/benchmarks.py` |
| R2 | calibration 改完测试失败：`git checkout -- src/condopt/calibration.py` |
| R3 | 外部脚本仍传旧参数：保留 `**legacy` + warning（本步已设计） |

---

## 6. 参考

R32（指标口径）、R42（真值/HV）、R12（噪声多目标）、R41（成本/多保真）、R54（评估协议）、`RECORDS_V2_IMPLEMENTATION.md`、`RECORDS_V2_APPLY_PLAN.md`、`REVIEW_v15.md`。
