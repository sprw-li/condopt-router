# ROUTER_CALIBRATION_HARNESS — 最小 LOPO-CV 校准 harness（可落成 `scripts/calibrate_router.py`）

> 维护者：algo-research ｜ v1（2026-11-05，Round 6）
> 对应：`BENCHMARK_SPEC.md` §6–7（矩阵/协议/校准方法）、`ROUTER_SCORETERM.md`（ScoreTerm 重构）、`ROUTER_CALIBRATION.md`（Round 4 方法论）。
> 文献：R32（benchmark 统计规范：多 seed/IQM/CI）、R8（SMAC3 Facade 作为"场景先验"的可校准性）、R17（推荐器配置化）、R24（规则需可解释）。
> 目标：一条命令跑出 `router-weights-YYYY-MM-DD.json` + 报告；**v0.5 阶段先在 8–12 个问题上验证流程**，不追求 full matrix。

---

## 1. 组件与数据流

```
problems (Tier A–F) ──┐
algorithms (registry) ─┼→ run_grid() ─→ runs.jsonl ─→ labels() ─→ LOPO-CV ─→ weights.json
seeds / budgets ──────┘                                              └→ report.md
```

四层，每层独立可测：

1. `run_grid`：`(problem, algorithm, budget, batch, seed) → RunRecord`（直接调 `registry.build` + 评测函数，**不经 Study**，快 10×）。
2. `aggregate`：每 `(problem, budget, batch)` 聚合 seeds → `IQM regret` / `IQM auc`。
3. `labels`：按 `ctx_key` 生成 `best/top2/tie`。
4. `calibrate`：`ScoreTerm` 权重拟合（随机搜索 + 坐标下降）+ LOPO-CV。

---

## 2. 问题与算法清单（v0.5 阶段）

| 问题 | 来源 | tags | 预算 |
|---|---|---|---|
| `branin`、`sphere2d`、`hartmann3` | `benchmarks.builtin_problems()` | continuous | 16 |
| `mixed_reaction`、`suzuki_like` | Tier B | mixed, chemistry | 16, 32 |
| `discrete_catalyst_v2`、`high_cardinality_solvent`（one-hot 路径） | Tier C | discrete/categorical | 24, 48 |
| `noisy_sphere`（σ=0.05/0.15） | Tier D | noisy | 24 |
| `constraint_easy`、`constraint_hard` | Tier D | constraints | 24 |
| `moo_yield_cost` | Tier F | multi_objective | 32 |

**算法**（`registry.names()` 中声明了所需能力的）：`random, sobol, lhs, grid, bayes, tpe, parego, constrained, qehvi`（按问题能力自动过滤：m=2 时 parego/qehvi，有约束时 constrained，等等）。

**数据量（最小可行）**：8 问题 × 4–6 算法 × 2 预算 × 8 seeds ≈ **512–768 runs**；单 run 16–32 次评估，`bayes` 单 run 约 2–4 s → 单机 8 进程 **10–20 min**。先跑 `--matrix smoke`。

---

## 3. 代码骨架

```python
# scripts/calibrate_router.py
"""Calibrate router ScoreTerm weights with leave-one-problem-out CV."""
from __future__ import annotations
import argparse, json, math
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np

from condopt import registry
from condopt.benchmarks import builtin_problems
from condopt.base import Objective
from condopt.dataset import Observation
from condopt.router import OptimizationContext, _terms_for, _legacy_score   # 见 ROUTER_SCORETERM.md

# ---------- 1. run grid ------------------------------------------------------
@dataclass
class RunRecord:
    problem: str; algorithm: str; budget: int; batch: int; seed: int
    best: float; trajectory: list[float]; regret: float; auc: float; runtime_s: float

def run_one(problem, algorithm, budget, batch, seed) -> RunRecord:
    opt = registry.build(algorithm, problem.space, problem.objective, seed=seed)
    best = -math.inf if problem.objective.direction == "max" else math.inf
    curve, done, rng = [], 0, np.random.default_rng(seed)
    while done < budget:
        n = min(batch, budget - done)
        configs = opt.ask(n)
        obs = []
        for c in configs:
            value = problem.fn(c)
            obs.append(Observation(x=c, y=value, source="benchmark"))
            best = max(best, value) if problem.objective.direction == "max" else min(best, value)
        opt.tell(obs); done += len(obs)
        curve.append(problem.regret(best) if hasattr(problem, "regret") else 0.0)
    return RunRecord(problem.name, algorithm, budget, batch, seed, float(best), curve,
                     float(problem.regret(best) or 0.0), float(np.trapz(curve) / max(len(curve), 1)), 0.0)

def run_grid(problems, algorithms, budgets, batches, seeds, out_path, workers=8):
    # 用 multiprocessing.Pool(workers) 并行 run_one；每完成一个写一行 JSONL
    ...

# ---------- 2. aggregate + labels -------------------------------------------
def iqm(values: list[float]) -> float:
    v = np.sort(values); k = len(v) // 4
    return float(np.mean(v[k:len(v)-k])) if len(v) > 2 * k else float(np.mean(v))

def build_labels(records, tie_tol=0.05) -> dict:
    """ctx_key=(problem,budget,batch) -> {best, top2, stats, is_tie}"""
    groups = {}
    for r in records:
        groups.setdefault((r.problem, r.budget, r.batch), []).append(r)
    labels = {}
    for key, rows in groups.items():
        stats = {a: iqm([r.regret for r in rows if r.algorithm == a]) for a in {r.algorithm for r in rows}}
        ranked = sorted(stats, key=stats.get)
        if len(ranked) < 2: continue
        tie = (stats[ranked[1]] - stats[ranked[0]]) < tie_tol * (stats[ranked[-1]] - stats[ranked[0]] + 1e-9)
        labels[key] = {"stats": stats, "ranked": ranked, "best": ranked[0],
                       "top2": set(ranked[:2]), "is_tie": bool(tie)}
    return labels

def build_contexts(problems: dict, records) -> dict:
    """ctx_key -> OptimizationContext（用 problem 的 space/objective 重建，不能只用 bench 行）"""
    contexts = {}
    for r in records:
        p = problems[r.problem]
        key = (r.problem, r.budget, r.batch)
        if key in contexts: continue
        contexts[key] = OptimizationContext.from_space(
            p.space, n_objectives=p.objective.n_objectives,
            n_constraints=len(getattr(p.objective, "constraints", ())),
            budget_total=r.budget, batch_size=r.batch,
            noise=getattr(p, "noise", "none"), smoothness="mixed")
    return contexts

# ---------- 3. objective + search (见 ROUTER_SCORETERM / ROUTER_CALIBRATION) --
def make_objective(contexts, labels, algorithms):
    def objective(w):
        loss = 0.0
        for key, ctx in contexts.items():
            scores = {a: _legacy_score(_terms_for(a, ctx), w)[0] for a in algorithms}
            chosen = max(scores, key=scores.get)
            lab = labels.get(key)
            if lab is None: continue
            regret = lab["stats"][chosen] - lab["stats"][lab["best"]]
            worst = max(lab["stats"].values()) - lab["stats"][lab["best"]] + 1e-9
            loss += (0.5 * (chosen != lab["best"] and not lab["is_tie"])
                     + 0.3 * (chosen not in lab["top2"])
                     + 0.2 * regret / worst)
        return loss / max(len(contexts), 1)
    return objective

def calibrate(objective, features, n_iter=300, seed=0):
    rng = np.random.default_rng(seed)
    best = {f: 1.0 for f in features}; best_val = objective(best)
    for _ in range(n_iter // 2):
        w = {f: float(np.clip(rng.normal(1.0, 0.35), 0.2, 2.5)) for f in features}
        val = objective(w)
        if val < best_val: best, best_val = w, val
    grid = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
    for _ in range(2):
        for f in features:
            for g in grid:
                w = dict(best); w[f] = g
                val = objective(w)
                if val < best_val - 1e-9: best, best_val = w, val
    return best, best_val

# ---------- 4. LOPO + output -------------------------------------------------
def lopo_validate(problems, records, algorithms, features):
    tasks = sorted({r.problem for r in records})
    folds = []
    for held in tasks:
        train = [r for r in records if r.problem != held]
        test = [r for r in records if r.problem == held]
        contexts_tr = build_contexts(problems, train); labels_tr = build_labels(train)
        w, _ = calibrate(make_objective(contexts_tr, labels_tr, algorithms), features)
        ctx_te = build_contexts(problems, test); labels_te = build_labels(test)
        folds.append(evaluate(w, ctx_te, labels_te, algorithms))
    return folds

def evaluate(w, contexts, labels, algorithms):
    top1 = top2 = 0; gaps = []
    for key, ctx in contexts.items():
        lab = labels.get(key)
        if lab is None: continue
        chosen = max({a: _legacy_score(_terms_for(a, ctx), w)[0] for a in algorithms}, key=lambda k: _legacy_score(_terms_for(k, ctx), w)[0])
        top1 += int(chosen == lab["best"] and not lab["is_tie"])
        top2 += int(chosen in lab["top2"])
        gaps.append(lab["stats"][chosen] - lab["stats"][lab["best"]])
    n = max(len(contexts), 1)
    return {"top1": top1 / n, "top2": top2 / n, "regret_gap": float(np.mean(gaps)) if gaps else 0.0}
```

---

## 4. CLI 与输出

```bash
py -3 scripts/calibrate_router.py \
  --matrix smoke \
  --problems branin sphere2d mixed_reaction suzuki_like discrete_catalyst_v2 noisy_sphere constraint_hard moo_yield_cost \
  --algorithms random sobol bayes tpe parego constrained qehvi \
  --budgets 16 32 --batches 1 4 --seeds 0 1 2 3 4 5 6 7 \
  --workers 8 \
  --out research/router-weights-2026-11-05.json \
  --report research/router-calibration-2026-11-05.md
```

**`runs.jsonl` 每行**：

```json
{"problem":"suzuki_like","algorithm":"bayes","budget":32,"batch":4,"seed":3,
 "best":107.2,"regret":0.07,"auc":0.61,"runtime_s":3.1,"trajectory":[0.1,0.3]}
```

**`router-weights-*.json`**（schema 见 `ROUTER_CALIBRATION.md` §2.2）：

```json
{"version":"router-weights-2026-11-05","base":{},"weights":{"dim_le_12":1.1,"dim_gt_20":0.7},
 "trained_on":{"matrix":"smoke","n_contexts":16,"seeds":8},
 "validation":{"lopo_top1":0.62,"lopo_top2":0.87,"regret_gap":0.05,"n_folds":8}}
```

**报告 `router-calibration-*.md`**：特征权重表、LOPO 每折 top-1/top-2/regret、按问题类型的命中率、失败案例（router 选错且差距 >20% 的 context）。

---

## 5. 预算与验收

| 项 | 值 |
|---|---|
| 最小矩阵（8 问题×4 算法×2 预算×8 seeds） | 512 runs ≈ 10–20 min（8 进程） |
| 校准搜索（150 随机 + 2 轮坐标） | <5 min（纯查表） |
| LOPO（8 折，每折重拟合） | ≈ 8 × 5 min = 40 min（可并行折） |
| 总计（第一轮） | **<1.5 h**（夜间跑） |

**验收**（写入 CI 的可选 gate）：

- `top1 ≥ 0.55`、`top2 ≥ 0.80`、`regret_gap ≤ 0.10`（第一轮目标；数据多后可提高）；
- 与默认权重（全 1）相比，LOPO 指标不劣化；
- 权重符号人工 review 通过（例如"高维 → GP 加分"必须 ≤0）；
- 同输入两次运行产出**逐字节相同**的 weights 文件。

---

## 6. 实现顺序（建议）

1. `run_grid` + `runs.jsonl`（半天，复用 `benchmarks.run_algorithm` 的评测逻辑）；
2. `aggregate/labels/contexts`（半天）+ 单测（tie 处理、LOPO 无泄漏）；
3. `calibrate/lopo/evaluate`（半天，依赖 `ROUTER_SCORETERM.md` 的 `_terms_for`）；
4. CLI + 报告（半天）；
5. 第一轮夜间跑 + 人工 review 权重符号。

---

## 7. 参考

R32（统计规范：IQM/CI/多 seed/LOPO）、R8（场景先验的可校准性）、R17（推荐器配置化）、R24（规则可解释）、`BENCHMARK_SPEC.md` §6–7、`ROUTER_SCORETERM.md`、`ROUTER_CALIBRATION.md`。
