# MF_BENCHMARK — `mf_suzuki` 代码级基准与 `mfbo` 验收

> 维护者：algo-research ｜ v1（2026-11-05，Round 6）
> 对应：`BENCHMARK_SPEC.md` Tier F、`MULTIFIDELITY_MVP.md`（触点与 `mfbo` 伪代码）、`BENCHMARK_TIER_CD.md`（`suzuki_like` 全保真定义）。
> 文献：R41（BOCA：保真度作为连续输入）、R39/R40（SH/Hyperband 成本口径）、R18（Olympus 的预算/成本协议）、R32（benchmark 可复现协议）。
> 目标：给出一份能直接写进 `src/condopt/benchmarks.py` 的多保真任务 + 可直接写进 `tests/test_mf.py` 的 5 条验收。

---

## 1. 任务定义

### 1.1 space（与全保真 `suzuki_like` 相同）

```python
space = SearchSpace.from_dict([
    {"name": "T",           "kind": "continuous", "low": 40.0, "high": 120.0, "unit": "degC"},
    {"name": "time_min",    "kind": "log",        "low": 5.0,  "high": 240.0, "unit": "min"},
    {"name": "cat_loading", "kind": "log",        "low": 0.1,  "high": 5.0,   "unit": "mol%"},
    {"name": "solvent",     "kind": "categorical", "choices": ["MeCN", "EtOH", "THF", "water", "DMF"]},
    {"name": "ligand",      "kind": "categorical", "choices": ["L1", "L2", "L3", "L4", "L5", "L6"]},
    {"name": "base_eq",     "kind": "integer",    "low": 1, "high": 4},
])
```

### 1.2 全保真目标（可核验：最优 110.0 @ (85, 60, 1.5, MeCN, L1, 2)）

```python
SB = {"MeCN": 5.0, "EtOH": 3.0, "THF": 2.0, "water": -2.0, "DMF": 4.0}
LB = {"L1": 4.0, "L2": 2.0, "L3": 3.0, "L4": 1.0, "L5": 0.0, "L6": 2.0}
BB = {1: 0.0, 2: 1.0, 3: 0.0, 4: -1.0}

def suzuki_yield(c: dict) -> float:
    return (20.0
            + 80.0 * math.exp(-((c["T"] - 85.0) / 20.0) ** 2
                              - (math.log(c["time_min"] / 60.0) / 0.9) ** 2
                              - (math.log(c["cat_loading"] / 1.5) / 0.7) ** 2)
            + SB[c["solvent"]] + LB[c["ligand"]] + BB[c["base_eq"]])
```

### 1.3 fidelity 与偏差项

```
levels = (0.25, 0.5, 1.0)            # 短时/中时/全时（或小试/中试/全量）
cost(s) = s                          # 成本与保真度成正比
bias(s) = 15.0 * (1 - s) ** 1.5      # 低保真系统性偏低：s=0.25→9.49，s=0.5→5.30，s=1→0
y_s = suzuki_yield(c) - bias(s) + noise
```

```python
FIDELITY = {"name": "scale", "levels": (0.25, 0.5, 1.0),
            "costs": (0.25, 0.5, 1.0), "continuous": False}

def mf_suzuki(c: dict, s: float) -> float:
    return suzuki_yield(c) - 15.0 * (1.0 - s) ** 1.5
```

**最优（全保真）**：`110.0` at `(T=85, time_min=60, cat_loading=1.5, MeCN, L1, base_eq=2)`；`s=0.5` 时最优值 `110 − 5.303 = 104.697`（偏差与 x 无关，最优点不变）。核验：`abs(mf_suzuki(opt, 1.0) - 110.0) < 1e-9`、`mf_suzuki(opt, 0.5) = 104.6967`。

---

## 2. 评估协议

```python
@dataclass
class MFBenchmarkProblem:
    name: str = "mf_suzuki"
    space: SearchSpace
    objective: Objective            # name="yield", direction="max", fidelity=FIDELITY
    levels: tuple[float, ...] = (0.25, 0.5, 1.0)
    costs: tuple[float, ...] = (0.25, 0.5, 1.0)
    noise_std: float = 0.0          # 绝对 σ（Tier D 用 0.5）
    replicates: int = 1

    def evaluate(self, config, fidelity, rng):
        base = mf_suzuki(config, fidelity) - 15.0 * (1.0 - fidelity) ** 1.5
        reps = [base + rng.normal(0.0, self.noise_std) for _ in range(self.replicates)]
        mean = float(np.mean(reps)); std = float(np.std(reps, ddof=1)) if len(reps) > 1 else 0.0
        return {"y": mean, "objective_std": std, "n_replicates": self.replicates,
                "fidelity": fidelity, "cost": float(fidelity)}
```

- **成本记账**：每次评估 `cost = s`；总成本 `Σ s`；**预算按成本**（不是次数）比较——这是多保真的意义所在。
- 同一个 x 可以出现在不同 fidelity（GP 输入含 `s`）。
- 初始设计：全部用 `min(levels)`（便宜）；`n_initial = max(2*(feature_dim+1), 6)`。

---

## 3. 指标

```
cost_aware_best(C) = max_{eval e : Σ_{e'≤e} cost(e') ≤ C} y(e)        # 在成本预算 C 内的最优
cost_to_best(τ)    = min{ C : cost_aware_best(C) ≥ τ }                # 达到 τ 所需成本
```

- 报告：`cost_aware_best` 曲线（x 轴成本、y 轴 best），而非"评估次数"曲线；
- 同时报告 `n_evals` 口径的 best-so-far（便于对照单保真算法）；
- 参考 baseline：`sobol@full`（只用全保真 Sobol）、`bayes@full`（只用全保真 BO）、`mf_bo`、`successive_halving`、`random@levels`（随机选 x 和 s）。

---

## 4. `mfbo` 的 5 条验收（可直接写 `tests/test_mf.py`）

```python
import numpy as np, pytest
from condopt.registry import build
from condopt.benchmarks import mf_suzuki_problem   # 上面定义的任务

OPT = {"T": 85.0, "time_min": 60.0, "cat_loading": 1.5,
       "solvent": "MeCN", "ligand": "L1", "base_eq": 2}

def test_mf_suzuki_optimum_verified():
    p = mf_suzuki_problem()
    assert p.evaluate(OPT, 1.0, np.random.default_rng(0))["y"] == pytest.approx(110.0, abs=1e-9)
    assert p.evaluate(OPT, 0.5, np.random.default_rng(0))["y"] == pytest.approx(104.6967, abs=1e-3)

def test_ask_with_fidelity_defaults_to_full_for_single_fidelity():
    # 单保真算法 (bayes) 的 ask_with_fidelity 必须全部返回 1.0
    prob = mf_suzuki_problem()
    opt = build("bayes", prob.space, prob.objective, seed=0)
    pairs = opt.ask_with_fidelity(3)
    assert len(pairs) == 3 and all(f == 1.0 for _, f in pairs)

def test_mfbo_uses_low_fidelity_early():
    # 前 10 个建议里, 低保真(s<1) 占比 > 50%, 且至少 1 个全保真
    prob = mf_suzuki_problem()
    opt = build("mf_bo", prob.space, prob.objective, seed=0)
    pairs = opt.ask_with_fidelity(10)
    fids = [f for _, f in pairs]
    assert sum(1 for f in fids if f < 1.0) > len(fids) / 2
    assert any(f == 1.0 for f in fids)

def test_mfbo_cost_aware_best_beats_full_fidelity_bo():
    # 同样成本预算 C=15 下, mf_bo 的 cost_aware_best ≥ 全保真 BO 的 1.05 倍(或差 ≥5)
    res = run_mf_suite(problems=["mf_suzuki"], algorithms=["mf_bo", "bayes_full"],
                       cost_budget=15.0, seeds=range(5))
    assert res["mf_bo"].cost_aware_best >= res["bayes_full"].cost_aware_best * 1.05

def test_mfbo_records_fidelity_in_observations():
    # tell 后 dataset 的 fidelity 与建议一致, 且 Study.cost_spent 正确
    study = make_mf_study(budget=10)
    pending = study.ask(4)
    study.tell([{"id": p["id"], "y": 100.0, "fidelity": p["fidelity"]} for p in pending])
    obs = study.dataset.observations
    assert [o.fidelity for o in obs] == [p["fidelity"] for p in pending]
    assert study.cost_spent == pytest.approx(sum(p["fidelity"] for p in pending))
```

**补充门槛（可进 benchmark 报表，不必单测）**：

| 指标 | 期望 |
|---|---|
| `mf_bo` vs `bayes@full`（同成本预算 20） | `cost_aware_best` 提升 ≥10%，或达到同 best 成本 ≤80% |
| `mf_bo` vs `random@levels` | 同成本预算下提升 ≥25% |
| `successive_halving`（ρ=0.9） | 同成本下 best ≥ 随机 +30% |
| 低保真误导（ρ=0.3 变体） | 允许更差；报告里必须显示"低保真相关性"诊断（供 router 决定是否启用 mf） |

---

## 5. 相关性诊断（决定要不要用多保真）

在 benchmark 里加一个辅助函数（真实场景也可用）：

```python
def fidelity_correlation(problem, n=100, rng=None):
    """同一批 x 在 s=min 与 s=1 下的 Spearman 相关。"""
    rng = rng or np.random.default_rng(0)
    xs = problem.space.sample_sobol(n, rng=rng)
    lo = [problem.evaluate(x, min(problem.levels), rng)["y"] for x in xs]
    hi = [problem.evaluate(x, 1.0, rng)["y"] for x in xs]
    from scipy.stats import spearmanr
    return float(spearmanr(lo, hi).statistic)
```

- `ρ ≥ 0.7`：多保真大概率有收益；
- `ρ < 0.5`：很可能负迁移 → router 应优先单保真（`MULTIFIDELITY_SPEC.md` §5 已约定 warning）。

---

## 6. 参考

R41（BOCA：保真度作为连续输入/两阶段）、R39/R40（SH/Hyperband 的成本预算口径）、R18（Olympus 的 budget/cost 协议）、R32（benchmark 可复现）、R11/R42（单目标最优值与 HV 口径）。
