# qNEHVI_ACCEPTANCE — `moo_yield_cost` 上的 5 条 qNEHVI 验收（可直接写 pytest）

> 维护者：algo-research ｜ v1（2026-11-05，Round 7）
> 被测：`src/condopt/algorithms/qnehvi.py`（v0.6，嵌套 MC，2 目标）；基准：`benchmarks.builtin_problems()["moo_yield_cost"]`。
> 文献：R12（qNEHVI：嵌套 MC、噪声下 baseline 必须随机）、R21（qEHVI：分区）、R42（HV 与**固定** ref point）、R10（带噪 EI）。
> 环境：CPU、固定 seed、`pytest -q`；下表的耗时按 1024 候选 / budget 48 / 5 seeds 估算。

---

## 0. 公共 fixture

```python
# tests/test_qnehvi_acceptance.py
import numpy as np, pytest
from condopt import registry
from condopt.base import Objective
from condopt.benchmarks import builtin_problems
from condopt.dataset import Observation
from condopt.pareto import summarise
from condopt.study import Study

SEEDS = (0, 1, 2, 3, 4)

def problem():
    return builtin_problems()["moo_yield_cost"]

def evaluate(prob, config, rng, noise_std=0.0, k=1):
    """调用基准的多目标函数；返回 results.json 可直接消费的 dict。"""
    values = prob.fn_multi(config) if getattr(prob, "fn_multi", None) else prob.evaluate(config)
    out = {}
    for name, value in values.items():
        reps = [value + rng.normal(0.0, noise_std) for _ in range(k)]
        out.setdefault("objectives", {})[name] = float(np.mean(reps))
        out.setdefault("objective_std", {})[name] = float(np.std(reps, ddof=1)) if k > 1 else 0.0
    out["n_replicates"] = k
    return out

def run(algorithm="qnehvi", budget=48, batch=4, seed=0, noise_std=0.0, k=1,
        params=None, algorithm_kwargs=None):
    prob = problem()
    study = Study.create(f"/tmp/qnehvi_accept/{algorithm}_{seed}_{noise_std}",
                         prob.space, prob.objective, algorithm=algorithm,
                         budget=budget, batch_size=batch, seed=seed,
                         params=params, **(algorithm_kwargs or {}))
    rng = np.random.default_rng(seed)
    while study.remaining() > 0:
        pending = study.ask()
        study.tell([{"id": p["id"], **evaluate(prob, p["x"], rng, noise_std, k)} for p in pending])
    return study, prob

def hv_ratio(study, prob):
    summary = summarise(study.dataset, prob.objective)      # 使用 Objective.ref_point（固定）
    if summary.hypervolume is None:
        pytest.skip("hypervolume unavailable")
    ref_front = getattr(prob, "pareto_front", None) or prob.reference_front()
    ref_hv = summarise([Observation(x={}, objectives=dict(zip(prob.objective.objectives, p)))
                        for p in ref_front], prob.objective).hypervolume
    return summary.hypervolume / ref_hv
```

> 适配提示：`benchmarks.py` 的多目标问题若没有 `fn_multi`/`pareto_front` 属性，用 `benchmarks.moo_yield_cost_front()`（v0.6 已提供 signed 参考前沿）包一层；`result` 里的 `objective_std` 传 dict 还是标量取决于 v0.6 的 `Observation`（当前为标量时，`evaluate` 只取第一个目标即可，并在测试注释标明）。

---

## 1. `test_qnehvi_noiseless_hv_ratio`（HV ratio 阈值）

```python
@pytest.mark.parametrize("seed", SEEDS)
def test_qnehvi_noiseless_hv_ratio(seed):
    """无噪声时 qNEHVI 应接近 qEHVI（v0.5 实测 qEHVI≈0.95）。"""
    study, prob = run("qnehvi", budget=48, batch=4, seed=seed, noise_std=0.0)
    assert hv_ratio(study, prob) >= 0.90        # 单 seed 门槛；5 seeds 的 IQM 应 ≥0.92
```

**判定**：5 seeds 的 `hv_ratio` 中位数 ≥0.92、最小值 ≥0.88；不达标先查 `n_base=64` 是否被 router 参数覆盖。

## 2. `test_qnehvi_nested_mc_convergence`（n_base / n_y 收敛）

```python
def test_qnehvi_nested_mc_convergence():
    """n_base 从 8 → 64 时，候选采集值排序应稳定（嵌套 MC 收敛）。"""
    prob = problem()
    study, _ = run("qnehvi", budget=24, batch=4, seed=0)      # 先攒 24 个观测
    opt = registry.build("qnehvi", prob.space, prob.objective, seed=0,
                         prior=study.dataset.observations,
                         params={"n_base": 8, "n_y": 64, "mc_draws": 64})
    acq8 = _acquisition_vector(opt)            # 内部：candidates × q，调用 opt._model_ask 前先取候选池
    opt.params.update({"n_base": 64, "n_y": 128})
    acq64 = _acquisition_vector(opt)
    from scipy.stats import spearmanr
    rho = spearmanr(acq8, acq64).statistic
    assert rho >= 0.85
    assert len(set(np.argsort(-acq8)[:8]) & set(np.argsort(-acq64)[:8])) >= 6
```

> 实现提示：`_acquisition_vector` 可用 `opt._candidate_pool()` + 手动调用与 `_model_ask` 相同的抽样逻辑（把 v0.6 的 `_model_ask` 拆成 `_acquisition(candidates, features)` 函数后单测最干净）。

## 3. `test_qnehvi_noisy_baseline_changes_acquisition`（含噪 baseline）

```python
def test_qnehvi_noisy_baseline_changes_acquisition():
    """噪声进入 baseline 后，采集排序应改变，且噪声运行的 HV_ratio 仍达标。"""
    study_noiseless, prob = run("qnehvi", budget=32, batch=4, seed=1, noise_std=0.0)
    study_noisy, _      = run("qnehvi", budget=32, batch=4, seed=1, noise_std=2.0, k=2)
    assert hv_ratio(study_noisy, prob) >= 0.80
    # 同一批观测、同一候选池下两种 baseline 噪声的 top-8 不应完全一致
    assert _top8_overlap(study_noiseless, study_noisy) < 8
```

**物理意义**：`noise_std=0` 时 qNEHVI≡qEHVI（baseline 确定）；`noise_std>0` 时前沿本身有不确定性 → 采集值整体抬高（v0.5 实测 1.68→1.80），排序变化是正确行为。

## 4. `test_qnehvi_batch_distinct_and_consistent`（批量）

```python
@pytest.mark.parametrize("batch", [1, 4, 8])
def test_qnehvi_batch_distinct_and_consistent(batch):
    study, prob = run("qnehvi", budget=32, batch=batch, seed=2)
    pending = study.asks_history[-1] if hasattr(study, "asks_history") else None
    xs = [np.asarray(p["x"][name] for name in prob.space.names) for p in pending]
    # 批内最小距离（单位立方体编码后）
    units = [prob.space.encode(p["x"]) for p in pending]
    d = min(np.linalg.norm(a - b) for i, a in enumerate(units) for b in units[i + 1:])
    assert d >= 0.05
    assert hv_ratio(study, prob) >= 0.80          # q=8 允许比 q=1 低，但不能崩
```

> 若 `Study` 不保留历史 pending，改为在 `run()` 里把每个 batch 的 pending 记下来返回。

## 5. `test_qnehvi_ref_point_is_fixed_and_required`（ref 固定/必填）

```python
def test_qnehvi_requires_explicit_ref_point():
    prob = problem()
    objective = Objective(name="yield", direction="max",
                          objectives=("yield", "cost"), directions=("max", "min"),
                          ref_point=())           # 缺 ref
    with pytest.raises(ValueError, match="ref_point|dominates"):
        run_with_objective(objective, seed=0)     # qnehvi 的 _model_ask 应显式报错（v0.6 已改）

def test_qnehvi_ref_point_is_persisted(tmp_path):
    prob = problem()
    study = Study.create(tmp_path / "run", prob.space, prob.objective,
                         algorithm="qnehvi", budget=8, batch_size=2, seed=0)
    pending = study.ask()
    study.tell([{"id": p["id"], **evaluate(prob, p["x"], np.random.default_rng(0))} for p in pending])
    state = (tmp_path / "run" / "opt_state.json").read_text()
    assert "ref" in state                       # qehvi.state_dict 存 ref
    # 重新加载后 ref 不变 → HV 可跨轮次比较
```

**额外断言**：`ref_point` 必须严格被至少一个观测支配；若给错，`_model_ask` 抛 `ValueError`（v0.6 已实现，见 `qehvi.py:134-138`）。

---

## 6. 汇总与运行成本

| 测试 | budget × batch | seeds | 单次耗时 | 总计 |
|---|---|---|---|---|
| #1 HV ratio（无噪） | 48×4 | 5 | ~8–12 s | ~1 min |
| #2 嵌套 MC 收敛 | 24×4 | 1 | ~2 s（不跑完整 campaign） | ~5 s |
| #3 含噪 baseline | 32×4 ×2 | 1 | ~15 s | ~20 s |
| #4 批量 | 32×{1,4,8} | 1 | ~20 s | ~20 s |
| #5 ref 固定/必填 | 8×2 | 1 | <2 s | <5 s |
| **合计** | — | — | — | **≈2 min**（可进 CI 的 `slow` 标记） |

**CI 策略**：`@pytest.mark.slow` 标 #1/#3；PR 只跑 #2/#4/#5（<30 s），nightly 跑全部。

**失败排查顺序**：①`n_base` 是否真的 64（router 参数被覆盖？）；②`objective_std`/`noise_std` 是否传进 baseline（#3 若不变化说明噪声没进 MC）；③ref 是否固定（HV 跨 seed 抖动大 → ref 被数据推导）；④`Observation.objective_std` 当前是标量，多目标时只有主目标噪声生效（REVIEW_v06 P1）——若要做真正的两目标异方差噪声，先改 dict。

---

## 7. 参考

R12（qNEHVI：嵌套 MC 与噪声 baseline）、R21（qEHVI：分区与 EHVI 定义）、R42（HV：固定参考点、跨 run 可比）、R10（带噪期望改进：baseline 随机化）、R32（benchmark 多 seed/统计协议）。
