# FAST_CI_SNIPPETS_V2 — qNEHVI #4/#5 + mf_suzuki 的可粘贴 pytest（v0.10 实测）

> 维护者：algo-research ｜ v1（2026-11-05，Round 11）
> 对应：`FAST_CI_SNIPPETS.md`（Round 10）、`QNEHVI_MF_ACCEPTANCE.md`（阈值）。
> 环境：v0.10、Windows/CPU、`auto_route=False`；时间用 `time.perf_counter()` 实测。

---

## 0. 实测耗时

| 用例 | 实测 |
|---|---|
| qNEHVI 批量 campaign（budget 12、batch 4、n_base=16/n_y=32） | **0.64 s** |
| qNEHVI 单次 acquisition（n_base=8/n_y=16） | **0.26 s** |
| qNEHVI ref 读取 / 坏 ref 报错 | <0.01 s / <0.3 s |
| `mf_suzuki` optimum 校验 | <0.01 s |
| `mfbo._best_scalar()` 全保真 incumbent | <0.01 s（v0.10 修复，实测 100 vs 低保证 200） |
| 快 CI 合计 | **≈1.5–3 s** |

---

## 1. 文件与 marker

```python
# tests/test_fast_ci_acceptance.py
import numpy as np, pytest
from condopt import registry
from condopt.benchmarks import builtin_problems, mf_suzuki
from condopt.base import Objective
from condopt.dataset import Observation
from condopt.study import Study

pytestmark = pytest.mark.fast_ci
```

---

## 2. qNEHVI #4：批量（q=1/4/8）

```python
@pytest.mark.parametrize("batch", [1, 4, 8])
def test_qnehvi_batch_distinct_and_tellable(tmp_path, batch):
    prob = builtin_problems()["moo_yield_cost"]
    study = Study.create(tmp_path / f"b{batch}", prob.space, prob.objective,
                         algorithm="qnehvi", budget=max(8, 2 * batch),
                         batch_size=batch, seed=0, auto_route=False,
                         params={"n_base": 8, "n_y": 16})
    pending = study.ask()
    units = [prob.space.encode(p["x"]) for p in pending]
    if len(units) > 1:
        d = min(np.linalg.norm(a - b) for i, a in enumerate(units) for b in units[i + 1:])
        assert d >= 0.02
    study.tell([{"id": p["id"], "objectives": prob.objectives_fn(p["x"]),
                 "objective_std": 1.0} for p in pending])
    assert study.remaining() >= 0
```

## 3. qNEHVI #5：ref 固定/必填、坏 ref 显式报错

```python
def test_qnehvi_ref_fixed_and_invalid_ref_raises(tmp_path):
    prob = builtin_problems()["moo_yield_cost"]
    assert tuple(prob.objective.ref_point) == (60.0, 30.0)

    study = Study.create(tmp_path / "ok", prob.space, prob.objective,
                         algorithm="qnehvi", budget=8, batch_size=4, seed=0,
                         auto_route=False, params={"n_base": 4, "n_y": 8})
    obs = []
    for _ in range(2):
        pending = study.ask()
        study.tell([{"id": p["id"], "objectives": prob.objectives_fn(p["x"]),
                     "objective_std": 1.0} for p in pending])
        obs.extend(study.dataset.observations)

    bad = Objective(name="yield", direction="max", objectives=("yield", "cost"),
                    directions=("max", "min"), ref_point=(1e6, 1e6))
    opt = registry.build("qnehvi", prob.space, bad, seed=0, prior=obs,
                         params={"n_base": 4, "n_y": 8})
    with pytest.raises(ValueError, match="ref_point|dominates"):
        opt.ask(1)
```

## 4. mf_suzuki：optimum / 单保真默认 / 初始设计最低档 / 全保真 incumbent

```python
OPT = {"T": 85.0, "time": 60.0, "cat_loading": 1.5,
       "solvent": "MeCN", "ligand": "L1", "base_eq": 2}

def test_mf_suzuki_optimum():
    assert mf_suzuki(OPT, 1.0) == pytest.approx(110.0, abs=1e-9)
    assert mf_suzuki(OPT, 0.5) == pytest.approx(104.6967, abs=1e-3)

def test_single_fidelity_defaults_to_full():
    prob = builtin_problems()["suzuki_like"]
    opt = registry.build("bayes", prob.space, prob.objective, seed=0)
    assert all(f == 1.0 for _, f in opt.ask_with_fidelity(3))

def test_mfbo_initial_design_lowest_fidelity(tmp_path):
    prob = builtin_problems()["suzuki_like"]
    study = Study.create(tmp_path / "mf", prob.space, prob.objective,
                         algorithm="mfbo", budget=6, batch_size=2, seed=0,
                         auto_route=False, params={"fidelity_levels": "0.25,0.5,1.0"})
    pending = study.ask(4)
    assert all(p["fidelity"] == 0.25 for p in pending)

def test_mfbo_best_scalar_uses_full_fidelity_only():
    """v0.10 修复：低保真虚高值不得污染 EI incumbent。"""
    prob = builtin_problems()["suzuki_like"]
    prior = [
        Observation(x={"T": 85.0, "time": 60.0, "cat_loading": 1.5,
                       "solvent": "MeCN", "ligand": "L1", "base_eq": 2},
                    y=200.0, fidelity=0.5),                # 低保证"虚高"
        Observation(x={"T": 40.0, "time": 5.0, "cat_loading": 0.1,
                       "solvent": "water", "ligand": "L5", "base_eq": 1},
                    y=100.0, fidelity=1.0),
    ]
    opt = registry.build("mfbo", prob.space, prob.objective, seed=0, prior=prior,
                         params={"fidelity_levels": "0.25,0.5,1.0"})
    assert opt._best_scalar() == pytest.approx(100.0)   # 实测 100.0
```

## 5. qNEHVI #2（MC 收敛）— 仍需先抽 `_acquisition`

```python
# 仅当 qehvi/qnehvi 抽出 `_acquisition(candidates, features)` 后启用
def test_qnehvi_nested_mc_convergence(tmp_path):
    from scipy.stats import spearmanr
    prob = builtin_problems()["moo_yield_cost"]
    study = Study.create(tmp_path / "run", prob.space, prob.objective,
                         algorithm="qnehvi", budget=12, batch_size=4, seed=0,
                         auto_route=False, params={"n_base": 8, "n_y": 16})
    while study.remaining() > 0:
        pending = study.ask()
        study.tell([{"id": p["id"], "objectives": prob.objectives_fn(p["x"]),
                     "objective_std": 1.0} for p in pending])
    opt = registry.build("qnehvi", prob.space, prob.objective, seed=0,
                         prior=study.dataset.observations,
                         params={"n_base": 8, "n_y": 16})
    cand, feat = opt._candidate_pool()
    acq8 = opt._acquisition(cand, feat)
    opt.params.update({"n_base": 64, "n_y": 32})
    acq64 = opt._acquisition(cand, feat)
    assert spearmanr(acq8, acq64).statistic >= 0.85
```
**抽取方案**（10 行）：`_model_ask` = `_candidate_pool` + `_acquisition` + `_greedy_batch`；`_acquisition` 里保留 n_base/n_y 的 MC。

---

## 6. 建议 CI 配置

```toml
[tool.pytest.ini_options]
markers = ["fast_ci: <15 s in PR", "slow: nightly"]
addopts = "-m 'not slow'"
```
- PR：`pytest -m fast_ci`（≈3 s）
- nightly：`pytest -m slow`（端到端 HV、多 seed）
- 注意：所有 Study 测试显式 `auto_route=False`（v0.10 已钉住显式算法，但测试仍建议显式）。

---

## 7. 参考

R12（qNEHVI）、R21（qEHVI）、R42（HV/固定 ref）、R41（BOCA/多保真）、R32（统计）、`QNEHVI_MF_ACCEPTANCE.md`、`REVIEW_v10.md`。
