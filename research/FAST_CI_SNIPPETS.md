# FAST_CI_SNIPPETS — qNEHVI #2/#4/#5 与 mf_suzuki 的 pytest 片段（v0.9 实测耗时）

> 维护者：algo-research ｜ v1（2026-11-05，Round 10）
> 对应：`FAST_CI_ACCEPTANCE.md`（Round 9 规格）、`QNEHVI_MF_ACCEPTANCE.md`（阈值实测）。
> 实测环境：Windows/CPU、v0.9、`auto_route=False`；时间 `time.perf_counter()`。
> **目标**：PR 门禁 <15 s，覆盖"批量 / ref 口径 / 多保真链路"；#2（MC 收敛）需先抽 `_acquisition`（见 §4）。

---

## 0. 实测耗时（v0.9）

| 用例 | 实测 | 备注 |
|---|---|---|
| qNEHVI 批量 campaign（budget 12、batch 4、n_base=16/n_y=32） | **0.64 s** | 含 3 次 ask/tell |
| qNEHVI 单次 acquisition（n_base=8/n_y=16） | **0.26 s** | 用于 #2 |
| qNEHVI ref 读取 / 坏 ref 报错 | **<0.01 s / <0.3 s** | #5 |
| `mf_suzuki` optimum 校验 | **<0.01 s** | 110.0 / 104.6967 |
| mfbo 迷你 campaign（cost 4、6 evals） | **~0.5–1 s** | 初始设计全 0.25（v0.9 已修） |
| **快 CI 合计** | **≈1.5–3 s** | 远低于 15 s |

---

## 1. 文件与 marker

```python
# tests/test_fast_ci_acceptance.py
import numpy as np, pytest
from condopt import registry
from condopt.benchmarks import builtin_problems, mf_suzuki
from condopt.dataset import Observation
from condopt.study import Study

pytestmark = pytest.mark.fast_ci   # pyproject: markers = ["fast_ci", "slow"]
```

---

## 2. qNEHVI #4：批量语义（q=1/4/8）

```python
@pytest.mark.parametrize("batch", [1, 4, 8])
def test_qnehvi_batch_distinct_and_tellable(tmp_path, batch):
    prob = builtin_problems()["moo_yield_cost"]
    study = Study.create(tmp_path / f"b{batch}", prob.space, prob.objective,
                         algorithm="qnehvi", budget=max(8, 2 * batch),
                         batch_size=batch, seed=0, auto_route=False,
                         params={"n_base": 8, "n_y": 16})
    pending = study.ask()
    assert len(pending) == min(batch, study.remaining() + len(pending))
    units = [prob.space.encode(p["x"]) for p in pending]
    if len(units) > 1:
        d = min(np.linalg.norm(a - b) for i, a in enumerate(units) for b in units[i + 1:])
        assert d >= 0.02                 # q=8 时惩罚较弱，门槛 0.02
    study.tell([{"id": p["id"], "objectives": prob.objectives_fn(p["x"]),
                 "objective_std": 1.0} for p in pending])
    assert study.remaining() >= 0
```

## 3. qNEHVI #5：ref 必填/固定/失效报错

```python
def test_qnehvi_ref_must_exist_and_dominate(tmp_path):
    from condopt.base import Objective
    prob = builtin_problems()["moo_yield_cost"]
    assert tuple(prob.objective.ref_point) == (60.0, 30.0)     # 固定 ref（不变）
    # 坏 ref：高于所有观测 → qNEHVI 在首次 model ask 时显式 ValueError
    bad = Objective(name="yield", direction="max", objectives=("yield", "cost"),
                    directions=("max", "min"), ref_point=(1e6, 1e6))
    study = Study.create(tmp_path / "ok", prob.space, prob.objective,
                         algorithm="qnehvi", budget=8, batch_size=4, seed=0,
                         auto_route=False, params={"n_base": 4, "n_y": 8})
    obs = []
    for _ in range(2):
        pending = study.ask()
        study.tell([{"id": p["id"], "objectives": prob.objectives_fn(p["x"]),
                     "objective_std": 1.0} for p in pending])
        obs.extend(study.dataset.observations)
    opt = registry.build("qnehvi", prob.space, bad, seed=0, prior=obs,
                         params={"n_base": 4, "n_y": 8})
    with pytest.raises(ValueError, match="ref_point|dominates"):
        opt.ask(1)
```

## 4. qNEHVI #2：嵌套 MC 收敛（需要先抽 `_acquisition`）

**现状**：`QNEHVI._model_ask` 只返回配置，无法直接比较采集向量。先做 10 行重构：

```python
# qehvi.py / qnehvi.py
def _acquisition(self, candidates, features) -> np.ndarray:
    ...                      # 把 _model_ask 里"算 acquisition"的部分搬到这里
def _model_ask(self, n):
    candidates, features = self._candidate_pool()
    acq = self._acquisition(candidates, features)
    return self._greedy_batch(candidates, acq, n)
```

测试：

```python
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
    assert len(set(np.argsort(-acq8)[:8]) & set(np.argsort(-acq64)[:8])) >= 6
```
**实测**：单次 acquisition 0.26 s（n_base=8）；n_base=64 约 1.5–2 s → #2 总耗时 ~2 s。

## 5. mf_suzuki：optimum / 单保真默认 / 初始设计最低档

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
```

---

## 6. 推荐 CI 配置

```toml
# pyproject.toml
[tool.pytest.ini_options]
markers = ["fast_ci: runs in PR (<15 s)", "slow: nightly only"]
addopts = "-m 'not slow'"
```
- PR：`pytest -m fast_ci`（≈3 s）
- nightly：`pytest -m slow`（端到端 HV、多 seed）
- **注意**：`auto_route=False` 必须显式传（v0.9 已据 `manual_selection` 修显式算法钉住，但测试里仍建议显式）。

---

## 7. 参考

R12（qNEHVI）、R21（qEHVI）、R42（HV/固定 ref）、R41（BOCA/多保真）、R32（多 seed/统计）、`QNEHVI_MF_ACCEPTANCE.md`、`REVIEW_v09.md`。
