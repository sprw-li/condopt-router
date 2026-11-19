# QNEHVI_ACQUISITION_EXTRACT — `_model_ask` → `_acquisition()` 逐行替换清单

> 维护者：algo-research ｜ v1（2026-11-05，Round 16）
> 对应：`QNEHVI_FAST_CI_FINAL.md`（diff 终稿）、`QNEHVI_APPLY_PLAN.md`（流程/回滚）。
> 本文件按**行号**给出"删哪些行、插什么代码"，保证行为不变（RNG 顺序、GP 拟合顺序、采样顺序一致）。

---

## 0. 前置（顺序不能反）

1. `git rev-parse HEAD` → 记 R0；
2. 先跑快照脚本（§4），生成 `tests/fixtures/{qehvi,qnehvi}_ask_snapshot.json` 并**提交**；
3. 之后才动代码。

---

## 1. `src/condopt/algorithms/qehvi.py`

### 1.1 删除段（旧 `_model_ask` 主体）

| 旧行号（v0.10.5） | 内容 | 处置 |
|---|---|---|
| 119-121 | `def _model_ask` + 早退 Sobol | **保留**（新 `_model_ask` 仍需要） |
| 122-124 | `_candidate_pool` + 空池早退 | 保留 |
| 126-143 | `draws/ref/valid/signed/boxes` 校验与分区 | **移入** `_acquisition` |
| 145-155 | 双 GP 拟合 + ensemble + `dhv_2d_batch().mean()` | **移入** `_acquisition` |
| 157-159 | `scale = self._local_penalty_scale(models[0])` | **移入** `_greedy_batch`（用 `self._last_models[0]`） |
| 160-177 | 贪心循环 | **移入** `_greedy_batch` |
| 178-181 | decode + 补 Sobol + return | 保留（新 `_model_ask` 尾部） |

### 1.2 新插入代码（插在 `_candidate_pool` 之前/之后均可，推荐 `_model_ask` 之前）

```python
    # -- acquisition ------------------------------------------------------
    def _objective_models(self, X, Y) -> list:
        """Fit one GP per objective (index order matters for RNG parity)."""
        return [self._fit(X, Y[:, index]) for index in range(2)]

    def _acquisition(self, candidates, features) -> np.ndarray:
        """EHVI acquisition vector; same RNG call order as the old _model_ask."""
        draws = int(self.params["mc_draws"])
        ref = self._signed_references()
        valid = np.all(np.isfinite(self._Y), axis=1)
        if not np.any(valid):
            raise ValueError(
                "qEHVI: no observation has a complete 2-objective vector; "
                "check the results file / objective names")
        X = self._X[valid]
        Y = self._Y[valid]
        signed = signed_scores(Y, self.objective)
        boxes = partition_2d(signed[non_dominated_mask(signed)], ref)
        if not boxes:
            raise ValueError(
                f"qEHVI: no observation dominates objective.ref_point="
                f"{self.objective.ref_point}; lower the reference point")
        models = self._objective_models(X, Y)
        ensemble = [
            self._sample_posterior_multi(model, features, draws).T for model in models
        ]
        points = np.stack(ensemble, axis=-1)
        self._last_models = models
        return dhv_2d_batch(points, boxes, ref).mean(axis=0)

    def _greedy_batch(self, candidates, acquisition, n) -> list[int]:
        """Greedy local-penalized selection; no RNG, identical to the old loop."""
        scale = self._local_penalty_scale(self._last_models[0])
        chosen: list[int] = []
        chosen_units: list[np.ndarray] = []
        available = np.ones(len(candidates), dtype=bool)
        for _ in range(n):
            if not available.any():
                break
            scores = acquisition.copy()
            if chosen_units:
                distances = np.sum(
                    (candidates[:, None, :] - np.asarray(chosen_units)[None, :, :]) ** 2,
                    axis=-1,
                )
                penalty = np.prod(1.0 - np.exp(-distances / (2.0 * scale * scale)), axis=1)
                scores = scores * penalty
            scores = np.where(available, scores, -np.inf)
            pick = int(np.argmax(scores))
            chosen.append(pick)
            chosen_units.append(candidates[pick])
            available[pick] = False
        return chosen
```

### 1.3 新 `_model_ask`（替换 119-181 的旧实现整体）

```python
    def _model_ask(self, n: int) -> list[dict]:
        if self._X is None or len(self._y) < 3:
            return self.space.sample_sobol(n, rng=self.rng)
        candidates, features = self._candidate_pool()
        if len(candidates) == 0:
            return self.space.sample_sobol(n, rng=self.rng)
        acquisition = self._acquisition(candidates, features)
        chosen = self._greedy_batch(candidates, acquisition, n)
        configs = [self.space.decode(candidates[i]) for i in chosen]
        while len(configs) < n:
            configs.extend(self.space.sample_sobol(n - len(configs), rng=self.rng))
        return configs
```

**行为不变检查**：① `_objective_models` 的 index 顺序 0→1（同旧循环）；② `_sample_posterior_multi` 调用顺序 0→1；③ `_greedy_batch` 无 RNG；④ `_local_penalty_scale` 用 `_last_models[0]`（与旧 `models[0]` 同一对象）；⑤ 补 Sobol 的 `self.rng` 调用次数不变（仅当 `len(configs)<n`）。

---

## 2. `src/condopt/algorithms/qnehvi.py`

### 2.1 删除段

| 旧行号（v0.10.5） | 内容 | 处置 |
|---|---|---|
| 104-124 | `_model_ask` 头 + 校验/取 ref/valid/signed | 头/早退保留，校验移入 `_acquisition` |
| 126-157 | `n_base/n_y` 嵌套 MC + 第三 GP `_local_penalty_scale` | 嵌套 MC 移入 `_acquisition`；第三 GP 删除（用 `_last_models[0]`） |
| 159-181 | 贪心循环 | **删除**（继承 `QEHVI._greedy_batch`） |
| 181 | `return configs` | 保留在新 `_model_ask` 尾部 |

### 2.2 新插入代码

```python
    def _acquisition(self, candidates, features) -> np.ndarray:
        """Nested-MC qNEHVI; same RNG call order as the old _model_ask."""
        n_base = int(self.params["n_base"])
        n_y = int(self.params["n_y"])
        ref = self._signed_references()
        noise = self._noise_std
        valid = np.all(np.isfinite(self._Y), axis=1)
        X = self._X[valid]
        Y = self._Y[valid]
        signed = signed_scores(Y, self.objective)
        if self._noise_missing and not getattr(self, "_noise_warned", False):
            warnings.warn(
                "qNEHVI: no observation-noise information (objective_std / noise_std are 0); "
                "the nested MC reduces to the noiseless qEHVI - set noise_std or record "
                "objective_std to get the noisy behaviour",
                RuntimeWarning,
                stacklevel=2,
            )
            self._noise_warned = True
        models = self._objective_models(X, Y)
        points = np.stack(
            [self._sample_posterior_multi(model, features, n_y).T for model in models],
            axis=-1,
        )
        self._last_models = models
        acquisition = np.zeros(len(candidates))
        effective = 0
        for _ in range(n_base):
            drawn = signed + noise * self.rng.standard_normal(signed.shape)
            boxes = partition_2d(drawn[non_dominated_mask(drawn)], ref)
            if not boxes:
                continue
            acquisition += dhv_2d_batch(points, boxes, ref).mean(axis=0)
            effective += 1
        if effective == 0:
            raise ValueError(
                "qNEHVI: no baseline draw produced a valid front above ref_point; "
                "lower the reference point or provide better observations")
        return acquisition / effective
```

### 2.3 新 `_model_ask`（与 qehvi 同构）

```python
    def _model_ask(self, n: int) -> list[dict]:
        if self._X is None or len(self._y) < 3:
            return self.space.sample_sobol(n, rng=self.rng)
        candidates, features = self._candidate_pool()
        if len(candidates) == 0:
            return self.space.sample_sobol(n, rng=self.rng)
        acquisition = self._acquisition(candidates, features)
        chosen = self._greedy_batch(candidates, acquisition, n)
        configs = [self.space.decode(candidates[i]) for i in chosen]
        while len(configs) < n:
            configs.extend(self.space.sample_sobol(n - len(configs), rng=self.rng))
        return configs
```

**行为不变检查**：① 先 baseline 抽样循环（`self.rng.standard_normal`），后贪心（无 RNG）；② 第三 GP 删除后 `scale` 用 `models[0]`——同 seed/同 X/Y 的 `_fit` 结果相同，lengthscale 不变；③ `_noise_warned` 只影响 warning 次数，不影响数值。

---

## 3. 快照脚本（`scripts/snapshot_ask.py`，约 35 行）

```python
"""Record ask() sequences for behaviour-preserving refactors (run before refactor)."""
import json, sys
import numpy as np
from condopt import registry
from condopt.benchmarks import builtin_problems
from condopt.dataset import Observation

def snapshot(algorithm: str, seed: int = 0, rounds: int = 3, q: int = 2) -> list:
    prob = builtin_problems()["moo_yield_cost"]
    opt = registry.build(algorithm, prob.space, prob.objective, seed=seed,
                         params={"n_base": 8, "n_y": 16})
    seq = []
    for _ in range(rounds):
        cfgs = opt.ask(q)
        seq.append([prob.space.encode(c).tolist() for c in cfgs])
        opt.tell([Observation(x=c, objectives=prob.objectives_fn(c)) for c in cfgs])
    return seq

if __name__ == "__main__":
    out = sys.argv[sys.argv.index("--out") + 1]
    algorithm = sys.argv[sys.argv.index("--algorithm") + 1]
    json.dump(snapshot(algorithm), open(out, "w"), indent=2)
    print("wrote", out)
```

对比脚本（重构后）：

```python
import json, numpy as np
for algo in ("qehvi", "qnehvi"):
    before = json.load(open(f"tests/fixtures/{algo}_ask_snapshot.json"))
    after = json.load(open(f"/tmp/after_{algo}.json"))
    delta = float(np.max(np.abs(np.array(before) - np.array(after))))
    print(algo, "IDENTICAL" if np.allclose(before, after, atol=1e-12, rtol=0) else "DIFF",
          "max_delta=", delta)
```

**预期**：`qehvi IDENTICAL max_delta= 0.0`、`qnehvi IDENTICAL max_delta= 0.0`。

---

## 4. 测试（#2/#4/#5，写入 `tests/test_qnehvi_fast_ci.py`）

```python
pytestmark = pytest.mark.fast_ci

def test_qnehvi_nested_mc_convergence(tmp_path):
    study, prob = _seed_study(tmp_path)
    opt = registry.build("qnehvi", prob.space, prob.objective, seed=0,
                         prior=study.dataset.observations, params={"n_base": 8, "n_y": 16})
    cand, feat = opt._candidate_pool()
    acq8 = opt._acquisition(cand, feat)
    opt.params.update({"n_base": 64, "n_y": 32})
    acq64 = opt._acquisition(cand, feat)
    assert spearmanr(acq8, acq64).statistic >= 0.85
    assert len(set(np.argsort(-acq8)[:8]) & set(np.argsort(-acq64)[:8])) >= 6

@pytest.mark.parametrize("batch", [1, 4, 8])
def test_qnehvi_batch_distinct(tmp_path, batch):
    study, prob = _seed_study(tmp_path, budget=max(8, 2 * batch), batch=batch)
    pending = study.ask()
    units = [prob.space.encode(p["x"]) for p in pending]
    if len(units) > 1:
        d = min(np.linalg.norm(a - b) for i, a in enumerate(units) for b in units[i + 1:])
        assert d >= 0.02
    study.tell([{"id": p["id"], "objectives": prob.objectives_fn(p["x"]),
                 "objective_std": 1.0} for p in pending])

def test_qnehvi_ref_fixed_and_invalid_raises(tmp_path):
    study, prob = _seed_study(tmp_path)
    assert tuple(prob.objective.ref_point) == (60.0, 30.0)
    bad = Objective(..., ref_point=(1e6, 1e6))
    opt = registry.build("qnehvi", prob.space, bad, seed=0, prior=study.dataset.observations)
    with pytest.raises(ValueError, match="ref_point|dominates"):
        opt.ask(1)
```

**耗时**：`_acquisition` 0.26 s（n_base=8）、#2 ≈2 s、#4 ≈1–2 s、#5 <0.3 s → **<5 s**。

---

## 5. 参考

R12（qNEHVI）、R21（qEHVI）、R42（HV/固定 ref）、R32（统计）、`QNEHVI_FAST_CI_FINAL.md`、`QNEHVI_APPLY_PLAN.md`。
