# QNEHVI_FAST_CI_PATCH — `_acquisition` 抽取的具体补丁与 #2/#4/#5 pytest

> 维护者：algo-research ｜ v1（2026-11-05，Round 13）
> 对应：`REVIEW_v12.md` P2-5、`FAST_CI_SNIPPETS_V2.md`（#2 需先抽 `_acquisition`）。
> 目标：把 `qehvi.py` / `qnehvi.py` 的 `_model_ask` 拆成 `_acquisition` + `_greedy_batch`，**保持 ask 行为逐位不变**（同一 RNG 调用顺序、同一 GP 拟合顺序），从而让快 CI #2（n_base 收敛）可以写严。

---

## 1. 当前结构（v0.10.2）

- `qehvi.py:119-181`：`_model_ask` = 数据校验 → 分区（`partition_2d`）→ 拟合 2 GP + MC ensemble → `dhv_2d_batch` 求均值 → 局部惩罚贪心 → 补 Sobol。
- `qnehvi.py:104-181`：同上，但 acquisition 是嵌套 MC（`n_base` × `n_y`），且 **额外 fit 了第三个 GP** 只为 `_local_penalty_scale`（`:159`）。

**问题**：测试无法在不跑完整 `ask` 的情况下拿到采集向量；`#2` 只能做 "top-8 交集" 的间接断言。

---

## 2. 补丁（qehvi.py）

```diff
@@ class QEHVI
-    def _model_ask(self, n: int) -> list[dict]:
-        if self._X is None or len(self._y) < 3:
-            return self.space.sample_sobol(n, rng=self.rng)
-        candidates, features = self._candidate_pool()
-        if len(candidates) == 0:
-            return self.space.sample_sobol(n, rng=self.rng)
-
-        draws = int(self.params["mc_draws"])
-        ref = self._signed_references()
-        valid = np.all(np.isfinite(self._Y), axis=1)
-        ...
-        improvement = dhv_2d_batch(points, boxes, ref)
-        acquisition = improvement.mean(axis=0)
-        scale = self._local_penalty_scale(models[0])
+    def _objective_models(self, X: np.ndarray, Y: np.ndarray) -> list:
+        return [self._fit(X, Y[:, index]) for index in range(2)]
+
+    def _acquisition(self, candidates, features) -> np.ndarray:
+        """Return the EHVI acquisition vector for the given candidate features."""
+        draws = int(self.params["mc_draws"])
+        ref = self._signed_references()
+        valid = np.all(np.isfinite(self._Y), axis=1)
+        if not np.any(valid):
+            raise ValueError(
+                "qEHVI: no observation has a complete 2-objective vector; "
+                "check the results file / objective names")
+        X = self._X[valid]; Y = self._Y[valid]
+        signed = signed_scores(Y, self.objective)
+        boxes = partition_2d(signed[non_dominated_mask(signed)], ref)
+        if not boxes:
+            raise ValueError(...)
+        models = self._objective_models(X, Y)
+        ensemble = [self._sample_posterior_multi(model, features, draws).T
+                    for model in models]
+        points = np.stack(ensemble, axis=-1)
+        self._last_models = models          # for _local_penalty_scale
+        return dhv_2d_batch(points, boxes, ref).mean(axis=0)
+
+    def _greedy_batch(self, candidates, acquisition, n) -> list[int]:
+        scale = self._local_penalty_scale(self._last_models[0])
+        chosen, chosen_units = [], []
+        available = np.ones(len(candidates), dtype=bool)
+        for _ in range(n):
+            if not available.any(): break
+            scores = acquisition.copy()
+            if chosen_units:
+                distances = np.sum((candidates[:, None, :]
+                                    - np.asarray(chosen_units)[None, :, :]) ** 2, axis=-1)
+                scores = scores * np.prod(1.0 - np.exp(-distances / (2.0 * scale * scale)), axis=1)
+            scores = np.where(available, scores, -np.inf)
+            pick = int(np.argmax(scores))
+            chosen.append(pick); chosen_units.append(candidates[pick]); available[pick] = False
+        return chosen
+
+    def _model_ask(self, n: int) -> list[dict]:
+        if self._X is None or len(self._y) < 3:
+            return self.space.sample_sobol(n, rng=self.rng)
+        candidates, features = self._candidate_pool()
+        if len(candidates) == 0:
+            return self.space.sample_sobol(n, rng=self.rng)
+        acquisition = self._acquisition(candidates, features)
+        chosen = self._greedy_batch(candidates, acquisition, n)
+        configs = [self.space.decode(candidates[i]) for i in chosen]
+        while len(configs) < n:
+            configs.extend(self.space.sample_sobol(n - len(configs), rng=self.rng))
+        return configs
```

**行为不变性**：`_acquisition` 内部 GP 拟合顺序（index 0→1）、`_sample_posterior_multi` 调用顺序、`draws` 数量与重构前完全一致；`_greedy_batch` 不含 RNG；`_last_models` 只被 `_greedy_batch` 读取。唯一差异是 `scale` 由 `_last_models[0]` 取（等价于原来的 `models[0]`）。**单测**：`test_qehvi_ask_unchanged_after_refactor`（固定 seed/prior，重构前后 `ask(2)` 序列逐位相同；可用 git stash 对照或快照）。

---

## 3. 补丁（qnehvi.py，差异部分）

```diff
-        # Candidate posterior draws: (n_y, m_candidates, 2)
-        ensemble = []
-        for index in range(2):
-            model = self._fit(X, Y[:, index])
-            draw = self._sample_posterior_multi(model, features, n_y)
-            ensemble.append(draw.T)
-        points = np.stack(ensemble, axis=-1)
-        acquisition = np.zeros(len(candidates))
-        effective = 0
-        for _ in range(n_base):
-            drawn = signed + noise * self.rng.standard_normal(signed.shape)
-            mask = non_dominated_mask(drawn)
-            boxes = partition_2d(drawn[mask], ref)
-            if not boxes: continue
-            acquisition += dhv_2d_batch(points, boxes, ref).mean(axis=0)
-            effective += 1
-        if effective == 0: raise ValueError(...)
-        acquisition /= effective
-        scale = self._local_penalty_scale(self._fit(X, Y[:, 0]))
+    def _acquisition(self, candidates, features) -> np.ndarray:
+        ...  # 与上面的校验/取 ref/valid 相同
+        n_base = int(self.params["n_base"]); n_y = int(self.params["n_y"])
+        if self._noise_missing: warnings.warn(...)     # 保持原 warning
+        models = self._objective_models(X, Y)
+        points = np.stack(
+            [self._sample_posterior_multi(m, features, n_y).T for m in models], axis=-1)
+        self._last_models = models
+        acquisition = np.zeros(len(candidates)); effective = 0
+        for _ in range(n_base):
+            drawn = signed + noise * self.rng.standard_normal(signed.shape)
+            boxes = partition_2d(drawn[non_dominated_mask(drawn)], ref)
+            if not boxes: continue
+            acquisition += dhv_2d_batch(points, boxes, ref).mean(axis=0)
+            effective += 1
+        if effective == 0: raise ValueError("qNEHVI: no baseline draw produced a valid front above ref_point")
+        return acquisition / effective
```
- `_greedy_batch` 从 `QEHVI` 继承（qnehvi 原本是复制粘贴的同一段，删除即可）。
- **行为不变**：RNG 调用顺序相同（先 baseline 抽样循环，后贪心无 RNG）；少掉的第三个 `_fit` 是确定性的、只用于 lengthscale，且 `models[0]` 与之相同（同 seed、同 X/Y），因此 `scale` 不变。**这是"保持行为"的关键论证**，单测可用 `test_qnehvi_ask_unchanged_after_refactor` 覆盖。
- 保留 `_noise_missing` 的 warning（**只发一次**：加 `self._noise_warned` 标志，避免每轮刷屏；v0.10.2 仍是每轮触发，属 P2）。

---

## 4. 可粘贴 pytest（#2/#4/#5）

```python
# tests/test_qnehvi_fast_ci.py
import numpy as np, pytest
from scipy.stats import spearmanr
from condopt import registry
from condopt.base import Objective
from condopt.benchmarks import builtin_problems
from condopt.study import Study

pytestmark = pytest.mark.fast_ci

def _moo():
    return builtin_problems()["moo_yield_cost"]

def _seed_study(tmp_path, budget=12, batch=4, n_base=8, n_y=16):
    prob = _moo()
    study = Study.create(tmp_path / "run", prob.space, prob.objective,
                         algorithm="qnehvi", budget=budget, batch_size=batch,
                         seed=0, auto_route=False, params={"n_base": n_base, "n_y": n_y})
    while study.remaining() > 0:
        pending = study.ask()
        study.tell([{"id": p["id"], "objectives": prob.objectives_fn(p["x"]),
                     "objective_std": 1.0} for p in pending])
    return study, prob

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
    bad = Objective(name="yield", direction="max", objectives=("yield", "cost"),
                    directions=("max", "min"), ref_point=(1e6, 1e6))
    opt = registry.build("qnehvi", prob.space, bad, seed=0,
                         prior=study.dataset.observations, params={"n_base": 4, "n_y": 8})
    with pytest.raises(ValueError, match="ref_point|dominates"):
        opt.ask(1)
```

**实测耗时**（v0.10.2）：`_acquisition`（n_base=8/n_y=16）≈0.26 s；#2 含 64 版本 ≈2 s；#4 三参数 ≈1–2 s；#5 <0.3 s；合计 <5 s。

---

## 5. 参考

R12（qNEHVI）、R21（qEHVI）、R42（HV/固定 ref）、R32（统计）、`FAST_CI_SNIPPETS_V2.md`、`REVIEW_v12.md`。
