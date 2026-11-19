# QNEHVI_FAST_CI_FINAL — `_acquisition` 抽取的最终 diff、行为不变验证与 #2 阈值

> 维护者：algo-research ｜ v1（2026-11-05，Round 14）
> 对应：`QNEHVI_FAST_CI_PATCH.md`（Round 13 draft）、`REVIEW_v12/v13`（P2-5 未落）、`FAST_CI_SNIPPETS_V2.md`。
> 目标：**最终版**可粘贴 diff（保持 ask 行为逐位不变）+ #2/#4/#5 pytest + #2 阈值建议。
> 实测基线（v0.10.3）：`_acquisition`（n_base=8/n_y=16）≈0.26 s；批量 campaign（budget 12、q=4）≈0.64 s。

---

## 1. 最终 diff（qehvi.py）

```diff
@@ QEHVI
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
-        if not np.any(valid):
-            raise ValueError(...)
-        X = self._X[valid]; Y = self._Y[valid]
-        signed = signed_scores(Y, self.objective)
-        boxes = partition_2d(signed[non_dominated_mask(signed)], ref)
-        if not boxes:
-            raise ValueError(...)
-        ensemble, models = [], []
-        for index in range(2):
-            model = self._fit(X, Y[:, index])
-            models.append(model)
-            ensemble.append(self._sample_posterior_multi(model, features, draws).T)
-        points = np.stack(ensemble, axis=-1)
-        acquisition = dhv_2d_batch(points, boxes, ref).mean(axis=0)
-        scale = self._local_penalty_scale(models[0])
-        ... greedy loop ...
-        return configs
+    def _objective_models(self, X, Y) -> list:
+        """Fit one GP per objective; shared by qEHVI and qNEHVI."""
+        return [self._fit(X, Y[:, index]) for index in range(2)]
+
+    def _acquisition(self, candidates, features) -> np.ndarray:
+        """EHVI acquisition vector; no RNG besides the posterior sampler."""
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
+            raise ValueError(
+                f"qEHVI: no observation dominates objective.ref_point="
+                f"{self.objective.ref_point}; lower the reference point")
+        models = self._objective_models(X, Y)
+        ensemble = [self._sample_posterior_multi(m, features, draws).T for m in models]
+        points = np.stack(ensemble, axis=-1)
+        self._last_models = models
+        return dhv_2d_batch(points, boxes, ref).mean(axis=0)
+
+    def _greedy_batch(self, candidates, acquisition, n) -> list[int]:
+        scale = self._local_penalty_scale(self._last_models[0])
+        chosen, chosen_units = [], []
+        available = np.ones(len(candidates), dtype=bool)
+        for _ in range(n):
+            if not available.any():
+                break
+            scores = acquisition.copy()
+            if chosen_units:
+                distances = np.sum((candidates[:, None, :] -
+                                    np.asarray(chosen_units)[None, :, :]) ** 2, axis=-1)
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

## 2. 最终 diff（qnehvi.py，只列变化段）

```diff
-        n_base = int(self.params["n_base"]); n_y = int(self.params["n_y"])
-        if self._noise_missing: warnings.warn(...)
-        ensemble = []
-        for index in range(2):
-            model = self._fit(X, Y[:, index])
-            ensemble.append(self._sample_posterior_multi(model, features, n_y).T)
-        points = np.stack(ensemble, axis=-1)
-        acquisition = np.zeros(len(candidates)); effective = 0
-        for _ in range(n_base):
-            drawn = signed + noise * self.rng.standard_normal(signed.shape)
-            boxes = partition_2d(drawn[non_dominated_mask(drawn)], ref)
-            if not boxes: continue
-            acquisition += dhv_2d_batch(points, boxes, ref).mean(axis=0)
-            effective += 1
-        if effective == 0: raise ValueError(...)
-        acquisition /= effective
-        scale = self._local_penalty_scale(self._fit(X, Y[:, 0]))
-        ... greedy loop ...
+    def _acquisition(self, candidates, features) -> np.ndarray:
+        n_base = int(self.params["n_base"]); n_y = int(self.params["n_y"])
+        if self._noise_missing and not getattr(self, "_noise_warned", False):
+            warnings.warn(...); self._noise_warned = True   # 只警告一次（P2 修复）
+        models = self._objective_models(X, Y)
+        points = np.stack([self._sample_posterior_multi(m, features, n_y).T
+                           for m in models], axis=-1)
+        self._last_models = models
+        acquisition = np.zeros(len(candidates)); effective = 0
+        for _ in range(n_base):
+            drawn = signed + noise * self.rng.standard_normal(signed.shape)
+            boxes = partition_2d(drawn[non_dominated_mask(drawn)], ref)
+            if not boxes: continue
+            acquisition += dhv_2d_batch(points, boxes, ref).mean(axis=0)
+            effective += 1
+        if effective == 0:
+            raise ValueError("qNEHVI: no baseline draw produced a valid front above ref_point")
+        return acquisition / effective
```
- `_greedy_batch` 从 `QEHVI` 继承，删除 `qnehvi.py:159-181` 的重复段。
- **行为不变**：baseline 抽样循环的 `self.rng` 调用顺序不变；少了第三个 `_fit`（原只为 lengthscale），但 `models[0]` 与其同 seed/同 X/Y → 同模型，`scale` 不变。

## 3. 行为不变验证（pytest 快照）

```python
def _run_ask_sequence(observe_seed=0):
    prob = builtin_problems()["moo_yield_cost"]
    opt = registry.build("qehvi", prob.space, prob.objective, seed=observe_seed)
    rng = np.random.default_rng(observe_seed)
    obs = []
    for _ in range(3):
        cfgs = opt.ask(2)
        new = [Observation(x=c, objectives=prob.objectives_fn(c)) for c in cfgs]
        opt.tell(new); obs += [c for c in cfgs]
    return obs

def test_qehvi_ask_sequence_unchanged():
    # 在重构前记录 SNAPSHOT（3 轮 × 2 点的配置 JSON），重构后逐位比较
    assert _run_ask_sequence() == pytest.approx(SNAPSHOT_QEHVI, rel=0, abs=1e-12)
```
（`SNAPSHOT_*` 由重构前代码生成、入库为常量；若环境浮点差异导致不逐位相等，用 `np.allclose(atol=1e-12)` 并只比较编码后的 unit 向量。）

## 4. #2 阈值建议

| 参数 | 建议 | 理由 |
|---|---|---|
| Spearman（n_base 8 vs 64） | **≥0.85**（保守），目标 0.90 | 候选池 1024、采集值有并列/近并列；0.90 在低 n_base 下可能 flaky |
| top-8 交集 | **≥6/8** | 与排序相关但更鲁棒 |
| n_base 对 | 8 vs 64（而非 4 vs 64） | 4 的方差过大，容易假阴性 |
| 若 CI 偶发失败 | 固定 `seed=0` + `n_y=16`（8 版本）与 `n_y=32`（64 版本） | 减少 MC 抖动；不要每次随机种子 |

## 5. #2/#4/#5 最终 pytest（可直接粘贴）

```python
pytestmark = pytest.mark.fast_ci

def test_qnehvi_nested_mc_convergence(tmp_path):
    study, prob = _seed_study(tmp_path)          # budget 12, batch 4, n_base 8/n_y 16
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

**耗时**：`_acquisition` 0.26 s；#2 ≈2 s；#4 三参数 ≈1–2 s；#5 <0.3 s；合计 <5 s（PR 可跑）。

## 6. 参考

R12（qNEHVI）、R21（qEHVI）、R42（HV/固定 ref）、R32（统计）、`QNEHVI_FAST_CI_PATCH.md`、`REVIEW_v13.md`。
