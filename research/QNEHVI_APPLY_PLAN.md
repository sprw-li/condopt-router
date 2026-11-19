# QNEHVI_APPLY_PLAN — 把 `QNEHVI_FAST_CI_FINAL` 的 diff 落地（逐文件顺序 + 回滚 + 验证）

> 维护者：algo-research ｜ v1（2026-11-05，Round 15）
> 目标：可直接照做的应用步骤。diff 正文见 `QNEHVI_FAST_CI_FINAL.md`；本文件只讲**顺序、检查点、回滚点、验证命令、预期输出**。
> 前置：v0.10.4、148 tests 全绿、工作树干净。

---

## 0. 准备（必须先做，否则无法证明"行为不变"）

```bash
# 0.1 确认基线
py -3 -m pytest -q                      # 期望：148 passed
git rev-parse HEAD                      # 记下 R0（回滚点）
git status --porcelain                  # 期望：空

# 0.2 生成"重构前"快照（行为不变验证的对照物）
py -3 scripts/snapshot_qehvi_ask.py --out tests/fixtures/qehvi_ask_snapshot.json
py -3 scripts/snapshot_qnehvi_ask.py --out tests/fixtures/qnehvi_ask_snapshot.json
```

`scripts/snapshot_*.py`（新文件，约 30 行，**在改动算法前提交**）：

```python
import json, numpy as np
from condopt import registry
from condopt.benchmarks import builtin_problems
from condopt.dataset import Observation

def snapshot(algorithm, seed=0, rounds=3, q=2):
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
    import sys
    out = sys.argv[sys.argv.index("--out") + 1]
    algo = "qehvi" if "qehvi" in out else "qnehvi"
    json.dump(snapshot(algo), open(out, "w"), indent=2)
```

**为什么先做**：refactor 后跑同一脚本，逐位/`allclose` 对比即可证明"ask 行为不变"；若先改代码，快照就失去对照意义。

---

## 1. qehvi.py（第一处改动）

**顺序**：
1. 加 `_objective_models(X, Y)`；
2. 加 `_acquisition(candidates, features)`（把 `_model_ask` 的校验/分区/MC 段搬入）；
3. 加 `_greedy_batch(candidates, acquisition, n)`（把贪心段搬入，`scale` 用 `self._last_models[0]`）；
4. 把 `_model_ask` 缩成 8 行（早退 → `_candidate_pool` → `_acquisition` → `_greedy_batch` → 补 Sobol）。

**检查点 C1**（此时不动 qnehvi）：
```bash
py -3 -m pytest -q tests/test_qehvi.py tests/test_pareto.py   # 期望 passed
py -3 -m pytest -q                                           # 期望 148 passed（未加新测试前）
```
**回滚点 R1**：`git checkout -- src/condopt/algorithms/qehvi.py`（或 `git stash`）。

## 2. qnehvi.py（第二处改动）

**顺序**：
1. 删除 `qnehvi.py` 内重复的贪心段（改为继承 `_greedy_batch`）；
2. 加 `_acquisition(candidates, features)`：保留 `n_base` 嵌套 MC；`_noise_missing` 的 warning 用 `self._noise_warned` 只发一次；模型用 `_objective_models`；
3. 把 `_model_ask` 缩到与 qehvi 同构。

**检查点 C2**：
```bash
py -3 -m pytest -q tests/test_qnehvi.py tests/test_qehvi.py   # 期望 passed
py -3 -m pytest -q                                           # 期望 148 passed
```

## 3. 行为不变验证（关键）

```bash
py -3 scripts/snapshot_qehvi_ask.py  --out /tmp/after_qehvi.json
py -3 scripts/snapshot_qnehvi_ask.py --out /tmp/after_qnehvi.json
py -3 - <<'PY'
import json, numpy as np
for algo in ("qehvi", "qnehvi"):
    before = json.load(open(f"tests/fixtures/{algo}_ask_snapshot.json"))
    after  = json.load(open(f"/tmp/after_{algo}.json"))
    ok = np.allclose(np.array(before), np.array(after), atol=1e-12, rtol=0)
    print(algo, "IDENTICAL" if ok else "DIFF", "max_delta=",
          float(np.max(np.abs(np.array(before) - np.array(after)))))
PY
```
**预期输出**：`qehvi IDENTICAL`、`qnehvi IDENTICAL`（max_delta=0.0）。若 qnehvi 有极小差异（浮点求和顺序），`atol=1e-12` 应仍通过；若 `DIFF` → 回滚 R2（`git checkout -- src/condopt/algorithms/qnehvi.py`）并检查 RNG 调用顺序。

## 4. 新增快 CI 测试

```bash
# 把 QNEHVI_FAST_CI_FINAL.md §5 的测试写入 tests/test_qnehvi_fast_ci.py
py -3 -m pytest -q tests/test_qnehvi_fast_ci.py -m fast_ci --durations=10
```
**预期**：3 个测试全过，总耗时 **<5 s**（`_acquisition` ≈0.26 s、#2 ≈2 s、#4 ≈1–2 s、#5 <0.3 s）。

**阈值（#2）**：`spearman >= 0.85`（保守）/ `top8 交集 >= 6`；若 CI 偶发失败，固定 `n_y`（8→16、64→32）并只比较 `argsort` 排名。

## 5. 全量回归与收尾

```bash
py -3 -m pytest -q                                  # 期望 151 passed（148 + 3）
py -3 -m pytest -q -m fast_ci --durations=10        # 期望 <15 s
```
**提交信息**（建议）：
```
refactor(qehvi,qnehvi): extract _acquisition/_greedy_batch for fast CI

- qehvi/qnehvi share _objective_models + _acquisition + _greedy_batch
- qnehvi warning once (self._noise_warned)
- behaviour unchanged: ask sequences identical to pre-refactor snapshots
- adds test_qnehvi_fast_ci (#2/#4/#5, <5 s)
```

---

## 6. 回滚点与应急

| 点 | 何时 | 操作 |
|---|---|---|
| **R0** | 改动前 | `git rev-parse HEAD` 记录；`git checkout R0 -- .` |
| **R1** | qehvi 改完但 qnehvi 未动 | `git checkout -- src/condopt/algorithms/qehvi.py` |
| **R2** | qnehvi 改完但快照不一致 | `git checkout -- src/condopt/algorithms/qnehvi.py` |
| **R3** | 测试文件已加但全量失败 | `git checkout -- tests/ scripts/` |

**风险点**：① `_acquisition` 里 GP 拟合顺序必须保持 index 0→1；② `_sample_posterior_multi` 的调用顺序（先 0 后 1）决定 RNG 流；③ `qnehvi` baseline 抽样循环必须在贪心之前；④ 不要改 `params` 默认值（`n_base/n_y/mc_draws`）。

---

## 7. 参考

R12（qNEHVI）、R21（qEHVI）、R42（HV/固定 ref）、R32（统计）、`QNEHVI_FAST_CI_FINAL.md`、`REVIEW_v14.md`。
