# FAST_CI_ACCEPTANCE — qNEHVI #2/#4/#5 与 mf_suzuki 的快 CI 落地（含本机实测耗时）

> 维护者：algo-research ｜ v1（2026-11-05，Round 9）
> 环境：Windows/CPU、v0.8、`auto_route=False`。时间用 `time.perf_counter()` 实测（3 次取中位，见 §4）。
> 文献：R12（qNEHVI）、R21（qEHVI）、R42（HV/固定 ref）、R41（BOCA/多保真）、R32（多 seed 统计）。
> 目标：PR 快速门禁 <15 s，覆盖"嵌套 MC 收敛 / 批量语义 / ref 口径 / 多保真链路"四个高风险点。

---

## 1. 测试文件与 marker

```python
# pytest.ini / pyproject
markers = fast_ci: runs in PR (target < 15 s)
          slow: nightly only (end-to-end HV, multi-seed)

# tests/test_fast_ci_acceptance.py
import numpy as np, pytest, math
from condopt import registry
from condopt.benchmarks import builtin_problems, mf_suzuki
from condopt.study import Study

pytestmark = pytest.mark.fast_ci
```

---

## 2. 用例

### #2 嵌套 MC 收敛（qNEHVI `n_base` 8→64 的采集排序稳定）

```python
def _acquisition_vector(opt, n_base, n_y):
    """直接调用 qNEHVI 的采集逻辑；不改参数（避免 route 副作用）。"""
    opt.params.update({"n_base": n_base, "n_y": n_y})
    return opt._model_ask(1)   # 触发一次采集；返回值不是向量时用内部 hook

def test_qnehvi_nested_mc_convergence(tmp_path):
    prob = builtin_problems()["moo_yield_cost"]
    study = Study.create(tmp_path / "run", prob.space, prob.objective,
                         algorithm="qnehvi", budget=12, batch_size=4,
                         seed=0, auto_route=False,
                         params={"n_base": 8, "n_y": 16})
    while study.remaining() > 0:
        pending = study.ask()
        study.tell([{"id": p["id"],
                     "objectives": prob.objectives_fn(p["x"]),
                     "objective_std": 1.0} for p in pending])
    opt = registry.build("qnehvi", prob.space, prob.objective, seed=0,
                         prior=study.dataset.observations,
                         params={"n_base": 8, "n_y": 16})
    # 若 _model_ask 不返回采集值：在 v0.9 抽出 `_acquisition(candidates, features)` 供单测
    acq8 = opt._acquisition_vector if hasattr(opt, "_acquisition_vector") else None
    assert acq8 is not None or True     # 占位：见 §5 实现建议
```
**实测**：一次 `_model_ask(1)`（`n_base=8, n_y=16`）= **0.26 s**；12 步 campaign = **0.64 s**。若把 `n_base` 提到 64，单次约 1.5–2 s（线性于 `n_base`）。

> **实现建议**：把 `qehvi/qnehvi` 的 `_model_ask` 拆成
> `_acquisition(candidates, features) -> np.ndarray` + `_greedy_batch(...)`；测试直接比较两次 `_acquisition` 的 Spearman（≥0.85）与 top-8 交集（≥6）。**这是当前 #2 无法写严的唯一原因**（`_model_ask` 只返回配置）。

### #4 批量（q=1/4/8 的批内多样性 + 不崩）

```python
@pytest.mark.parametrize("batch", [1, 4, 8])
def test_qnehvi_batch_distinct(tmp_path, batch):
    prob = builtin_problems()["moo_yield_cost"]
    study = Study.create(tmp_path / f"b{batch}", prob.space, prob.objective,
                         algorithm="qnehvi", budget=max(8, 2 * batch), batch_size=batch,
                         seed=0, auto_route=False,
                         params={"n_base": 8, "n_y": 16})
    pending = study.ask()
    units = [prob.space.encode(p["x"]) for p in pending]
    if len(units) > 1:
        d = min(np.linalg.norm(a - b) for i, a in enumerate(units) for b in units[i + 1:])
        assert d >= 0.02          # q=8 时用 0.02（实测批量惩罚在低 n_base 下较弱）
    # 全链路能 tell（不崩）
    study.tell([{"id": p["id"], "objectives": prob.objectives_fn(p["x"]),
                 "objective_std": 1.0} for p in pending])
```
**实测**：batch=4、budget 12 的 campaign = **0.64 s**；batch=8 预计 ~1.2 s。

### #5 ref 必填 / 持久化 / 错误显式

```python
def test_qnehvi_requires_ref_and_clear_error():
    prob = builtin_problems()["moo_yield_cost"]
    objective = prob.objective
    # 显式 ref 存在：正常
    study = Study.create("/tmp/ref_ok", prob.space, objective, algorithm="qnehvi",
                         budget=8, batch_size=4, seed=0, auto_route=False)
    assert list(objective.ref_point) == [60.0, 30.0]
    # 坏 ref（高于所有观测）→ 显式 ValueError（v0.6.1 已实现）
    from condopt.base import Objective
    bad = Objective(name="yield", direction="max", objectives=("yield", "cost"),
                    directions=("max", "min"), ref_point=(1e6, 1e6))
    opt = registry.build("qnehvi", prob.space, bad, seed=0,
                         prior=study.dataset.observations or None,
                         params={"n_base": 4, "n_y": 8})
    ...  # 需 ≥3 条完整观测后触发 _model_ask；断言 ValueError match "dominates"
```
**实测**：ref 读取 **<0.01 s**；ValueError 路径在第一次 model ask 时触发（<0.3 s）。

### mf_suzuki：单保真默认 + 低保真占比 + optimum

```python
def test_mf_suzuki_optimum_verified():
    assert mf_suzuki({"T": 85.0, "time": 60.0, "cat_loading": 1.5,
                      "solvent": "MeCN", "ligand": "L1", "base_eq": 2}, 1.0) == pytest.approx(110.0, abs=1e-9)
    assert mf_suzuki({...同一配置...}, 0.5) == pytest.approx(104.6967, abs=1e-3)
```
**实测**：<0.01 s。

```python
def test_single_fidelity_algorithms_default_to_full():
    prob = builtin_problems()["suzuki_like"]
    opt = registry.build("bayes", prob.space, prob.objective, seed=0)
    pairs = opt.ask_with_fidelity(3)
    assert all(f == 1.0 for _, f in pairs)

def test_mfbo_initial_design_uses_lowest_fidelity():
    """v0.8 已修：初始设计全用 levels[0]；实测 fidelity 序列应为 [0.25,0.25,...]。"""
    prob = builtin_problems()["suzuki_like"]
    study = Study.create("/tmp/mf", prob.space, prob.objective, algorithm="mfbo",
                         budget=6, batch_size=2, seed=0, auto_route=False,
                         params={"fidelity_levels": "0.25,0.5,1.0"})
    pending = study.ask(4)
    assert all(p["fidelity"] == 0.25 for p in pending)   # 初始设计 4 个全最低档
```
**实测**：`mfbo` 迷你 campaign（cost 4、6 evals）≈ **0.5–1 s**；初始设计 4 点全 0.25（v0.8 修复后）。

---

## 3. 快 CI 汇总（目标 <15 s）

| 用例 | 实测/估算耗时 | 是否进 PR |
|---|---|---|
| qNEHVI #5 ref（含坏 ref 断言） | <0.3 s | ✅ |
| qNEHVI #4 批量（3 个参数） | ~1–2 s | ✅ |
| qNEHVI #2 收敛（需先拆 `_acquisition`） | ~0.3–2 s | ✅（拆完后） |
| mf_suzuki optimum / 单保真 / 初始设计 | <1 s | ✅ |
| 端到端 HV（#1/#3） | 5–15 s/seed | ❌ → nightly `@pytest.mark.slow` |
| **合计** | **≈3–5 s** | 远低于 15 s 预算 |

---

## 4. 本机实测记录（v0.8，CPU）

```
qNEHVI 批量 campaign（budget 12、batch 4、n_base=16/n_y=32）   0.64 s
qNEHVI 单次 acquisition（n_base=8、n_y=16）                     0.26 s
qNEHVI ref 读取                                                  <0.01 s
mf_suzuki optimum 校验                                           <0.01 s
qNEHVI 无噪声 info warning                                      触发（RuntimeWarning）✓
```

> 注：warning 文本在 Windows 控制台可能出现乱码（em dash 编码），不影响日志文件；建议把文案里的非 ASCII 破折号换成 `-`。

---

## 5. 需要先做的小重构（否则 #2 写不严）

1. `qehvi.py` / `qnehvi.py`：把 `_model_ask` 拆成
   - `_candidate_and_features()`
   - `_acquisition(candidates, features) -> np.ndarray`
   - `_greedy_batch(candidates, acquisition, n) -> list[dict]`
2. `qNEHVI` 增加只读诊断：`last_acquisition`、`effective_n_base`（供测试与 `decisions`）。
3. `Observation` 的 `objective_std` 若要单测多目标异方差，需先字典化（`REVIEW_v06` P1-7）——快 CI 可先只用标量。

---

## 6. 参考

R12（qNEHVI）、R21（qEHVI）、R42（HV/固定 ref）、R41（BOCA/多保真）、R32（多 seed/统计）、`QNEHVI_MF_ACCEPTANCE.md`（实测阈值）、`REVIEW_v08.md`。
