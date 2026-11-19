# ROUTER_WEIGHTS_ACCEPTANCE — 权重机制验收与 `runs.jsonl → weights.json` 接口

> 维护者：algo-research ｜ v1（2026-11-05，Round 8）
> 被测：`router.set_algorithm_weights` / `router.algorithm_weights` / `cli --weights`（v0.6.1）。
> 对应：`ROUTER_SCORETERM_PATCH.md`（长期方案）、`REVIEW_v07.md` P1-1/P2-3/P2-4。
> 文献：R8（场景先验可配置化）、R17（推荐器配置）、R32（多 seed 统计口径）、R24（权重可解释）。
> 结论：机制"能用"，但有三个必须钉住的语义：**乘在总分上（负分反直觉）**、**NaN 未拦截**、**权重不写进 study**。

---

## 1. 机制现状（v0.6.1）

```python
# router.py:36-52
_ALGORITHM_WEIGHTS: dict[str, float] = {}          # 进程级全局；空 = 默认
def set_algorithm_weights(weights):                # 校验：名称必须在 registry；factor >= 0
def algorithm_weights() -> dict

# router.py:801-806（route 内，能力过滤之后）
for name in scores:
    factor = _ALGORITHM_WEIGHTS.get(name)
    if factor is not None and factor != 1.0:
        scores[name] = round(scores[name] * factor, 2)

# cli.py:128-142
def _apply_router_weights(value):                  # 读 JSON：{"weights": {...}} 或裸 dict
# 仅 route / init 子命令有 --weights
```

**语义**：`final_score(a) = round(base_score(a) * factor_a, 2)`，作用于**总分的绝对值**；不修改 reasons/warnings；不参与能力过滤（过滤在前）。

---

## 2. 验收用例（可直接写 pytest，含本机实测）

```python
# tests/test_router_weights.py
import json, math, pytest
from condopt.router import (OptimizationContext, route, set_algorithm_weights,
                            algorithm_weights)
from condopt.cli import _apply_router_weights

CTX = OptimizationContext(n_continuous=6, budget_total=24, batch_size=1,
                          noise="medium", domain="chemistry")

@pytest.fixture(autouse=True)
def reset_weights():
    set_algorithm_weights(None)
    yield
    set_algorithm_weights(None)

def test_empty_weights_are_bitwise_identical():
    """实测：空权重 vs 默认 12 个算法分数字典完全相同。"""
    base = dict(route(CTX).candidate_scores)
    set_algorithm_weights({})
    assert dict(route(CTX).candidate_scores) == base
    assert algorithm_weights() == {}

def test_factor_scales_final_score():
    """实测：{"bayes":1.5} → bayes 80 → 120（round 到 2 位）。"""
    base = route(CTX).candidate_scores["bayes"]
    set_algorithm_weights({"bayes": 1.5})
    got = route(CTX).candidate_scores["bayes"]
    assert got == pytest.approx(round(base * 1.5, 2))

def test_zero_factor_disables_algorithm():
    """factor=0 → 分数 0；若其他算法正分更高，则不会被选中。"""
    set_algorithm_weights({"bayes": 0.0})
    rec = route(CTX)
    assert rec.candidate_scores["bayes"] == 0.0
    assert rec.algorithm != "bayes"

def test_unknown_algorithm_rejected():
    with pytest.raises(ValueError, match="unknown algorithm"):
        set_algorithm_weights({"nope": 1.0})

def test_negative_rejected():
    with pytest.raises(ValueError, match="must be >= 0"):
        set_algorithm_weights({"bayes": -1.0})

@pytest.mark.xfail(reason="REVIEW_v07 P2-3：nan 未被拦截（nan<0 为 False）")
def test_nan_rejected():
    with pytest.raises(ValueError):
        set_algorithm_weights({"bayes": float("nan")})

def test_weights_cannot_bypass_capability_filter():
    """能力过滤在前：mfbo 没有 multi_objective 能力，权重再大也不能在 MOO 场景被选。"""
    moo = OptimizationContext(n_continuous=2, n_objectives=2, budget_total=32)
    set_algorithm_weights({"mfbo": 100.0})
    rec = route(moo)
    assert rec.algorithm != "mfbo"

def test_cli_weights_file_and_bare_dict(tmp_path):
    """实测：CLI 文件 {"weights":{"bayes":1.5}} → bayes 80→120。"""
    path = tmp_path / "w.json"
    path.write_text(json.dumps({"weights": {"bayes": 1.5}}), encoding="utf-8")
    _apply_router_weights(str(path))
    set_algorithm_weights(None)
    _apply_router_weights(str(path))
    assert algorithm_weights() == {"bayes": 1.5}
    assert route(CTX).candidate_scores["bayes"] == pytest.approx(
        round(route(CTX).candidate_scores["bayes"], 2))

def test_cli_bare_dict_accepted(tmp_path):
    path = tmp_path / "w.json"
    path.write_text(json.dumps({"bayes": 1.2}), encoding="utf-8")
    _apply_router_weights(str(path))
    assert algorithm_weights() == {"bayes": 1.2}

def test_cli_missing_file_exits():
    with pytest.raises(SystemExit):
        _apply_router_weights("/nonexistent/weights.json")

def test_deterministic_same_weights_same_result():
    set_algorithm_weights({"bayes": 1.5, "tpe": 0.8})
    assert route(CTX).to_dict() == route(CTX).to_dict()
```

**已知行为（需要文档化，不要当 bug 隐藏）**：`{"bayes": 0}` 时若其他算法仍有负分（如 `parego=-165`），分数 0 会让 bayes 反而排到前面；`{"bayes": 2}` 会让负分算法更负（见 `REVIEW_v07` P1-1）。在 ScoreTerm 化（`ROUTER_SCORETERM_PATCH.md`）之前，建议约定：**只对正分算法设置 >1 的权重；0 表示禁用**。

**P0 关联（务必在验收前修）**：`Study.create(algorithm=..., auto_route=True)` 会静默改算法（实测 requested=mfbo → pending=sobol），因此"用 `--weights` 校准后跑 Study 验收"时必须 `auto_route=False` 或等 P0-1 修复，否则测的是 router 而不是目标算法。

---

## 3. `runs.jsonl → weights.json` 脚本接口

### 3.1 输入（`runs.jsonl` 每行，来自 `calibrate_router.py --collect`）

```json
{"problem":"moo_yield_cost","algorithm":"qehvi","budget":32,"batch":4,"seed":3,
 "regret":0.08,"hv_ratio":0.94,"status":"ok"}
```

- `regret`：越低越好（无 HV 的单目标）；`hv_ratio`：越高越好（多目标，**必须基于 true objectives**，见 `REVIEW_v07` P0-2）；
- 聚合键：`(problem, budget, batch)`；每键每组算法至少 `min_seeds=5`，否则该 context 不参与标签。

### 3.2 转换公式（`fit_weights(runs, ...)`）

```python
def fit_weights(runs, beta=0.8, factor_range=(0.6, 1.6), min_seeds=5):
    # 1) 聚合：每个 context 内按算法算 IQM 指标
    scores = {}                       # (problem,budget,batch) -> {algo: iqm_utility}
    for key, rows in groupby(runs, key=lambda r: (r["problem"], r["budget"], r["batch"])):
        for algo, group in groupby(rows, key=lambda r: r["algorithm"]):
            if len(group) < min_seeds: continue
            if "hv_ratio" in group[0]:
                scores.setdefault(key, {})[algo] = iqm([r["hv_ratio"] for r in group])   # 越大越好
            else:
                scores.setdefault(key, {})[algo] = -iqm([r["regret"] for r in group])   # 转成越大越好
    # 2) 每个 context 归一化到 [0,1]，再跨 context 取平均 utility
    util = {}
    for key, algo_scores in scores.items():
        lo, hi = min(algo_scores.values()), max(algo_scores.values())
        span = hi - lo if hi > lo else 1.0
        for algo, value in algo_scores.items():
            util.setdefault(algo, []).append((value - lo) / span)
    mean_util = {algo: float(np.mean(v)) for algo, v in util.items()}
    # 3) 相对最优算法的 log-ratio → factor（clamp）
    best = max(mean_util, key=mean_util.get)
    factors = {}
    for algo, u in mean_util.items():
        raw = math.exp(beta * (mean_util[best] - u))          # u 越低 → factor 越大（补偿性）
        factors[algo] = round(min(max(raw, factor_range[0]), factor_range[1]), 3)
    return factors
```

- **方向**：`factor_a = exp(β·(u_best − u_a))` —— 校准因子是"**补偿**弱算法的负分/低分"，让它有机会被选中；若想"奖励强算法"，用 `exp(β·(u_a − u_mean))` 并只用正分算法（见 §2 约定）。
- `β=0.8`、`factor_range=(0.6,1.6)` 是保守默认（避免权重把能力过滤外的算法拉进来——不会，但会让分数失真）。
- **只输出 `factors` 里非 1.0 的项**；空 dict = 默认。

### 3.3 输出 `weights.json`

```json
{
  "version": "router-weights-2026-11-08",
  "weights": {"tpe": 1.25, "sobol": 0.9, "qehvi": 1.1},
  "trained_on": {"matrix": "r8-smoke", "n_contexts": 18, "min_seeds": 5,
                 "problems": ["branin", "mixed_reaction", "..."], "beta": 0.8},
  "validation": {"lopo_top1": 0.58, "lopo_top2": 0.83, "regret_gap": 0.07, "n_folds": 9}
}
```

### 3.4 脚本 CLI（可直接落 `scripts/calibrate_router.py`）

```bash
# 采集（Study/runner 均可；注意 auto_route=False，且多目标用 true HV）
py -3 scripts/calibrate_router.py --collect --matrix smoke --workers 8 \
    --out runs/bench/r8_runs.jsonl

# 拟合 + LOPO
py -3 scripts/calibrate_router.py --fit runs/bench/r8_runs.jsonl \
    --beta 0.8 --factor-range 0.6 1.6 --min-seeds 5 --folds lopo \
    --out research/router-weights-2026-11-08.json \
    --report research/router-calibration-2026-11-08.md

# 路由端验收
py -3 -m condopt route --space configs/space.example.json --weights research/router-weights-2026-11-08.json --json
```

### 3.5 验收门槛

| 指标 | 门槛 | 备注 |
|---|---|---|
| 空权重 = 默认 | 逐位一致 | §2 `test_empty_weights_are_bitwise_identical` |
| LOPO top-1 | ≥0.55 | 9 问题 × 8 seeds 的第一轮目标 |
| LOPO top-2 | ≥0.80 | |
| regret gap | ≤0.10 | |
| factor 范围 | ⊂ [0.6, 1.6] | 超出则说明数据不足/异常，告警 |
| 非法输入 | unknown/negative 必抛；**NaN 应抛（待修）** | P2-3 |
| 可回滚 | 删文件/空 dict 即回到 v0.6.1 行为 | |

---

## 4. 参考

R8（Facade/场景选择显式化）、R17（推荐器配置化）、R32（IQM/多 seed/LOPO）、R24（权重需可解释）、`ROUTER_SCORETERM_PATCH.md`、`ROUTER_CALIBRATION_HARNESS.md`、`REVIEW_v07.md`（P0-1/P1-1/P2-3/P2-4）。
