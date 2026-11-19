# ROUTER_SCORETERM_PATCH — router.py 的补丁级 ScoreTerm 落地（保持 v0.6 行为逐位一致）

> 维护者：algo-research ｜ v1（2026-11-05，Round 7）
> 对应：`ROUTER_SCORETERM.md`（Round 5 设计）、`ROUTER_CALIBRATION_HARNESS.md`（Round 6 校准脚本骨架）、`CAPABILITIES_SPEC.md`。
> 文献：R8（SMAC3 Facade：规则是"场景先验"）、R17（推荐器可配置化）、R32（多 seed/统计协议）、R24（规则应可解释）。
> **硬约束**：本补丁在默认权重（全 1.0）下，`route()` 的输出（algorithm/params/scores/warnings）必须与 v0.6 **逐位一致**——现有测试不改一行。

---

## 1. 补丁结构（三个小 PR）

### PR-1：`ScoreTerm` + `_terms_*`（纯重构，不改行为）

```python
# router.py 顶部
from dataclasses import dataclass, field

@dataclass(frozen=True)
class ScoreTerm:
    feature: str            # 稳定名字，见 §2 特征目录
    delta: float            # 未加权贡献
    reason: str
    weightable: bool = True

_WEIGHTS: dict[str, float] = {}          # 空 = 全 1.0（v0.6 等价）

def _legacy_score(terms: list[ScoreTerm], weights: dict[str, float] | None = None):
    """完全复刻 _score_* 的返回值：round(score,2) 由 route() 统一做。"""
    weights = _WEIGHTS if weights is None else weights
    score = 0.0
    reasons: list[str] = []
    warnings: list[str] = []
    for term in terms:
        w = 1.0 if not term.weightable else float(weights.get(term.feature, 1.0))
        score += w * term.delta
        (warnings if term.delta < 0 else reasons).append(term.reason)
    return score, reasons, warnings
```

**逐条迁移**（v0.6 的每个常数都对应一个 term；`weightable=False` 用于"不适用"语义）：

| 现有规则（v0.6 的 `_score_*` 内常数） | feature 名 | weightable |
|---|---|---|
| 各算法初值（45/35/28/25/22/15/5…） | `base_<algo>` | ✓ |
| `effective_dim ≤ 12` +18 / `≤15` +8 / `≤20` -10 / `>20` -60 | `dim_le_12` / `dim_13_15` / `dim_16_20` / `dim_gt_20` | ✓ |
| 预算档位（<8/-40、8–11/-10、12–19/+4、≥20/+12、>60/-10） | `budget_lt_8` / `budget_8_11` / `budget_12_19` / `budget_ge_20` / `budget_gt_60` | ✓ |
| 噪声（medium/high）+5、TPE high +4 | `noise_medium` / `noise_high` | ✓ |
| smooth +12 / rough -12 | `smooth` / `rough` | ✓ |
| 批量 >16 -8、≥8 +12（sobol）、≥8 +16（lhs） | `batch_gt_16` / `batch_ge_8_sampling` | ✓ |
| categorical 相关（TPE +8/+20） | `categorical_present` / `discrete_frac_ge_0.5` | ✓ |
| prior ≥6 +10、prior <1.5·eff -10 | `prior_ge_6` / `prior_lt_1.5dim` | ✓ |
| 多目标（bayes -200、parego +70） | `moo_unsupported`（-200，**weightable=False**）/ `moo_supported` | 固定 |
| 约束（bayes -8/parego -5） | `constraints_present` | ✓ |
| multi_fidelity -5（bayes） | `multi_fidelity_present` | ✓ |
| all_discrete & card ≤200（-15/+55） | `all_discrete_small` | ✓ |
| `domain=="chemistry"` +5（**无文献依据**） | `domain_chemistry` | ✓（建议校准后置 0） |
| `n_objectives==2` +5、`>=4` -15 | `moo_2obj` / `moo_ge4obj` | ✓ |

```python
def _terms_bayes(ctx) -> list[ScoreTerm]:
    terms = [ScoreTerm("base_bayes", 45.0, "GP-BO 基线")]
    if ctx.n_objectives > 1:
        terms.append(ScoreTerm("moo_unsupported", -200.0,
                               "多目标不适用（请用 parego/qehvi/qnehvi）", weightable=False))
    eff = ctx.effective_dim
    if eff <= 12:
        terms.append(ScoreTerm("dim_le_12", 18.0, f"特征维 {eff} ≤ 12，GP 代理可靠"))
    elif eff <= 15:
        terms.append(ScoreTerm("dim_13_15", 8.0, f"特征维 {eff} 处于 GP 舒适区边缘"))
    elif eff <= 20:
        terms.append(ScoreTerm("dim_16_20", -10.0, f"特征维 {eff} 偏高，GP 需要更多初始点"))
    else:
        terms.append(ScoreTerm("dim_gt_20", -60.0,
                               f"特征维 {eff} > 20（one-hot 放大维度），GP 不可靠"))
    ...
    return terms

# 兼容层：旧函数名保留（外部/测试可能 import 它们）
def _score_bayes(ctx):
    return _legacy_score(_terms_bayes(ctx))
```

**顺序与行为一致性**：`reason/warning` 顺序必须与 v0.6 相同（测试若断言 warnings 文本顺序会失败）。迁移时**逐行对照**：先复制 v0.6 的 `_score_*`，再把每个 `score += x` 改成 `terms.append(...)`，保持 append 顺序。

### PR-2：权重加载（默认空 → 行为不变）

```python
import json, os, warnings

_DEFAULT_WEIGHTS: dict[str, float] = {}          # 空 = 全 1.0

def load_weights(path: str | None = None) -> dict[str, float]:
    global _WEIGHTS
    path = path or os.environ.get("CONDOPT_ROUTER_WEIGHTS")
    if not path:
        _WEIGHTS = dict(_DEFAULT_WEIGHTS); return _WEIGHTS
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        weights = {str(k): float(v) for k, v in payload.get("weights", {}).items()}
        if any(not (0.0 <= v <= 3.0) for v in weights.values()):
            raise ValueError("weight outside [0, 3]")
        _WEIGHTS = weights
    except Exception as exc:
        warnings.warn(f"router weights ignored ({exc}); using defaults")
        _WEIGHTS = dict(_DEFAULT_WEIGHTS)
    return _WEIGHTS
```

`router.py` 模块加载时调用一次 `load_weights()`；`route()` 顶部可再调（支持测试注入）。

### PR-3：`Recommendation.terms`（可解释性，可选）

```python
@dataclass
class Recommendation:
    ...
    terms: list[dict] = field(default_factory=list)   # [{"feature":..,"delta":..,"weight":..,"contribution":..}]
```
只在 `route()` 的 best 分支填充（不影响 `to_dict()` 的旧字段与测试）。

---

## 2. 权重 JSON（schema 见 `ROUTER_CALIBRATION.md` §2.2）

```json
{
  "version": "router-weights-2026-11-07",
  "weights": {"dim_le_12": 1.0, "dim_gt_20": 1.0, "noise_high": 1.0,
               "domain_chemistry": 0.0, "prior_ge_6": 1.0},
  "trained_on": {"matrix": "smoke", "n_contexts": 24, "seeds": 8},
  "validation": {"lopo_top1": 0.0, "lopo_top2": 0.0, "regret_gap": 0.0, "n_folds": 8}
}
```

---

## 3. 单测（与 PR-1 同批）

```python
def test_terms_reproduce_legacy_v06_scores():
    # 对 16 个手写 context：_terms_* + _legacy_score(w={}) == v0.6 的 _score_*
    # 数值与 reasons/warnings 顺序都逐位比较（快照字符串）
def test_term_features_are_unique_per_algorithm():
    # 每个 _terms_* 的 feature 不重复（除有意复用，如 dim_*）
def test_weights_scale_contributions():
    w = {"dim_le_12": 2.0}
    # 该 term 贡献翻倍；其余不变
def test_zero_weight_removes_term():
    # domain_chemistry=0 → 化学 domain 不再加分
def test_bad_weights_fall_back():
    # 损坏 JSON / 越界 → 默认 + warning
def test_route_unchanged_at_default_weights():
    # 12 个 context 的 to_dict()（不含新 terms 字段）与 v0.6 快照一致
def test_capability_filter_still_runs_before_scoring():
    # n_objectives=2 时 bayes/parego/... 的 moo_unsupported 不参与（本来就被过滤）
```

**CI gate**：PR-1/2/3 合并前 `pytest` 全绿 + `test_route_unchanged_at_default_weights` 通过。

---

## 4. 最小校准实验（Tier A–F，1 天内可跑）

### 4.1 矩阵

| 项 | 值 |
|---|---|
| 问题 | `branin`, `sphere2d`, `hartmann3`, `mixed_reaction`, `suzuki_like`, `discrete_catalyst_v2`, `noisy_sphere`, `constraint_hard`, `moo_yield_cost`（9 个，覆盖 A–F） |
| 算法 | `random, sobol, lhs, bayes, tpe, parego, qehvi, constrained`（按问题能力自动过滤；m=2 用 parego/qehvi，约束用 constrained） |
| 预算 | 16、32 |
| 批量 | 1、4 |
| seeds | 0–7（8 个） |
| run 数 | 9 × ~4 × 2 × 2 × 8 ≈ **~460 runs**（能力过滤后） |
| 单 run 成本 | `bayes` 16 次评估 ≈ 1.5 s；32 次 ≈ 3 s；`qehvi` 每条 ~3–5 s（MC 128） |
| 预计总时长 | 8 进程 **15–25 min**（qehvi 占大头） |

### 4.2 步骤

```bash
# 1) 采集（并行；输出 JSONL）
py -3 scripts/calibrate_router.py --matrix smoke --workers 8 \
  --out runs/bench/r7_smoke_runs.jsonl

# 2) 聚合 + LOPO-CV（w=1 基线 + 拟合权重）
py -3 scripts/calibrate_router.py --fit runs/bench/r7_smoke_runs.jsonl \
  --search random+cd --n-iter 200 --folds lopo \
  --out research/router-weights-2026-11-07.json \
  --report research/router-calibration-2026-11-07.md
```

### 4.3 结果 JSON（`runs.jsonl` 每行）

```json
{"problem":"moo_yield_cost","algorithm":"qehvi","budget":32,"batch":4,"seed":3,
 "regret":0.08,"hv_ratio":0.94,"auc":0.81,"runtime_s":4.2,"status":"ok"}
```

### 4.4 验收

| 指标 | 门槛（第一轮，数据量小） |
|---|---|
| LOPO top-1 | ≥0.55 |
| LOPO top-2 | ≥0.80 |
| regret gap vs best | ≤0.10 |
| 默认权重（w=1）回归 | 与 v0.6 命中率相同或更好 |
| 权重符号 | 人工 review 通过（`domain_chemistry` 建议置 0） |

**说明**：9 个问题做 LOPO，每折留 1 个问题（训练 8 个）→ 只能评价"跨问题泛化"的粗粒度权重；第一轮**只校准 8–10 个通用 feature**（`dim_*`、`budget_*`、`noise_*`、`batch_*`、`categorical_present`、`prior_ge_6`），其余保持 1.0。

---

## 5. 风险与回滚

1. **理由顺序/文本变化** → 快照测试会抓到；PR-1 必须先逐位对齐。
2. **权重文件被 agent 误改** → schema 校验 + `[0,3]` 限制 + 启动 warning；文件纳入版本管理。
3. **校准过拟合** → LOPO + 只调通用 feature + 保留旧权重文件；`--allow-fail` 时只告警不替换。
4. **性能**：`_terms_*` 每次 `route()` 构造 list（~20 个对象）→ 可忽略（`route` 只在 ask/route 命令调用）。

---

## 6. 参考

R8（Facade 作为场景先验，可被数据校准）、R17（推荐器配置化）、R32（多 seed/IQM/CI 与 LOPO）、R24（规则可解释性）、`ROUTER_SCORETERM.md`、`ROUTER_CALIBRATION_HARNESS.md`。
