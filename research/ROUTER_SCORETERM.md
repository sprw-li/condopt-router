# ROUTER_SCORETERM — router.py 的补丁级重构与最小校准实验

> 维护者：algo-research ｜ v1（2026-11-05，Round 5）
> 对应：`ROUTER_CALIBRATION.md`（Round 4 方法论）、`CAPABILITIES_SPEC.md`（先过滤后打分）、`router.py`（v0.4：`_score_*` 仍返回 `(score, reasons, warnings)`）。
> 目标：**纯重构 + 可加载权重**，保证 `weights=1.0` 时输出与 v0.4 逐位一致；再用一轮最小实验拿到第一版 `router-weights-*.json`。

---

## 1. 补丁结构

### 1.1 新增 `ScoreTerm` 与 terms 函数（保留旧函数）

```python
# router.py 顶部
@dataclass(frozen=True)
class ScoreTerm:
    feature: str
    delta: float
    reason: str
    weightable: bool = True

def _terms_bayes(ctx) -> list[ScoreTerm]:
    terms = [ScoreTerm("base_bayes", 45.0, "GP-BO 基线")]
    if ctx.effective_dim <= 12:
        terms.append(ScoreTerm("dim_le_12", 18.0, f"特征维 {ctx.effective_dim} ≤ 12，GP 代理可靠"))
    elif ctx.effective_dim <= 20:
        terms.append(ScoreTerm("dim_13_20", -10.0, "特征维偏高，GP 需要更多初始点"))
    else:
        terms.append(ScoreTerm("dim_gt_20", -60.0, "特征维 >20（one-hot 放大维度），GP 不可靠"))
    if ctx.budget_total >= 20:
        terms.append(ScoreTerm("budget_ge_20", 12.0, "预算 ≥20，足够 BO 冷启动"))
    elif ctx.budget_total < 8:
        terms.append(ScoreTerm("budget_lt_8", -40.0, "预算 <8，样本效率优势来不及体现"))
    if ctx.n_objectives > 1:
        terms.append(ScoreTerm("moo_unsupported", -200.0, "多目标不适用", weightable=False))
    ...
    return terms

def _legacy_score(terms, weights) -> tuple[float, list[str], list[str]]:
    score = 0.0; reasons = []; warnings = []
    for t in terms:
        w = 1.0 if not t.weightable else float(weights.get(t.feature, 1.0))
        score += w * t.delta
        (warnings if t.delta < 0 else reasons).append(t.reason)
    return round(score, 2), reasons, warnings

# 向后兼容：旧函数名保留，内部走 terms（w=1 时数值不变）
def _score_bayes(ctx):
    return _legacy_score(_terms_bayes(ctx), _WEIGHTS)
```

**逐算法迁移清单**（v0.4 的 `_score_*` 全部照搬为 `_terms_*`）：`bayes/parego/tpe/sobol/lhs/random/grid`。注意保留 `weightable=False` 的项（如 `moo_unsupported=-200`、`single_objective_penalty=-200`）：这些是"不适用"语义，不能被权重放大/缩小。

### 1.2 权重加载与默认表

```python
_DEFAULT_WEIGHTS: dict[str, float] = {}          # 空 = 全 1.0（v0.4 等价）
_WEIGHTS = dict(_DEFAULT_WEIGHTS)

def load_weights(path: str | None = None) -> dict[str, float]:
    global _WEIGHTS
    path = path or os.environ.get("CONDOPT_ROUTER_WEIGHTS")
    if not path:
        _WEIGHTS = dict(_DEFAULT_WEIGHTS); return _WEIGHTS
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        weights = {str(k): float(v) for k, v in payload["weights"].items()}
        if any(not (0.0 <= v <= 3.0) for v in weights.values()):
            raise ValueError("weight out of [0,3]")
        _WEIGHTS = weights
    except Exception as exc:                      # 坏文件不阻塞
        warnings.warn(f"router weights ignored: {exc}")
        _WEIGHTS = dict(_DEFAULT_WEIGHTS)
    return _WEIGHTS
```

**JSON schema**：`{"version":"router-weights-YYYY-MM-DD","base":{...},"weights":{feature: w}, "trained_on":..., "validation":...}`（见 `ROUTER_CALIBRATION.md` §2.2）。`base` 项可并入 `weights`（如 `base_bayes` 作为 feature），MVP 只支持 `weights`。

### 1.3 `route()` 的改动（3 行）

```python
score, reasons, warnings = rule(ctx)      # 旧
# →
terms = _TERMS[best_name](ctx); score, reasons, warnings = _legacy_score(terms, _WEIGHTS)
```

`Recommendation` 增加（可选）`terms: list[dict]`，供 `report --json` 展示 `feature/delta/weight/贡献`，便于审计。

### 1.4 默认权重表（第一版，全部 1.0）

| feature | 默认 | 备注 |
|---|---|---|
| 全部 `weightable=True` 的 term | 1.0 | 与 v0.4 完全等价 |
| `moo_unsupported` / `single_objective_penalty` | 1.0（`weightable=False`） | 不参与校准 |
| `domain_chemistry`（+5） | 建议校准后置 0 | 无文献依据（`REVIEW_v01` §5），权重 0 即删除该直觉项 |

---

## 2. 单测（必须与重构同 PR）

```python
def test_terms_reproduce_legacy_scores():       # 对 12 个采样 context：_score_* == _legacy_score(_terms_*, {})
def test_scoreterm_weights_scale_contributions():# 权重 0 时该 feature 的贡献消失
def test_weights_json_roundtrip(tmp_path):      # 写/读/校验
def test_bad_weights_file_falls_back():         # 损坏 JSON / 越界权重 → 默认 + warning
def test_weightable_false_is_ignored():         # moo_unsupported 不随权重变化
def test_router_candidate_scores_unchanged_at_default_weights(): # 快照对比
```

---

## 3. 最小校准实验（1 天，先用 smoke/frontier 而非 full）

### 3.1 数据采集（半天，可并行）

```bash
# 1) 现有 benchmark runner 输出 JSON（Tier A–F 中的 8–10 个问题）
py -3 scripts/benchmark.py --budget 16 --batch 1 --seeds 0 1 2 3 4 5 6 7 --json > runs/bench/smoke_b16.json
py -3 scripts/benchmark.py --budget 32 --batch 4 --seeds 0 1 2 3 4 5 6 7 --json > runs/bench/frontier_b32q4.json
```
预算：8 问题 × 4 算法（random/sobol/bayes/parego）× 2 配置 × 8 seeds ≈ **512 runs**；每 run 16–32 次评估 → CPU 单机 ~10–20 min（`bayes` 的候选池 1024、n=32 时单 run 约 2–4 s）。

### 3.2 标签与拟合（半天）

1. 每个 `(task, budget, batch)` 聚合 8 seeds → `IQM regret`；
2. 排序 → `best`/`top2`；`|IQM(best)−IQM(second)| < 5%` 记为 tie（不进 top-1 监督）；
3. `contexts` 用 `OptimizationContext.from_space(task.space, ...)` 重建（不能只用 bench 行）；
4. LOPO-CV（留一 task）拟合 weights（随机搜索 150 + 坐标下降 2 轮）；
5. 产出 `router-weights-YYYY-MM-DD.json` + `router-calibration-YYYY-MM-DD.md`。

### 3.3 预期与门槛

| 指标 | 期望（第一轮） | 不达标时 |
|---|---|---|
| LOPO top-1 | 0.55–0.70 | 数据少 → 增加 seeds/budget，或只校准 5–8 个关键 feature |
| LOPO top-2 | 0.80–0.90 | 同上 |
| regret gap | ≤8% | 先不替换默认权重（保留为候选文件） |
| 权重符号 | 人工 review 全通过 | 发现反号 → 改规则（如删 `domain_chemistry`），不要拟合 |

**注意**：Tier A/F 的问题数少（2–3 个/类），LOPO 方差大；第一轮建议**只校准跨任务通用的 feature**（`dim_*`、`budget_*`、`noise_*`、`batch_*`、`categorical_frac`），不要校准 task 特有项。

---

## 4. 实施顺序

1. `ScoreTerm` + `_terms_*` + `_legacy_score`（纯重构，0.5 天）→ 现有 102 测试全绿 + 6 条新单测；
2. 权重加载 + `Recommendation.terms`（2 h）；
3. benchmark 数据采集（半天，可与 1 并行）；
4. 校准脚本 `scripts/calibrate_router.py`（1 天，见 `ROUTER_CALIBRATION.md` §6）；
5. 第一版权重文件 + 报告（半天）。

---

## 5. 参考

R8（SMAC3 Facade：规则是"先验策略"，可被场景化数据校准）、R17（BayBE 推荐器切换的配置化）、R32（benchmark 统计：IQM/CI/LOPO）、R24（权重需有依据/可解释）、`BENCHMARK_SPEC.md` §7、`ROUTER_CALIBRATION.md`。
