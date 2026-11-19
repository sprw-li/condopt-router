# ROUTER_CALIBRATION — router 权重的 ScoreTerm 化与自动校准

> 维护者：algo-research ｜ v1（2026-11-05，Round 4）
> 对应：`BENCHMARK_SPEC.md` §7（校准方法论）、`CAPABILITIES_SPEC.md`（先过滤后打分）、`router.py`（v0.3 的 `_score_*` 硬编码规则）。
> 文献：R8（SMAC3 Facade：场景化入口是"规则"的产品形态）、R17（BayBE 推荐器切换）、R32（benchmark 协议与报告规范）、R24（方法学综述，权重应有依据）。
> 目标：把"手写常数"变成"**可加权特征项 + 数据拟合权重**"，并用 leave-one-problem-out 防过拟合；产出 `router-weights-*.json` 供 `router.py` 加载。

---

## 1. 现状与重构目标

现状（v0.3）：`router.py:213-378` 每个 `_score_*` 返回 `(score, reasons, warnings)`，score 里是散落的常数（45、+18、-200、+70…）。问题：

1. 权重与理由**混在一起**，无法单独调参；
2. 调一个数会牵连一堆理由文本，校准不可复现；
3. `docs/ROUTING_POLICY.md` 自承"权重未校准"。

**重构目标**：`score(algorithm, ctx) = base[algorithm] + Σ_feature w_feature · delta(algorithm, feature, ctx)`。

- `delta` 是算法的特征贡献（可为 0/正/负），`w` 是全局校准权重（默认全 1 = 现状）；
- 能力过滤（`_capability_filter`）保持"硬过滤"，**不参与加权**；
- 理由文本挂在 `ScoreTerm.reason` 上，最终报告显示 `w·delta` 与 reason。

---

## 2. `ScoreTerm` 与权重 JSON Schema

### 2.1 Python 结构

```python
@dataclass(frozen=True)
class ScoreTerm:
    feature: str          # 见 §2.3 特征目录
    delta: float          # 该算法在此特征下的得分贡献（未加权）
    reason: str           # 人类可读理由（进 rationale）
    weightable: bool = True   # False = 硬约束/不可校准项（如 -200 的"不适用"）

def rule_bayes(ctx) -> list[ScoreTerm]:
    terms = [ScoreTerm("base_bayes", 45.0, "GP-BO 基线")]
    if ctx.n_objectives > 1:
        terms.append(ScoreTerm("moo_unsupported", -200.0, "多目标不适用", weightable=False))
    if ctx.effective_dim <= 8:
        terms.append(ScoreTerm("dim_le_8", 18.0, f"特征维 {ctx.effective_dim} ≤ 8，GP 可靠"))
    ...
    return terms

def score_algorithm(name, ctx, weights) -> float:
    total = 0.0
    for t in rule(name, ctx):
        w = 1.0 if not t.weightable else weights.get(t.feature, 1.0)
        total += w * t.delta
    return total
```

### 2.2 权重文件 JSON Schema

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "RouterWeights",
  "type": "object",
  "additionalProperties": false,
  "required": ["version", "base", "weights", "trained_on", "validation"],
  "properties": {
    "version": {"type": "string", "pattern": "^router-weights-[0-9]{4}-[0-9]{2}-[0-9]{2}(-r[0-9]+)?$"},
    "base": {"type": "object", "additionalProperties": {"type": "number"}},
    "weights": {"type": "object", "additionalProperties": {"type": "number", "minimum": 0.0, "maximum": 3.0}},
    "trained_on": {
      "type": "object",
      "required": ["matrix", "n_contexts", "seeds"],
      "properties": {"matrix": {"enum": ["smoke", "frontier", "full"]},
                     "n_contexts": {"type": "integer", "minimum": 1},
                     "seeds": {"type": "integer", "minimum": 1},
                     "benchmark_version": {"type": "string"}}
    },
    "validation": {
      "type": "object",
      "required": ["lopo_top1", "lopo_top2", "regret_gap"],
      "properties": {"lopo_top1": {"type": "number"}, "lopo_top2": {"type": "number"},
                     "regret_gap": {"type": "number"}, "n_folds": {"type": "integer"}}
    },
    "notes": {"type": "string"}
  }
}
```

**加载规则**：`router.py` 启动时读 `CONDOPT_ROUTER_WEIGHTS` 环境变量/`configs/router_weights.json`；缺失或校验失败 → 用内嵌默认（全 1），并 warning。权重文件**不改变能力过滤**（避免"校准把不支持的算法选进来"）。

### 2.3 特征目录（与现有规则一一对应；feature 名稳定，便于跨版本比权重）

| feature | 触发 | 当前对应规则 |
|---|---|---|
| `base_<algo>` | 恒 true | 各 `_score_*` 的初值 |
| `dim_le_8` / `dim_9_15` / `dim_16_20` / `dim_gt_20` | `ctx.effective_dim` | bayes/parego/tpe 的维度项 |
| `budget_lt_8` / `budget_8_11` / `budget_12_19` / `budget_ge_20` / `budget_gt_60` | `budget_total` | 各 `_score_*` 预算项 |
| `batch_ge_4` / `batch_ge_8` / `batch_gt_16` | `batch_size` | 批量项 |
| `noise_low/medium/high` | `ctx.noise` | 噪声项 |
| `smooth` / `rough` | `ctx.smoothness` | 光滑度项 |
| `categorical_frac_ge_0.3` | `discrete_frac` | TPE/bayes 分类项 |
| `prior_ge_6` | `n_prior` | 暖启动项 |
| `n_prior_lt_1.5dim` | 观测不足 | 过拟合告警项 |
| `multi_objective` / `n_objectives_2` / `n_objectives_ge_4` | `n_objectives` | parego 项 |
| `constraints` | `n_constraints>0` | 约束降分项 |
| `multi_fidelity` | `multi_fidelity` | 多保真降分项 |
| `all_discrete_small` | `all_discrete & card<=200` | 枚举项 |
| `domain_chemistry` | `domain=="chemistry"` | 建议删除/权重设 0（无文献依据，见 REVIEW） |

---

## 3. 校准数据与标签

### 3.1 数据来源

跑 `BENCHMARK_SPEC.md` §6 的 `frontier` 矩阵（`task × algo × budget × batch × seeds`），每条记录：

```json
{"task": "mixed_reaction", "algo": "bayes", "budget": 32, "batch": 4, "seed": 3,
 "iqm_regret": 0.08, "auc": 0.72, "runtime_s": 1.9, "tags": ["mixed","chemistry"]}
```

### 3.2 上下文（context）定义

两档，用于不同目的：

- **task-level**：`ctx_key=(task, budget_bucket, batch_bucket)` — 标签最可靠，但只能评估"同任务内选算法"；
- **regime-level**（用于 router 泛化）：`ctx_key=(tags, effective_dim_bucket, budget_bucket, batch_bucket, noise)` — router 的输入就是这些。

### 3.3 标签生成

```python
def make_labels(records):
    labels = {}
    for key, group in groupby(records, key=lambda r: ctx_key(r)):
        stats = {algo: iqm([r["iqm_regret"] for r in group if r["algo"] == algo])
                 for algo in algorithms}
        ranked = sorted(stats, key=lambda a: stats[a])        # 越小越好
        top1, top2 = ranked[0], ranked[:2]
        tie = abs(stats[top1] - stats[top2[1]]) < 0.05 if len(top2) > 1 else False
        labels[key] = {"ranked": ranked, "stats": stats,
                       "best": top1, "top2": set(top2), "is_tie": tie}
    return labels
```
**规则**：`is_tie=True` 的 context 只进 top-2 监督（不进 top-1），并要求每个 context ≥8 seeds。

---

## 4. 目标函数与搜索

### 4.1 目标

```python
def objective(w, contexts, labels):
    total = 0.0
    for key, ctx in contexts.items():
        scores = {algo: score_algorithm(algo, ctx, w) for algo in algorithms}
        chosen = max(scores, key=scores.get)
        lab = labels[key]
        regret = lab["stats"][chosen] - lab["stats"][lab["best"]]
        worst = max(lab["stats"].values()) - lab["stats"][lab["best"]] + 1e-9
        loss = 0.0
        if not lab["is_tie"] and chosen != lab["best"]:
            loss += 0.5
        if chosen not in lab["top2"]:
            loss += 0.3
        loss += 0.2 * regret / worst
        total += loss
    return total / len(contexts)
```

### 4.2 搜索：随机搜索 + 坐标下降

```python
def calibrate(contexts, labels, n_iter=300, seed=0):
    rng = np.random.default_rng(seed)
    names = sorted(all_features)                    # ~20–40 个特征
    best = {f: 1.0 for f in names}
    best_val = objective(best, contexts, labels)
    # 1) 随机搜索（全局）
    for _ in range(n_iter // 2):
        w = {f: float(np.clip(rng.normal(1.0, 0.35), 0.2, 2.5)) for f in names}
        val = objective(w, contexts, labels)
        if val < best_val: best, best_val = w, val
    # 2) 坐标下降（局部精修，2 遍）
    grid = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
    for _ in range(2):
        for f in names:
            for g in grid:
                w = dict(best); w[f] = g
                val = objective(w, contexts, labels)
                if val < best_val - 1e-9: best, best_val = w, val
    return best, best_val
```

**说明**：score 对 w 是线性的，目标是非凸的分段常数 → 随机搜索 + 坐标下降足够（无需梯度）；单次评估 <1 s，300 次 <5 min。

### 4.3 LOPO-CV（必须做）

```python
def lopo_cv(contexts, labels, n_folds=None):
    tasks = sorted({c.task for c in contexts})
    reports = []
    for held in tasks:                                # leave-one-task-out
        train = {k: v for k, v in contexts.items() if k.task != held}
        test = {k: v for k, v in contexts.items() if k.task == held}
        w, _ = calibrate(train, {k: labels[k] for k in train})
        report = evaluate(w, test, labels)            # top1/top2/regret_gap
        reports.append(report)
    return mean(reports), per_fold(reports)
```

---

## 5. 验收阈值

| 指标 | 门槛 | 说明 |
|---|---|---|
| held-out top-1 命中率 | ≥ 0.60 | tie context 不计入 |
| held-out top-2 命中率 | ≥ 0.85 | |
| 相对 oracle 的 IQM regret 损失 | ≤ 5% | `(chosen − best) / (worst − best)` 的均值 |
| 与手写默认权重相比 | 不劣化 | 回归门禁 |
| 权重符号审查 | 100% | 人工 review：如"高维→bayes 加分"必须为正，否则改规则而不是拟合 |
| 权重稳定性（bootstrap） | 特征权重 95% 区间不含 0 的项进"确认"清单 | 防止过拟合 |
| 运行时间 | 300 iter × 30 context < 5 min | |

**注**：上下文数量少（<20）时 top-1 指标方差大，报告必须给 bootstrap CI，且不要把"权重文件版本"当 immutable（每次 benchmark 更新可重训，但保留旧版本可回滚）。

---

## 6. `scripts/calibrate_router.py` 设计

```
usage: py -3 scripts/calibrate_router.py \
  --bench runs/bench/frontier.jsonl \
  --out research/router-weights-2026-11-05.json \
  --report research/router-calibration-2026-11-05.md \
  --search random+cd --n-iter 300 --folds lopo --seed 0
```

内部流程（每个函数可单测）：

1. `load_bench(path)` → `records`（校验字段、去重复 run、检查每个 `(task,algo,budget,batch)` 的 seed 数）；
2. `build_contexts(records)` → `ctx_key → OptimizationContext`（用 task 的空间/目标重建，不能只用 bench 行）；
3. `make_labels(records)` → `labels`（§3.3）；
4. `calibrate(...)` → `weights`（§4.2）；
5. `lopo_cv(...)` → `validation`（§4.3）；
6. `write_weights(...)`（JSON Schema 校验）/ `write_report(...)`（Markdown 表：特征权重 + LOPO 指标 + 每任务 top-1 命中）；
7. **退出码**：验收不达标 → 非零退出（CI 可用），但在 `--allow-fail` 下只 warning。

**确定性**：固定 seed、输入排序稳定；同输入两次运行产生逐字节相同的 weights 文件（`json.dumps(..., sort_keys=True)`）。

---

## 7. 测试与实施顺序

```python
def test_scoreterm_sum_matches_old_rule():     # w=1 时 weighted sum == 旧 _score_*
def test_weights_json_schema_roundtrip():      # 写入→schema 校验→加载
def test_calibrate_improves_on_synthetic():    # 构造 3 个 context、已知最优权重，LOPO 后 top1 提升
def test_lopo_no_leakage():                    # held-out task 的 records 不参与训练
def test_router_falls_back_on_bad_weights():   # 损坏文件 → 默认权重 + warning
```

**实施顺序**：①把 `_score_*` 重构为 `list[ScoreTerm]`（纯重构，w=1 保持一致，单测保证）；②`router.py` 加载权重文件；③`benchmark` 产出 `frontier.jsonl`；④`calibrate_router.py`；⑤接入 `docs/ROUTING_POLICY.md` 更新说明。

---

## 8. 参考

R8（SMAC3 Facade：场景/能力/策略分层，规则即"先验策略"）、R17（BayBE 推荐器切换的可配置化）、R32（benchmark 统计规范：IQM/CI/多 seed）、R24（方法学：规则应有依据，权重应可解释）、`BENCHMARK_SPEC.md` §7、`CAPABILITIES_SPEC.md` §6。
