# REPLICATES_ALPHA — 复孔/重复测量如何映射到观测与 GP alpha

> 维护者：algo-research ｜ v1（2026-11-05，Round 5）
> 对应：`CONSTRAINTS_NOISE_IMPL.md` §4（Round 3 原理）、`bayes.py::_fit`（v0.4，alpha 仍固定为参数）、`dataset.py::Observation`（无 std/replicates 字段）。
> 文献：R2（Snoek：噪声/重复测量处理）、R10（带噪约束）、R29（log-EI：噪声下采集函数的稳定性）、R24（噪声 BO 的方法学位置）。
> 结论：**v0.4 尚未实现复孔**——`bayes.py` 只有固定 `alpha` 参数（`ParamSpec("alpha", "float", 1e-6, ...)`），`Observation` 没有 `std` 字段。本文给最小补丁。

---

## 1. 数据 schema

### 1.1 `Observation`（`dataset.py:23-113`）

```python
@dataclass
class Observation:
    ...
    fidelity: float = 1.0
    cost: float = 0.0
    # ---- 新增（全部有默认值，旧 JSONL 可读）----
    objective_std: dict[str, float] = field(default_factory=dict)   # 每个目标的重复测量标准差
    constraint_std: dict[str, float] = field(default_factory=dict)  # 每个约束的重复测量标准差
    n_replicates: int = 1                                            # 该观测由 k 次重复取均值得来
```

- `to_dict()`：仅当非空/`>1` 时写出（保持旧 JSONL 简洁）；
- `from_dict()`：缺省 `{}` / `1`；
- 校验：`n_replicates >= 1`；std 非负；`std` 的 key 必须是该 study 的目标/约束名（在 `Study.tell` 里校验，`Observation` 本身不依赖 Objective）。

### 1.2 `results.json` 兼容（`study.py::_to_observation`，v0.4 在 ~line 450-490）

支持三种写法（全部归一化成 `Observation`）：

```jsonc
// A) 旧：单值
{"id": "r1-1", "y": 93.2}
// B) 新：单目标 + 复孔
{"id": "r1-1", "y": 93.2, "y_std": 0.6, "n_replicates": 3}
// C) 新：多目标/多约束 + 复孔
{"id": "r1-1", "objectives": {"yield": 93.2, "cost": 11.4},
 "objective_std": {"yield": 0.6, "cost": 0.1},
 "constraints": {"purity": 95.4}, "constraint_std": {"purity": 0.3},
 "n_replicates": 3}
```

解析规则：

```python
def _parse_replicates(payload):
    n = int(payload.get("n_replicates", 1))
    obj_std = dict(payload.get("objective_std", {}))
    con_std = dict(payload.get("constraint_std", {}))
    if "y_std" in payload and payload.get("y") is not None:
        name = self.objective.name
        obj_std.setdefault(name, float(payload["y_std"]))
    # 允许 std 以 "value±std" 字符串给出（人工录入友好）
    for key in list(payload.get("constraints", {}) or {}):
        val = payload["constraints"][key]
        if isinstance(val, str) and "±" in val:
            mean, std = val.split("±")
            payload["constraints"][key] = float(mean)
            con_std.setdefault(key, float(std))
    return n, obj_std, con_std
```

---

## 2. alpha 的数学

### 2.1 观测模型

```
单次测量: y_ij = f(x_i) + ε_ij,   ε ~ N(0, σ_rep²(x_i))
k 次重复取均值: ȳ_i = f(x_i) + ε̄,  Var(ε̄) = σ_rep² / k
```

sklearn `GaussianProcessRegressor(alpha=α, normalize_y=True)` 把 `α` 加到**标准化后**的 K 对角线上：

```
α = Var(ε̄) / y_std² = (σ_rep² / k) / y_std²
```

其中 `y_std` 是**训练目标的标准差**（sklearn 内部用 `_y_train_std`）。注意：

1. 若直接用 **k 次重复的原始值**训练（不聚合），等价于把同一 x 复制 k 次 → 后验方差被错误压缩；**必须聚合**。
2. 若目标取 log（乘性噪声）：`α_log ≈ ln(1 + CV²) ≈ CV²`（CV<0.3）；`CV = σ_rep / |y|`。
3. 多目标：每个目标的 `y_std` 不同 → **每个目标单独算 α**（ParEGO 标量化后另算，见 §2.3）。

### 2.2 有/无重复的数据

```python
def estimate_alpha(observations, y_std, default):
    """返回 (alpha, source)。source 写进 decisions 便于审计。"""
    variances = []
    for obs in observations:
        if obs.n_replicates > 1 and obs.objective_std:
            # 目标名与 objective.name 对应（多目标时由调用方按目标分别调用）
            for name, std in obs.objective_std.items():
                variances.append((float(std) ** 2) / obs.n_replicates)
    if not variances:
        return float(default), "default"
    return float(np.mean(variances)) / (float(y_std) ** 2 + 1e-12), "replicates"
```

- 无复孔：用 router 的噪声档位默认（`none=1e-8, low=1e-6, medium=5e-3, high=3e-2`），与 `REVIEW_v01` P1-5 一致；
- 有复孔但 `std` 缺失：用 `default` 并在 report 里提示"复孔缺少 std"；
- `y_std` 的取法：`np.std(y_agg)`（聚合后的训练目标），在 `_fit` 里即可拿到。

### 2.3 约束与 ParEGO

- **约束 GP**：`alpha_c = (σ_c²/k)/c_std²`，`c_std = np.std(c_values)`；带噪约束（`Constraint.noisy=True`）时**必须**用这个（否则 `P(feasible)` 过度自信，见 `CONSTRAINT_IMPL.md` §2）。
- **ParEGO**：标量化 `g_λ` 的噪声方差近似 `Var(g) ≈ Σ (∂g/∂z_i)² · Var(z_i)`；MVP 可用"各目标 α 的最大值"或"s_i 的加权平均"，并在 report 标注近似。

---

## 3. 接入点（v0.4 代码位置）

### 3.1 `bayes.py::_fit`（v0.4 约 line 409-430）

```python
def _fit(self, X, y):
    ...
    model = GaussianProcessRegressor(
        kernel=kernel,
        alpha=float(self.params["alpha"]),     # ← 当前：固定参数
        normalize_y=True,
        ...
    )
```

改为：

```python
def _fit(self, X, y):
    alpha = self._alpha_for(y)                 # 新增
    model = GaussianProcessRegressor(kernel=kernel, alpha=alpha, normalize_y=True, ...)

def _alpha_for(self, y: np.ndarray) -> float:
    """复孔优先；无复孔用参数默认。结果缓存到 self._alpha_info 供 report/decisions。"""
    y_std = float(np.std(y)) if len(y) > 1 else 1.0
    alpha, source = estimate_alpha(self._observations(), y_std, self.params["alpha"])
    self._alpha_info = {"alpha": alpha, "source": source, "y_std": y_std}
    return alpha
```

- `self._observations()`：`GaussianProcessBO` 目前只存 `_X/_y/_units`（`bayes.py:172-189`），**需要把原始 `Observation` 列表也存下来**（或存 `objective_std/n_replicates` 两个数组）——最小改动：`tell` 里 `self._rep_var.append(...)`，`_fit` 用它。
- **`_fit` 的调用点**：`_model_ask`（主 GP）、`constrained.py:147-148`（目标 + 每个约束 GP，需按约束的 std 传不同 alpha → 给 `_fit` 加可选参数 `alpha=None, rep_var=None`）。
- **约束 GP 的 alpha**：`constrained._fit(self._X, self._C[:, j])` 目前复用目标 alpha → 改为每个约束单独估计。

### 3.2 采集函数的连带效应

- `logei`（已实现渐近式）在噪声下的表现依赖正确的 α；α 偏小 → GP 过拟合噪声 → EI 把幸运观测当真实改进（`REVIEW_v01` P1-5 的实测依据）。
- 高噪声场景推荐同时把 `acquisition="logei"` + 正确 α；**不要**用固定 κ 的 UCB 代替（R23）。
- 噪声多目标未来用 qNEHVI（R12）。

### 3.3 `Study.tell` 与报告

| 位置 | 改动 |
|---|---|
| `study.py::_to_observation` | 解析 `objective_std/constraint_std/n_replicates/y_std`（§1.2） |
| `study.py::tell` | 校验 std 的 key ∈ objective.objectives/constraints；`n_replicates` 与 std 同时出现 |
| `Study.report` | 显示"平均复孔数 k、alpha 估计值/来源"；若 30% 以上观测缺 std → warning |
| `Study.decisions` | 每次 ask 写 `alpha_info`（alpha/source/y_std） |
| `benchmarks.py` | 噪声问题的 evaluator 生成 `n_replicates` 与 `objective_std`（Tier D 已定义） |

### 3.4 `Observation` 之外要不要改 `Dataset`？

不需要：`Dataset` 只是容器；`scores()` 仍按 `y` 取。若要"按 std 加权平均同 x 观测"，属于 v0.6 的高级功能（重复点合并），MVP 不做。

---

## 4. 验收

```python
def test_observation_replicate_roundtrip():       # to_dict/from_dict 保留 std/n
def test_results_parses_y_std_and_pm():           # "95.4±0.3" 与 y_std 两种写法人
def test_alpha_from_replicates():                 # k=3, std=0.6, y_std=2.0 → α=(0.36/3)/4=0.03
def test_alpha_matches_when_no_replicates():      # 无复孔 → 返回参数默认 + source=default
def test_alpha_improves_noisy_sphere():           # noise_std=0.15, k=3 时 best_so_far 优于 k=1 固定 alpha=1e-4（3 seeds）
def test_constraint_gp_uses_constraint_alpha():   # 带噪约束的 P(feasible) 不塌缩到 0/1
def test_study_report_shows_alpha_info():         # report/decisions 含 alpha 与来源
```

**benchmark 门槛**（`BENCHMARK_TIER_CD.md`）：`noisy_sphere` σ=0.05/0.15，`replicates=1/3`；正确 α 映射后，`k=3` 版本在 budget=20 的 `simple_regret` 比 `k=1`（同样总测量次数）低 ≥10%，且跨 seed 方差下降。

---

## 5. 实施顺序（1–2 天）

1. `Observation` 三字段 + `to_dict/from_dict` + 单测（半天）；
2. `Study._to_observation` 解析 + 校验 + 单测（1–2 h）；
3. `bayes.tell` 存 rep_var；`bayes._fit` 用 `_alpha_for`；`constrained._fit` 支持 per-constraint alpha（半天）；
4. benchmark `noisy_sphere` 的复孔变体 + 验收（半天）；
5. report/decisions 透出 alpha_info（1 h）。

---

## 6. 参考

R2（重复测量/噪声处理）、R10（带噪约束与观测噪声）、R29（log-EI 在高噪声下的稳定实现）、R24（噪声 BO 的方法学位置）、R31（采集函数优化在高噪声下的表现）。
