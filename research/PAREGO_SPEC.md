# PAREGO_SPEC — 多目标 ParEGO 的代码级实现规格（可直接照抄）

> 维护者：algo-research ｜ v1（2026-11-05，Round 3）
> 对应：`MULTI_OBJECTIVE_AND_NOISE.md` §1（算法原理）、`REVIEW_v01.md` P1-10（多目标静默降级）、`BENCHMARK_SPEC.md` Tier F（`moo_yield_cost`）。
> 目标：在**不引入 BoTorch** 的前提下，用现有 `GaussianProcessBO` 的 GP/EI/候选池机制实现 ParEGO，2–3 天可完成 + 单测覆盖。
> 文献：R11（Knowles 2006，增广 Tchebycheff + 随机权重）、R24（综述定位）、R2（GP/EI 复用）。

---

## 1. 前置改动（必须先做，否则多目标无法表达）

当前 `Objective`（`base.py:139-184`）只有一个 `direction`，而多目标每个目标的优化方向可能不同（收率 max、成本 min）。**先扩展 `Objective`（向后兼容）**：

```python
@dataclass
class Objective:
    name: str = "score"
    direction: str = "max"                     # 旧的单目标方向，保留
    objectives: tuple[str, ...] = ()
    # --- new ---
    directions: tuple[str, ...] = ()           # 每个目标的方向；空 = 全部用 direction
    ref_point: tuple[float, ...] = ()          # nadir：每个目标"低于此值无意义"（原始单位，含方向语义）
    ideal_point: tuple[float, ...] = ()        # 可选：每个目标的理想值；空 = 用观测最好值

    def __post_init__(self) -> None:
        if not self.objectives:
            self.objectives = (self.name,)
        if not self.directions:
            self.directions = (self.direction,) * len(self.objectives)
        if len(self.directions) != len(self.objectives):
            raise ValueError("directions 长度必须等于 objectives 长度")
        for d in self.directions:
            if d not in ("max", "min"):
                raise ValueError(f"invalid direction {d!r}")
        if self.ref_point and len(self.ref_point) != len(self.objectives):
            raise ValueError("ref_point 长度必须等于 objectives 长度")
        if self.ideal_point and len(self.ideal_point) != len(self.objectives):
            raise ValueError("ideal_point 长度必须等于 objectives 长度")

    @property
    def signs(self) -> tuple[float, ...]:
        return tuple(1.0 if d == "max" else -1.0 for d in self.directions)

    def to_dict(self) -> dict:   # 追加新字段
        ...
```

`Study.create` 的 `context_from_space(..., n_objectives=objective.n_objectives)` 已支持多目标计数；`router` 需要 `parego` 注册后才不会降级（`CAPABILITIES_SPEC.md` §5）。

**方向统一**：ParEGO 内部把所有目标转成"最大化分数"：`s_i = sign_i · y_i`（`sign_i = +1` if max else `-1`）。之后所有公式都在 `s` 上做。

---

## 2. 数据与归一化

### 2.1 数据矩阵

```python
# self._Y : np.ndarray (n_obs, m)   —— 方向统一后的"分数"（越大越好）
def tell(self, observations):
    super().tell(observations)                  # 校验 + t 计数
    if not observations:
        return
    Y_new = np.array([self._multi_scores(o) for o in observations])   # (k, m)，含 NaN 掩码
    self._Y = Y_new if self._Y is None else np.vstack([self._Y, Y_new])
    self._X = ...                                # 复用父类：features / units
```

```python
def _multi_scores(self, obs) -> np.ndarray:
    m = len(self.objective.objectives)
    vals = []
    for name, sign in zip(self.objective.objectives, self.objective.signs):
        if name in obs.objectives:
            v = obs.objectives[name]
        elif obs.y is not None and name == self.objective.name:
            v = obs.y
        else:
            return np.full(m, np.nan)            # 缺目标 → 标记，不参与标量化
        vals.append(sign * float(v))
    return np.asarray(vals, dtype=float)
```

### 2.2 归一化（**关键决策：用固定范围，不用漂移 min/max**）

```
n_i = (s_i − lo_i) / (hi_i − lo_i)          # 每维 → [0,1]，再 clip 到 [0,1]
```

- `lo_i = sign_i · ref_point_i`（用户给的"无意义阈值"，**固定**；如收率 30%、成本 20）；
- `hi_i = sign_i · ideal_point_i`（固定）或 `max(观测最好值, lo_i + ε)`（**只在首次确定后冻结**）。
- **不要每轮用观测 min/max 重新归一化**：λ 固定时 `g` 的历史值会随之改变，等于每轮换目标函数，EI 的"best"不可比、且 Pareto 归档混乱（R11 的实践也是固定窗口/固定范围）。若用户没给 `ref_point`：用"首次 10 个点的 min 再留 10–20% 余量"作为 lo，并写进 study state（可复现）。
- 退化处理：`hi_i - lo_i < 1e-12` → 该目标权重强制 0（退化目标），`m_eff` 减一，λ 在剩余维重新归一化（见 §6.2）。
- `ref_point` 在 CLI/skill 里暴露为必填（默认启发式：`lo_i = 观测最差值 − 0.1·range`）。

归一化后的矩阵记为 `Z ∈ [0,1]^{n×m}`。

---

## 3. 标量化：增广 Tchebycheff

```
g_λ(z) = max_i ( λ_i · z_i ) + ρ · Σ_i λ_i · z_i        ,  ρ = 0.05
```

- 第一项：Tchebycheff（能覆盖非凸前沿）；第二项：`ρ` 倍加权和（保证弱 Pareto 支配可区分，避免权重极端时并列）。R11 取 `ρ=0.05`。
- `g` 的值域约 `[0, 1+ρ]`，所以 EI 的 `xi=0.01`（父类默认）正好合适，不需要额外缩放。
- 权重 `λ ∈ Δ^{m-1}`（单纯形）：`λ_i ≥ 0, Σλ = 1`。

```python
def _scalarize(self, Z: np.ndarray, lam: np.ndarray, rho: float) -> np.ndarray:
    return np.max(lam[None, :] * Z, axis=1) + rho * (Z @ lam)
```

**自检**（单测）：`m=2, z=(1,0), λ=(0.5,0.5), ρ=0.05` → `g = 0.5 + 0.05*(0.5) = 0.525`。

---

## 4. 权重采样（每轮/每点）

| 策略 | 公式/参数 | 适用 |
|---|---|---|
| **默认：每点独立 Dirichlet** | `λ ~ Dirichlet(α=1_m)`，`α=1` | 批量时天然分散；实现最简单 |
| 极端权重偏好 | `α=0.5`（接近角点，强调单目标端） | 2 目标、想覆盖前沿两端 |
| 均匀偏好 | `α=2~4`（集中在中心） | 目标少且希望均衡 |
| 权重记忆（可选） | 保留最近 `K=20` 个 λ，拒绝与已有 λ 夹角 < `15°` 的重采样（最多 10 次） | 小预算下避免重复方向、减少前沿"洞" |

**批量策略**：`ask(n)` 一次采 `n` 个独立 λ（**不是同一个 λ**）。每个点用对应 λ 的标量化目标选点；这样同批点在目标空间自然分散（R11/ParEGO 的批量扩展实践）。

---

## 5. 与现有 `GaussianProcessBO` 的复用（代码骨架）

`GaussianProcessBO` 的关键内部接口（当前 `bayes.py`）：
- `_fit(X, y) -> GaussianProcessRegressor`（含 ARD/alpha/normalize_y）
- `_candidate_pool() -> (units, features)`（Sobol + 局部 + 去重）
- `_acquisition(model, features) -> np.ndarray`（EI/logEI/UCB/PI，**内部读 `self._y` 作为 best**）
- `_sample_posterior(model, features)`（TS）
- `_initial_design_left()`、`ask()` 的初始设计逻辑

**复用方式：子类 + 临时替换 `self._y`**

```python
# src/condopt/algorithms/parego.py
from .bayes import GaussianProcessBO
from ..base import AlgorithmSpec, ParamSpec, Objective
from ..registry import register
import numpy as np

@register
class ParEGO(GaussianProcessBO):
    spec = AlgorithmSpec(
        name="parego", family="model_based",
        summary="多目标 ParEGO：每轮随机权重 + 增广 Tchebycheff 标量化 + GP/EI，复用单目标内核。",
        when="2–3 个目标、预算 20–200、每次实验昂贵、需要覆盖 Pareto 前沿时。",
        params=GaussianProcessBO.spec.params + (
            ParamSpec("rho", "float", 0.05, low=0.0, high=0.5,
                      description="增广 Tchebycheff 的加权和系数"),
            ParamSpec("weight_alpha", "float", 1.0, low=0.1, high=10.0,
                      description="Dirichlet 浓度；<1 偏极端权重，>1 偏均匀"),
            ParamSpec("weight_memory", "int", 20, low=0, high=200,
                      description="保留最近多少个 λ 用于去重；0 表示不去重"),
            ParamSpec("min_weight_angle_deg", "float", 15.0, low=0.0, high=90.0,
                      description="重采样 λ 的最小夹角（度）"),
        ),
        capabilities=frozenset({"batch", "mixed", "categorical", "continuous",
                                 "noisy", "expensive", "model_based", "multi_objective"}),
        references=(
            "Knowles, ParEGO: A Hybrid Algorithm with On-Line Landscape Approximation "
            "for Expensive Multiobjective Optimization Problems, IEEE TEC 2006",
            "Shahriari et al., Taking the Human Out of the Loop, Proc. IEEE 2016",
        ),
    )

    def __init__(self, *args, **kwargs):
        self._Z: np.ndarray | None = None         # 归一化后的 (n, m)
        self._Y: np.ndarray | None = None         # 方向统一后的原始 (n, m)
        self._lam_history: list[np.ndarray] = []
        self._normalizer = None                   # (lo, hi) 固定窗口
        super().__init__(*args, **kwargs)

    # ---- data ---------------------------------------------------------
    def tell(self, observations):
        super().tell(observations)                # 父类：校验/t/X/units/y（多目标时 y 无意义）
        if not observations:
            return
        Y_new = np.array([self._multi_scores(o) for o in observations])
        self._Y = Y_new if self._Y is None else np.vstack([self._Y, Y_new])
        self._maybe_init_normalizer()
        self._Z = self._normalize(self._Y)

    def _maybe_init_normalizer(self):
        if self._normalizer is not None:
            return
        m = len(self.objective.objectives)
        if self.objective.ref_point:
            lo = np.asarray(self.objective.ref_point, float) * np.asarray(self.objective.signs)
            hi = (np.asarray(self.objective.ideal_point, float) * np.asarray(self.objective.signs)
                  if self.objective.ideal_point else self._Y.max(axis=0) + 1e-9)
        else:                                     # 无 ref：用首次数据 + 余量，之后冻结
            lo = self._Y.min(axis=0)
            hi = self._Y.max(axis=0)
            hi = np.where(hi - lo < 1e-9, lo + 1.0, hi)   # 退化目标给单位宽度
        lo, hi = lo - 1e-9, hi + 1e-9
        self._normalizer = (lo, hi)

    def _normalize(self, Y):
        lo, hi = self._normalizer
        Z = (Y - lo) / (hi - lo)
        return np.clip(Z, 0.0, 1.0)

    # ---- propose ------------------------------------------------------
    def ask(self, n=1):
        if n <= 0:
            return []
        out = []
        design_left = self._initial_design_left()
        if design_left > 0:
            k = min(n, design_left)
            out.extend(self.space.sample_sobol(k, rng=self.rng))
            n -= k
        if n == 0:
            return out
        if self._Z is None or len(self._Z) < 2:
            out.extend(self.space.sample_sobol(n, rng=self.rng))
            return out
        candidates, features = self._candidate_pool()       # 父类：Sobol+局部+去重
        if len(candidates) == 0:
            out.extend(self.space.sample_sobol(n, rng=self.rng))
            return out
        chosen_units = []
        for _ in range(n):
            lam = self._sample_weight()
            g = self._scalarize(self._Z, lam, float(self.params["rho"]))
            gp = self._fit(self._X, g)                      # 复用父类 GP（ARD/alpha/normalize_y）
            self._y = g                                     # ★ 父类 _acquisition 用 self._y 作为 best
            acq = self._acquisition(gp, features)
            if chosen_units:                                # 批内局部惩罚（见 SPRINT2_SPEC）
                acq = acq * self._local_penalty(candidates, chosen_units)
            pick = int(np.argmax(acq))
            chosen_units.append(candidates[pick])
            out.append(self.space.decode(candidates[pick]))
            self._lam_history.append(lam)
        return out

    def _sample_weight(self):
        m = len(self.objective.objectives)
        alpha = float(self.params["weight_alpha"])
        for _ in range(10):
            lam = self.rng.dirichlet(np.full(m, alpha))
            if not self._lam_history or self.params["weight_memory"] <= 0:
                return lam
            recent = np.asarray(self._lam_history[-int(self.params["weight_memory"]):])
            cos = (recent @ lam) / (np.linalg.norm(recent, axis=1) * np.linalg.norm(lam) + 1e-12)
            if np.degrees(np.arccos(np.clip(cos.max(), -1, 1))) >= float(self.params["min_weight_angle_deg"]):
                return lam
        return lam
```

**必须注意的复用坑**

1. `self._y` 在父类里既是"GP 训练目标"又是 `_acquisition` 的 `best` 来源；ParEGO 每次换 λ 都要重设 `self._y = g` **再**调 `_acquisition`。`tell()` 之后 `self._y` 是主目标的 score（父类行为），不影响（下次 ask 会覆盖）。
2. `_fit` 里 `normalize_y=True` → 标量化的尺度自动处理，但 `logei` 的 `xi` 是标量化单位（g∈[0,1+ρ]），默认 0.01 正好，不要用原始目标单位。
3. `_candidate_pool` 的 `local` 中心用 `np.argsort(-self._y)`（父类）→ ParEGO 里 `self._y` 是上一次的 `g`，**会误导局部池**。建议在 ParEGO 里覆写 `_candidate_pool` 或先把 `self._y` 设成"当前 λ 下的 g"再调 pool（与上面代码顺序一致：先 `_scalarize` → `self._y = g` → `_candidate_pool()`）。**若沿用父类顺序（pool 在循环外）**，需接受局部中心偏差，或用"当前 Pareto 归档点"作中心。
4. `_thompson_batch` 在 ParEGO 里不适用（TS 对多目标需要标量化后 TS；`acquisition="ts"` 时按 λ 标量化 + 父类 TS 即可）。

---

## 6. 边缘情况（逐条给处理）

### 6.1 单目标退化（m=1）
- `n=1` 时 ParEGO 等价于普通 BO；不要重复实现：`if m == 1: return super().ask(n)`；单测断言两者在相同 seed 下轨迹一致。

### 6.2 目标同质化（强相关 / 完全相同）
- 检测：用 `Z` 的 Pearson 相关矩阵，若 `|corr| > 0.95` 的 pair 覆盖所有目标 → 警告 + 建议退化为单目标（或按用户意图只保留主目标）。
- 完全相同（两列差值 <1e-9）：该目标在归一化后权重自动为 0（见 §2.2）。
- 不要静默：写进 `state_dict()` 的 `warnings`，`route`/`report` 展示。

### 6.3 目标尺度退化（hi==lo）
- 该维 `λ_i=0`，在剩余维重新归一化 λ；若全部退化（所有目标常数）→ 抛 `ValueError`（无优化意义）。

### 6.4 离散 / 混合空间
- ParEGO 与空间编码无关：候选池/decode 由父类处理（见 `space.py` features/encode）。权重只作用在目标空间，不影响空间层。
- 纯离散空间 `cardinality` 小（≤200）时优先 `grid`（router 已有规则），ParEGO 在 `card > budget` 时才有价值。

### 6.5 噪声
- 目标值是重复测量均值（`Observation` 记录 `std`/重复次数）；`alpha` 用 `mean(σ_rep²)/y_std²`（见 `CONSTRAINTS_NOISE_IMPL.md` §3）。
- 每个 λ 的标量化值噪声 = `Σ (∂g/∂z_i)² σ_i²` 的近似；简单做法：直接用父类 alpha，并在报告里标注"ParEGO 未显式建模噪声，前沿点有不确定性"。
- 后续升级：`qNEHVI` 或"每目标 qNEI + ParEGO 标量化"（`MULTI_OBJECTIVE_AND_NOISE.md` §3）。

### 6.6 缺失目标值（失败实验）
- `_multi_scores` 返回 NaN 的观测：从 `Z` 中剔除（保持 `X` 与 `Z` 同索引，剔除要同时剔除 `_X/_units`），并在 `decisions` 记录 `dropped_incomplete`。
- 若某目标系统性缺失（>30%），拒绝运行并提示用户。

### 6.7 参考点设置错误
- `ref_point` 必须"严格差于所有观测"：若 `lo_i ≥ min(s_i)` → 归一化会 clip 到 0，警告并按 `lo_i = min(s_i) - 0.1*range` 调整（记录到 state）。
- `ideal_point` 必须 `hi_i > lo_i`，否则视为未提供。

### 6.8 预算极小
- `n < m+2` 时无法估计前沿：直接返回 Sobol/LHS 点（与父类一致），警告"预算不足以形成 Pareto 前沿"。
- `n_initial` 建议 `max(2*(d_feat+1), 2*m)`，但受预算 ≤40% 限制。

---

## 7. 参数表（可直接写进 `ParamSpec` 默认值）

| 参数 | 默认 | 范围 | 说明 |
|---|---|---|---|
| `rho` | 0.05 | 0–0.5 | 增广项；R11 经典值 |
| `weight_alpha` | 1.0 | 0.1–10 | Dirichlet 浓度 |
| `weight_memory` | 20 | 0–200 | λ 去重窗口 |
| `min_weight_angle_deg` | 15 | 0–90 | 重采样最小夹角 |
| `n_initial` | 0（auto） | 0–200 | auto = `max(3, min(feature_dim+1, 20))`（父类） |
| `n_candidates` | 1024 | 64–65536 | 父类候选池 |
| `acquisition` | `logei` | ei/logei/ucb | 标量化后单目标 EI |
| `alpha` | 1e-4 | 1e-12–1.0 | 复用父类；有重复测量时按 §6.5 |
| `batch_strategy` | `local_penalty`（新） | greedy/believer/CL/local_penalty | 批量见 `SPRINT2_SPEC.md` |

---

## 8. 报告与归档（多目标专用，先做最小版）

- `Study.report()` 对多目标：输出"当前 Pareto 归档"（非支配集，最多 10 个点 + 目标值），而不是单一 best。
- `Study.summary()`：`best` 字段对多目标改为 `pareto: [...]`，同时保留 `best_by_primary`（主目标最优）以兼容旧 CLI。
- 归档去重：按归一化目标空间的 ε-支配（`ε=0.01`）合并近似重复点。
- 概率 Pareto（噪声/推荐用）见 `MULTI_OBJECTIVE_AND_NOISE.md` §5，v0.2 再做。

---

## 9. 验收测试（PR 必须带）

```python
def test_scalarization_hand_computed():      # §3 自检
def test_weights_on_simplex():               # ask 时 λ 满足 Σ=1, λ≥0
def test_normalization_frozen():             # 第二轮 tell 不改变已有 Z 的 lo/hi
def test_parago_single_objective_matches_bo():   # m=1 时与 bayes 同 seed 同轨迹
def test_parago_batch_distinct():            # batch=4 的 4 个点互不相同且 λ 分散
def test_parago_ref_point_required():        # 无 ref_point 时使用冻结 min/max 并写 state
def test_parego_on_moo_yield_cost():         # HV_ratio ≥ 0.85（budget=32, seeds=5）
def test_degenerate_objective_weight_zero(): # 常数目标的 λ_i == 0
def test_incomplete_observations_dropped():  # NaN 目标被剔除且记录
```

**benchmark 门槛**：`BENCHMARK_SPEC.md` Tier F `moo_yield_cost`（参考前沿 101 点、端点 `(109.0, 11.5)` / `(45.77, 9.0)`）上，`HV_ratio ≥ 0.85`、前沿覆盖 ≥60%（10×10 网格）。

---

## 10. 参考

R11（ParEGO 原文：增广 Tchebycheff、ρ=0.05、随机权重）、R24（多目标 BO 综述定位）、R2（GP/EI/ARD 复用）、R12/R21（后续 qEHVI/qNEHVI 升级路线）、R29（标量化后 EI 用 log-EI）。
