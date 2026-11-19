# SPRINT2_SPEC — ARD / 局部惩罚 / TS 批量 / Sobol 2^k 的实现要点

> 维护者：algo-research ｜ v1（2026-11-05，Round 3）
> 对应：`REVIEW_v01.md` P1-1（ARD）、P1-3（constant-liar）、P1-4（TS 批量）、P1-7（Sobol）。文献：R2（ARD/GP 工程）、R15（高维 lengthscale 先验）、R35（local penalization）、R36（Sobol 生成器/2^k）、R25（TS）。
> 每个子项给：**公式 → 参数 → 代码骨架 → 测试 → 验收**。按 §1→§4 顺序交付（互不依赖）。

---

## 1. ARD（每特征一个 length_scale）

### 1.1 现状问题

`_make_kernel` 用标量 `Matern(length_scale=0.3)` → 各向同性；`feature_dim`（数值维 + one-hot 展开）的所有维度共享一个尺度。后果：categorical 一列决定全局距离、高基数 one-hot 支配核、无法表达"哪些变量相关"。

### 1.2 公式与初始化

```
k(x,x') = σ_f² · Matern_ν( Σ_i (x_i − x_i')² / ℓ_i² )
```

**lengthscale 初值**（按 R15 的维度缩放先验，中位数）：

```
ℓ_i(0) = exp( √2 + 0.5·ln(d) )     , d = feature_dim > 8
ℓ_i(0) = 0.3                        , d ≤ 8（低维用短尺度，更局部）
```

**边界**：每维 `[1e-2, 1e2]`（绝对值，**不要**由 `length_scale` 参数推导——见 REVIEW P2-1）。`d>20` 时上界可放到 `1e3`（让"无关维"退化为平坦）。

**先验**（可选，sklearn 不支持核超参先验，只能靠初值 + 边界；若要真先验用 MAP 实现或换 BoTorch）：至少把初值设成上面公式，并在 `n < 2d` 时 `optimizer=None` 冻结。

### 1.3 代码骨架

```python
def _make_kernel(kind: str, feature_dim: int, ls0: float = 0.3):
    d = max(1, int(feature_dim))
    ls_init = 0.3 if d <= 8 else float(np.exp(0.5 * np.log(d) + 0.5))
    ls = np.full(d, ls_init if ls0 is None else ls0)
    bounds = np.tile(np.array([1e-2, 1e2]), (d, 1))   # shape (d, 2)
    if kind == "matern52":
        base = Matern(length_scale=ls, length_scale_bounds=bounds, nu=2.5)
    elif kind == "matern32":
        base = Matern(length_scale=ls, length_scale_bounds=bounds, nu=1.5)
    else:
        base = RBF(length_scale=ls, length_scale_bounds=bounds)
    return ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3)) * base
```

冻结规则（加到 `_fit`）：

```python
freeze = len(y) < 2 * self.space.feature_dim
model = GaussianProcessRegressor(
    kernel=kernel, alpha=alpha, normalize_y=True,
    optimizer=None if freeze else "fmin_l_bfgs_b",
    n_restarts_optimizer=0 if freeze else 3,
    random_state=self.seed,
)
```

### 1.4 诊断（必须做，否则 ARD 出问题不可见）

- `model.kernel_.k2.length_scale` 是数组 → 报告"撞边界比例"：
  `frac_at_bound = mean(ls <= 1.1*lo or ls >= 0.9*hi)`；
- **把 lengthscale 映射回变量名**（feature 布局 = 数值变量依次 + 每个离散变量的 one-hot 块）：

```python
def feature_slices(space):
    out, i = [], 0
    for v in space.variables:
        if v.is_numeric:
            out.append((v.name, i, i + 1)); i += 1
        else:
            k = len(v.choices); out.append((v.name, i, i + k)); i += k
    return out   # [(name, start, end), ...]
```
- 若某离散变量的 one-hot 块 lengthscale 都很小 → 该变量对目标影响大；都很大 → 影响小（可考虑固定/剔除）。
- 这些统计写进 `Study.decisions` 的 payload，router 可据此建议"降维/换 TPE"。

### 1.5 测试与验收

```python
def test_ard_has_one_lengthscale_per_feature():   # len(ls) == space.feature_dim
def test_ard_frozen_when_few_observations():      # optimizer is None
def test_ard_recovers_relevant_dim():             # 合成：y 只依赖 x1 → 其余 ℓ 远大于 ℓ1（或撞上界）
def test_ard_improves_mixed_reaction():           # 3 seeds 平均 best-so-far 不劣于标量核
```
验收：`mixed_reaction` demos、`suzuki_like`（`BENCHMARK_TIER_CD.md`）上 ARD 的 3-seed 平均 best-so-far ≥ 现版本；lengthscale 诊断输出可在 `report` 里看到。

---

## 2. 局部惩罚（local penalization）替代 constant-liar

### 2.1 论文公式与工程简化

R35 的惩罚（对每个已选点 `x_j`）：

```
φ_j(x) = ½ · erfc( (L·‖x − x_j‖ − M_j) / (√2 · s_j) )
acq_batch(x) = acq(x) · Π_j φ_j(x)
```

- `L`：目标函数的 Lipschitz 常数估计（见 §2.2）；
- `M_j`：与"当前最优值 M 和 x_j 处预测 μ_j 的差"相关的阈值（论文用 `M_j = M − m_j` 形式，`m_j` 为局部下界估计）；
- `s_j`：`x_j` 处的后验标准差。

**工程简化版**（推荐先实现，足够好、无 Lipschitz 估计误差）：

```
penalty_j(x) = 1 − exp( − ‖x − x_j‖² / (2·ℓ_pen²) )
acq_batch(x)  = acq(x) · Π_j penalty_j(x)
```

- `penalty_j(x_j)=0`（完全抑制重复点）、距离 ≫ ℓ_pen 时 →1；
- `ℓ_pen`：取 ARD lengthscale 的**中位数 × 0.5**，并 clip 到 `[0.03, 0.25]`（单位立方体）；
- 比 constant-liar 好：**不伪造数据**（不压后验方差、不引入假观测），批量点分散性可控。

### 2.2 Lipschitz 常数估计（若要用论文原式）

```python
def estimate_lipschitz(units, y, model=None):
    # 1) 数据法：相邻点值差 / 距离
    d = np.linalg.norm(units[:, None, :] - units[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    nn = d.min(axis=1)
    dy = y[:, None] - y[None, :]
    L_data = np.max(np.abs(dy) / np.maximum(d, 1e-9))
    # 2) 后验梯度法（更稳）：在候选池上有限差分取 ||∂μ/∂x|| 的最大值
    return float(np.clip(L_data, 1e-3, 1e3))
```
实践中取 `L = 2 ×` 数据法估计（保守），或直接用简化版（推荐）。

### 2.3 批量选点伪代码（q≥4 默认）

```python
def _model_ask_local_penalty(self, n):
    candidates, features = self._candidate_pool()
    model = self._fit(self._X, self._y)
    chosen_units, out = [], []
    for i in range(n):
        acq = self._acquisition(model, features)
        if chosen_units:
            d2 = np.sum((candidates[:, None, :] - np.asarray(chosen_units)[None, :, :]) ** 2, axis=-1)
            penalty = np.prod(1.0 - np.exp(-d2 / (2.0 * self._l_pen ** 2)), axis=1)
            acq = acq * penalty
        pick = int(np.argmax(acq))
        chosen_units.append(candidates[pick]); out.append(self.space.decode(candidates[pick]))
        # 不重新拟合 GP（省时间）；若要更强，可用 kriging believer 更新一次
    return out
```

`_l_pen` 计算：

```python
if model.kernel_.k2.length_scale is scalar:
    ls = float(model.kernel_.k2.length_scale)
else:
    ls = float(np.median(model.kernel_.k2.length_scale))
self._l_pen = float(np.clip(0.5 * ls, 0.03, 0.25))
```

### 2.4 选型规则（写进 router/参数策略）

| batch | 策略 |
|---|---|
| 1 | 单点 EI/logEI |
| 2–3 | kriging believer（质量好、成本低） |
| ≥4 | **local penalization**（默认）；q≥12 且算力低 → TS |
| 噪声高 | qNEI/qNEI+penalty（`CONSTRAINTS_NOISE_IMPL.md`） |

### 2.5 测试与验收

```python
def test_local_penalty_batch_distinct():        # batch=8 两两距离 ≥ 0.05*range
def test_local_penalty_penalty_zero_at_selected():  # penalty(x_j) ≈ 0
def test_local_penalty_vs_constant_liar():      # mixed_reaction budget=32, batch=8, 5 seeds：LP 不劣于 CL
```
验收：`mixed_reaction_v2` 与 `suzuki_like` 上 batch=8 的 IQM best-so-far ≥ CL；`noisy_sphere`（σ=0.05）上 LP 不劣于 greedy。

---

## 3. TS 批量：独立抽样

### 3.1 问题

当前 `_thompson_batch` 用**同一条**后验样本取 top-n → 批量点高度相关（同一函数在相邻点取值接近）。正确做法：**每个 batch 点独立抽一条后验函数**再取 argmax（R25 的并行 TS 语义）。

### 3.2 高效实现（一次 Cholesky 抽 n 条）

```python
def _sample_posterior_multi(self, model, features, n_draws):
    mu, cov = model.predict(features, return_cov=True)
    cov = 0.5 * (cov + cov.T) + _JITTER * np.eye(len(features))
    try:
        chol = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eigh(cov)
        chol = vecs @ np.diag(np.sqrt(np.clip(vals, 0.0, None)))
    Z = self.rng.standard_normal((len(features), n_draws))
    return mu[:, None] + chol @ Z          # (m, n_draws)

def _thompson_batch(self, model, candidates, features, n):
    draws = self._sample_posterior_multi(model, features, n)   # (m, n)
    picks = []
    for i in range(n):
        col = draws[:, i].copy()
        if picks:
            col[picks] = -np.inf
        picks.append(int(np.argmax(col)))
    return [self.space.decode(candidates[i]) for i in picks]
```

（`features` 需与 `candidates` 对齐；当前调用处已构造。）

### 3.3 参数与注意

- `n_draws = n`（batch 大小）；m=1024 候选时一次 `1024×1024` Cholesky + `1024×n` 抽样，CPU 上 <0.2 s；
- 若 `n` 很大（>16）且算力低 → 退化为"抽 1 条 + 局部惩罚"；
- **可复现**：抽样使用 `self.rng`（Study 每轮推进 RNG，见已修 P1-8），`state_dict` 已持久化 rng；
- TS 不需要 `_acquisition`，但仍要 `self._y` 只用于 candidate pool 的局部中心（可接受）。

### 3.4 测试与验收

```python
def test_ts_batch_independent_draws():   # 两次 ask 的 batch 内点距离分布合理（不退化到 1 个簇）
def test_ts_improves_over_mean_argmax(): # 合成 2D 上 ts 的 best-so-far ≥ "mu argmax"
```
验收：`sphere2d`/`branin` 上 `ts` 与 `ei` 的 3-seed 平均差距 <10%（当前实现 `ts`≈贪心，修好后应有探索性）。

---

## 4. Sobol 2^k 初始设计

### 4.1 为什么是 2^k

`scipy.stats.qmc.Sobol.random(n)` 只在 `n` 为 2 的幂时保持"每层均匀"的平衡性（否则 scipy 报 warning 且平衡性变差）。**增量调用**（每轮 ask 一点点）也会破坏序列性质。正确做法：**一次性预生成 2^k 个点，按需切片**（R36）。

### 4.2 实现要点

```python
import math
from scipy.stats import qmc

def sobol_design(space, n_total, seed):
    k = max(1, int(math.ceil(math.log2(max(n_total, 2)))))
    units = qmc.Sobol(d=space.dim, scramble=True, seed=seed).random_base2(k)  # (2^k, d)
    return units[:n_total]      # 取前缀；2^k 集合是平衡的，前缀也足够低差异

# 在 ask 中：
if self._init_queue is None:
    n0 = int(self.params["n_initial"]) or max(5, min(20, 2 * (self.space.feature_dim + 1)))
    n0 = min(n0, max(1, int(0.4 * remaining_budget)))
    self._init_queue = [self.space.decode(u) for u in sobol_design(self.space, n0, self.seed)]
    # 过滤掉已在 prior 中测过的点（暖启动场景）
    seen = {self._key(u) for u in self._units}
    self._init_queue = [c for c in self._init_queue if self._key(self.space.encode(c)) not in seen]
k = min(n, len(self._init_queue))
out = self._init_queue[:k]; self._init_queue = self._init_queue[k:]
```

**参数**

| 参数 | 值 | 说明 |
|---|---|---|
| `n_initial` | `max(5, min(20, 2*(feature_dim+1)))` | 被 P0-6 修正后的口径；高基数 categorical 空间随 feature_dim 增大 |
| 预算上限 | `min(n0, 0.4*budget)` | 初始设计不超 40% 预算 |
| 2^k 取整 | `k = ceil(log2(n0))` | 生成后取前缀 |
| `scramble` | True | 随机化 → 无偏、可复现（固定 seed） |

### 4.3 `state_dict` 持久化

```python
# state_dict: {"init_queue": self._init_queue}；load 时恢复
# 这样进程重启/重载 study 不会从头再生成（避免重复点）
```

### 4.4 测试与验收

```python
def test_sobol_design_power_of_two_balance():  # n_total=8，每维每层恰 1 个点
def test_sobol_ask_returns_requested_n():       # 两次 ask(4) 合计 8 点，不重复
def test_sobol_initial_design_respects_prior():# 暖启动时跳过已测点
def test_sobol_design_reproducible():          # 同 seed 两次生成一致
```
验收：`branin` budget=16 时，用 2^k 初始设计的 `bayes` 的 5-seed 平均 best-so-far ≥ 旧实现（预期小幅提升；关键是不再出现"初始点不是 2^k 导致覆盖不均"）。

---

## 5. 交付顺序（建议 PR 拆分）

1. **PR-A**（1 天）：ARD + 诊断（§1）——影响所有后续算法，先做。
2. **PR-B**（半天）：local penalization（§2）+ `batch_strategy="local_penalty"` 默认 q≥4。
3. **PR-C**（半天）：TS 独立抽样（§3）。
4. **PR-D**（半天）：Sobol 2^k 初始设计（§4）+ state 持久化。
5. 每个 PR 带对应单测；全部完成后跑 `BENCHMARK_TIER_CD.md` smoke（`mixed_reaction_v2`、`noisy_sphere`、`discrete_catalyst_v2`）回归。

---

## 6. 参考

R2（ARD 与 GP 工程默认）、R15（高维 lengthscale 先验）、R35（local penalization 原式与 Lipschitz 常数）、R36（Sobol 生成器与 2^k）、R25（TS 与并行采样）、R3（采集函数/批量框架）。
