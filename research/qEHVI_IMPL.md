# qEHVI_IMPL — 2D EHVI/qEHVI 的逐行实现规格（可照抄）

> 维护者：algo-research ｜ v1（2026-11-05，Round 5）
> 对应：`qEHVI_SPEC.md`（Round 4 概览）、`pareto.py`（signed-score + `hypervolume_2d`）、`REVIEW_v03.md`（P0：ref point 必须固定）。
> 文献：R21（qEHVI：可微 EHVI + 分区）、R42（HV 与固定参考点）、R12（qNEHVI：噪声版，本文不做）、R11（ParEGO 对照）、R35（批量局部惩罚）。
> 本文所有数值都用现仓库 `pareto.hypervolume_2d` + `scipy.integrate.dblquad` 实测核对过（见 §5/§6）。

---

## 1. 记号与坐标变换

`pareto.py` 内部统一 **signed score**（越大越好）：`s = sign ⊙ y`，`sign_j = +1` if max else `−1`。

```
r_signed = sign ⊙ ref_point_raw          # Objective.ref_point（原始单位）→ signed
P        = 非支配的已观测 signed scores（m×2，m 可为 0）
HV(S)    = 面积(∪_{p∈S} [r, p])
DHV(y)   = HV(P ∪ {y}) − HV(P)   （≥0）
```

**ref 的硬性要求**

1. `r_signed` 必须**严格被每个用于计算的 `p` 支配**（`p_j > r_j` 逐分量）。不满足的观测不贡献 HV，计算前过滤：`P = P[all(P > r, axis=1)]`。
2. ref 在 `Study` 创建时确定并写入 `problem.json`，**之后不变**（否则跨轮次 HV 不可比，见 `REVIEW_v03` P0-3）。
3. 没有显式 `ref_point` 时：`r = nadir − 5%·span`（`pareto.summarise` 当前行为），并在报告里标注"参考点由数据推导，HV 不可跨 run 比较"。

---

## 2. Partition bins 的精确构造（2D）

### 2.1 前置处理

```python
def prepare_front(scores: np.ndarray, ref: np.ndarray):
    """排序 + 去重 + 过滤，返回 P（obj1 严格降序、obj2 严格升序）。"""
    P = np.asarray(scores, float)
    P = P[np.all(P > ref, axis=1)]                # 必须严格支配 ref
    if len(P) == 0:
        return np.empty((0, 2))
    # 去重（完全相同的点只留一个）
    P = np.unique(P, axis=0)
    # 只保留非支配点
    P = P[non_dominated_mask_2d(P)]
    # 排序：obj1 降序；obj1 相同时 obj2 降序（排序后 obj2 必升序）
    order = np.lexsort((-P[:, 1], -P[:, 0]))
    return P[order]
```

排序后性质：`P[0,0] > P[1,0] > ...`，且 `P[:,1]` 严格升序（非支配 + 去重保证）。

### 2.2 bins

令 `P ∈ R^{n×2}`，`p_{n+1} = r`（只在 x 方向用它的 `r_1`）：

```
for i in 0..n-1:
    l_i = ( P[i+1,0] if i+1 < n else r_0 ,  r_1 )
    u_i = ( P[i,0],                          P[i,1] )
```

```python
def partition_2d(P, ref):
    n = len(P)
    boxes = []
    for i in range(n):
        lx = P[i + 1, 0] if i + 1 < n else ref[0]
        boxes.append((np.array([lx, ref[1]]), np.array([P[i, 0], P[i, 1]])))
    return boxes
```

**性质（必须单测）**

- `Σ_i (u_i − l_i)` 的逐分量乘积之和 `= HV(P)`；
- bins 两两不重叠（x 区间互不相交，左闭右开在实现里用 `>`/`≥` 区分，浮点下按 `min` 裁剪即可）。

### 2.3 边界情况表

| 情况 | 处理 |
|---|---|
| `P` 为空（无明显改进历史） | `boxes=[]`；`DHV(y) = Π(y−r)` if `y>r` else 0（等价于"新点独自撑起 HV"） |
| 单点前沿 `P={p}` | 一个 bin：`l=(r_0,r_1)`, `u=(p_0,p_1)` |
| 候选 `y` 被 `P` 支配 | `DHV=0`（公式会自然给出 0，但建议提前 `if any(p>=y).all(): return 0` 早退） |
| 候选 `y` 与某 `p` 相等 | `DHV=0` |
| 候选 `y ≤ r` 任一分量 | `DHV=0`（**必须**先判，否则 `Π(y−r)` 为负会污染） |
| `P` 中有重复点 | `np.unique` 去重 |
| `P` 中同一 obj1（理论上非支配集不可能，除非重复/浮点） | 排序时 obj2 降序，`non_dominated_mask_2d` 会过滤掉低 y 者 |
| `P` 中有点不严格支配 `r` | 先过滤掉（它们不进 HV 计算） |
| 目标数 m>2 | 本规格不覆盖；用 MC/标量化（见 `qEHVI_SPEC` §3.3 切换条件） |

---

## 3. `DHV(y)` 公式与实现

```
DHV(y) = Π_{j=1,2} (y_j − r_j)  −  Σ_{b} Π_{j=1,2} max(0, min(y_j, u_{b,j}) − l_{b,j})
```

推导：`Π(y−r)` 是 `[r,y]` 的面积；bins 覆盖旧支配域 `∪[r,p]`，第二项是 `[r,y]` 落入旧域的面积；差 = 新域面积。

```python
def dhv_2d(y, boxes, ref):
    y = np.asarray(y, float)
    if np.any(y <= ref):
        return 0.0
    total = float(np.prod(y - ref))
    for lower, upper in boxes:
        clipped = np.minimum(y, upper) - lower
        if np.all(clipped > 0.0):
            total -= float(np.prod(clipped))
    return max(total, 0.0)          # 数值兜底（理论上旧域内已保证 ≥0）
```

**向量化版**（候选池 m 个点，一次算完）：

```python
def dhv_2d_many(Y: np.ndarray, boxes, ref: np.ndarray) -> np.ndarray:
    """Y: (m, 2) → (m,)"""
    Y = np.asarray(Y, float)
    out = np.zeros(len(Y))
    valid = np.all(Y > ref, axis=1)
    if not valid.any():
        return out
    Yv = Y[valid]
    total = np.prod(Yv - ref, axis=1)
    for lower, upper in boxes:
        clipped = np.minimum(Yv, upper) - lower
        positive = np.all(clipped > 0.0, axis=1)
        total[positive] -= np.prod(clipped[positive], axis=1)
    out[valid] = np.clip(total, 0.0, None)
    return out
```

---

## 4. 独立正态下的 1D 期望（解析 EHVI 用）

单目标后验 `Y_j ~ N(μ_j, σ_j²)`，两目标**独立**（独立 GP 近似）：

```
E[(Y − a)_+] = σ φ(z) + (μ − a) Φ(z),        z = (μ − a)/σ
E[max(0, min(Y,u) − l)] = E[(Y−l)_+] − E[(Y−u)_+]
E[Π_j (Y_j − r_j)] = Π_j E[Y_j − r_j] = Π_j (μ_j − r_j)      (独立性)
```

```python
from scipy.special import log_ndtr, erfcx

def expected_relu(mu, sigma, a):
    """E[(Y-a)_+]，数值稳定（小 z 也不下溢）。"""
    mu, sigma, a = float(mu), float(sigma), float(a)
    z = (mu - a) / sigma
    return sigma * _phi(z) + (mu - a) * norm.cdf(z)

def expected_hinge(mu, sigma, l, u):
    """E[max(0, min(Y,u)-l)] = E[(Y-l)_+] - E[(Y-u)_+]"""
    return expected_relu(mu, sigma, l) - expected_relu(mu, sigma, u)
```

于是**解析 EHVI**（不 clamp 版本）：

```
EHVI_analytic(x) = Π_j (μ_j − r_j)
                 − Σ_b Π_j [ E[(Y_j − l_{b,j})_+] − E[(Y_j − u_{b,j})_+] ]
```

**关于 `max(0, DHV)` 的 clamp**：解析式去掉了 `max(0,·)`；当候选后验以压倒性概率落在 `y > r` 区域（`min_j (μ_j−r_j)/σ_j ≥ 3`，即 `P(y≤r) < 0.3%`）时误差可忽略；否则（候选靠近 ref）改用 MC（§5.2）或数值积分。**推荐实现**：默认 MC，解析式作为"快速路径 + 单测对照"。

**数值核对（本仓库实测）**：`P=[(3,1)]`、`r=(0,0)`、`μ=(2.2,2.2)`、`σ=(0.6,0.6)`：

```
解析（无 clamp）= 2.676459
scipy dblquad 数值积分 = 2.676537
MC（S=4e5）        = 2.677839
```

---

## 5. qEHVI 的两种估计

### 5.1 单点 EHVI

```python
def ehvi_2d_point(mu, sigma, boxes, ref, rng, S=128, mode="mc"):
    if mode == "analytic":
        zmin = min((mu[j] - ref[j]) / sigma[j] for j in range(2))
        if zmin >= 3.0:
            total = float(np.prod(mu - ref))
            for lower, upper in boxes:
                total -= float(np.prod([
                    expected_hinge(mu[j], sigma[j], lower[j], upper[j]) for j in range(2)]))
            return max(total, 0.0)
        # 否则 fall through 到 MC
    Y = mu[None, :] + sigma[None, :] * rng.standard_normal((S, 2))
    return float(np.mean(dhv_2d_many(Y, boxes, ref)))
```

### 5.2 q 点（批量）：`HV(P ∪ Y_s)` 扫描

```python
def qehvi_2d(Yq, mu, sigma, boxes, ref, P, rng, S=64):
    """Yq: (q, m_cand, 2) 候选的后验均值/方差已按候选组织；此处给伪代码思路。"""
    # 每个 MC 样本 s：抽 q 个候选点的联合值 y_s (q,2)
    Ys = mu[None, ...] + sigma[None, ...] * rng.standard_normal((S,) + mu.shape)  # (S,q,m_cand,2)
    base_hv = hv_2d(P, ref)
    out = np.zeros(mu.shape[0])
    for s in range(S):
        # 对每个候选点：HV(P ∪ {y_s[i]}) - HV(P)   （贪心近似）
        hv_new = hv_2d(np.vstack([P, Ys[s]]), ref)
        out += np.maximum(0.0, hv_new - base_hv)
    return out / S
```

**批量贪心（推荐 v0.5 实现）**：

```
X_chosen = []
for i in 1..q:
    acq = qEHVI_1pt(candidate_pool, P ∪ X_chosen, ...)     # 或单点 EHVI + 惩罚
    acq *= local_penalty(X_chosen)                          # 见 SPRINT2_SPEC §2
    x_i = argmax(acq); X_chosen.append(x_i)
```

- q=1：MC `S=128`；q≥4：`S=64` + 局部惩罚（`ℓ_pen = clip(0.5·median(lengthscale), 0.03, 0.25)`）；
- 联合优化 q 点（`q×2` 维）在 2D 下收益有限、成本高，**不推荐**；
- 每个候选点的 `μ,σ` 来自两个独立 GP 的 `predict(..., return_std=True)`；若要用相关抽样，改成 `return_cov=True` 联合抽（两目标协方差需另有模型，v0.5 不做）。

---

## 6. 6 个数值自检用例（输入 → 期望输出）

`r = (0,0)`；`P` 为 signed scores；`DHV` 用 `HV(P∪{y}) − HV(P)` 实测（现仓库 `pareto.hypervolume_2d`）。**全部写成单测**。

| # | P（已测非支配点） | 候选 y | 期望 DHV | 期望 HV(P) |
|---|---|---|---|---|
| 1 | `[(3,1),(2,2)]` | `(3,2)` | **1.0** | 5.0 |
| 2 | `[(3,1),(2,2)]` | `(4,2)` | **3.0** | 5.0 |
| 3 | `[(3,1),(2,2)]` | `(1,1)`（被支配） | **0.0** | 5.0 |
| 4 | `[(3,1),(2,2)]` | `(0,0)`（等于 ref） | **0.0** | 5.0 |
| 5 | `[(3,1),(2,2)]` | `(5,0.5)`（y2 低于前沿、但新 x 区域有效） | **1.0** | 5.0 |
| 6 | `[(3,1),(2,2),(1,3)]` | `(4,1.5)` | **2.0** | 6.0 |
| 6b | 同 #6 | `(1.5,4.0)`（另一侧） | **2.0** | 6.0 |
| 单点前沿 | `[(2,2)]` | `(3,1)` | **1.0** | 4.0 |

```python
def test_dhv_cases():
    ref = np.zeros(2)
    P = np.array([[3.,1.],[2.,2.]])
    boxes = partition_2d(prepare_front(P, ref), ref)
    assert dhv_2d(np.array([3.,2.]), boxes, ref) == pytest.approx(1.0)
    assert dhv_2d(np.array([4.,2.]), boxes, ref) == pytest.approx(3.0)
    assert dhv_2d(np.array([1.,1.]), boxes, ref) == pytest.approx(0.0)
    assert dhv_2d(np.array([0.,0.]), boxes, ref) == pytest.approx(0.0)
    assert dhv_2d(np.array([5.,0.5]), boxes, ref) == pytest.approx(1.0)
    # 3 点前沿
    P3 = prepare_front(np.array([[3.,1.],[2.,2.],[1.,3.]]), ref)
    boxes3 = partition_2d(P3, ref)
    assert dhv_2d(np.array([4.,1.5]), boxes3, ref) == pytest.approx(2.0)
    assert dhv_2d(np.array([1.5,4.0]), boxes3, ref) == pytest.approx(2.0)
    # 单点前沿
    P1 = prepare_front(np.array([[2.,2.]]), ref)
    assert dhv_2d(np.array([3.,1.]), partition_2d(P1, ref), ref) == pytest.approx(1.0)
```

另外的**期望值自检**（§4 的 1D 函数）：

| # | 输入 | 期望 |
|---|---|---|
| 7 | `E[(Z)_+]`, `Z~N(0,1)` | **0.3989422804**（=1/√2π） |
| 8 | `E[(Z−0.5)_+]`, `Z~N(0,1)` | **0.1977965574** |
| 9 | `E[min(Z,3)] = E[(Z)_+]−E[(Z−3)_+]`, `Z~N(2.2,0.6)` | **2.1745807873** |
| 10 | 解析 EHVI（#§4 场景） | **≈2.676459**（数值积分 2.676537，差 <1e-4） |

---

## 7. 与 ParEGO 的切换阈值

| 条件 | 选择 | 依据 |
|---|---|---|
| m=2，budget ≥ 40，noise ∈ {none, low}，无约束 | **qEHVI**（q=1 起步） | R21：2D 下 EHVI 的前沿覆盖/HV 通常优于随机标量化；本文门槛来自样本效率经验 |
| m=2，budget < 40 | ParEGO | 标量化冷启动更快；EHVI 的 MC 方差在小样本时更明显 |
| m=2，noise medium/high | ParEGO（或 qNEHVI，未实现） | EHVI 假设观测即真值；噪声使 `P` 不可靠 |
| m=3 | ParEGO（v0.5 不实现 3D 分区） | 分区数量增长；R21 支持但实现成本高 |
| m>3 | ParEGO / NSGA-II | 同上 |
| 有约束 | ParEGO + PoF（v0.5） | 约束 qEHVI 留待 v0.6 |
| 任何情况 | 必须同时跑 ParEGO 对照 | 校准门槛：qEHVI 的 `HV_ratio` 相对 ParEGO 提升 ≥3% 才切换为默认 |

**router 落地**：`_score_qehvi(ctx)` 在满足行 1 时 +（≥ ParEGO 分数），否则 -（低于 ParEGO）；`qehvi` 的 `capabilities` 必须含 `multi_objective`/`batch`/`categorical`/`continuous`（与 `parego` 相同），并额外声明 `needs_ref_point`（v0.5 用 `Objective.ref_point` 必填校验，缺失时 warning + 自动 nadir 外推）。

---

## 8. 集成到 condopt（文件级）

| 文件 | 改动 |
|---|---|
| `src/condopt/pareto.py` | 新增 `prepare_front`、`partition_2d`、`dhv_2d`、`dhv_2d_many`、`expected_relu/expected_hinge`、`ehvi_2d_point` |
| `src/condopt/algorithms/qehvi.py`（新） | `QEHVI(Optimizer)`：`tell` 存 `_Y`/`_xs`（缺目标容错）；`ask`：`prepare_front(score_matrix, ref_signed)` → 每目标 `fit_gp`（复用抽出的 `fit_gp` 函数）→ 候选池（复用 `space` + 局部扰动，或复制 `bayes._candidate_pool`）→ 每候选 `predict(return_std=True)` → `dhv`/`ehvi` 打分 → 贪心 + 惩罚 → `decode` |
| `src/condopt/algorithms/bayes.py` | 抽出模块级 `fit_gp(space, X, y, params, seed)`（供 bayes/parego/qehvi 共用）；`parego.py` 当前每 proposal 重建 GP 可一并简化 |
| `src/condopt/router.py` | `_score_qehvi` 规则 + `_params_for("qehvi")`；切换阈值见 §7 |
| `tests/test_qehvi.py` | 上述 6+4 个数值用例 + `test_ehvi_matches_numeric_integration`（dblquad，容差 5e-3） |

**实现顺序**：①`pareto` 的纯函数 + 单测（半天）；②`qehvi` 的 q=1 + 单点 EHVI（1 天）；③批量贪心 + 惩罚（半天）；④router 切换 + benchmark 对照（半天）。

---

## 9. 参考

R21（qEHVI 分区与 MC）、R42（HV/固定参考点）、R11（ParEGO）、R12（qNEHVI：噪声多目标，下一步）、R35（批量局部惩罚）、R29（log-EI：EHVI 内部 `E[(·)_+]` 的稳定计算思想）。
