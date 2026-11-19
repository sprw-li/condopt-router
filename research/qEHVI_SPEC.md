# qEHVI_SPEC — 2D EHVI/qEHVI 的代码级规格

> 维护者：algo-research ｜ v1（2026-11-05，Round 4）
> 对应：`MULTI_OBJECTIVE_AND_NOISE.md` §2（原理）、`REVIEW_v03.md` P0-3（固定 ref point）、`pareto.py`（现有 HV/支配工具）。
> 文献：R21（qEHVI 原始论文：可微 EHVI + 分区）、R12（qNEHVI：噪声版）、R42（HV 指标/固定参考点）、R11（ParEGO 对照）。
> 本文只做 **2 目标**（M=2）的精确分区；3 目标给递归思路；q 点用"单点 EHVI + 贪心/局部惩罚"，避免一次性上完整 qEHVI 的复杂度。

---

## 1. 记法与现有代码的对接

`pareto.py` 的内部约定：**所有目标已转成 signed score（越大越好）**，支配 = 逐分量 `>=` 且至少一个 `>`；参考点 `r` 也在 signed 空间。沿用之。

```
P = {p_1..p_n}  已观测的非支配 signed scores（n×2）
r = 参考点（signed，必须严格被每个 p_i 支配：p_i > r 逐分量）
HV(S) = Λ( ∪_{y∈S} [r, y] )            # 2D 面积
EHVI(x) = E[ max(0, HV(P ∪ {f(x)}) - HV(P)) ]
```

**前置条件**：`Objective` 增加 `ref_point`（见 `REVIEW_v03.md` P1-3 / `PAREGO_SPEC.md` §1），单位是**原始目标单位**；在 `pareto.py` 里用 `signed_scores` 转成 signed 后作为 `r`。**ref 必须创建时固定并持久化**（`Study` 写入 `problem.json`）——否则不同 run 的 HV/EHVI 不可比（`REVIEW_v03.md` P0-3）。

---

## 2. 2D 分区（partition）——把"旧支配域"拆成互不重叠的箱

### 2.1 算法

把 `P` 按第 1 目标**降序**排（2D 非支配集此时第 2 目标必升序）；令 `p_{n+1} = r`：

```python
def partition_2d(scores: np.ndarray, ref: np.ndarray):
    """返回箱列表 [(lower(2,), upper(2,))]，覆盖 ∪_i [ref, p_i] 且两两不重叠。"""
    P = scores[np.argsort(-scores[:, 0])]         # x 降序
    boxes = []
    for i in range(len(P)):
        lower = np.array([P[i + 1, 0] if i + 1 < len(P) else ref[0], ref[1]])
        upper = np.array([P[i, 0], P[i, 1]])
        boxes.append((lower, upper))
    return boxes
```

**性质**：`Σ_i Π(u_i − l_i) = HV(P)`；每个箱内部对任意候选 `y` 的"新增面积"可解析计算。

### 2.2 单点 DHV（EHVI 的被积函数）

对候选 `y`（要求 `y > r` 逐分量，否则 DHV=0）：

```
DHV(y) = Π_j (y_j − r_j) − Σ_b Π_j max(0, min(y_j, u_{b,j}) − l_{b,j})
```

实现：

```python
def dhv_2d(y: np.ndarray, boxes, ref: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    if np.any(y <= ref):
        return 0.0
    total = float(np.prod(y - ref))
    for lower, upper in boxes:
        clipped = np.minimum(y, upper) - lower
        if np.all(clipped > 0):
            total -= float(np.prod(clipped))
    return max(total, 0.0)
```

### 2.3 数值自检（必须写成单测）

| 用例 | 输入 | 期望 |
|---|---|---|
| HV of P | `P=[(3,1),(2,2)]`, `r=(0,0)` | `HV=5`（箱：(2,0)-(3,1) 面积 1；(0,0)-(2,2) 面积 4） |
| 改进 1 | `y=(3,2)` | `DHV=1`（`Π=6`，落入旧域 5） |
| 改进 2 | `y=(4,2)` | `DHV=3`（`Π=8−5`） |
| 被支配 | `y=(1,1)` | `DHV=0` |
| 三点前沿 | `P=[(3,1),(2,2),(1,3)]`, `r=(0,0)` | `HV=5`（面积 3+2+... 与手算一致） |

```python
def test_dhv_2d_hand_computed():
    ref = np.zeros(2); P = np.array([[3.,1.],[2.,2.]])
    boxes = partition_2d(P, ref)
    assert abs(sum(np.prod(u-l) for l,u in boxes) - 5.0) < 1e-12
    assert abs(dhv_2d(np.array([3.,2.]), boxes, ref) - 1.0) < 1e-12
    assert abs(dhv_2d(np.array([4.,2.]), boxes, ref) - 3.0) < 1e-12
    assert abs(dhv_2d(np.array([1.,1.]), boxes, ref) - 0.0) < 1e-12
```

---

## 3. qEHVI（q 点）的 MC 估计

### 3.1 单点 EHVI（v0.3.1，先做这个）

```
EHVI(x) ≈ (1/S) Σ_s max(0, DHV(y^s(x)))
```

- 每个目标一个 GP（**独立 GP 近似**，忽略目标间相关；BoTorch 用联合多输出模型，v0.3 可接受近似）。
- `y^s(x)`：从两个 GP 的后验各抽一条（`predict(return_cov=True)` + Cholesky，或对单点用 `predict(return_std=True)` 独立抽样——单点抽样足够，因候选池逐点评估）。
- `S=128`（2D、q=1）；采样固定 `self.rng` → 可复现。

```python
def ehvi_2d(models, candidates_units, candidate_features, boxes, ref, rng, S=128):
    scores = np.zeros(len(candidate_features))
    for j, model in enumerate(models):
        mu_j, sd_j = model.predict(candidate_features, return_std=True)
        draws_j = mu_j[None, :] + sd_j[None, :] * rng.standard_normal((S, len(mu_j)))
        scores += draws_j if j == 0 else 0.0          # placeholder
        if j == 0:
            Y = draws_j                                  # (S, m_cand)
        else:
            Y = np.vstack([Y, draws_j])
    Y = Y.reshape(len(models), S, -1)                    # (2, S, m_cand)
    acq = np.zeros(Y.shape[2])
    for s in range(S):
        for i in range(Y.shape[2]):
            acq[i] += max(0.0, dhv_2d(Y[:, s, i], boxes, ref))
    return acq / S
```
（实现时向量化 `dhv_2d`：`clipped = np.minimum(y, upper) - lower` 对 `(S, m_cand, 2)` 广播。）

### 3.2 q 点（v0.3.2）

对每个 MC 样本 `s`，把 q 个候选点与旧前沿合并后算 HV：

```
ΔHV_s = HV(P ∪ {y^s_1..y^s_q}) − HV(P)
qEHVI(X) ≈ (1/S) Σ_s max(0, ΔHV_s)
```

- `HV(P ∪ {q 点})` 用 `pareto.hypervolume_2d`：把 `P` 与 q 个样本点拼成点集、排序扫描（O((n+q) log(n+q))），无需构造 q 的并集分区。
- 批量选点：**贪心 + 局部惩罚**（不直接联合优化 q×2 维）：
  1. 选 `x_1 = argmax EHVI(x)`；
  2. 对剩余点乘惩罚 `Π(1−exp(−‖x−x_j‖²/(2ℓ_pen²)))`，再选 `x_2`，…；
  3. 或每点独立从后验抽"函数样本"求 argmax（TS 风格），成本更低。
- `S`：q=1 用 128；q≥4 用 64（或 32 + 惩罚）。

### 3.3 与 ParEGO 的切换条件（写进 router）

| 条件 | 推荐 | 理由 |
|---|---|---|
| m=2，budget ≥ 40，噪声 none/low | **qEHVI** | 前沿覆盖与 HV 更好（R21 实验结论） |
| m=2，budget < 40，或噪声 medium/high | **ParEGO** | 样本少时标量化更稳；噪声大需要 qNEHVI（未实现） |
| m=3，budget ≥ 60，噪声 low | qEHVI（需 M=3 递归分区，v0.4） | — |
| m>3 | ParEGO / NSGA-II | EHVI 分区数量爆炸（R21） |
| 有约束 | ParEGO + PoF 或 qNEHVI-constrained（v0.5） | — |

router 落地：`qehvi` 注册时声明 `capabilities={"multi_objective","batch","categorical","continuous",...}`；`_score_qehvi(ctx)` 只在 `m==2 and budget>=40 and noise in ("none","low")` 时给高分，否则低分；`ParEGO` 仍是默认兜底。

---

## 4. 与现有代码的集成（文件级）

| 文件 | 改动 |
|---|---|
| `src/condopt/pareto.py` | 加 `partition_2d`、`dhv_2d`、`hypervolume_2d`（已存在，复用）；加 `front_scores(dataset, objective)`（已有 `score_matrix`） |
| `src/condopt/algorithms/qehvi.py`（新） | `QEHVI(Optimizer)`：`tell` 存 `_Y`/`_xs`；`ask` 复盘：滤非支配 → 固定 ref → 每目标 fit GP（复用 `bayes._make_kernel/_fit` 或独立小函数）→ 候选池（复用父类逻辑需抽成 mixin 或复制 40 行）→ `ehvi_2d` 打分 → 贪心+惩罚 → decode |
| `src/condopt/router.py` | 加 `_score_qehvi` 规则 + `_params_for("qehvi")`；`_required_capabilities` 不变（multi_objective 已覆盖） |
| `src/condopt/benchmarks.py` | `moo_yield_cost` 已有；用固定 ref 计算 `HV_ratio`（`pareto.hypervolume_ratio`） |
| `Study` | `pareto_summary` 增加 `hv_ratio`（相对 benchmark 参考前沿时） |

**注意**：现有 `GaussianProcessBO` 是单目标（`self._y`）；qEHVI 需要 m 个 GP。最省事的实现是**独立类**（不继承 `GaussianProcessBO`），共享 `_make_kernel/_fit` 静态函数（建议把 `_fit` 抽成模块级 `fit_gp(space, X, y, params, seed)`，供 bayes/parego/qehvi 三方复用——顺手解掉 `parego.py` 每 proposal 重写 GP 的问题）。

---

## 5. 参数与验收

| 参数 | 默认 | 说明 |
|---|---|---|
| `ref_point` | 必填（Objective） | signed 空间严格劣于所有观测；**创建时固定** |
| `n_initial` | `max(2*(feature_dim+1), 2*m)` ≤20 | 冷启动 |
| `n_candidates` | 1024（compute=high 2048） | 候选池 |
| `mc_samples` | 128（q=1）/64（q≥4） | S |
| `local_penalty` | True（q>1） | `ℓ_pen = clip(0.5·median(ls), 0.03, 0.25)` |
| `batch_strategy` | greedy+penalty | 见 §3.2 |

**验收**

1. `test_dhv_2d_hand_computed`（§2.3 全部用例）；
2. `test_ehvi_monotone`：同一候选，`EHV` 随其后验均值提高/方差增大而提高（定性）；
3. `moo_yield_cost`（固定 ref，`BENCHMARK_TIER_CD.md` §6.4）：budget=48、5 seeds 时 `HV_ratio(qehvi) ≥ HV_ratio(parego) + 0.03`；若达不到，保留 ParEGO 为默认并记录数据（R21 的结论依赖问题结构）；
4. 运行时间：`S=128, n_cand=1024` 单次 ask <1.5 s（CPU）；
5. 与 ParEGO 的切换规则在 router 单测里覆盖（`test_router_prefers_qehvi_when_m2_lownoise_budget40`）。

---

## 6. 参考

R21（qEHVI：分区 + 可微 MC + 固定 ref）、R12（qNEHVI：噪声版；本文不做）、R42（HV 指标与固定参考点）、R11（ParEGO 对照）、R35（批量局部惩罚）。
