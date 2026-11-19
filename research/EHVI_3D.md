# EHVI_3D — 3+ 目标 EHVI 的递归分区与 MC 近似

> 维护者：algo-research ｜ v1（2026-11-05，Round 6）
> 对应：`qEHVI_IMPL.md`（2D 分区/DHV/解析期望）、`qehvi.py`（v0.5 仅支持 2 目标）、`PAREGO_SPEC.md`（标量化路线）。
> 文献：R21（qEHVI：分区 + 可微 MC，M 维推广）、R42（HV 指标/固定 ref）、R12（qNEHVI：噪声版 M 维）、R11（ParEGO 对照）。
> 结论：**v0.6 只做 m=3**，递归分区 + 每候选 MC（或解析无 clamp 版）；m≥4 继续 ParEGO/NSGA-II。理由：box 数 `O(n^{M-1})`，m=3、n_pareto≤30 时约 435 个 box，可接受；m=4 起爆炸。

---

## 1. 记号

沿用 `qEHVI_IMPL.md`：signed scores（越大越好）、固定 `r`（`Objective.ref_point × signs`）、`HV(P)=Λ(∪[r,p])`、`DHV(y)=HV(P∪{y})−HV(P)`。

M=3 时 HV 是 3D 体积；分区把旧支配域拆成 3D 盒子 `[l_b, u_b]`。

---

## 2. 递归分区算法（M≥3）

**核心思想**：按第 1 维切片；切片 `(p_{i+1,0}, p_{i,0}]` 内的点只能被"第 1 维 ≥ 该切片"的 Pareto 点支配，所以横截面（第 2..M 维）的旧支配域由前 `i+1` 个点的投影决定 → 递归降 1 维。

```python
def partition_nd(P: np.ndarray, ref: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """P: (n, M) 非支配 signed scores（n>=1, M>=2），ref: (M,)。"""
    P = np.asarray(P, float); ref = np.asarray(ref, float)
    M = P.shape[1]
    if M == 2:
        return partition_2d(P, ref)                     # 复用 v0.5 的 2D 实现
    order = np.lexsort([P[:, j] for j in reversed(range(1, M))][::-1] + [-P[:, 0]])
    # 简化：按第 0 维降序；并列时按后续维降序（保证横截面包含"支配者"）
    P = P[np.argsort(-P[:, 0], kind="stable")]
    boxes, n = [], len(P)
    for i in range(n):
        x_low = P[i + 1, 0] if i + 1 < n else ref[0]
        x_high = P[i, 0]
        if x_high <= x_low + 1e-12:      # 并列/零宽切片：跳过（其横截面已被前一刀覆盖）
            continue
        # 横截面：所有"第 0 维 >= x_high"的点的第 1..M-1 维（即 P[0..i+1]）
        sub = P[: i + 1, 1:]
        if len(sub) == 0:
            continue
        sub_mask = non_dominated_mask(sub)               # 投影后重新取非支配
        for l_sub, u_sub in partition_nd(sub[sub_mask], ref[1:]):
            boxes.append((np.concatenate([[x_low], l_sub]), np.concatenate([[x_high], u_sub])))
    return boxes
```

**注意**：

1. **并列第 0 维**：两个点 `p_i, p_{i+1}` 的 `x0` 相等时切片宽度为 0，跳过是对的——横截面会由后续更宽的切片覆盖（见 §3 例子）。
2. **横截面必须重新取非支配**：投影到低维后，原本非支配的点可能互相支配（例如 `(1,1,2)` 与 `(2,1,1)` 在第 1 维投影后 `(1,2)` vs `(1,1)` 后者被支配）。
3. **ref 逐维传递**：递归时 `ref[1:]`。
4. 返回的 boxes 两两不重叠，且 `Σ Π(u_b − l_b) = HV(P)`（单测断言）。

**复杂度**：单次分区 box 数 `K = O(n^{M-1}/(M-1)!)`；m=3 时 `K≈n²/2`（n=30 → 435；n=50 → 1225）；分区时间 `O(K·M)`。m=4 时 `K≈n³/6`（n=30 → 4500），开始不可接受。

---

## 3. 3D 数值自检（可直接写 pytest）

**用例**：`r=(0,0,0)`，`P = [(3,1,1), (2,2,1), (2,1,2)]`（三个点两两非支配）。

- 旧支配域 HV = `|A∪B∪C|`，由容斥：`3+4+4 −2−2−2 +2 = **7**`；
- 递归分区（第 0 维降序，切片）：
  1. `i=0`（`x0∈(2,3]`）：横截面 `{(1,1)}` → 2D 分区 1 个 box → 3D box `[(2,0,0),(3,1,1)]`，体积 **1**；
  2. `i=1`（`x0∈(2,2]` 零宽）→ 跳过；
  3. `i=2`（`x0∈(0,2]`）：横截面 `{(1,1),(2,1),(1,2)}` → 非支配 `{(2,1),(1,2)}` → 2D 分区 2 个 box（`[(1,0),(2,1)]` 体积 1；`[(0,0),(1,2)]` 体积 2）→ 3D 体积 = `(2−0)×(1+2)= **6**`；
  - 合计 `1+6 = 7` ✓，box 数 3。

```python
def test_partition_3d_volume_matches_hv():
    P = np.array([[3.,1.,1.],[2.,2.,1.],[2.,1.,2.]]); ref = np.zeros(3)
    boxes = partition_nd(P, ref)
    vol = sum(np.prod(u - l) for l, u in boxes)
    assert vol == pytest.approx(7.0)
    assert len(boxes) == 3
```

---

## 4. EHVI（3D）的两种计算方式

### 4.1 MC（推荐先做，最稳）

对每个候选 `x`：抽 `S` 条 3 维后验样本（每目标独立 GP），

```
EHVI_3D(x) ≈ (1/S) Σ_s max(0, DHV_3D(y_s)),   DHV_3D(y) = Π_j (y_j−r_j) − Σ_b Π_j max(0, min(y_j,u_{b,j})−l_{b,j})
```

- `DHV_3D` 是 2D 公式的直接推广（逐分量乘积换成 M 个），复杂度 `O(K·M)`；
- `S=128`（m=3、候选 1024、K≈400 → `128×1024×400×3 ≈ 1.6e8` 次乘加，numpy 向量化后 ~2–4 s/ask，可接受；算力低降 `S=64`）；
- 批量：贪心 + 局部惩罚（同 2D）。

### 4.2 解析（更快，依赖独立性 + clamp 处理）

独立正态下 `E[Π_j g_j(Y_j)] = Π_j E[g_j]`，所以：

```
EHVI_3D_unclamped(x) = Π_j (μ_j − r_j) − Σ_b Π_j [ E[(Y_j−l_bj)_+] − E[(Y_j−u_bj)_+ ] ]
```

- 1D 期望函数与 2D 相同（`qEHVI_IMPL.md` §4）；
- **clamp 问题更严重**：`max(0, DHV)` 在"候选有任一分量 ≤ ref"时为 0；m=3 时该事件概率更大（每个目标各有尾部）；
- 实用规则：`z_min = min_j (μ_j−r_j)/σ_j`；`z_min ≥ 3` 用解析（误差 <1%），否则用 MC；或对 `z_min∈[1,3]` 用"解析 + 单侧修正"（用 1D 的 `E[(Y−r)_+·1{Y>r}]` 组合，复杂度 O(M)）。

**精度/成本折中表**

| 方式 | 每 ask 成本（m=3, K≈400, n_cand=1024） | 精度 | 建议 |
|---|---|---|---|
| MC `S=128` | 2–4 s | 方差 ~2–5% | **默认** |
| 解析（`z_min≥3`） | 0.1–0.3 s | <1% | 快速路径 |
| 解析 + MC 混合 | 0.5–1.5 s | <1% | 生产推荐 |

---

## 5. 与 ParEGO / 2D qEHVI 的切换阈值

| 条件 | 推荐 | 依据 |
|---|---|---|
| m=2，无/低噪 | 2D qEHVI | 已有实现，`HV_ratio≈0.95`（v0.5 实测） |
| m=2，中/高噪 | qNEHVI（`qNEHVI_SPEC_V2.md`） | R12 |
| m=3，预算 ≥80，n_pareto ≤50，无/低噪 | **3D EHVI（MC）** | R21 的 M 维推广；box 数可控 |
| m=3，预算 <80 或 n_pareto >50 | ParEGO | EHVI 的 MC 方差在少样本时更大 |
| m=3，中/高噪 | ParEGO（qNEHVI 3D 暂不做） | 3D qNEHVI 的 nested MC 成本 `n_base×n_y×K` 爆炸 |
| m≥4 | ParEGO / NSGA-II | `K=O(n^{M-1})` 不可行 |
| 有约束 | ParEGO + PoF（约束 EHVI 留待 v0.7） | — |

**router 落地**：`_score_ehvi3d(ctx)` 只在 `m==3 and budget>=80 and noise in ("none","low")` 时给高于 ParEGO 的分；`n_pareto` 用 `pareto.summarise` 实时算。`qehvi` 与 `qnehvi` 保持 m=2 硬约束（`__init__` raise，已实现）。

---

## 6. 实现清单

| 文件 | 改动 |
|---|---|
| `pareto.py` | 新增 `partition_nd`（递归，M=2 走现有 `partition_2d`）；`dhv_nd_batch(points, boxes, ref)`；`hypervolume_3d`（扫描线或 MC，报告用） |
| `algorithms/ehvi3d.py`（新） | 复用 `QEHVI` 的 `tell/_signed_references/_candidate_pool/_greedy_batch`；只覆写 `_model_ask`：m 个 GP、`partition_nd`、MC `dhv_nd` |
| `router.py` | `_score_ehvi3d` + `_params_for`；`m==3` 时能力过滤需允许 `ehvi3d`（capabilities 加 `multi_objective`，与 parego/qehvi 同） |
| `tests/test_pareto_nd.py` | 上述 3D 体积用例 + box 不重叠断言 + 与 MC HV 对照（容差 1e-9） |
| `tests/test_ehvi3d.py` | 3 目标合成问题上 `EHVI_3D` 的 argmax 落在"能扩展前沿"的点（定性）+ MC 收敛 |

**验收**：3D 分区体积 = HV（1e-9）；`tests` 全绿；在 3 目标合成问题（`yield/cost/purity` 三目标，参考前沿由网格枚举）上，`ehvi3d` 的 `HV_ratio` ≥ ParEGO + 3%（budget=100、5 seeds）。

---

## 7. 参考

R21（qEHVI：M 维分区与 MC）、R42（HV：固定 ref、M 维体积）、R12（qNEHVI：噪声版；3D 暂不做）、R11（ParEGO：m≥4/噪声场景的兜底）、R34（约束版本留待后续）。
