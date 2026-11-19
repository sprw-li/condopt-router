# MULTI_OBJECTIVE_AND_NOISE — 多目标 BO 与噪声 BO 的可实现规格

> 维护者：algo-research ｜ v1（2026-11-05，Round 2）
> 对应 `REVIEW_v01.md` P1-10 / P1-5；`CAPABILITIES_SPEC.md` 的 `objects.multi_objective` / `noise.models_observation_noise` 能力位。
> 目标：给 condopt 下一轮实现（不依赖 BoTorch）提供**公式 + 伪代码 + 参数 + 验收**。
> 约定：所有目标先统一为**最大化**（`direction=min` 时取负），并归一化到 `[0,1]`；参考点 `r` 在归一化空间取 `(0,0,...)`（对应"业务上无意义"的阈值，需用户显式给）。

---

## 1. ParEGO（先做这个，1–2 天可落地）

**为什么先做**：复用现有单目标 GP + EI 全套代码，只多一层"每轮随机加权标量化"；2–3 目标样本效率通常够用（R11）。

### 1.1 公式

对观测 `y(x) ∈ R^m`（已归一化、已取向）与权重 `λ ∈ Δ^{m-1}`（单纯形）：

```
g_λ(y) = max_i (λ_i · y_i) + ρ · Σ_i λ_i · y_i        (ρ = 0.05，增广 Tchebycheff)
```

- `max_i` 项保证能找到非凸 Pareto 前沿上的点（加权求和做不到）；
- `ρ` 项保证弱 Pareto 支配（避免权重极端时目标值并列）；R11 的取值 `ρ=0.05`，实现时做成参数。
- 每个 iteration：`λ ~ Dirichlet(α=1)`；**本轮内固定 λ**，用 `g_λ` 重算**全部历史观测**的标量值（不能只算新点，因为 λ 变了），在该标量上拟合 GP 并最大化 EI。

### 1.2 算法伪代码

```python
class ParEGO(Optimizer):
    params: rho=0.05, weight_alpha=1.0, acquisition="logei", n_initial, ref_point

    def tell(self, observations):
        super().tell(observations)
        self._Y = np.array([normalize(o.objectives) for o in self._all])   # 归一化到 [0,1]

    def ask(self, n=1):
        out = []
        for _ in range(n):                       # 批量时每点独立采 λ，天然分散
            lam = self.rng.dirichlet(np.full(m, self.params["weight_alpha"]))
            g = np.max(lam * self._Y, axis=1) + self.params["rho"] * (lam * self._Y).sum(axis=1)
            gp = fit_gp(self._X, g)              # 复用 bayes._fit
            x = maximize_acquisition(gp, self.space, acq=self.params["acquisition"])
            out.append(self.space.decode(x))
            # 关键：把 x 以"幻想值"加入，避免同批重复（局部惩罚更佳）
        return out
```

### 1.3 归一化（必须固定口径）

```
y_norm_i = (y_i - ideal_i) / (nadir_i - ideal_i + eps)
ideal_i  = max over observed (or user "best plausible")
nadir_i  = min over observed (or user "unacceptable" threshold)
```
- 用**用户给的 `ref_point`**（"低于此值无意义"，如收率 30%）作为 `nadir`，比用观测 min 稳定（观测 min 会随数据漂移，导致历史 `g` 值变化）。
- 归一化参数写进 `Study` 的 decisions/state（可复现）。
- 若某目标范围退化（max=min），该维权重固定为 0（否则除零）。

### 1.4 参数与失效

| 参数 | 默认 | 说明 |
|---|---|---|
| `rho` | 0.05 | 太小 → 前沿覆盖差；太大 → 接近加权和、丢失非凸前沿 |
| `weight_alpha` | 1.0 | Dirichlet 浓度；<1 更极端权重（强调各目标端），>1 更均匀 |
| `n_initial` | `max(2*(d+1), 2*m)` | 多目标冷启动要更多点 |
| 批量 | 每点独立 λ | 若同批同 λ，会扎堆 |

**失效**：①两个目标强相关 → 前沿退化为点；②目标尺度差异大且没归一化 → 某一目标完全支配；③权重方向覆盖不全 → 前沿有洞（缓解：保留最近 K 轮的 λ，对新 λ 加"最小夹角"约束）；④噪声大 → 用重复测量均值再标量化（R10）。

### 1.5 验收

在 `BENCHMARK_SPEC.md` 的 `moo_yield_cost`（参考前沿 101 个点）上：
- `HV_ratio = HV(算法前沿)/HV(参考前沿) ≥ 0.85`（budget 32、3 seeds）；
- 前沿覆盖：把 2D 目标空间分 10×10 格，非空格子数 ≥ 参考前沿的 60%；
- 噪声 10% 版本：`HV_ratio` 下降 <10%。

---

## 2. EHVI / qEHVI（2–3 目标的主武器）

### 2.1 超体积与改进

最大化、参考点 `r`（归一化空间 `r=0`）：

```
HV(S)   = Λ( ∪_{y∈S} [r, y] )                    # Λ = 体积
EHVI(x) = E[ max(0, HV(S ∪ {f(x)}) - HV(S)) ]
```

### 2.2 分区（partitioning）——把改进算成"箱体积之和"

把已观测的**非支配集** `P = {p_1..p_n}`（按第 1 目标降序排，则第 2 目标升序）在旧支配域 `∪_i [r, p_i]` 上分解成互不重叠的箱 `b_i = [l_i, u_i]`（**2D 精确算法**）：

```
p_{n+1} = r
for i = 1..n:
    l_i = ( p_{i+1,1}, r_2 )
    u_i = ( p_{i,1},   p_{i,2} )
```

对候选点 `y`（需 `y ≻ r` 才有改进）：

```
DHV(y) = Π_j (y_j - r_j)  −  Σ_i Π_j max(0, min(y_j, u_{i,j}) − l_{i,j})
```

（`Π(y−r)` 是 `[r,y]` 的总体积；第二项是它落入旧支配域的部分。）

**2D 数值自检**（写单测）：`P={(3,1),(2,2)}, r=(0,0)` → `HV(P)=5`；`y=(3,2)` → `DHV=1`；`y=(4,2)` → `DHV=3`。

**M>2 的递归**：按第 1 目标排序，对每个 `p_i` 先切出 `(p_{i+1,1}, p_{i,1}]` 的切片，再在剩余 `M-1` 维上对"第 1 目标 ≥ p_{i,1} 的点"递归分区；实现参考 BoTorch 的 `botorch/utils/multi_objective/partitioning.py`（`partition_non_dominated_space_2d`/`NondominatedPartitioning`）。M>3 时箱数量指数增长，改用 MC 近似或标量化。

### 2.3 qEHVI 的 MC 估计（q 个点）

单样本 `s`：从联合后验抽 `y^s ∈ R^{m×q}`（各目标独立 GP 抽样是常用近似；有联合模型时用联合抽样）。

```
ΔHV_s = HV( P ∪ {y^s_1..y^s_q} ) - HV(P)
qEHVI(X) ≈ (1/S) Σ_s max(0, ΔHV_s)
```

- `q ≤ 4`：`HV(P ∪ {q 点})` 用"2^q 个 `[r,y_i]` 盒的并集体积"（inclusion–exclusion 或扫描线）精确算；
- `q > 4`：改用"单点 EHVI + 局部惩罚"（见 `REVIEW_v01.md` P1-3 的惩罚形式），或逐点贪心 sequential greedy；
- `S`：q=1 用 128–256，q≥4 用 64–128（方差与耗时折中）。

### 2.4 参考点与归一化（最容易出错的地方）

1. 每个目标先按**用户给的可接受阈值**归一化：`z_i = (y_i - ref_i) / (ideal_i - ref_i)`，`ref_i` 一下视为"无意义"（收率 30%、成本上限…）。
2. 归一化后 `r = (0,...,0)`；**ref point 一旦设定就固定**（不要随观测漂移），否则 HV 曲线不可比。
3. 若某目标优化空间大，可对 `z_i` 取 log（`log1p`）后再归一化——注意 EHVI 的单调性不受影响，但报告 HV 时要回到原尺度。

### 2.5 验收

- 2D：`DHV` 与"暴力网格计算 HV 差"一致（单测，tol 1e-9）；
- 在 `moo_yield_cost` 上 `qEHVI` 的 `HV_ratio ≥ ParEGO`，且 Pareto 集覆盖更均匀；
- 3 目标兼容性测试（Tier F 扩展）。

---

## 3. qNEHVI（噪声多目标，2–3 目标）

**与 qEHVI 的唯一区别**：`P` 不再是确定的观测值，而是**后验随机变量**。

```
qNEHVI(X) = E_{f|D}[ E_{y|f,X}[ HV(P(f) ∪ {y}) - HV(P(f)) ] ]
```

### 3.1 嵌套 MC 估计（可直接实现）

```python
def qnehvi(model_list, X_cand, X_base, ref, n_base=64, n_y=128):
    # 1) baseline 概率化：抽 n_base 组 baseline 值
    F_base = sample_joint(model_list, X_base, n_base)        # (n_base, n_b, m)
    pareto_sets = [nondominated(F_base[k]) for k in range(n_base)]
    hv_old = [hypervolume(P, ref) for P in pareto_sets]
    # 2) 候选点抽样
    Y = sample_joint(model_list, X_cand, n_y)                # (n_y, q, m)
    total = 0.0
    for k in range(n_base):
        P = pareto_sets[k]
        hv_new = hypervolume(np.vstack([P, Y[k]]), ref)      # 或逐箱增量
        total += max(0.0, hv_new - hv_old[k])
    return total / n_base
```

- **`X_baseline` 选择**：`n_obs ≤ 200` 用全部已评估点；否则取"最近 200 个 + 当前最优点"，并 `prune_baseline`（去掉被支配的，只留非支配点以降 n）。
- 样本量：`n_base=32–64`、`n_y=64–128`（BoTorch 默认量级；正确性比精度重要，先能跑）。
- **批量 q>1** 用同一 `Y[k]` 内的 q 点联合 HV（同 §2.3）；q>4 仍建议"q=1 + 局部惩罚"。
- 近似替代（若实现成本高）：每目标单独 `qNEI` + ParEGO 标量化，`HV_ratio` 一般只差 2–5%。

### 3.2 参数与失效

| 参数 | 建议 | 说明 |
|---|---|---|
| `X_baseline` | 全部（n≤200） | 太大时耗时 O(n_base·n_b) |
| `n_base` | 32 | 噪声越大越要大 |
| `n_y` | 64–128 | q 大时降 |
| `ref` | 固定 | 同 §2.4 |
| `prune_baseline` | True | 只保留非支配 baseline |

**失效**：①`n_base` 太小 → 采集函数方差大、推荐抖动；②baseline 含大量被支配点 → 浪费算力；③目标间强相关但用独立 GP 抽样 → 低估联合方差（可接受近似，需在文档标注）。

---

## 4. 噪声 BO：noisy EI 与噪声约束（R10）

### 4.1 观测模型与"best"的随机性

```
y_i = f(x_i) + ε_i,   ε_i ~ N(0, σ_i²)          # σ_i 可来自重复测量
```
单目标下 `f* = max_j f(x_j)` 是**随机变量**。`qNEI`（BoTorch 的 `qNoisyExpectedImprovement`）就是：

```
qNEI(X) = E_{f|D}[ max(0, max_{i=1..q} f(x_i) − max_{j∈baseline} f(x_j)) ]
```

MC 实现（无需嵌套，只需对 baseline 抽样）：

```python
def qnei(model, X_cand, X_base, n_base=32, n_y=64):
    F_base = sample_joint(model, X_base, n_base)      # (n_base, n_b)
    f_star = F_base.max(axis=1)                        # (n_base,)
    total = 0.0
    for k in range(n_base):
        y = sample_joint(model, X_cand, 1).ravel()     # 或一次抽 n_y 取平均
        total += max(0.0, max(y) - f_star[k])
    return total / n_base
```

**与当前 condopt 的关系**：`REVIEW_v01.md` P1-5 指出 `alpha` 是归一化 y 的噪声方差、router 高噪档偏小。在 `qNEI` 落地前，先把 `alpha` 调到 `(σ_rep/y_std)²` 并用 `logei`（P1-2 修完后）——这是低成本改进。

### 4.2 带噪约束（Letham 式）

**问题**：`EI(x) · P(c(x) ≤ 0)` 在约束带噪时有偏——`c` 的观测噪声会让 `P(feasible)` 剧烈震荡，且乘积假设目标与约束独立。

**Letham 2019（R10）的做法**：对 `(f, c)` 建联合模型，直接估计

```
EI_c(x) = E_{f, c | D}[ max(0, f(x) − f*) · 1{ c(x) ≤ 0 } ]
```

其中 `f*` 也按 §4.1 的 baseline 随机处理。MC 估计：

```python
def constrained_noisy_ei(model_f, model_c_list, X_cand, X_base, n_base=64, n_y=128):
    # 独立 GP 近似（联合模型可用 multitask GP 替换）
    Fb = sample_joint(model_f, X_base, n_base)              # (n_base, n_b)
    Cb = [sample_joint(mc, X_base, n_base) for mc in model_c_list]
    f_star = Fb.max(axis=1)
    total = 0.0
    for k in range(n_base):
        fc = sample_joint(model_f, X_cand, 1).ravel()
        cc = np.array([sample_joint(mc, X_cand, 1).ravel() for mc in model_c_list])
        feas = np.all(cc <= 0, axis=0)
        total += np.max(np.where(feas, fc - f_star[k], -np.inf))
    return total / n_base
```
- **噪声进入 `σ`**：`model_c.predict(..., return_std=True)` 的 `std` 应包含观测噪声（对 measurement 噪声建模），否则 `P(feasible)` 会过度自信。实现上：`alpha` 用重复测量的方差，或用 `WhiteKernel` 学噪声。
- **保守裕度**（工程近似，便宜）：把可行性判据改为 `μ_c + κ·σ_c ≤ 0`（`κ=1.0–1.5`），等价于"按观测值判定的保守版"；κ 按风险偏好设。
- **小可行域**：可行比例 <5% 时，前 30% 预算只优化 `P(feasible)`（或"到可行边界的距离"），找到可行域后再切目标（`BENCHMARK_SPEC.md` Tier E 的 `constraint_hard` 就是为这个设计）。

### 4.3 验收

1. **qNEI 正确性**：1D 噪声问题，`qNEI` 与"暴力 MC 积分（大量样本）"相对误差 <5%；
2. **噪声 benchmark**：`noise_20pct` 上 `qNEI` 的 `simple_regret` 应优于现有 `EI + alpha=1e-4` ≥10%；
3. **带噪约束**：`constraint_hard` 上 `EI_c`（联合 MC）的可行点到达次数 ≤ 纯 PoF 乘积的 60%；
4. **不静默**：约束优化输出的每个推荐必须带 `P(feasible)`（点估计 + 区间）。

---

## 5. 概率 Pareto 报告（给实验者的交付物）

噪声下"Pareto 前沿"本身是随机的；不要只给一条点估计前沿。

### 5.1 公式

对每个候选/已观测点 `x`，用 `M` 组后验抽样（各目标联合，或独立近似）：

```
P_nd(x)  = (1/M) Σ_m 1{ f^m(x) 在 S 中非支配 }                 # S = 已观测 ∪ 候选池
P_dom(x, y) = (1/M) Σ_m 1{ f^m(x) ≻ f^m(y) }
E_HVC(x) = (1/M) Σ_m [ HV(P^m ∪ {f^m(x)}) − HV(P^m) ]          # 期望超体积贡献
```

报告表（按 `P_nd` 降序，取前 10）：

| x（条件） | P(非支配) | E[HV 贡献] | P(可行) | 目标均值 ± 标准差 |
|---|---|---|---|---|
| T=88, t=62, MeCN… | 0.94 | 0.021 | 0.99 | y=98.2±1.1, cost=11.1±0.2 |

- 阈值：默认列 `P_nd ≥ 0.5`；高风险场景可要求 `≥ 0.8`。
- `E_HVC` 用于"下一个实验该测哪个"——它与 qNEHVI 的排序高度相关，可交叉验证。
- **校准**：在合成问题上检验 `P_nd` 的可靠性（Brier score < 0.15；把 `P_nd` 分 5 桶，实际非支配频率与桶均值差 <0.1）。

### 5.2 实现要点

1. 抽样数量：`M = 128`（小 m 可 64）；报告用同一批样本算 `P_nd`/`P_dom`/`E_HVC` 以保持一致。
2. 目标相关性：独立 GP 抽样会低估"同高同低"概率 → `P_nd` 偏大；有 `MultiTaskGP`/联合模型时用联合抽样。文档要标注近似。
3. `HV` 用 §2.2 的箱分解或扫描线（m≤3 精确），m≥4 用 MC/近似。
4. 输出 schema（给 skill/LLM 消费）：
```json
{"pareto": [{"x": {...}, "p_nondominated": 0.94, "ehv_contribution": 0.021,
             "p_feasible": 0.99, "objectives": {"yield": [98.2, 1.1], "cost": [11.1, 0.2]}}],
 "ref_point": {"yield": 30.0, "cost": 20.0}, "n_samples": 128, "method": "independent-gp-mc"}
```

---

## 6. condopt 落地计划（文件级）

| 阶段 | 文件 | 内容 | 依赖 |
|---|---|---|---|
| v0.2a | `src/condopt/algorithms/parego.py` | ParEGO 采样器（复用 `bayes` 的 GP/EI）；`ObjectCaps(multi_objective=True, max_objectives=None)` | 现有机制 |
| v0.2a | `src/condopt/multiobjective.py` | 归一化、非支配过滤、2D/3D 精确 HV、2D 分区 `DHV` | 纯 numpy |
| v0.2b | `src/condopt/algorithms/qehvi.py` | q=1 EHVI（MC over 独立 GP 抽样）+ 局部惩罚批量 | 上面两个 |
| v0.2c | `src/condopt/algorithms/qnehi.py` | qNEHVI（嵌套 MC，§3.1）；`X_baseline`/prune | `MultiTaskGP` 或独立 GP 近似 |
| v0.2c | `src/condopt/algorithms/noisy_ei.py` | `qNEI` + 带噪约束 `EI_c`（§4） | 同上 |
| v0.3 | `src/condopt/reporting.py` | 概率 Pareto 表（§5） | HV 工具 |
| 配套 | `router.py` | MOO context 下：`parego`（m≤3, n<40）/`qehvi`（m≤3, n≥40）/`qnehvi`（noise high）；PARETO report | `CAPABILITIES_SPEC` |
| 配套 | `benchmarks.py` | Tier F：`moo_yield_cost`、`mf_suzuki` | `BENCHMARK_SPEC` §3 |

**注意**：现有 `Bayes` 采样器只吃单目标（`Observation.score(objective.name)`）。ParEGO 要绕过 `Optimizer._fit` 的 y 提取，改成"自己构造标量目标数组"。建议在 `base.Optimizer` 加一个 `self._scalar_values()` 钩子（默认返回主目标 score），ParEGO 覆写为 `g_λ`。

---

## 7. 默认参数总表

| 场景 | 算法 | 关键参数 |
|---|---|---|
| 2–3 目标、样本少（n<40）、无噪声 | ParEGO | `rho=0.05, α=1`，每点独立 λ，`n_initial=max(2(d+1),2m)` |
| 2–3 目标、n≥40 | qEHVI | `ref=用户阈值归一化`, `S=128`（q=1）, `q>4` 用局部惩罚 |
| 2–3 目标、噪声 high | qNEHVI | `n_base=32–64, n_y=64–128`, `X_baseline=全部(n≤200)+prune` |
| 单目标、噪声 high | qNEI | baseline=全部，`M=32–64`；先修 `alpha`（R2/R10） |
| 带噪约束 | `EI_c` | 联合 MC（§4.2），`κ=1.0` 保守裕度；可行比例 <5% 时两阶段 |
| 报告 | 概率 Pareto | `M=128`，阈值 0.5/0.8，附 `P(feasible)` 与 `E[HVC]` |

---

## 8. 参考

- R11（ParEGO：增广 Tchebycheff、ρ=0.05、随机权重）
- R12（qNEHVI：噪声多目标的嵌套 MC、baseline 积分）
- R21（qEHVI：可微 EHVI、分区算法）
- R10（Letham：带噪约束的贝叶斯优化、截断后验、避免 EI×PoF 偏差）
- R29（log-EI：噪声/极小改进下采集函数的数值稳定；qNEI/EHVI 内部要用 log 版本）
- R24（综述：多目标/约束/噪声的方法学定位）
- R3（并行、约束、多保真的统一框架）
- R1（EI 的原始定义）
