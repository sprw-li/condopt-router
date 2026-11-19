# CONSTRAINTS_NOISE_IMPL — 带约束/噪声 BO 的代码级实现

> 维护者：algo-research ｜ v1（2026-11-05，Round 3）
> 对应：`MULTI_OBJECTIVE_AND_NOISE.md` §4（原理）、`BENCHMARK_TIER_CD.md` §4/§5（`noisy_sphere`、`constraint_easy/hard`）。
> 文献：R34（Gardner 2014，EI-CDF 基线）、R10（Letham 2019，带噪约束）、R23（GP-UCB 的探索系数）、R2（GP/alpha 工程细节）。
> 目标：在 condopt 现有 `GaussianProcessBO` 上**加两个采集函数**（`ei_cdf`、`noisy_ei_constrained`）和一个观测聚合层，不做大重构。

---

## 1. 观测模型

### 1.1 目标与约束

```
目标:  y = f(x) + ε,     ε ~ N(0, σ_y²(x))
约束:  c_j(x) = g_j(x) + ε_j,  ε_j ~ N(0, σ_j²(x)),  可行 ⇔ g_j(x) ≤ 0
```

- `σ` 可以随 x 变化（乘性噪声时对 y 取 log：`log y = log f + ε`, `σ_log ≈ CV`）。
- 每个观测可带 `k` 次重复（**复孔**）：记录 `y_mean`、`y_std`、`k`；GP 用 `y_mean` 训练，`alpha` 用 `σ²/k` 折算。

### 1.2 当前 condopt 的数据模型（已支持）

`Observation(x, y, objectives, constraints, fidelity, cost, meta)`：`constraints={"c_j": value}` 已经能存；缺的是"约束值的 std"与"样本量"。**建议扩展**（向后兼容）：

```python
@dataclass
class Observation:
    ...
    objective_std: dict[str, float] = field(default_factory=dict)  # {"yield": 0.4}
    constraint_std: dict[str, float] = field(default_factory=dict) # {"c1": 0.2}
    n_replicates: int = 1
```

`Study.tell` 从结果 JSON 里透传这三个字段（缺省 0 / 1）。

---

## 2. 基线：EI-CDF（Gardner 2014，R34）

```
EI_c(x) = EI(x) · Π_j P(c_j(x) ≤ 0),      P(c_j ≤ 0) = Φ( -μ_j(x) / σ_j(x) )
```

- 每个约束一个 GP（对 `c_j` 的训练值建模），`μ_j, σ_j` 来自 GP 后验。
- **优点**：实现 10 行；对"确定性约束"（测量误差远小于约束余量）效果好；候选池可先过滤明显不可行点。
- **问题（无噪声时也存在的近似）**：假设目标与约束独立；`P(feasible)` 是**逐点乘积**，不是联合可行性；当可行域很小时乘积迅速趋 0 → 数值下溢/梯度消失。
- **带噪时更糟**：约束 GP 的 `σ_j` 把观测噪声当"不确定性"（除非 alpha 设对），导致 `P(feasible)` 随重复测量剧烈震荡；而 `EI×PoF` 在约束噪声大时**系统性低估**期望改进（因为乘积不是期望的乘积）。

**适用**：约束可视为确定性（或 σ_c/|余量| < 0.2）、可行域 ≥10% 体积。

---

## 3. 推荐：Letham 式带噪约束 EI（R10）

### 3.1 目标

```
EI_c(x) = E_{f, g | D}[ max(0, f(x) − f*) · 1{ g_1(x) ≤ 0, ..., g_J(x) ≤ 0 } ]
```

其中 `f* = max_{j∈baseline} f(x_j)` 在噪声下也是随机变量（与 qNEI 相同）。

### 3.2 MC 估计（可直接实现，独立 GP 近似）

```python
def constrained_noisy_ei(model_y, models_c, X_cand, X_base,
                         n_base=64, n_y=128, margin=0.0):
    """最大化版本。margin>0 表示保守（要求 g + margin <= 0）。"""
    # 1) baseline 抽样：f* 的分布
    Fb = sample_joint(model_y, X_base, n_base)          # (n_base, n_b)
    f_star = Fb.max(axis=1)                              # (n_base,)
    # 2) 候选点联合抽样（目标 + 每个约束）
    Y = sample_joint(model_y, X_cand, n_y)               # (n_y, n_cand)
    C = np.stack([sample_joint(mc, X_cand, n_y) for mc in models_c], axis=0)  # (J, n_y, n_cand)
    # 3) 逐 baseline 样本估计
    total = 0.0
    for k in range(n_base):
        imp = np.maximum(0.0, Y - f_star[k])             # (n_y, n_cand)
        feasible = np.all(C + margin <= 0.0, axis=0)     # (n_y, n_cand) bool
        total += np.mean(np.where(feasible, imp, 0.0))
    return total / n_base                                # (n_cand,)
```

要点：

1. **必须联合抽样**：同一组 MC 样本里，`f` 与 `g_j` 相关也没关系（独立 GP 近似下等价于对每个样本用同一随机数池）；`sample_joint` 实现用 `model.predict(..., return_cov=True)` + Cholesky（与 `bayes._sample_posterior` 相同，但要"一次抽 n_y 条"）。
2. **`f*` 用 baseline 抽样**：与 `qNEI` 一致；`X_base` 建议取全部已测点（n≤200），噪声大时 `n_base=64`。
3. **`margin`（保守裕度）**：`g + κ·σ_c ≤ 0` 的等价实现；工程上直接 `margin = κ * σ_c_median`（`κ=1.0` 保守）。风险偏好高时 `κ=0`（按后验均值判可行）、安全边界场景 `κ=1.5`。
4. **噪声估计**：`σ_c` 来自重复测量；没有复孔时用 GP 学到的噪声（`alpha`）+ 先验；**不要**把 `σ_c` 当 0（会过度自信）。
5. 计算量：`n_cand=1024, n_y=128, n_base=64, J=2` → 约 `64×128×1024×2 ≈ 1.7e7` 次浮点运算，numpy 向量化后 <0.5 s/轮，可接受。可先对候选池过滤 `P(feasible) < 0.05` 的点再算。

### 3.3 与 EI-CDF / 罚函数的对比

| 维度 | EI-CDF（R34） | 带噪约束 MC（R10 风格） | 罚函数（additive penalty） |
|---|---|---|---|
| 公式 | `EI·Π PoF` | `E[max(0, f−f*)·1{feasible}]` | `EI on y − λ·Σ max(0, c_j)` |
| 噪声约束 | 差（PoF 抖） | **好**（显式积分） | 差（罚权 λ 难定、有偏） |
| 小可行域 | 乘积下溢 | 需大 `n_y`；可两阶段 | 容易全不可行 |
| 目标/约束相关 | 忽略 | 可建模（联合 GP） | 忽略 |
| 调参 | 无 | `n_base/n_y/margin` | `λ`（量纲强依赖） |
| 实现成本 | 10 行 | ~60 行 + 抽样工具 | 5 行 |
| 建议 | 确定性约束、可行域 ≥10% | **默认（噪声/小可行域）** | 仅作兜底/对照 |

**结论**：`noisy` 或 `constraint_hard`（可行 6.2%）场景用 §3.2；`constraint_easy`（可行 25%、σ_c 小）用 EI-CDF 足够；罚函数只在"约束无法建模（离散黑箱规则）"时用，并且要报告 `λ` 的选择依据。

### 3.4 两阶段（可行域 <5%）

```
阶段 A（前 30% 预算）：只优化 P(feasible)（= PoF 或"到可行边界距离"），找到一个可行点后进入阶段 B
阶段 B：带约束 EI（上面的 MC），并保证每批至少 1 个"探索可行性"点
```
`BENCHMARK_TIER_CD.md` 的 `constraint_hard`（6.2%）用于验证：两阶段在预算 30 时首个可行点出现次数应显著少于单阶段 EI-CDF。

---

## 4. 复孔 / 重复测量 → alpha 与观测聚合

### 4.1 聚合

```python
def aggregate(reps: list[dict]) -> dict:
    return {
        "mean": {k: float(np.mean([r[k] for r in reps])) for k in reps[0]},
        "std":  {k: float(np.std([r[k] for r in reps], ddof=1)) for k in reps[0]},
        "n": len(reps),
    }
```
- 同一 `x` 的 `k` 次测量 → GP 训练用均值；`Observation.objective_std` 存 `std`。
- **不要**把 k 次重复当 k 个独立训练点塞进 GP（相当于重复计数，后验方差被错误压缩）；除非把噪声调到 `alpha` 里。

### 4.2 alpha 映射（sklearn `normalize_y=True` 下的正确公式）

sklearn 先把训练 y 标准化为 `(y − mean)/std`，`alpha` 加在**标准化空间**的对角线。若观测噪声方差为 `σ_obs²`（对训练目标而言；若用 k 次均值则为 `σ_rep²/k`）：

```
alpha = (σ_obs / y_std)²          # y_std = 训练 y 的标准差（sklearn 内部计算）
```

实现建议：

```python
def estimate_alpha(observations, y_std, default=1e-4):
    vars = []
    for o in observations:
        if o.n_replicates > 1 and o.objective_std:
            vars.append((o.objective_std[name] ** 2) / o.n_replicates)
    if not vars:
        return default
    return float(np.mean(vars)) / (y_std ** 2 + 1e-12)
```

- 没有复孔时：按噪声档位给默认（`none=1e-8, low=1e-6, medium=5e-3, high=3e-2`），与 `REVIEW_v01.md` P1-5 一致。
- 目标取 log 后：`alpha_log ≈ ln(1+CV²) ≈ CV²`（CV<0.3 时）。
- 约束 GP 的 alpha：同理用 `constraint_std²/k / c_std²`；约束的"测量噪声"要保留，因为 `P(g≤0)` 判据依赖它。

### 4.3 报告

每个推荐点输出：`pred_mean`、`pred_std`、`P(feasible)`（均值与区间）、`n_replicates`、`alpha_used`。`Study.decisions` 记录 alpha 估计来源（replicate/default），便于审计。

---

## 5. 实现骨架（加在 `bayes.py` 或新文件 `algorithms/constrained.py`）

```python
# 采集函数选择：ParamSpec("acquisition", choices=("ei","logei","ucb","pi","ts","ei_cdf","noisy_ei_c"))
# 新增 ParamSpec("constraint_margin", "float", 0.0, 0..3)
# 新增 ParamSpec("n_base", "int", 64, 8..512)；("n_y","int",128, 16..1024)

def _constraint_models(self, X, obs):
    models = []
    for name in self.objective.constraints:
        c = np.array([o.constraints.get(name, np.nan) for o in obs])
        mask = ~np.isnan(c)
        models.append(self._fit(X[mask], c[mask]))   # 复用父类 GP（alpha 按 §4.2）
    return models

def _acq_constrained(self, y_model, c_models, features):
    if self.params["acquisition"] == "ei_cdf":
        acq = self._acquisition(y_model, features)          # 复用父类 EI/logEI
        for cm in c_models:
            mu, sd = cm.predict(features, return_std=True)
            acq *= norm.cdf((0.0 - mu) / np.maximum(sd, 1e-12))   # g<=0 可行
        return acq
    # noisy_ei_c (MC)
    return constrained_noisy_ei(y_model, c_models, features, self._X,
                                n_base=int(self.params["n_base"]),
                                n_y=int(self.params["n_y"]),
                                margin=float(self.params["constraint_margin"]))
```

批量：对 q 个候选点，`noisy_ei_c` 里对 q 取 max（与 qEI 一致）；或 q=1 + 局部惩罚（`SPRINT2_SPEC.md`）。

**候选过滤**：`constraints` 便宜时，先算 `P(feasible)`，把 `<0.02` 的点从候选池剔除（省算力，也别让它污染 argmax）。

---

## 6. 参数与验收

| 参数 | 默认 | 说明 |
|---|---|---|
| `acquisition` | `noisy_ei_c`（有约束时） | 约束确定性且可行域大 → `ei_cdf` |
| `constraint_margin` | 0.0（正常）/1.0（保守） | 单位是约束值单位；对安全边界用 1.5 |
| `n_base` | 64 | baseline 抽样数 |
| `n_y` | 128 | 候选抽样数；q>1 时适当降 |
| `alpha` | 由 §4.2 估计 | 复孔时必用 |

**验收**

1. `noisy_sphere`（σ=0.05/0.15，replicates 1/3）：`alpha` 估计与真值误差 <50%；`qNEI`/`logei` 在 20 次评估内 best ≥ 0.9；
2. `constraint_easy`：EI-CDF 与 MC 版本 `best_so_far` 差距 <5%；两者都 ≥ 无约束 BO 的可行最优；
3. `constraint_hard`（可行 6.2%）：MC 版本的"首个可行点评估次数"显著低于 EI-CDF（Welch t-test p<0.05，10 seeds），且最终 best ≥ 90% 约束最优（93.096）；
4. 约束值带噪（σ_c=0.5）：EI-CDF 明显劣于 MC 版本（≥10% best 差）；
5. 所有推荐输出都带 `P(feasible)`；未建模约束时 `acquisition` 回退并 warning。

---

## 7. 参考

R34（EI-CDF：`EI·P(feasible)` 的原始文献与适用条件）、R10（Letham：带噪约束的截断后验/期望改进，本文 MC 估计是其可落地近似）、R2（噪声/重复观测与 GP alpha）、R23（UCB 的理论背景；约束场景不做固定 κ UCB）。
