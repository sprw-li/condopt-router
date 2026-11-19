# BENCHMARK_SPEC — condopt 基准问题集、评估协议与 router 权重校准

> 维护者：algo-research ｜ v1（2026-11-05，Round 2）
> 配套：`REVIEW_v01.md`（P0-2：现有 benchmark 的 optimum 写错）、`REFERENCES.md` R18/R26/R32（Olympus / 初始设计 / HPOBench）。
> 目标读者：实现 `src/condopt/benchmarks.py` 与 `scripts/benchmark.py` 的人。**本文所有"最优值"都给了核验方法（穷举/解析/网格）与实际核验结果**，可直接写进代码不踩 P0-2 的坑。

---

## 1. 设计原则

| 原则 | 要求 | 依据 |
|---|---|---|
| 可核验最优 | 每个问题必须给出 `optimal_config` + `optimum`，并有自检测试（详见 §3 各行"核验"列） | R18（Olympus）、R32（HPOBench） |
| 覆盖真实失效模式 | 不是只测"收敛"，要覆盖：噪声、混合空间、categorical 高基数、约束、多目标、多保真、批量 | R18/R24 |
| 公平协议 | 相同初始设计（同 seed 的 Sobol）、相同总评估数、相同超参预算 | R26（初始设计影响 SMBO 结果）、R32 |
| 可统计 | ≥20 seeds、IQM + bootstrap CI、成对检验；禁止只报 mean±std | R32、HPO 社区惯例 |
| 可校准 | 结果要能直接用于拟合 router 权重（§7），而不只是"打表" | R8/R17（能力/场景选择） |
| 运行成本可控 | 分 smoke（CI，分钟级）/ calibration（夜间）/ full（周级）三档 | R18 的 benchmark 分层思想 |

**成本约定**：`budget` 一律指**评估次数**（`n_evals`），不是墙钟；多保真问题额外记录 `total_cost = Σ cost(fidelity)`，报告两种口径（§5.4）。

---

## 2. 组件架构（建议重构 `benchmarks.py`）

```python
@dataclass(frozen=True)
class BenchmarkTask:
    name: str
    space: SearchSpace
    objectives: tuple[Objective, ...]          # 1 个=单目标，2 个=多目标（Pareto）
    constraints: tuple[ConstraintSpec, ...]    # ConstraintSpec(name, fn, noise_std, direction="<=")
    noise: NoiseModel                          # NoiseModel(kind="gaussian"|"multiplicative", std=..., cv=...)
    fidelity: FidelitySpec | None              # levels, cost per level, "as input" or "discrete"
    optimal_config: dict | None                # 单目标：最优点
    optimum: float | None                      # 单目标：最优值
    pareto_front: list[tuple[float, ...]] | None   # 多目标：数值核验过的参考前沿
    tags: tuple[str, ...]                      # ("continuous"/"mixed"/"discrete"/"noisy"/"constrained"/"multi_objective"/"multi_fidelity")
    budget_range: tuple[int, int]              # 推荐预算区间（用于矩阵，§6）
    notes: str = ""

class Evaluator(Protocol):
    """可替换评估器：synthetic / emulator / live。"""
    def evaluate(self, configs: list[dict], rng: np.random.Generator,
                 fidelity: float | None = None) -> list[EvaluationResult]: ...
```

`EvaluationResult` 至少含：`objectives: dict[str,float]`、`constraints: dict[str,float]`、`cost: float`、`fidelity: float`、`true_objectives`（无噪声真值，仅 synthetic 有，用于 oracle 曲线/校准诊断）。

**关键实现要求**

1. **噪声必须在 Evaluator 里加**（不是加在算法里），且每个 seed 独立；同一 `(config, seed)` 可重复得到相同噪声（`rng` 派生）。
2. `true_objectives` 允许计算"oracle regret"（噪声问题的真实性能），但算法只看 `objectives`（带噪观测）。
3. 同一问题的**初始设计**由协议层统一生成（`n0 = max(5, min(20, 2*(d_eff+1)))`，Sobol scramble，seed 固定），通过 `Optimizer.__init__(prior=...)` 注入，保证所有算法冷启动一致。

---

## 3. 基准问题集（A–F 六层）

> 下表的 optimum 均由脚本核验：`mixed_reaction`/`discrete_catalyst` 为穷举；新增问题为"解析最优 + 网格数值复核"（括号内给出复核值）。**实现时请把 `optimal_config` 也写进代码并加 `test_optima_are_consistent`**。

### Tier A：连续合成（回归基线，无噪声）

| name | d | space | 目标（max 或 min） | optimum | 核验 | 用途 |
|---|---|---|---|---|---|---|
| `branin` | 2 | 单位立方 → [-5,10]×[0,15] | min | **0.397887** | 解析已知 | 经典多峰 |
| `hartmann3` | 3 | 单位立方 | max（返回负值） | **-3.86278** | 经典已知值 | 中等复杂度 |
| `hartmann6` | 6 | 单位立方 | max（负值） | **-3.32237** | 标准基准值（实现后网格复核 ≥-3.34） | 6D 多峰，测 ARD |
| `ellipsoid_d5` | 5 | 单位立方 | max `1 - Σ w_i (u_i-0.3)²/Σw`，`w_i=10^{i/4}` | **1.0**（u=0.3^5） | 解析 | 各维尺度差异大 |
| `rastrigin_d4` | 4 | 单位立方 → [-5.12,5.12]⁴ | min | **0.0**（原点 = u=0.5⁴） | 解析/已知 | 强多峰，检验 BO 失效边界 |
| `sphere2d_log` | 2 | log 空间 [1e-3,1e2] | max | **0.0**（在 log 中点附近定义） | 解析 | 检 log 编码 |

**A 层验收线**：`branin`/`hartmann3` 在 budget=20、batch=1、无噪声时，bayes 的 IQM 应优于 sobol（否则 GP 内核有问题，参见 `REVIEW_v01.md` P1-1/P1-2）。

### Tier B：混合空间（"化学形状"，单目标）

| name | 变量（6D） | 公式 / 说明 | optimum | 核验 |
|---|---|---|---|---|
| `mixed_reaction_v2` | T cont 40–120；time cont 5–120；conc log 0.02–1.0 M；solvent cat 4；equiv int 1–4；inert bool | 同现有 `_mixed_reaction`，**修正 optimum** | **102.0** at T=82, time=55, conc=10^-0.8, MeCN, equiv=2, inert=True | 穷举（801×1151×穷举离散） |
| `suzuki_like` | T cont 40–120；time log 5–240 min；cat_loading log 0.1–5 mol%；solvent cat 5；ligand cat 6；base_eq int 1–4 | `y = 20 + 80·exp(-((T-85)/20)² - (ln(t/60)/0.9)² - (ln(load/1.5)/0.7)²) + sb[solvent] + lb[ligand] + bb[base_eq]`，`sb={MeCN:5,EtOH:3,THF:2,water:-2,DMF:4}`，`lb={L1:4,L2:2,L3:3,L4:1,L5:0,L6:2}`，`bb={1:0,2:1,3:0,4:-1}` | **110.0** at (85, 60, 1.5, MeCN, L1, 2) | 解析（网格复核 109.94，差值为网格分辨率） |
| `mixed_reaction_logy` | 同 v2 | 目标乘性噪声：`y_obs = y·(1+ε), ε~N(0,0.1²)`，测试 log 变换 | 102.0 | 同 v2 |

**B 层要点**：`suzuki_like` 的连续最优在 log 变量中心（60 min、1.5 mol%），能检验"log 编码后 GP 是否把最优放在中心"；categorical 维度含 5/6 级，检验 one-hot 与 ARD（`REVIEW_v01.md` P1-1/P1-6）。

### Tier C：离散 / 高基数分类

| name | 变量 | 组合数 | 公式 | optimum | 核验 |
|---|---|---|---|---|---|
| `discrete_catalyst_v2` | metal cat 3；ligand cat 4；additive cat 4；temp int 40–120 | `base[metal] + lig[ligand] + add[additive] + 3·1{Pt,X3} - 0.002(T-80)²`，`base={Ni:60,Pd:72,Pt:68}`，`lig={A:3,B:1,C:6,D:2}`，`add={X1:2,X2:-1,X3:4,X4:0}` | **82.0** at (Pd, C, X3, 80) | 穷举 3×4×4×81 |
| `high_cardinality_solvent` | T cont 60–120；solvent cat **12 级**（每级带 2D 描述符 `(polarity, protic)`） | `y = 50 + 25·exp(-((pol-0.8)/0.35)² - ((protic-0.2)/0.6)²) + 20·exp(-((T-90)/25)²)` | **92.371** at S04（`pol=0.8,protic=0`），T=90 | 穷举 12×601 |
| `all_discrete_grid` | 2 个 cat(3,4) + 1 int 1–5 | `y = 10 - Σ 惩罚`（每维一个偏好值） | 组合数 60，可穷举 | 穷举 |

**C 层用途**：①验证 grid/穷举在 `card ≤ budget` 时的优势（router 现有规则）；②`high_cardinality_solvent` **必须同时实现 one-hot 与描述符两条特征路径**，用来证明描述符方案（R22 Gryffin）优于 12 维 one-hot。

### Tier D：噪声（同问题、不同噪声档）

对 Tier B 的 `mixed_reaction_v2` 生成 3 个变体：

- `noise_none`：`y_obs = y`
- `noise_5pct`：多重复测量模型——每次评估 `k=3` 次取均值，观测噪声 `N(0,(0.05·y)²)`；`Observation` 里记录 `std`
- `noise_20pct`：`N(0,(0.20·y)²)`，`k=1`

**验收**：①`bayes` 在 20% 噪声下不应显著差于 `sobol`（当前实现会因 alpha 偏小而过拟合，见 P1-5）；②重复测量聚合开启后，`alpha` 估计应接近 `σ²/y_std²`（单测）。

### Tier E：约束

统一目标：`y = 60 + 35·exp(-((T-110)/20)² - ((time-100)/45)²)`（无约束最优 **95.0** at (110,100)）。
约束（椭圆可行域）：`g(T,time) = ((T-95)/rT)² + ((time-80)/rt)² - 1 ≤ 0`

| name | rT, rt | 实测可行比例（MC 30 万点） | 约束最优（网格 2001×4001） | 说明 |
|---|---|---|---|---|
| `constraint_easy` | 25, 60 | **25.0%** | **95.0** at (110, 100)（约束**不**在最优处 binding） | 测"过滤不影响最优" |
| `constraint_hard` | 12.5, 30 | **6.2%** | **93.096** at (105.9, 94.8)（约束 binding） | 测可行性感知搜索 |

**验收**：①约束优化必须记录 `n_feasible` 与"首个可行点出现在第几次评估"；②`constraint_hard` 上，纯目标优化 + 事后过滤的算法应显著差于 PoF 加权（否则说明约束逻辑未生效）。

### Tier F：多目标与多保真

| name | 目标 | 前沿/最优 | 核验 |
|---|---|---|---|
| `moo_yield_cost` | max `y`（同 Tier B 内核，T∈[60,115]）+ min `cost = 0.1·T + 0.05·t` | Pareto 前沿 **101 个点**；端点 `(y=109.0, cost=11.5)`（T=85,t=60）与 `(y=45.77, cost=9.0)`（T=60,t=60） | 枚举 T 网格 221 × t 网格 161 × 类别组合后做非支配过滤 |
| `mf_suzuki` | 同 `suzuki_like`，fidelity `s∈[0.1,1]`（"短时试跑"），`y_s = y - 15·(1-s)^{1.5}`，`cost(s)=s` | 全保真 optimum 110.0；`s=0.5` 时最优值 ≈ 110 - 15·0.354 ≈ 104.7 | 解析（bias 项与 x 无关） |

**F 层验收**：多目标报告 HV（固定 ref point：目标归一化到 `[0,1]` 后取 `(1.1, 1.1)` 或 `nadir+0.1·range`）；多保真报告 "单位成本 best-so-far"。

---

## 4. 统一评估协议（Protocol）

```python
def run_task(task, algorithm, *, budget, batch_size, seed,
             n0=None, params=None, initial_design=None) -> RunRecord:
    rng = np.random.default_rng(seed)
    init = initial_design or make_initial_design(task.space, seed, n0)
    obs = []
    for cfg in init:                      # 统一初始设计，不计入算法差异
        obs.append(task.evaluator.evaluate([cfg], rng)[0])
    opt = registry.build(algorithm, task.space, task.objectives[0],
                         params=params, seed=seed, prior=to_observations(obs))
    best = best_value(obs, task)
    trajectory = [best]
    while len(obs) < budget:
        n = min(batch_size, budget - len(obs))
        cfgs = opt.ask(n)
        results = task.evaluator.evaluate(cfgs, rng)      # 批量并行 = 同时评估
        opt.tell(to_observations(results))
        obs.extend(results)
        trajectory.append(best_value(obs, task))          # best-so-far（每批记录）
    return RunRecord(task.name, algorithm, budget, batch_size, seed, trajectory, obs, wall_clock)
```

**协议细节（必须写进代码注释，避免"跑得不一样"）**

1. **初始设计不计入算法差异**：所有算法共享同一批 `n0` 点（由 seed 决定），算法从 `prior` 开始。`n0` 建议：连续 d≤6 用 8；混合 6–10 维用 10；categorical 高基数用 12–15（与 `REVIEW_v01.md` P0-6 一致）。
2. **`best` 定义**：单目标 `max(score)`；多目标返回非支配集；噪声问题可选 `best_mean`（对重复测量的均值）。
3. **批量语义**：`ask(n)` 一次性给 n 个，`tell` 一次性回；异步实验的 benchmark 用"随机延迟回传"模拟（可选，见 Tier D 异步变体）。
4. **重跑**：每个 `(task, algo, budget, batch, seed)` 组合只跑一次；跑两次不同即视为环境不干净（写入 CI 检查）。
5. **失败处理**：Evaluator 抛异常 → 该次 run 标记 failed（不静默丢弃），统计里单独报失败率。

---

## 5. 指标定义（可直接实现）

记第 n 次评估后的 best-so-far 为 `b_n`，`opt` 为核验过的真实最优（多目标用参考前沿计算 HV）。

### 5.1 单目标

```
best_so_far(n)      = b_n                                  # 原尺度，画曲线用
simple_regret(n)    = (opt - b_n) / (opt - b_0)             # 归一化，0 最好；b_0 = 初始设计最优
sample_efficiency   = clamp((b_B - b_0) / (opt - b_0), 0, 1)  # B = 总预算
threshold_hit(k)    = 1[ b_B ≥ b_0 + k·(opt - b_0) ]         # k∈{0.5,0.8,0.95}
auc_best_so_far     = trapz(b_n, n) / (B · (opt - b_0))     # 样本效率（面积归一化）
n_to_threshold(k)   = min{ n : b_n ≥ b_0 + k·(opt-b_0) }    # 达到阈值所需评估数（censored）
```

**报告首选**：`simple_regret(B)` 的 IQM + `auc_best_so_far` 的 IQM；曲线画 `best_so_far(n)` 的中位数与 25/75 分位带。

### 5.2 多目标

```
HV_n            = hypervolume(nondominated(b_1..b_n), ref_point)   # 固定 ref point、目标先归一化
HV_ratio(n)     = HV_n / HV(front_ref)                             # 以参考前沿 HV 归一化
IGD_n           = mean_{p in front_ref} min_{q in front_n} ||p-q||  # 需要 front_ref
EHV_contribution(x) = E[HV(S ∪ {x}) - HV(S)]                       # 概率 Pareto 报告用
```

### 5.3 噪声

- 每个配置的噪声重复次数 `k` 固定（Tier D 用 k=3）；`best` 用重复均值。
- 报告 `best_so_far` 的"乐观偏差"：`b_n` 与 `true_best(x)`（无噪声真值）的差——合成问题才有。
- 报告"推荐点的真实性能" `true_y(x_best)`（oracle 可比）。

### 5.4 成本感知（多保真）

```
cost_best_so_far(C) = best b 满足 Σ cost(evaluations up to b) ≤ C
sample_efficiency_per_cost = (b_B - b_0) / (opt - b_0) / total_cost
```

### 5.5 算法自身健康指标（可选，用于 router 诊断）

`mean_fit_ms`、`mean_suggest_ms`、`gp_lengthscale_at_bound_frac`、`n_candidates_effective`、`reroute_count`、`failed_runs`。这些是 `REVIEW_v01.md` P2-3 建议写入 decisions 的字段，benchmark 也要采集。

---

## 6. 预算 × 批量 × 种子矩阵

**三档，避免一次性跑爆**：

| 档位 | 问题集 | 算法 | 预算 | 批量 | seeds | 估计 run 数 | 用途 |
|---|---|---|---|---|---|---|---|
| `smoke`（CI，<5 min） | A:branin, A:sphere2d_log, B:mixed_reaction_v2, C:discrete_catalyst_v2, D:noise_20pct | random, sobol, bayes | 16 | 1 | 3 | 45 | 每 PR 回归 |
| `frontier`（夜间，~1–2 h） | A(6) + B(3) + C(3) + D(3) + E(2) | random, sobol, lhs, bayes | {8,16,32} | 1,4 | 8 | ~17×4×3×2×8 = **3264** | router 校准主数据 |
| `full`（周级） | 全部（含 F、多保真、高基数） | 全部注册算法 | {8,16,32,64} | 1,4,8,16 | 20 | 数千–万 | 发版/论文级 |

**批量矩阵要点**：`batch ∈ {1,4,8,16}` 与 `budget` 解耦——同一 budget 下比较 batch 大小对 `best_so_far` 的影响；`batch > budget/2` 的组合无意义（跳过）。高基数 categorical 空间强制 `batch ≥ 4`（天然并行）。

**硬件/时间预算估算**：每次 `ask` 的 bayes 成本 ≈ `n_candidates × feature_dim` 的 GP 预测 + `n-1` 次重拟合；`n_candidates=1024, d=10` 时单次 ask ≈ 0.1–0.5 s（CPU）。3264 runs × 16 evals × 0.3 s ≈ 4.4 h → 建议并行 4–8 进程（`scripts/benchmark.py --workers`）。

---

## 7. Router 权重校准（可执行方法）

### 7.1 现状与目标

当前 router 是**手写加法打分**：`score_a(ctx) = Σ_k rule_k(a, ctx)`，规则里是常数（45、+18、-35…）。`docs/ROUTING_POLICY.md` 已承认"权重未校准"。目标：让"router 选择的算法"在基准上尽量接近"该场景的最优算法"。

### 7.2 步骤

**Step 1：生成校准数据**（用 `frontier` 矩阵，含 batch/noise 变体）

```python
rows = []   # one row per (task, algo, budget, batch, seed)
for task in tasks:
    for algo in algorithms:
        for budget in budgets:
            for batch in batches:
                for seed in seeds:
                    rec = run_task(task, algo, budget=budget, batch_size=batch, seed=seed)
                    rows.append(dict(task=task.name, algo=algo, budget=budget, batch=batch,
                                     seed=seed, regret=norm_regret(rec), auc=auc(rec),
                                     tags=task.tags))
```

**Step 2：生成"ground truth"排序**（每个 context 一个标签）

context 的粒度决定校准难度，建议两档：

- `ctx_key = (task, budget_bucket, batch_bucket)`：任务级（最细、最可靠，但无法泛化到新任务）；
- `ctx_key = (tags, feature_dim_bucket, budget_bucket, batch_bucket)`：场景级（可泛化，用于 router）。

对每个 `ctx_key`，按算法聚合（IQM 或中位 regret），得到排序 `best_algo`。**要求**：该 context 至少 8 个 seed、且 top1 与 top2 的 IQM 差 ≥ 5%（否则标记为 tie，不参与 top-1 监督，只算 top-2）。

**Step 3：把规则重构成"可加权特征"**

把 `_score_bayes/_score_sobol/...` 改为返回三类项：

```python
@dataclass
class ScoreTerm:
    feature: str        # "dim_le_8" | "budget_ge_20" | "noise_high" | "batch_ge_8" | "categorical" | ...
    delta: float        # 当前经验值
    reason: str
    weightable: bool = True
```

`score_a(ctx) = base_a + Σ_k w_k · Σ_terms delta_{k}`。**保持可解释性**：最终报告里每个 term 显示 `w_k * delta` 与理由。

**Step 4：拟合权重**（目标函数）

```python
def router_objective(w, data) -> float:
    total = 0.0
    for ctx in data.contexts:
        scores = {a: router_score(a, ctx, w) for a in algorithms}
        chosen = max(scores, key=scores.get)
        label = ctx.best_algo
        rank_ok = chosen == label
        top2_ok = chosen in ctx.top2
        regret = ctx.iqm_regret[chosen] - ctx.iqm_regret[label]      # 相对最优算法的损失
        total += (0.5 * (1.0 - rank_ok) + 0.3 * (1.0 - top2_ok)
                  + 0.2 * regret / (ctx.iqm_regret_worst - ctx.iqm_regret_best + 1e-9))
    return total / len(data.contexts)      # 越小越好
```

**搜索方法**：规则数 k≈20–40，非凸但低维 → 随机搜索 200–500 组 + 坐标下降精修；每组评估 <1 s（纯查表）。搜索空间约束：`w_k ∈ [0.5, 2.0]`（乘法校准，保持原规则的符号与相对大小），base 项允许 ±10 的加法偏移。

**Step 5：防过拟合（必须做）**

- **Leave-one-problem-out（LOPO）CV**：每次留出一个 task 的全部 seeds 作为测试；报告 held-out top-1/top-2 命中率与 regret 损失。
- 只用 `frontier` 数据训练，`full` 数据做最终验证。
- 记录权重的**符号变化**：如果某条规则权重被学反（如"高维 → bayes 加分"），说明规则本身错，应改规则而不是拟合权重。

**Step 6：验收阈值（建议写进 CI）**

| 指标 | 门槛 |
|---|---|
| held-out top-1 命中率 | ≥ 0.60 |
| held-out top-2 命中率 | ≥ 0.85 |
| 相对 oracle 的 regret 损失（IQM） | ≤ 5% |
| 与旧手写权重相比 | 不劣化（回归门禁） |
| 规则符号合理性 | 100%（人工 review 权重表） |

**Step 7：落盘格式**

```json
{
  "version": "router-weights-2026-11-05",
  "trained_on": {"matrix": "frontier", "n_contexts": 42, "seeds": 8},
  "base": {"bayes": 45.0, "sobol": 25.0, "lhs": 22.0, "random": 15.0, "grid": 5.0},
  "weights": {"dim_le_8": 1.0, "dim_ge_20": 0.6, "noise_high": 1.4, "budget_ge_20": 1.2,
              "batch_ge_8": 1.3, "categorical_frac": 0.9, "prior_ge_6": 1.1},
  "validation": {"lopo_top1": 0.64, "lopo_top2": 0.88, "regret_gap": 0.041}
}
```

`router.py` 从该 JSON 加载（缺省用代码内默认），权重文件纳入版本管理；每次校准产出新文件，旧文件保留（可回滚）。

---

## 8. 报告与结果格式

- **原始轨迹**：`runs/bench/<task>/<algo>__b<budget>_ba<batch>_s<seed>.jsonl`（每行一次评估：`n, x, obs, true, cost, t`）。
- **汇总**：`bench_summary.csv` 列：`task, algo, budget, batch, n_seeds, iqm_regret, iqm_auc, ci_low, ci_high, iqm_hv_ratio, failed_rate, mean_suggest_ms`。
- **图**：每个 task 一张 `best_so_far` 曲线（中位数 + 25/75 带），一张算法 IQM 排名热图（task × algo）。
- **统计**：IQM（去掉最好/最差各 25% 后取均值，抗重尾）+ bootstrap 95% CI（1000 次重采样）；成对比较用 Wilcoxon signed-rank + Holm 校正；效应量用 Cliff's delta。**禁止**把 different ref point 的 HV 放在一起比。

---

## 9. 实现清单（给 Lead 的 sprint 建议）

1. **S1（半天）**：修正现有 5 个问题的 `optimal_config/optimum` + 自检测试（`REVIEW_v01.md` P0-2）；给 `BenchmarkProblem` 加 `tags`、`noise`、`optimal_config`。
2. **S1（半天）**：`run_task` 统一初始设计（`prior` 注入）+ `RunRecord` 落盘 + 归一化 regret/auc。
3. **S2（1–2 天）**：加 Tier B `suzuki_like`、Tier C `high_cardinality_solvent`（含描述符路径）、Tier D 噪声变体、Tier E 约束变体。
4. **S2（1 天）**：`smoke` 档接入 CI（5 问题 × 3 算法 × 3 seeds × budget16，<5 min），设回归门槛（IQM regret 不劣化 >10%）。
5. **S3（2–3 天）**：`frontier` 矩阵 + router 权重校准脚本（§7）+ 权重 JSON 落盘 + LOPO 报告。
6. **S4**：Tier F（MOO/多保真）在 `MULTI_OBJECTIVE_AND_NOISE.md` 的算法实现之后接入。

**验收**：`smoke` 在 CI 稳定 <5 min；`frontier` 能产出 `router-weights-*.json` 且 LOPO top-2 ≥0.85；benchmark 报告里每个最优值都有 `optimal_config` 复核。
