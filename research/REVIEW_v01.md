# REVIEW_v01 — 对 `src/condopt` v0.1 的文献级实现审查

> 审查者：algo-research ｜ 日期：2026-11-05 ｜ 对应实现：condopt v0.1（38 单测版本）
> **审查快照**：`src/condopt/**` 读取时间为 2026-11-05 14:20–14:31。期间该目录被并发改写（14:26–14:28 四个文件出现过临时编码损坏，随后恢复；14:30 测试 `import condopt` 正常）。**行号以 14:30 快照为准**，打补丁前请先 `grep` 确认锚点仍在。
> 每条格式：**`文件:行` → 问题 → 文献依据 → 修复建议（可落地代码/公式）→ 优先级**。
> 优先级定义：**P0 = 行为错误/正确性风险，会直接导致实验推荐偏离或数据/报告失真**；P1 = 显著影响样本效率或可解释性，应在下一轮修；P2 = 工程质量/体验。

---

## 0. 总结（先看这张表）

| 编号 | 位置 | 一句话 | 优先级 |
|---|---|---|---|
| P0-1 | `algorithms/bayes.py:333-341` | TS 的"后验采样"既没做后验条件化、量纲也没乘 `y_std`，`ts` 实际上退化为"后验均值 + 小抖动" | **P0** |
| P0-2 | `benchmarks.py:148,162` | 两个化学基准的 `optimum` 写错（101.6 应为 **102.0**；81.0 应为 **78.0**），regret 恒为 0、benchmark 结论不可信 | **P0** |
| P0-3 | `router.py:94-109,221-232` + `space.py:298` | router 用变量个数 `dim` 判断 GP 是否可靠，但 GP 实际输入是 `feature_dim`（one-hot 膨胀）。3 个 20 级 categorical + 1 连续 → `dim=4, feature_dim=61`，router 仍给 bayes 80 分 | **P0** |
| P0-4 | `study.py:296-306` | `override` 记录后，下一次 `ask` 会被 `auto_route` 立刻改回 router 的选择——override 形同虚设 | **P0** |
| P0-5 | `study.py:349-377` | `tell` 收到部分结果时把**整批 pending 清空**（未回测的建议静默丢失）；重复 `tell` 会向 dataset 追加重复观测（非幂等） | **P0** |
| P0-6 | `bayes.py:182-186` + `router.py:386-389` | 初始化点数用 `dim+1`；对 61 维 feature 只给 5 个初始点，GP 超参优化必然过拟合 | **P0** |
| P1-1 | `bayes.py:36-47,73` | 核是**标量各向同性** length_scale（sklearn 默认），没有 ARD；categorical one-hot 与连续维共享一个尺度 | P1 |
| P1-2 | `bayes.py:318-325` | `logei` 只是 `log(EI)`，EI 先下溢成 0 再 clamp 到 1e-300 → 在真正需要 log 的区域所有候选并列为 -690.8 | P1 |
| P1-3 | `bayes.py:214-226` | constant-liar 的伪观测**以零噪声**进入 GP，且"谎言值"取 `max(y)-std(y)`（不够悲观）→ 批量点仍会扎堆 | P1 |
| P1-4 | `bayes.py:233-237` | TS 批量 = 同一条后验样本的 top-n → 批量内点彼此高度相关（扎堆） | P1 |
| P1-5 | `bayes.py:299-305` + `router.py:393-398` | `alpha` 是**归一化 y 的噪声方差**，但 router 映射（高噪 5e-3）偏小约 4–10 倍；也没有用重复测量方差或 WhiteKernel | P1 |
| P1-6 | `space.py:371-383` | `features` 把所有离散变量都 one-hot（ordinal 丢序）；categorical 高基数维度爆炸；无 descriptor 机制 | P1 |
| P1-7 | `space.py:396-410` / `sampling.py:146-166` | Sobol 每次调用新建采样器（增量调用破坏平衡性）；dedupe 后可能返回 `< n` 个点且无提示 | P1 |
| P1-8 | `study.py:310-317` + `base.py:202-204` | 每轮 `ask` 都用同一个 `seed` 重建 Optimizer → RNG 流重置，候选池随机部分每轮重复 | P1 |
| P1-9 | `router.py:203-210,476-535` | `AlgorithmSpec.capabilities` 只是一组无结构字符串，router **完全没有能力过滤**，靠硬编码规则 dict | P1 |
| P1-10 | `router.py:214-218,478-482` + `base.py:139-173` | 多目标只是"扣 200 分 + 警告"，然后静默按主目标优化；没有 ParEGO/EHVI 路径 | P1 |
| P1-11 | `bayes.py:203-231` | 批量>4 仍用逐点贪心 + 每次重拟合 GP；未用 qEI/local-penalization，q≥8 时质量与耗时都变差 | P1 |
| P1-12 | `bayes.py:239-286` | 候选池 50% uniform + 50% Sobol（两者信息重复），local 池以 `top-5 by y` 为中心（噪声下选错中心），无 pending 排除 | P1 |
| P2-1 | `bayes.py:38` | length_scale_bounds 由当前 `length_scale` 参数推导，改初始值会把可行区间整体搬走 | P2 |
| P2-2 | `bayes.py:320` | `xi=0.01` 是原始 y 单位，量纲敏感（y≈100 时无意义，y∈[0,1] 时过大） | P2 |
| P2-3 | `bayes.py:298` | 超参 `n_restarts_optimizer` 固定 1；`ConvergenceWarning` 被完全吞掉，没有"lengthscale 撞边界"诊断 | P2 |
| P2-4 | `space.py:189-194` | continuous 的 `validate_value` 不检查 `[low, high]`，越界值能进 encode（u>1） | P2 |
| P2-5 | `space.py:357-360` | `decode` 对 categorical 用 `int(u*k)`，与 `encode` 的 `idx/(k-1)` 不对称（可逆但语义不同），易踩坑 | P2 |
| P2-6 | `study.py:318-325,341-344` | `opt_state.json` 只存"当前算法"的状态；reroute 后切回原算法会丢计数器（Sobol 从头开始） | P2 |
| P2-7 | `study.py:345-346` / `_write_suggestions` | `suggestions.csv` 每次覆盖写，历史建议不留档 | P2 |
| P2-8 | `study.py:441-466` | `summary()` 里 `route(ctx)` 被调用两次（doubling 计算 + 日志不一致风险） | P2 |
| P2-9 | `study.py:423-438` | `override` 不校验能力兼容（例如把不支持 batch 的算法设到 batch=8 的 study） | P2 |
| P2-10 | `space.py:431-467` | `sample_grid` 按字典序排列 → 预算小于网格时只覆盖第一个维度的低端；粗化可能重复 level | P2 |
| P2-11 | `space.py:371-389` | `features_many` 逐点 `validate_config`（Python 循环），1024 候选池每轮调用两次，是 ask 热点 | P2 |
| P2-12 | `cli.py:1`（BOM）+ 14:26 编码事故 | cli.py 带 UTF-8 BOM；14:26 的批量改写曾把 4 个文件写成 BOM+乱码（`import` 失败），建议加 `test_no_bom`/`iconv` 检查 | P2 |

**总体评价**：骨架（ask/tell、空间层、注册表、study 持久化、router 可解释性）方向是对的，和 `ARCHITECTURE_PATTERNS.md` 的建议高度一致；主要问题集中在 **GP 内核的数值正确性（P0-1/P1-1/P1-2/P1-5）** 与 **router 的维度/能力建模（P0-3/P1-9）**。修完 P0 再做 benchmark 校准，性价比最高。

---

## 1. P0 详述

### P0-1 `algorithms/bayes.py:333-341` — TS 后验采样错误（两个 bug 叠加）

**现状**

```python
def _sample_posterior(self, model, features):
    mu = model.predict(features)                          # 原始 y 单位
    cov = model.kernel_(features) + _JITTER * np.eye(...) # 归一化 y 单位 + 未条件化
    return mu + chol(cov) @ rng.standard_normal(len(features))
```

**问题 1（量纲）**：`_fit` 里 `normalize_y=True`，sklearn 用 `y_std` 归一化后拟合；`model.kernel_(.)` 返回的是**归一化空间**的协方差，而 `model.predict` 返回**原始单位**的均值。二者相加 = 把 O(1) 的噪声加到 O(100) 的均值上。实测（demo `mixed_reaction`，8 个初始点，`y_std=6.89`）：

```
code cov diag = 0.926  (归一化)     →  T S 采样噪声 σ≈1
model.predict(..., return_cov=True) diag mean = 33.6  (原始单位, 已含 y_std²=47.5 与条件化)
```

即 TS 的不确定性被低估 ~6 倍（标准差）。20 次独立抽样中 `argmax(TS) == argmax(μ)` 出现 6 次——它更像"带小抖动的贪心"，不是 Thompson sampling。

**问题 2（未条件化）**：后验协方差应为 `k(x,x') − k(x,X)(K+αI)⁻¹k(X,x')`，代码用的是先验协方差 `k(x,x')`，在已采样区域**不会**降低不确定性。

**文献依据**：R25（TS 定义：从后验抽函数）、R3（第 7 节并行/TS）、R12（BoTorch `MaxPosteriorSampling` 实现）。

**修复（最小改动，直接用 sklearn 的后验协方差）**

```python
def _sample_posterior(self, model: GaussianProcessRegressor, features: np.ndarray) -> np.ndarray:
    # sklearn 的 return_cov 已按 _y_train_std**2 折算回原始 y 单位，且包含后验条件化
    mu, cov = model.predict(features, return_cov=True)
    cov = 0.5 * (cov + cov.T)
    jitter = 1e-9
    for _ in range(5):
        try:
            chol = np.linalg.cholesky(cov + jitter * np.eye(len(features)))
            break
        except np.linalg.LinAlgError:
            jitter *= 10.0
    else:  # 兜底：特征分解
        vals, vecs = np.linalg.eigh(cov)
        chol = vecs @ np.diag(np.sqrt(np.clip(vals, 0.0, None)))
    return mu + chol @ self.rng.standard_normal(len(features))
```

**验收**：①单测断言 `np.std(samples) ≈ np.sqrt(np.diag(predict(return_cov=True)).mean())`（相对误差 <20%）；②在 1D 二次函数上，`ts` 与 `ei` 的 best-so-far 差距 <10%（当前 `ts` 实际是贪心，会更快停滞）。

### P0-2 `benchmarks.py:148,162` — 基准的最优值写错，regret 失效

**现状**：`mixed_reaction.optimum=101.6`、`discrete_catalyst.optimum=81.0`。我按实现公式穷举：

```
mixed_reaction true max = 102.0 at T=82, time=55, conc=10^-0.8, MeCN, equiv=2, inert=True
discrete_catalyst true max = 78.0 at (Pd, C, temp=80)
```

**问题**：`BenchmarkProblem.regret()` 用 `max(0, sign*optimum - best)`，当 `optimum` 低于真实最优时 regret 恒为 0 → "样本效率/regret" 这类指标全失真；demo 声称"收敛到 101.5/101.6 ≈ optimum"其实是**没到真正最优**（102.0）。

**文献依据**：R18（Olympus：emulator 必须给出可核验的最优/参考值）、R32（HPOBench：每个任务带 verified optimum 与 protocol）。

**修复**

1. 直接改数值：`mixed_reaction optimum=102.0`，`discrete_catalyst optimum=78.0`；并在 `builtin_problems()` 里加自检：

```python
def _assert_optimum(problem: BenchmarkProblem) -> None:
    assert problem.optimum is not None
    # 用 problem 自带的 argmax 点复算，容差 1e-6
    value = problem.evaluate(problem.optimal_config)
    assert abs(value - problem.optimum) < 1e-6, (problem.name, value, problem.optimum)
```

2. 给 `BenchmarkProblem` 加 `optimal_config: dict`（既是自检锚点，也是 benchmark 报告里"oracle"的来源）。
3. regret 用归一化版本（与预算无关、可跨问题平均）：

```
simple_regret(n)     = (opt − best_n) / (opt − best_0)        # 越小越好，0 表示已到最优
sample_efficiency(n) = clamp((best_n − best_0) / (opt − best_0), 0, 1)
```

**验收**：单测 `test_benchmark_optima_match_recomputed`；benchmark 表里 `mixed_reaction` 的 bayes 结果不再"必达 optimum"。

### P0-3 `router.py:94-109,221-232` + `space.py:298` — 用 `dim` 而不是 `feature_dim` 判断 GP 可用性

**现状**：`OptimizationContext.from_space` 只记录 `n_dims=space.dim`（变量个数），`_score_bayes` 按 `n_dims<=12` 加 18 分。但 `bayes` 实际在 `space.features`（one-hot 展开）上建 GP。实测：

```
demo mixed_reaction:          dim=6   feature_dim=10   → router: bayes 85 分
3×20 级 categorical + 1 连续:  dim=4   feature_dim=61   → router: bayes 80 分（!）
```

后者的 GP 有 61 个输入维、初始设计 `n_initial = max(3, min(4+1,20)) = 5` 个点——5 个点拟合 61 维 GP，超参完全不可识别。

**文献依据**：R13（categorical one-hot 会改变模型输入维度）、R15（高维要按维度缩放 lengthscale 先验）、R24/R25（综述：有效维度而非名义维度决定方法选择）、R7（高维离散空间用森林代理更稳）。

**修复（三步）**

1. `OptimizationContext` 增加 `feature_dim: int = 0`（`from_space` 填 `space.feature_dim`），并在 `__post_init__` 里若为 0 则回退到 `n_dims`。
2. router 判断改用 `effective_dim = max(feature_dim, n_dims)`；阈值参照文献：`≤8` 舒适，`8–20` 边缘（需要更多初始点/ARD），`>20` 扣分并推荐 TuRBO/CMA 路线（R14/R5）。
3. 增加"**观测不足**"判定：`n_observations < 1.5 * feature_dim` 时警告"GP 超参会过拟合，建议冻结超参或换 TPE/森林"（见 P0-6）。

```python
eff = max(ctx.feature_dim or ctx.n_dims, ctx.n_dims)
if eff <= 8:   score += 18
elif eff <= 15: score += 10
elif eff <= 24: score -= 5;  warnings.append(f"有效特征维 {eff} 偏高，GP 需 ARD + 更多初始点")
else:           score -= 35; warnings.append(f"有效特征维 {eff} > 24，建议 TuRBO/CMA-ES/TPE")
if ctx.n_observations < 1.5 * eff:
    score -= 15; warnings.append("观测数不足 1.5×有效维，GP 超参不可识别")
```

**验收**：新增单测 `test_high_cardinality_space_penalises_bayes`（上面 4 变量/61 特征空间必须不再给 bayes 最高分，或至少给出 warning 且 `n_initial ≥ 15`）。

### P0-4 `study.py:296-306` — 用户 override 被 auto_route 立刻覆盖

**现状**

```python
algorithm, params = self.algorithm, dict(self.params)   # 用户 override 后的值
if self.auto_route:
    rec = route_round(ctx, ...)
    if rec.algorithm != algorithm:                       # router 不认同 → 直接切回
        self.log_decision(action="auto_reroute", ...); algorithm, params = rec.algorithm, rec.params
```

`override(grid, reason=...)` 之后下一次 `ask` 会立刻 log 一条 `auto_reroute` 切回 bayes——**override 只生效到下一次 ask 之前**，审计链里还看不出"用户意图被覆盖"。

**文献依据**：R27（ChemOS：人工干预必须显式、可回滚）、R17（BayBE：人工指定点/推荐器要能 pin 住）。

**修复：加 `pinned` 语义**

```python
# Study.__init__ 里
self.pinned: bool = False

def override(self, algorithm, params=None, reason="", pinned=True):
    ...
    self.algorithm, self.params = algorithm, merged
    self.pinned = bool(pinned)
    self.log_decision(action="override", ..., payload={"previous": previous, "pinned": self.pinned})

def ask(self, n=None):
    ...
    if self.auto_route and not self.pinned:
        rec = route_round(...); ...
    else:
        rec = route(ctx)                      # 只为 report/alternatives，不改变选择
        if self.pinned:
            self.log_decision(action="pinned_skip_reroute", algorithm=algorithm, reason="override 已固定")
```

再给 CLI 加 `--unpin`（或 `override --algorithm auto --reason ...`），让 agent 能显式交还控制权。

**验收**：单测 `test_override_survives_next_ask`：override→ask→断言 `study.algorithm` 仍为 grid，且 decisions 里**没有** `auto_reroute`。

### P0-5 `study.py:349-377` — 部分 `tell` 丢 pending + 非幂等

**现状**

```python
self._pending = []          # 无论收到几条，整批清空
```

- 实验台只回了 8 条里的 6 条（做了 6 个孔位、2 个失败/未做）→ 另外 2 条建议从 `pending.json` 静默消失，既没有 `expired/failed` 记录，也没有审计痕迹。
- 同一批结果因网络/人工重试被 `tell` 两次 → `dataset.jsonl` 追加两份相同观测，`_rounds` 也多一轮，`best`/stall 检测被污染。
- `import_observations` 也不去重（同一批历史数据导入两次 → 两份）。

**文献依据**：R9（Optuna：Trial 状态机 + FAIL/PRUNED 状态是 ask/tell 语义的一部分）、R8（SMAC3 RunHistory：`status` 字段区分 SUCCESS/FAILED/CRASHED，重复 run 用 `(config, instance, seed, budget)` 去重）。

**修复**

```python
def tell(self, results, *, allow_partial: bool = True):
    told = []
    known = {o.meta.get("suggestion_id") for o in self._dataset().observations
             if o.meta.get("suggestion_id")}
    for item in results:
        sid = str(item.get("id", "")) if isinstance(item, Mapping) else ""
        if sid and sid in known:
            raise ValueError(f"suggestion {sid!r} already told (tell is not idempotent-by-design; "
                             "use import for re-runs)")
        obs = self._to_observation(item, {p["id"]: p for p in self._pending})
        told.append(obs)
    ...
    self._pending = [p for p in self._pending if p["id"] not in told_ids]   # 只删已回测的
    if self._pending and not allow_partial:
        raise PendingError(...)
    # 未回测的 pending 记成 expired 或保留在 pending，二选一，必须写进 decisions/rounds
```

同时：
- `_pending` 条目加 `"created_t": time.time()`，report 里显示"已 pending N 小时"。
- `rounds` 记录 `n_told`、`n_pending_left`，stall 检测基于 `n_told>0` 的轮次。
- 把 `dataset` 去重键（`encode(x)` 四舍五入 + `fidelity`）写进 `Observation.meta`，提供 `Dataset.dedupe()` 给 import 使用。

**验收**：单测 `test_partial_tell_keeps_remaining_pending`、`test_double_tell_raises`、`test_import_dedupes_identical_rows`。

### P0-6 `bayes.py:182-186` + `router.py:386-389` — 初始设计点数与特征维脱节

**现状**

```python
n_initial = int(self.params["n_initial"]) or max(3, min(self.space.dim + 1, 20))   # 用 dim
```

- demo 空间 `dim=6, feature_dim=10` → 7 个初始点，勉强；
- 高基数 categorical 空间（`feature_dim=61`）→ 5 个点；
- 且 `_fit` 在 `len(y) >= 6` 时就开启超参优化：**5–7 个点优化 10–61 个 lengthscale，必然过拟合**（`ConvergenceWarning` 又被静默）。

**文献依据**：R2（Snoek：小样本时用先验固定超参，不是硬优化）、R26（Bossek：初始设计策略本身显著影响 SMBO 结果——设计点数应按"模型复杂度"而非变量数设置）、R15（高维先验）。

**修复**

```python
# 1) 初始化点数按 feature_dim 且受预算约束
eff = self.space.feature_dim
n0 = max(5, min(20, 2 * (eff + 1)))
n0 = min(n0, max(3, int(0.4 * remaining)))     # 初始化不超过预算 40%
# 2) 样本不足以支撑超参学习时冻结（用先验/MAP 值，不调 lengthscale）
freeze_hypers = len(y) < 2.0 * max(1, eff)
# sklearn 做法：optimizer=None 时直接用初始 theta
model = GaussianProcessRegressor(kernel=kernel, alpha=alpha, normalize_y=True,
                                 optimizer=None if freeze_hypers else "fmin_l_bfgs_b",
                                 n_restarts_optimizer=0 if freeze_hypers else 3,
                                 random_state=self.seed)
```

**验收**：benchmark 里把 `mixed_reaction` 的 `n_initial` 从 7 提到 12（预算 24），观察 5–10 seeds 的 best-so-far 是否不劣化；新增单测 `test_gp_freezes_hypers_when_n_lt_2_feature_dim`。

---

## 2. P1 详述

### P1-1 `bayes.py:36-47,73` — 没有 ARD，所有特征共用一个 length_scale

`_make_kernel` 用 `Matern(length_scale=ls)`（标量）→ sklearn 视为**各向同性**核，10 个 feature 维（含 one-hot）共用一个尺度。后果：①categorical 与连续变量的相关性权重无法区分；②高基数 one-hot 直接支配距离；③`length_scale=0.807`（实测拟合值）与 61 维空间耦合时毫无意义。

**文献依据**：R2（ARD 是 BO 在 ML 调参上 work 的关键之一）、R15（每个维度独立 lengthscale + 维度缩放先验）、R13（categorical 需要独立处理）。

**修复**

```python
def _make_kernel(kind: str, feature_dim: int, ls0: float = 0.3):
    # ARD：每个特征维一个 lengthscale；初值按维数缩放（R15 的先验中位数 exp(sqrt(2)+0.5 ln d)）
    d = max(1, feature_dim)
    ls_init = ls0 if d <= 8 else float(np.exp(0.5 * np.log(d) + 0.5))
    ls = np.full(d, ls_init)
    bounds = np.tile(np.array([1e-2, 1e2]), (d, 1))   # 绝对边界，不随参数漂移
    if kind == "matern52":  base = Matern(length_scale=ls, length_scale_bounds=bounds, nu=2.5)
    elif kind == "matern32": base = Matern(length_scale=ls, length_scale_bounds=bounds, nu=1.5)
    else:                    base = RBF(length_scale=ls, length_scale_bounds=bounds)
    return ConstantKernel(1.0, (1e-3, 1e3)) * base
```

同时把 `k2.length_scale` 数组化后，`ParamSpec("length_scale")` 语义变为"初始值"（在 `when/description` 里写清楚）。

**验收**：在 `mixed_reaction` 上 ARD 版本的 3-seed 平均 best-so-far 应 ≥ 现版本；检查拟合后的 lengthscale 数组（categorical one-hot 维通常学到更大/更小的尺度，说明 ARD 在起作用）。

### P1-2 `bayes.py:318-325` — `logei` 不是 log-EI

```python
ei = imp * norm.cdf(z) + std * norm.pdf(z)
if mode == "logei": return np.log(np.maximum(ei, 1e-300))
```

EI 在 `z<-38` 时下溢为 0，log 变成常数 **-690.776**，所有候选并列 → 优化器只能靠 tie-break。实测：

```
z=-40: EI(direct)=0  code_logEI=-690.776  stable_logEI=-808.299
z=-45: EI(direct)=0  code_logEI=-690.776  stable_logEI=-1013.419
```

**文献依据**：R29（Ament et al.：log-EI 必须在对数空间计算，并给出数值稳定的 `log h(z)` 递推/闭式，使 EI 在极小改进区仍然有序且有梯度）、R3（EI 公式）。

**修复（已验证的稳定实现，相对误差 ≤1e-15）**

```python
from scipy.special import log_ndtr, erfcx
_LOG2PI = math.log(2.0 * math.pi)

def _log_h(z: np.ndarray) -> np.ndarray:
    """log(z*Phi(z) + phi(z))，全区间数值稳定。"""
    log_phi = -0.5 * z * z - 0.5 * _LOG2PI
    out = np.empty_like(z, dtype=float)
    pos = z > 0
    out[pos] = np.logaddexp(np.log(z[pos]) + log_ndtr(z[pos]), log_phi[pos])
    t = -z[~pos]
    # 1 - t*sqrt(pi/2)*erfcx(t/sqrt(2))  = h(z)/phi(z)，用 log1p 保精度
    out[~pos] = log_phi[~pos] + np.log1p(-t * math.sqrt(math.pi / 2.0) * erfcx(t / math.sqrt(2.0)))
    return out

def _log_ei(mu, best, sigma, xi=0.0):
    z = (mu - best - xi) / sigma
    return np.log(np.maximum(sigma, 1e-300)) + _log_h(z)
```

注意 `xi` 的尺度问题见 P2-2（建议内部用 `xi_rel * y_std`）。

**验收**：单测比对 `np.exp(_log_ei(z)) ≈ direct_ei(z)`（z∈[-30,5]，rtol 1e-12），且 `z=-45` 时仍单调可排序；在噪声 benchmark 上 logEI vs EI 的 best-so-far 差异（期望 logEI 略好）。

### P1-3 `bayes.py:214-226` — constant liar 的伪观测以零噪声进入 GP

```python
if strategy == "constant_liar": y_lie = float(np.max(self._y)) - float(np.std(self._y))
...
X_aug = vstack([X_aug, x_new]); y_aug = concat([y_aug, [y_lie]])
model = self._fit(X_aug, y_aug)     # 伪观测和真观测同权、零噪声
```

问题：①伪观测把后验方差在 `x_new` 处压到 ~0（等于"假装已经测过"），后续点会避开一个其实完全未知的区域；②"谎言值" `max-std` 只比当前最优低 1 个标准差，对最大化目标而言仍很乐观 → 后续点继续扎堆；③sklearn 无法给单点不同噪声（`alpha` 是标量）。

**文献依据**：R2（Kriging believer / constant liar 原意：believer 用后验均值、CL 用悲观常数；论文也指出二者都是近似）、R30（Ginsbourger：**bumpiness/local penalization** 是更干净的并行替代，不需要伪造数据）、R25。

**修复（推荐顺序）**

1. **首选：局部惩罚（不伪造数据）**——在单点采集函数上乘距离惩罚：

```python
def _local_penalty(self, units, chosen_units, l_pen):
    if not chosen_units:
        return np.ones(len(units))
    d2 = ((units[:, None, :] - chosen_units[None, :, :]) ** 2).sum(-1)   # (m, k)
    return np.prod(1.0 - np.exp(-d2 / (2.0 * l_pen ** 2)), axis=1)

# penalty 的尺度：取 ARD lengthscale 的中位数 * 0.5（或单位立方体 0.1~0.2 常数）
```
2. 若保留 constant liar：伪观测改用**悲观值** `min(self._y) - 1.0*std`，并给伪观测所在行加"大噪声"——sklearn 做不到逐点噪声，可近似为把 `alpha` 临时设大（每次拟合 `alpha_fake=1e-1`），或者在 `X_aug` 里复制该点 3 次并加微小扰动以升高其噪声；不推荐。
3. Kriging believer（q≤2）保留，但明确只在 batch_strategy=kriging_believer 时使用。

**验收**：`test_bo_batch_is_distinct` 已存在，补充"batch 内最小成对距离 ≥ 0.05×range"的断言；在 `mixed_reaction` 上 batch=8 比较 greedy/believer/CL/local-penalty 的 best-so-far（期望 local-penalty ≥ CL）。

### P1-4 `bayes.py:233-237` — TS 批量用单条样本的 top-n

```python
sample = self._sample_posterior(model, features)
order = np.argsort(-sample)[:n]
```

同一条函数样本的 top-n 点会在空间上彼此接近（同一条后验函数在相邻点取值高度相关）→ batch 内多样性差。

**文献依据**：R25（TS 的并行版本：每个 batch 点独立抽函数再取 argmax，或抽一条 + 惩罚）、R3（并行策略）。

**修复**

```python
def _thompson_batch(self, model, candidates, features, n):
    picks: list[int] = []
    for _ in range(n):
        s = self._sample_posterior(model, features)
        if picks:
            s[picks] = -np.inf                      # 允许重复候选但降权
        picks.append(int(np.argmax(s)))
    return [self.space.decode(candidates[i]) for i in picks]
```
（若算力允许，也可一次抽 `n` 条独立样本、对每条 argmax；前者省内存。）

**验收**：batch=8 时 batch 内平均最近邻距离应显著大于现状；`test_bo_batch_is_distinct` 增加距离断言。

### P1-5 `bayes.py:299-305` + `router.py:393-398` — alpha 语义与噪声档位不匹配

`normalize_y=True` 时，sklearn 的 `alpha` 是**归一化 y 空间**的噪声方差：`alpha ≈ (σ_meas / y_std)²`。router 现状：

| ctx.noise | acquisition | alpha | 问题 |
|---|---|---|---|
| high | ucb | 5e-3 | σ/y_std ≈ 7% —— 化学重复测量 CV 10–30% 时对应 alpha 0.01–0.09，偏小 2–20 倍；且 UCB 不处理噪声（R10/R12 建议噪声下用 EI/qNEI 系） |
| medium | ei | 1e-4 | σ≈1% 太小 |
| none | ei | 1e-8 | 合理 |

**文献依据**：R2（噪声/重复观测处理）、R10（噪声约束 BO：必须显式建模观测噪声）、R12（qNEI：噪声下 EI 的最优基线是随机的）、R1。

**修复**

1. 在 `Observation` 层面支持重复测量聚合：同一 `x` 测 k 次 → 用均值 + 组内方差；`alpha = mean(σ_rep²) / y_std²`。
2. 没有重复数据时，按噪声档位给保守值并写进文档：

```python
NOISE_ALPHA = {          # alpha in normalized-y variance
    "none": 1e-8, "low": 1e-6, "medium": 5e-3, "high": 3e-2,
}
# 高噪场景优先 logEI/EI（先实现 qNEI 再切换），不要用固定 kappa 的 UCB
```
3. 未来在 `acquisition` 里加 `qnei`（P1-10 的下一轮），高噪直接切。

**验收**：`test_noisy_high_switches_acquisition_and_alpha` 已存在；补充"alpha 与 y_std 关系"的文档与单测（给定重复观测 → alpha≈σ²/y_std²）。

### P1-6 `space.py:371-383` — one-hot 的虚假距离、ordinal 失序、高基数爆炸

**量化"虚假距离"**（值得写进文档）：

- 两个不同类别：`E||onehot_i − onehot_j||² = 2`（k 个 hot 维里有 2 个翻转）；
- 两个连续维的随机两点（单位区间）：`E(u−v)² = 1/6 ≈ 0.167`；
- 若 k=4，则类别差异（2.0）≈ 连续维差异的 **12 倍（平方距离）**；即使各向同性核，categorical 变量也会主导核函数。

现状还有两个具体问题：

1. **ordinal 被 one-hot**（`features` 对所有离散一视同仁）→ 丢失 `A < B < C` 的单调性（R13 的整数松弛/顺序编码更适合 ordinal）。
2. **无 descriptor 机制** → 20 种溶剂 = 20 维 one-hot；混合 3 个就 60 维。R22（Gryffin）的核心贡献恰恰是"categorical 用专家特征的连续嵌入"，化学场景这是最大增量点。

**修复**

1. ordinal 在 `features` 里用**rank 标量**（与 `encode` 相同），不用 one-hot；
2. categorical 用 `k−1` dummy（去掉一列，避免共线）或保留 k 列但配 ARD（P1-1 修完后 ARD 能自动给冗余列大 lengthscale）；
3. 给 `Variable` 增加可选 `descriptors: Mapping[str, tuple[float, ...]]`：

```python
# 当 k > 8 或用户提供 descriptors 时，features 用描述符向量（标准化后）替代 one-hot
if v.descriptors and len(v.choices) > 8:
    vec = np.asarray(v.descriptors[str(value)], dtype=float)
    parts.extend(self._scale_descriptor(v, vec))   # 每维 min-max 到 [0,1]
```
4. router 在 `n_categorical>0 且 max_cardinality>8` 时建议"描述符化或换 TPE"（R4/R22）。

**验收**：`test_ordinal_features_are_rank_scalar`、`test_descriptor_features_replace_onehot`；benchmark 里加一个 12 级溶剂 + 描述符的问题（见 `BENCHMARK_SPEC.md` Tier C），TPE/bayes 对比。

### P1-7 `space.py:396-410` / `sampling.py:146-166` — Sobol 增量使用与"返回不足 n"

- `sample_sobol` 每次 `qmc.Sobol(seed=新随机数)` 重新开始：**两次 ask 的 Sobol 点不是同一条低差异序列**，平衡性/分层性丢失；`qmc.Sobol.random(n)` 只在 `n` 为 2 的幂时保持平衡（scipy 会 warning，代码把 warning 压掉了）。
- `SobolSearch.ask` 的 dedupe 会 `continue` 掉已见点，**返回可能少于 n 个**；`Study.ask` 不检查长度，CLI 直接打印"待测建议（<n 条）"。

**文献依据**：R6（LHS/空间填充设计）、Sobol 原始定义（`sampling.py` 已引）、scipy `qmc` 文档（平衡性要求 2 的幂）。

**修复**

1. `SearchSpace.sample_sobol` 支持传入/返回 sampler 状态（或要求调用方传 `n` 为 2 的幂后取前 n 个）；`SobolSearch` 已保存 `_asked` 并 `fast_forward`——**这条路径是对的**，问题只在 `SearchSpace.sample_sobol` 被 `bayes.ask` 用于初始设计时每次重启。

```python
# 方案：初始化设计一次预生成 2^k 个点，按 _initial_design_left 取切片
units = qmc.Sobol(d=dim, scramble=True, seed=self.seed).random_base2(
            int(math.ceil(math.log2(max(n_initial, 2)))))
```

2. `SobolSearch.ask` dedupe 后不足 n 时，用 `space.sample_uniform(n - len(out))` 补齐（并记录 `dropped_duplicates`）。
3. `Study.ask` 断言 `len(configs) == n`（或允许不足但写 warning 到 decisions）。

**验收**：`test_sobol_ask_returns_requested_n`、`test_initial_design_is_low_discrepancy`（把两次 ask 的点并起来看梯度分布）。

### P1-8 `study.py:310-317` + `base.py:202-204` — 每轮 ask 重置 RNG 流

```python
optimizer = registry.build(..., seed=self.seed, prior=observations)   # 同一 seed
self.rng = np.random.default_rng(self.seed)                            # 每次新建
```

`bayes` 的候选池里 `rng.random(...)` 与 Sobol seed 每轮从同一起点重放 → 每轮候选池的"随机部分"是同一套点（减去已评估的），局部扰动的噪声方向也重复。表现为"探索不足、候选池逐渐被掏空"。

**修复**：在 `state_dict()` 里持久化 `np.random.Generator` 的 bit generator state（JSON 化：`rng.bit_generator.state` 是 dict）：

```python
def state_dict(self):
    s = super().state_dict()
    s["rng"] = self.rng.bit_generator.state
    return s

def load_state_dict(self, state):
    super().load_state_dict(state)
    if "rng" in state:
        self.rng.bit_generator.state = state["rng"]
```

Study 已经在写/读 `opt_state.json`，只需补上；同时 `Study.ask` 后要把更新后的 state 写回（当前写的是 ask 之后的状态，顺序没问题）。

**验收**：单测：同 seed 两次独立运行结果一致；同一 study 连续两轮 ask 的候选池随机部分不相同。

### P1-9 `router.py` — capabilities 未参与过滤（详见 `CAPABILITIES_SPEC.md`）

现状：`AlgorithmSpec.capabilities` 是 `frozenset[str]`，且**词汇不统一**（`bayes` 用 `"noisy"/"expensive"/"model_based"`，`random` 用 `"high_dim"/"cheap"/"model_free"`，`grid` 用 `"exhaustive"`），router 靠 `_rules()` 里硬编码的 5 个函数。

问题：加一个新算法必须改 router 的三处（`_rules`、`_params_for`、`_build_plan`），与 `docs/ROUTING_POLICY.md` 声称的"注册表 + 规则"不一致；且无法表达"这个算法不支持约束/多目标/多保真"。

**修复**：见 `CAPABILITIES_SPEC.md`（结构化 `Capabilities` + `required_capabilities(ctx)` + 先过滤后打分）。最小改动：给 `AlgorithmSpec` 加 `caps: Capabilities`（dataclass），router 入口先 `feasible = [s for s in specs if s.caps.satisfies(required(ctx))]`，再做打分；规则函数只处理"偏好"，不处理"可行"。

**文献依据**：R8（SMAC3 Facade：不同场景入口对应不同能力组合）、R17（BayBE Recommender 能力声明）、R9（Optuna Sampler/Pruner 组合）。

### P1-10 多目标静默降级

`router._score_bayes` 对 `n_objectives>1` 直接 `-200`，然后 `route()` 返回 sobol/lhs 等单目标算法 + 一条 warning；`Study.tell` 仍然只把 `objective.name` 一个值喂给模型（`Observation.score`），其余目标只存在 `objectives` dict 里——**报告的"Pareto 前沿"其实没有任何算法在维护**。

**修复（分两步）**

1. **v0.1.x（本周）**：`Study.create` 在 `objective.n_objectives>1` 且无 MOO 算法时**显式失败**（或在 `ask` 前 `raise NotImplementedError`），错误信息给出 ParEGO 路线图；不要静默按主目标跑。
2. **v0.2**：实现 ParEGO（`MULTI_OBJECTIVE_AND_NOISE.md` §1），router 加 `supports_multi_objective` 能力位与 `parego` 注册项，之后 `route` 才允许 MOO 场景。

**文献依据**：R11（ParEGO）、R12（qNEHVI）、R21（qEHVI）、R24（综述）。

### P1-11 `bayes.py:203-231` — 批量选点仍是"逐点贪心重拟合"

q=8 时每选一个点重拟合一次 GP（`_fit` 每次含超参优化，n=20 时约 0.1–0.5 s）→ 一轮 ask 秒级；且逐点贪心在 q 大时多样性差。R3 的 qEI 是"联合后验"，R12 的 qNEHVI 也是。

**修复（不引入 qEI 也能改善）**：q>8 时改用"单点 EI + 局部惩罚"（`P1-3` 的惩罚实现），一次拟合 + k 次打分；q≤8 保留贪心（believer）。把 `batch_strategy` 的 `choices` 加 `"local_penalty"` 并设为 `batch>=8` 的默认。等 `qEI` 实现后升级。

### P1-12 `bayes.py:239-286` — 候选池构成

- `n_uniform = n_rest//2`（`rng.random`）+ `n_sobol`：两者都是全局均匀点，信息重复；建议 100% scramble Sobol（或 70% Sobol + 30% 网格/边界点）。
- `local` 池以 `argsort(-self._y)[:5]` 为中心：噪声大时 top-5 可能是幸运观测 → 用后验均值/可行域内最优作中心更稳。
- 没有排除 pending 点（Study 当前串行所以问题不大，但 Optimizer 契约应自洽）。
- `local_sigma=0.15` 对所有维度固定：d=2 时太宽，d=20 时太窄；建议 `sigma = clip(0.5 * (1/local_frac)**(1/d), 0.05, 0.3)` 或按 lengthscale 中位数。

**修复示例**：`local_centers = argsort(-model.predict(self._units)[0])[:5]`；pool 组成 `70% Sobol + 20% local + 10% boundary/grid`；`local_sigma` 与均值函数/核尺度联动。

---

## 3. P2（工程/体验，择机）

- **P2-1 `bayes.py:38`**：`bounds = (max(1e-2, ls*0.05), max(2e-1, ls*20))` 随 `length_scale` 参数漂移。改为绝对区间 `(1e-2, 1e2)`（与 P1-1 一起改）。
- **P2-2 `bayes.py:320`**：`xi=0.01` 是原始 y 单位；建议内部换算 `xi_eff = xi_rel * y_std`（`xi_rel` 默认 0.01），或在 `_acquisition` 里先标准化 y。
- **P2-3 `bayes.py:298,306-308`**：`n_restarts_optimizer=1`；`ConvergenceWarning` 被完全吞掉。建议：n≥15 时 restart=3；记录 `kernel_.theta` 与每个 lengthscale 是否贴边（`<1.1*bound_low` 或 `>0.9*bound_high`），写入 `Study.decisions` 的 payload，作为 router 重路由信号。
- **P2-4 `space.py:189-194`**：continuous 的 `validate_value` 只查 finite，不查 `[low,high]`。`import` 历史数据时可能把越界点带进 GP。加 `if not (low <= v <= high): raise`（或 `warn + clip` 并记录）。
- **P2-5 `space.py:160-187`**：`encode` 的 categorical 是 `idx/(k-1)`、`decode` 是 `int(u*k)`；二者可逆但不对称，`features` 又用 one-hot。建议在 `Variable` 上显式提供 `unit_from_index`/`index_from_unit` 一对函数并单测，避免后续本地扰动语义混乱。
- **P2-6 `study.py:318-325,341-344`**：`opt_state.json` 只存单算法状态。改为 `{"states": {algo: state}}`，`load_state_dict` 前按算法名取；这样 reroute 来回切换不丢 Sobol 的 `_asked`。
- **P2-7 `study.py:345-346`**：`suggestions.csv` 每次覆盖。改为 `suggestions.csv`（当前批）+ `suggestions_history.jsonl`（追加）。
- **P2-8 `study.py:441-466`**：`summary()` 里 `route(ctx)` 调两次。提成局部变量 `next_route = route(ctx)`。
- **P2-9 `study.py:423-438`**：`override` 不做能力检查。用 `CAPABILITIES_SPEC` 的 `required_capabilities(ctx)` 校验，不满足时 warning（不要 raise，允许用户自己承担）。
- **P2-10 `space.py:431-467`**：`sample_grid` 字典序 → 预算小于网格时只覆盖第一维低端。建议用 seeded shuffle（`rng.permutation`）打散；粗化那步 `keep = linspace(0, m-1, m-1)` 会**恰好删掉中间端**，建议改为按位置均匀删除并断言不重复。
- **P2-11 `space.py:371-389`**：`features_many` 逐点 validate（Python 循环）。给 `SearchSpace` 增加 fast path（信任已 validate 的 decode 结果），或向量化 one-hot（`np.eye` 索引）。ask 每轮两次 1024 点调用，是主要 CPU 热点之一。
- **P2-12 `cli.py:1` BOM + 编码事故**：`cli.py` 目前带 UTF-8 BOM（Python 可执行，但 `ast.parse`、部分工具会报错）；且 14:26–14:28 的批量改写曾把 `space.py/study.py/dataset.py/cli.py` 写成 BOM + 乱码，导致 `import condopt` 直接 SyntaxError。建议：① 统一 `encoding="utf-8"`（无 BOM）写入；② CI 加一项 `python -c "import ast,pathlib; [ast.parse(p.read_text('utf-8')) for p in pathlib.Path('src').rglob('*.py')]"`；③ 所有中文写入走 `write` 工具，禁止 shell heredoc（本次事故的根因）。

---

## 4. 混合空间专项结论（供 `space.py` 改造参考）

1. **unit 编码（1 维/变量）+ features（one-hot）双表示**是合理设计（BayBE/Nevergrad 同构），但要**明确唯一权威**：`decode` 只从 unit 来，`features` 只给模型用（当前注释已这么写，保持）。
2. **categorical 的距离陷阱**（见 P1-6 量化）：在 unit 空间里类别按 `idx/(k-1)` 线性排布 → `local_sigma=0.15` 的扰动对 k=4 是"35% 概率翻到相邻类"，对 k=20 只有 7%——**同一参数在不同基数下语义不同**。建议局部扰动改在 features 空间（one-hot + 数值）定义，或按 `1/(k-1)` 归一化。
3. **高基数 categorical**：唯一可扩展的路线是"描述符/嵌入"（R22）；其次是 TPE/森林（R4/R7）。**不要**让 GP 硬吃 60 维 one-hot。
4. **ordinal**：用 rank 标量（R13 的整数松弛思想），并允许单调性先验（如 `x_k = k/(K-1)` 直接进核）。
5. **bool**：2 维 one-hot 是浪费；1 维 0/1 数值即可（当前 one-hot，可优化）。
6. **条件空间**：当前 `SearchSpace` 不支持条件参数（如"溶剂=A 时温度范围不同"）。`docs` 里没有承诺，但 skill 层迟早需要；设计见 `ARCHITECTURE_PATTERNS.md` §2.1 的 `ConditionalParameter`（v0.2）。

---

## 5. Router 打分与文献对照（逐条挑刺）

| 规则 | 现状 | 文献/事实对照 | 建议 |
|---|---|---|---|
| `dim<=12` → bayes +18 | 用变量数 | 应看 `feature_dim`（P0-3）；R15 指出高维真正问题是先验而非维度本身 | 改 `effective_dim`，并区分 8/15/24 三档 |
| `noise medium/high` → bayes +5 | 提升 bayes | 噪声下 BO 的优势依赖**噪声建模**（R10/R12），当前实现没建模 → 提升缺乏依据 | 加条件：只有 `alpha/replicate/qNEI` 就绪时才 +5 |
| `noise high` → `acquisition=ucb, alpha=5e-3` | 高噪用 UCB | UCB 的 κ 需按尺度调，且不处理观测噪声；R25 建议 TS/qNEI | 高噪 → `ei/logei`（+ 更大 alpha）或 qNEI；κ 仅作探索参数 |
| `smoothness=="smooth"` → bayes +12 | 用户自报 | 自报"光滑"不可靠（化学里常见阈值/相变） | 保留但降权（+6），并在 report 里标注"依据用户自述" |
| `domain=="chemistry"` → bayes +5，理由"反应条件通常平滑" | 无依据断言 | 文献没有此结论；这是直觉 | 删除或用 `notes` 中的用户先验；不要给领域贴标签加分 |
| `budget_total>=20` → +12 | 预算充足 | 合理（R26：初始设计占比不能过高） | 改成 `budget_total / n0` 比值判断 |
| `batch_size>16` → 仅 warning | 批量过大 | R3：qEI 贪心在 q 大时质量下降；应改用 local-penalty/TS | 批量>8 时同时把 `batch_strategy` 切到惩罚式 |
| `n_dims>20` → bayes -35，高维 → sobol/random +20 | 高维走采样 | 与 R14（TuRBO）/R5（CMA-ES）不符：高维应推荐"局部 BO 或进化"，不是纯随机 | 高维先给 warning+建议，v0.2 加 tuRBO/cmaes 后再改分 |
| `grid` 在 `card<=200` +55 | 穷举 | 合理（R26 初始设计/析因）；但 `card>remaining*2` -15 也合理 | 保持；补"grid 顺序应 shuffle"（P2-10） |
| `route_round` 停滞→sobol | 跳出局部 | R25/R24：模型失效时用模型无关探索是对的；但"回切"策略缺失 | exploration 批次数要限 1 批，之后强制回主算法；把 `design["n"]` 真正用上（P2：现在 Study.ask 忽略） |
| 置信度 margin 20/8 | 经验 | 无文献依据 | 用 benchmark 校准（`BENCHMARK_SPEC.md` §5），或改成分位数 |

---

## 6. Study / CLI 契约专项（对 Lead 三个问题的直接回答）

1. **ask 返回不足 n**：`SobolSearch` dedupe 会少返；`bayes` 只在候选池耗尽时补 Sobol（不会少）；`Study.ask` 不校验；CLI 不报错。→ 修 P1-7（补点 + 断言 + warning）。
2. **tell 幂等**：当前**不幂等**（重复追加 dataset；`rounds` 也重复）。→ 修 P0-5（按 `suggestion_id` 拒绝重复；`import` 按 encode 去重）。
3. **pending 隐患**：部分 tell 丢 pending（P0-5）；`ask` 直接抛 `PendingError` 对异步实验不友好（未来要 `allow_pending`/部分 tell 后再 ask 剩余 n）；`_pending` 无时间戳/过期语义；`opt_state` 单算法（P2-6）。→ 建议 `ask` 支持 `n_remaining = n - len(pending)` 语义（v0.2：`ask_more()`）。
4. **budget 记账**：`remaining = budget − len(dataset) − len(pending)`；`import_observations` 可把 used 推过 budget（不报错）→ 建议 import 时检查并裁剪/警告。
5. **CLI**：`_read_results` 只接受 JSON array 或 JSONL；`tell` 无 `--allow-partial`；`ask` 在 pending 非空时报错信息不含 pending ids（用户不知道要补哪些 id）→ 错误信息里带 `pending: [{id, x}]`。

---

## 7. 修复路线图（建议）

**Sprint 1（1–2 天，纯 bugfix，不改架构）**
1. P0-2 benchmark optimum 修正 + 自检测试（30 min，收益最大）
2. P0-1 TS 采样（用 `return_cov=True`，30 min）
3. P0-4 override pinned（1 h）
4. P0-5 部分 tell + 幂等（2 h）
5. P0-3 feature_dim 进 context + router 阈值（1 h）
6. P0-6 初始化点数/冻结超参（1 h）

**Sprint 2（3–5 天，GP 质量）**
P1-1 ARD、P1-2 logEI、P1-3/11 局部惩罚、P1-4 TS 批量、P1-5 alpha/重复测量、P1-8 RNG 状态、P1-7 Sobol。

**Sprint 3（1 周，路由与空间）**
P1-9/CAPABILITIES、P1-6 ordinal+descriptor、P1-10 多目标（ParEGO）、P1-12 候选池、`BENCHMARK_SPEC.md` 的校准流程。

**验收 gate（每个 sprint 结束）**
- 全单测 + 新增单测通过；
- benchmark（`BENCHMARK_SPEC.md` Tier A/B，budget=24，batch=4，seeds≥10）中 `bayes` 对 `sobol` 的 IQM 优势 ≥ 上一版本，且 `ts` 不再被 greedy 打平；
- `mixed_reaction` 有 seed 能达到 ≥101.9（真实最优 102.0）。

---

## 8. 证据附录（本审查实际运行过的检查）

| 检查 | 方法 | 结果 |
|---|---|---|
| benchmark 真实最优 | 穷举 `mixed_reaction`（T×time 网格 801×1151，其余取解析最优）/`discrete_catalyst`（3×4×81） | 102.0 / 78.0；声明值 101.6 / 81.0 |
| feature_dim | `SearchSpace.feature_dim` | demo 10；3×20 级 categorical+1 连续 = 61（dim=4） |
| TS 协方差 | 8 点 GP（`normalize_y=True`），比较 `kernel_(features)` 与 `predict(return_cov=True)` | 0.926 vs 33.6（diag mean）；采样噪声低估约 6× |
| TS 行为 | 20 次独立抽样 `argmax` | 6/20 与后验均值 argmax 相同（近贪心） |
| logEI | z∈[-45,3] 对比 direct EI / 代码 logEI / 稳定 logEI | z≤-38 代码恒 -690.776；稳定实现 rtol≤1e-15 |
| router 决策 | `route(ctx)` 实测 | demo：bayes 85；高基数 61 维：bayes 80；多目标：sobol（静默降级）；高噪：ucb/alpha 5e-3；budget=10：bayes n_initial=4 |
| 静态检查 | `ast.parse` 全部 `src/condopt/**/*.py` | 14:30 快照全部通过（14:26–14:28 曾有 4 文件编码损坏，现已恢复；`cli.py` 仍带 BOM） |
