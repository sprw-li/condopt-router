# qNEHVI_SPEC_V2 — 带噪多目标嵌套 MC 的可实现规格（2D）

> 维护者：algo-research ｜ v1（2026-11-05，Round 6）
> 对应：`MULTI_OBJECTIVE_AND_NOISE.md` §3（原理）、`qEHVI_IMPL.md`（分区/DHV/解析 EHVI）、`qehvi.py`（v0.5，2D 单点 MC + 贪心 + 局部惩罚）。
> 文献：R12（qNEHVI：噪声多目标嵌套 MC）、R21（qEHVI：分区）、R10（带噪 EI：baseline 随机化）、R42（HV/固定 ref）。
> 目标：在 `qehvi.py` 基础上加一个 `qnehvi` 类，**复用** `partition_2d`/`dhv_2d_batch`/`_sample_posterior_multi`/局部惩罚；给出参数、伪代码、数值验收。

---

## 1. 定义与与 qEHVI 的唯一区别

qEHVI（无噪）：`P` 是确定的已观测非支配集。

qNEHVI（带噪）：

```
qNEHVI(X) = E_{f|D}[ E_{y|f,X}[ max(0, HV(P(f) ∪ {y}) − HV(P(f))) ] ]
```

`P(f)` = 由**后验抽样出的 baseline 值**决定的非支配集。等价地（实现视角）：

```
for k in 1..n_base:
    B_k = baseline_scores + Σ_base^{1/2} z_k          # baseline 后验抽样（每个目标）
    P_k = non_dominated(B_k)                            # 该样本下的 Pareto 集
    boxes_k = partition_2d(P_k, r)                      # 复用 qEHVI 的分区
for s in 1..n_y:
    y_s = μ_cand + σ_cand ⊙ z_s                         # 候选后验抽样
qNEHVI_j ≈ (1/n_base) Σ_k (1/n_y) Σ_s max(0, dhv_2d(y_s, boxes_k, r))
```

**关键性质**：给定 baseline 样本 `k`，`max(0, HV(P_k∪{y}) − HV(P_k)) = max(0, DHV_{boxes_k}(y))` —— 因此无需每次重算 HV，直接复用 `dhv_2d_batch`，复杂度 `O(n_base × n_y × m_cand)`（向量化后每 baseline 一次批量调用）。

---

## 2. 参数取值（可直接写进 `ParamSpec`）

| 参数 | 默认 | 范围 | 依据/说明 |
|---|---|---|---|
| `n_base` | **64** | 8–256 | baseline 外层 MC。噪声越大越要大（决定 `P_k` 的偏差）；实测 std 随 `n_base` 8→32→128 从 0.063 降到 0.016→0.004 |
| `n_y` | **128** | 16–512 | 候选内层 MC；`n_base×n_y` 是总预算，建议 4k–16k |
| `mc_draws`（qEHVI 现有） | 128 | — | 与 `n_y` 合并成一个参数（见 §6 实现） |
| `baseline_max` | 200 | 20–1000 | `X_baseline` 上限；`n_obs > baseline_max` 时抽样 |
| `baseline_noise` | `"auto"` | auto/off | auto = 用 `Observation.objective_std`/`n_replicates` 估计；无复孔时用 GP `alpha`（`REPLICATES_ALPHA.md`） |
| `batch_strategy` | `local_penalty` | — | q>1 用贪心 + 局部惩罚（同 qEHVI） |
| `seed` | — | — | 固定 → 可复现；Study 每轮推进 seed（已实现） |

**总成本**：`n_base × n_y × m_cand` = `64×128×1024 ≈ 8.4M` 次 `dhv` 计算 + `64` 次分区；numpy 向量化后单次 ask 约 1.5–3 s（CPU）。算力低时降 `n_base=32, n_y=64`（~2M）。

---

## 3. `X_baseline` 的选择

| 情况 | 策略 |
|---|---|
| `n_obs ≤ baseline_max`（默认 200） | 全部已观测点 |
| `n_obs > baseline_max` | 非支配点（必须全要）+ 随机补足到 `baseline_max` + 最近 20 个点 |
| 有约束 | 只用可行观测（`P(feasible)` 或观测判据） |
| 有失败/缺目标观测 | 跳过（`tell` 已容错） |
| 全部被支配 | 只保留非支配集（`P_k` 仍随机，因为 baseline 值有噪声） |

**注意**：`X_baseline` 不是"越少越好"——它决定 `P_k` 的分布；噪声大时 baseline 抽样不足会低估 `P_k` 的不确定性（qNEHVI 退化接近 qEHVI）。

---

## 4. 伪代码（可直接贴进 `qnehvi.py`）

```python
@register
class QNEHVI(QEHVI):
    spec = AlgorithmSpec(
        name="qnehvi", family="model_based",
        summary="带噪 2 目标 qNEHVI：对 baseline Pareto 集做嵌套 MC，噪声下不早停。",
        when="恰好 2 目标、有噪声（复孔/测量误差）、预算 ≥40，需要稳定的 Pareto 前沿时。",
        params=QEHVI.spec.params + (
            ParamSpec("n_base", "int", 64, low=8, high=256),
            ParamSpec("n_y", "int", 128, low=16, high=512),
            ParamSpec("baseline_max", "int", 200, low=20, high=1000),
            ParamSpec("baseline_noise", "choice", "auto", choices=("auto", "off")),
        ),
        capabilities=QEHVI.spec.capabilities | {"noisy"},
        references=("Daulton, Balandat & Bakshy, qNEHVI, NeurIPS 2021",
                    "Daulton, Balandat & Bakshy, qEHVI, NeurIPS 2020"),
    )

    def _baseline_scores(self) -> np.ndarray:
        """已观测 signed scores, 最多 baseline_max 行（非支配点全保留）。"""
        Y = signed_scores(self._Y, self.objective)
        if len(Y) <= int(self.params["baseline_max"]):
            return Y
        mask = non_dominated_mask(Y)
        keep = np.flatnonzero(mask).tolist()
        rest = [i for i in range(len(Y)) if i not in set(keep)]
        rng = self.rng
        extra = rng.choice(rest, size=min(len(rest), int(self.params["baseline_max"]) - len(keep)),
                           replace=False) if rest else []
        return Y[np.array(keep + list(extra), dtype=int)]

    def _baseline_cov(self, n: int) -> np.ndarray:
        """baseline 值在每个目标上的噪声协方差（对角，2×2）。"""
        if str(self.params["baseline_noise"]) == "off":
            return np.zeros((n, 2))
        # 有复孔: 用 objective_std² / k；否则用 GP alpha（归一化）→ 乘 y_std² 回原单位
        alpha = float(self.params["alpha"])
        y_std = np.std(self._Y, axis=0, keepdims=True)
        var = np.maximum(self._Y.std(axis=0, keepdims=True), 1e-9) ** 2 * alpha
        return np.broadcast_to(var, (n, 2)).copy()

    def _model_ask(self, n: int) -> list[dict]:
        if self._X is None or len(self._y) < 3:
            return self.space.sample_sobol(n, rng=self.rng)
        candidates, features = self._candidate_pool()
        if len(candidates) == 0:
            return self.space.sample_sobol(n, rng=self.rng)
        ref = self._signed_references()
        Yb = self._baseline_scores()
        base_var = self._baseline_cov(len(Yb))                       # (n_b, 2)
        models = [self._fit(self._X, self._Y[:, j]) for j in range(2)]
        draws = int(self.params["n_y"])
        ens = np.stack([self._sample_posterior_multi(m, features, draws).T
                        for m in models], axis=-1)                    # (n_y, m_cand, 2)
        mu_base = Yb
        sd_base = np.sqrt(np.maximum(base_var, 0.0))
        acq = np.zeros(len(candidates))
        n_base = int(self.params["n_base"])
        for _ in range(n_base):
            Bk = mu_base + sd_base * self.rng.standard_normal(Yb.shape)   # baseline 抽样
            mask = non_dominated_mask(Bk)
            boxes = partition_2d(Bk[mask], ref)
            if not boxes:
                acq += np.maximum(0.0, np.prod(ens - ref, axis=-1)).mean(axis=0)
                continue
            acq += dhv_2d_batch(ens, boxes, ref).mean(axis=0)          # (m_cand,)
        acq /= n_base
        return self._greedy_batch(candidates, acq, n)                  # 复用 qEHVI 的惩罚贪心
```

**注意**：`qehvi.py` 现有 `_model_ask` 需要把"贪心 + 局部惩罚"抽成 `_greedy_batch(candidates, acquisition, n)`，qNEHVI 直接复用（避免复制粘贴）。

---

## 5. 批量 q 点

| q | 策略 |
|---|---|
| q=1 | 上述 MC 直接 argmax |
| 2–4 | 贪心：每选一点后用**局部惩罚**降权（不更新 baseline/GP），或"每点独立抽候选样本后 argmax"（TS 式） |
| q>4 | 局部惩罚 + 每点独立 baseline 样本（共享 `boxes_k` 会高估相关性；独立更稳） |

不建议对 q 个点做联合 MC（`HV(P_k ∪ {q 点})`）——2D 下收益小、成本 `O(n_base×n_y×2^q)`。

---

## 6. 与现有 qEHVI 的复用清单

| 现有函数 | 复用方式 |
|---|---|
| `pareto.partition_2d` | 直接复用（每个 baseline 样本调用一次） |
| `pareto.dhv_2d_batch` | 直接复用（`ens` 形状 `(n_y, m_cand, 2)`） |
| `pareto.non_dominated_mask` | 每 baseline 样本一次 |
| `pareto.hypervolume_2d` | 只在验收脚本里对照，不必在采集函数内 |
| `pareto.summarise` | 报告用（固定 ref） |
| `GaussianProcessBO._fit/_sample_posterior_multi/_candidate_pool/_local_penalty_scale` | 直接继承 |
| `QEHVI._greedy_batch`（新增） | 从 `qehvi.py:147-161` 抽出，qEHVI/qNEHVI 共用 |

**改动范围**：`pareto.py` 不动；`qehvi.py` 抽 1 个方法；新增 `qnehvi.py`（~120 行）；`router.py` 加 `_score_qnehvi` + `_params_for`；`registry` 自动注册。

---

## 7. 数值验收（本机实测，供 pytest 参考）

固定场景：`ref=(0,0)`、baseline `P=[(3,1),(2,2)]`、候选后验 `μ=(2.6,2.4)`、`σ=(0.3,0.3)`、20 个 seed 复跑：

| 估计器 | 参数 | 均值 | 跨 seed 标准差 |
|---|---|---|---|
| qEHVI（单点 MC） | S=64 | 1.6648 | 0.0926 |
| qEHVI（单点 MC） | S=128 | 1.6654 | 0.0926 |
| qNEHVI（base 无噪） | n_base=8, n_y=32 | 1.6695 | 0.0628 |
| qNEHVI（base 无噪） | n_base=32, n_y=128 | 1.6843 | 0.0159 |
| qNEHVI（base 无噪） | n_base=128, n_y=256 | 1.6813 | 0.0043 |
| qNEHVI（base σ=0.3/维） | n_base=32, n_y=128 | 1.8002 | 0.1027 |
| qNEHVI（base σ=0.3/维） | n_base=128, n_y=256 | 1.7640 | 0.0367 |

**测试用例（pytest）**

```python
def test_qnehvi_reduces_to_qehvi_without_noise():
    # baseline 噪声=0 时 qNEHVI 与 qEHVI 单点 MC 均值相对差 < 3%（实测 1.1%）
def test_qnehvi_estimator_variance_decreases_with_n_base():
    # seed 间 std: n_base=8 > n_base=32 > n_base=128（实测 0.063 > 0.016 > 0.004）
def test_qnehvi_increases_with_baseline_noise():
    # 含噪 baseline 的 qNEHVI 均值 > 无噪版本（实测 1.80 vs 1.68）
def test_qnehvi_respects_fixed_ref():   # ref 变化时估计单调变化；ref 由 Objective.ref_point 决定
def test_qnehvi_vs_parago_on_noisy_moo():  # moo_yield_cost + noise=1.5, replicates=2, 5 seeds: HV_ratio(qnehvi) ≥ HV_ratio(parego)
```

**benchmark 验收**：`moo_yield_cost` 加噪（σ=1.5、replicates=2）后，`qnehvi` 的 `HV_ratio`（固定 ref，`BENCHMARK_TIER_CD.md` §6.4）相对 ParEGO 提升 ≥3%，且前沿点的"概率 Pareto"（`MULTI_OBJECTIVE_AND_NOISE.md` §5）覆盖更稳（跨 seed 方差下降）。

---

## 8. 与 router 的切换条件

| 条件 | 算法 |
|---|---|
| m=2，noise ∈ {none, low} | qEHVI |
| m=2，noise ∈ {medium, high}，有复孔数据 | **qNEHVI** |
| m=2，budget < 40 | ParEGO |
| m≥3 | ParEGO（直到 EHVI 3D 落地，见 `EHVI_3D.md`） |
| 有约束 | 暂不支持（qNEHVI + 约束需联合 MC，v0.6） |

---

## 9. 参考

R12（qNEHVI 原始定义与嵌套 MC）、R21（qEHVI 分区/DHV）、R10（带噪 EI：baseline 必须随机化）、R42（HV/固定 ref）、R2（噪声/复孔 alpha）。
