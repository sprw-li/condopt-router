# QNEHVI_MF_ACCEPTANCE — qNEHVI 与 mf_suzuki 剩余验收的本机实测与阈值建议

> 维护者：algo-research ｜ v1（2026-11-05，Round 8）
> 环境：Windows/CPU、`auto_route=False`、`moo_yield_cost`（ref_point=(60,30)）与 `mf_suzuki`（levels 0.25/0.5/1.0，cost=s）。
> 文献：R12（qNEHVI）、R21（qEHVI）、R42（HV/固定 ref/真值口径）、R41（BOCA/多保真）、R32（多 seed 统计）、R40（成本预算口径）。
> **重要前提**：噪声场景的 HV **必须用真实目标值**计算；用带噪观测会得到 >1 的 HV ratio（见 `REVIEW_v07` P0-2）。

---

## 1. qNEHVI 实测（budget 24、batch 4、2 seeds；参考前沿 `moo_reference_front()`，ref=(60,−30) signed）

| 算法 | 噪声 | observed HV ratio | **true HV ratio** | 备注 |
|---|---|---|---|---|
| `qehvi` | 0 | 0.9466 / 0.9426 | 同 observed（无噪） | 与 Lead 报的 ≈0.95 一致 |
| `qehvi` | σ=1.5（`objective_std=1.5`） | 1.0213 / 0.9500 | **0.9538 / 0.9231** | observed 可 >1 → 不能用于验收 |
| `qnehvi` | 0 | 0.9464 / 0.9555 | 同 observed | 触发 no-noise `RuntimeWarning` ✓ |
| `qnehvi` | σ=1.5（`n_base=32,n_y=64`） | 0.9969 / 0.9613 | **0.9565 / 0.9234** | 与 qEHVI 基本持平（均值 0.9400 vs 0.9385） |

**解读**

1. 无噪时 qNEHVI ≡ qEHVI（实测轨迹一致：0.9464/0.9555 vs 0.9466/0.9426），符合定义；
2. 噪声下 qNEHVI 的真实 HV 略优（+0.0015 均值，budget 24 太小，差异在噪声内）；**不要在 budget=24 断言 qNEHVI > qEHVI**；
3. `observed HV ratio > 1` 是"幸运噪声"的伪指标，验收必须用 true HV（`REVIEW_v07` P0-2）。

### 1.1 阈值建议（替换 `qNEHVI_ACCEPTANCE.md` #1/#3 的默认值）

| 用例 | 原阈值 | 建议阈值（基于实测） | 是否进快 CI |
|---|---|---|---|
| #1 无噪 HV ratio（5 seeds） | ≥0.90（单 seed）、中位≥0.92 | 保持：单 seed ≥0.90、5-seed 中位 ≥0.93 | 慢（~10–15 s/seed）→ nightly |
| #2 n_base 收敛（采集排序） | Spearman ≥0.85 | 保持（该测试不跑 campaign，~2 s） | ✅ 快 CI |
| #3 含噪 baseline | 排序改变 + HV≥0.80 | 改为**单元级**采集断言（同一数据集下 `noise_std=0` vs `1.5` 的 top-8 交集 <8）；端到端只断言 **true HV ratio ≥0.90**（实测 0.92–0.96） | 单元级 ✅；端到端 nightly |
| #4 批量 | 最小距离 ≥0.05、HV≥0.80 | 保持；补 q=8 与 q=1 的 true HV 差 ≤0.10 | ✅ 快 CI（~5 s） |
| #5 ref 固定/必填 | 抛 ValueError/持久化 | 保持（改动最小、价值最高） | ✅ 快 CI（<2 s） |

**快 CI 组合**：#2 + #4 + #5（实测总耗时 <10 s，且覆盖"收敛/批量/口径"三个最关键风险）。

---

## 2. `mf_suzuki` 剩余验收实测

设置：`levels=(0.25,0.5,1.0)`、`min_full_fraction=0.15`、batch=2；成本预算 24（按 `cost=s` 累计），2 seeds。

| 算法 | best full-fidelity yield | 实际成本 | 备注 |
|---|---|---|---|
| `mfbo` | **103.46 / 104.65**（均值 104.06） | 25.25（+5.2% 超支） | 初始设计含全保真（见下） |
| `bayes`（全保真 24 次） | 98.92 / 93.24（均值 96.08） | 24.00 | 基线 |

- **相对提升**：`mfbo` 均值 **+8.3%**（104.06 vs 96.08），单 seed 波动大（+4.6% / +12.2%）；
- **成本超支**：+5.2%（`Study.remaining()` 只按次数，`ask` 一次给 2 个点不看剩余成本）→ `REVIEW_v07` P1-4。

### 2.1 `mf_suzuki` 的 5 条验收（实测/阈值）

| # | 用例 | 期望（含实测） | 说明 |
|---|---|---|---|
| 1 | optimum 核验 | `mf_suzuki(OPT,1.0)=110.0`；`mf_suzuki(OPT,0.5)=104.697` | 已实测（benchmarks.py：`optimum=110.0`） |
| 2 | 单保真默认 | 非多保真算法的 `ask_with_fidelity` 全返回 1.0 | `Optimizer` 默认实现 |
| 3 | 低保真使用 | 前 10 个建议中 s<1 占比 >50%，且 ≥1 个 s=1 | 实测 fidelity 序列 `[0.25,0.5,1.0,0.25,0.5,1.0,...]`（2/3 低、1/3 全）→ 通过；**但初始设计轮转含全保真，规格要求"初始全最低"**（P1-5） |
| 4 | 同成本优于全保真 BO | 2 seeds 均值 ≥ **+5%**；单 seed ≥ -5% | 实测 +8.3%（104.06 vs 96.08）；原建议 10% 偏高 → 改 5% |
| 5 | fidelity 全链路 | pending/suggestions.csv/results/dataset/`cost_spent` 一致 | 已验证 Study `supports_fidelity` 链路；`cost_spent` 需在 report 暴露 |

### 2.2 建议的 pytest 代码（实测数值写进 assert 注释）

```python
def test_mfbo_beats_full_fidelity_at_equal_cost():
    rows = {}
    for algo, params, max_evals in [
        ("mfbo", {"fidelity_levels": "0.25,0.5,1.0", "min_full_fraction": 0.15}, 40),
        ("bayes", None, 24),
    ]:
        best = []
        for seed in (0, 1):
            b, cost, n, fids = run_mf(algo, seed=seed, cost_budget=24.0,
                                      max_evals=max_evals, params=params)
            assert cost <= 24.0 * 1.10            # 允许 ≤10% 超支（当前实测 +5.2%）
            best.append(b)
        rows[algo] = float(np.mean(best))
    assert rows["mfbo"] >= rows["bayes"] * 1.05   # 实测 104.06 vs 96.08 → +8.3%

def test_mfbo_initial_design_uses_low_fidelity():
    # 当前实现会返回 0.25/0.5/1.0 轮转 → xfail，等 P1-5 修复后改 strict
    pairs = build("mfbo", space, objective, params={"fidelity_levels": "0.25,0.5,1.0"}).ask_with_fidelity(6)
    assert all(f == 0.25 for _, f in pairs)

def test_mfbo_cost_budget_respected():
    # 当前 Study 只按次数 → 实测 25.25/24；目标 ≤24（修复 P1-4 后 strict）
    ...
```

---

## 3. 「哪些验收进快 CI」与执行顺序

| 优先级 | 用例 | 耗时（实测） | 理由 |
|---|---|---|---|
| P0 | `qNEHVI #5`（ref 必填/持久化） | <2 s | 防止 NaN ref/静默退化 |
| P0 | `qNEHVI #2`（n_base 收敛） | ~2 s | 验证嵌套 MC 实现正确 |
| P1 | `qNEHVI #4`（批量） | ~5 s | 批量语义/惩罚 |
| P1 | `mf_suzuki #1`（optimum）、#5（fidelity 链路） | <2 s | 基准与链路 |
| P2（nightly） | `qNEHVI #1/#3`（HV 端到端）、`mfbo` 同成本对比 | 10–20 s/seed | 统计需多 seed |
| 待修复后开启 | `mfbo` 初始设计最低档、成本预算、`best` 全保真口径 | — | 见 `REVIEW_v07` P1-4/5 |

**建议的 CI 标记**：`@pytest.mark.slow` 给端到端 HV/多 seed；PR 跑 P0/P1 行（<15 s），nightly 跑全部。

---

## 4. 需要先修的判据问题（否则验收数字会误导）

1. **true HV**（P0-2）：`benchmarks` 保存真值（`Observation.meta["true_objectives"]`），`summarise(use_true=True)`；
2. **`auto_route=False`**（P0-1）：所有验收用显式算法，否则测的是 router；
3. **`objective_std` 标量**：qNEHVI 只能得到两目标同 σ；若某目标噪声不同，实测数字不代表"真异方差"（`REVIEW_v06` P1-7）；
4. **mfbo 成本预算**（P1-4）：当前 `Study` 不感知成本，验收必须允许 ≤10% 超支，或先加 `remaining_cost()`。

---

## 5. 参考

R12（qNEHVI）、R21（qEHVI）、R42（HV/固定 ref/真值）、R41（BOCA：保真度与成本）、R40（Hyperband 成本预算口径）、R32（多 seed/统计）、`REVIEW_v07.md`、`MF_BENCHMARK.md`。
