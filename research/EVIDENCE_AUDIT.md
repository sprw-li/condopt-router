# 证据声明复核（Evidence Audit）

`scripts/verify_evidence.py` 会**真的重跑**选定的基准脚本，检查 `DEFAULTS.md` / `SCENARIOS.md`
引用的关键数字是否仍成立（`docs/SCENARIOS.md` 只证明“选了哪个算法”，这里证明“数字没漂”）。

```powershell
py -3 scripts/verify_evidence.py --quick   # 两条最便宜的
py -3 scripts/verify_evidence.py           # 全部（约 10 分钟）
```

## 声明与最近一次结果

| 声明 | 断言 | 最近实测 | 状态 |
|---|---|---|---|
| `cmaes_beats_bayes_10d` | 10 维光滑连续上 cmaes 优于 bayes（rosenbrock10 与 ackley10） | cmaes −2.70 vs bayes −5.87（缩减预算 50、2 seeds） | ✅ |
| `descriptor_hv_difference_is_not_detected` | 描述符的 HV 差异**未检出**（配对 t 不显著；A4 的 TOST 口径下 90% CI (−14.62, +7.36) 跨越 ±δ ⇒ **不确定**，不能写“等价”） | mean −3.63、sd 11.52、t=−0.704 < 2.776；equivalent=false（CI 跨边界） | OK |
| `descriptors_raise_coverage_not_quality` | 描述符提升覆盖而非最好值（MOO+约束） | desc 可行点 37.0、可行 HV 1045.1 | OK |
| `descriptors_only_help_a_constraint_aware_optimizer` | 描述符必须与约束感知算法成对使用 | constrained 可行率 0.550→0.875；bayes 最优值 85.5→83.0 | OK |
| `staged_transfer_can_lose_when_the_veto_misses` | 否决漏判时分阶段会输给 fixed_k1 | discrete_catalyst biased：staged −1.03（否决 0）vs fixed_k1 −0.32 | OK |
| `staged_api_matches_the_hand_rolled_policy` | API 与手工配方逐位一致 | staged_audit == staged_recheck（三个源质量） | OK |
| `transfer_veto_band_is_safe_from_0_4` | 否决线 ≥0.4 才起保护作用 | t0.2 worst −0.53 vs t0.4/0.6 +0.71 | OK |
| `staged_transfer_avoids_the_biased_loss` | 分阶段迁移避开有偏源亏损 | biased staged +0.71 vs k3 −1.54 / k5 −0.33 | OK |
| `audit_is_blind_to_the_size_of_the_bias` | 审计分与偏差大小无关（一票否决器） | 偏差 0~−18 审计恒同，收益 +2.49 → −0.51 | OK |
| `audit_cannot_separate_a_biased_source` | 审计对中等偏差无分辨力（有偏源高分却负收益） | biased audit=0.65 delta=−0.54 | OK |
| `prior_reliability_is_observable_when_powered` | 有功效时 reliability 缩放可观测 | r=1.0 +44.9 vs r=0.4 +39.3 | OK |
| `warm_start_beats_cold_after_plateau_fix` | 修复平台期 bug 后好源暖启动优于冷启动 | cold 73.81 vs k=3 81.10（+7.3） | OK |
| `constrained_beats_bayes_on_hard_feasibility` | 硬可行域上 constrained 更优且能找到可行点 | constrained 92.97（3/3）vs bayes 87.80（1/3） | OK |
| `noise_advice_matches_measurement` | 高噪声下 UCB+大 kappa 不比默认 EI 好 | noise levels 0/5/12：EI 优于 UCB（ucb>ei 无） | OK |
| `tiny_budget_keeps_the_surrogate` | 极小预算下代理类不逊于纯空间填充 | parego 523.3 vs sobol 27.6（metric_mean） | OK |
| `confirm_batch_three_is_not_worse` | 默认 3 次复孔在 branin 上不劣于 2 次 | confirm3 6.39 vs confirm2 2.59 | OK |
| `mid_budget_policies_agree_below_20` | 预算 12 时两种停止口径结果一致 | 两口径 15.72/15.72（spread 0） | OK |
| `transfer_audit_grades_bad_sources_low` | 一致源 ≥0.8、坏源 ≤0.6 | consistent 1.0 / worst_bad 0.3 | OK |
| `constrained_beats_bayes_on_constraints` | 带硬约束时 constrained 优于 bayes | constrained 90.36 vs bayes 83.13 | ✅ |
| `mfparego_matches_momfbo_two_objectives` | 2 目标 + 多保真（无约束）上 mfparego 不落后 | mfparego 674.5 vs momfbo 669.5（3 seeds） | OK |
| `mfparego_matches_or_beats_momfbo` | 3 目标 + 多保真上 mfparego 不落后（≥0.97×） | mfparego 27318 vs momfbo 26998 | ✅ |
| `deliverable_hv_is_tracked_and_bounded` | 约束多目标的可行 HV 单独记录且 ≤ raw HV | feasible 80.94 vs raw 143.03 | ✅ |
| `mfparego_matches_momfbo_under_constraints` | 2 目标 + 约束 + 多保真上 mfparego 不落后（≥0.97×） | mfparego 640.99 vs momfbo 636.27 | ✅ |
| `retune_beats_stopping_on_stagnation` | 停滞时调参继续优于直接停止 | tune −324.69 vs stop −335.76（缩减预算下差距收窄） | ✅ |
| `feasible_full_evidence_always_delivered` | 约束 + 多保真下总能拿到可行全保真证据 | aware 臂 `has_feasible_full_rate=1.0` | ✅ |

## 设计说明

- 每次复核用**更小预算/更少 seeds**（只要能判定方向），因此数字与 `DEFAULTS.md` 中
  的正式基准不完全相同，这是刻意的成本取舍；
- 断言写成**方向性**（优于 / 不落后于 / 比例达标）而不是精确数值，避免噪声导致的假失败；
- 实测：6 条声明全部重跑只需 **50 s**（各 2.9–14.2 s），因此 release_check 直接跑**全量**
  `verify_evidence.py`（门禁名 `evidence_claims`）；`--quick` 保留给开发期的 4 条快检；全量 12 条约 230 s。
- `tests/test_evidence_claims.py` 只校验声明列表完整、脚本存在、文档同步（不在 CI 跑基准）。
