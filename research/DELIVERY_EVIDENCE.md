# 交付证据：四种问题形态的自检通过率

`delivery_evidence_policy`（交付必须带证据）最初由**约束场景**的数据定出
（可行 HV vs raw HV 80.94 vs 143.03、可行全保真证据率 0.40 → 1.00）。这一轮验证它在
**约束之外**是否同样成立 —— 走真实适配器路径（`run_campaign`），budget 24，3 seeds：

| 问题 | 形态 | 选中算法 | 交付自查 | 观测数 |
|---|---|---|---|---|
| `moo_yield_cost` | 2 目标，无约束，单保真 | parego | **3/3** | 23.3 |
| `mf_moo` | 2 目标，无约束，多保真 | mfparego | **3/3** | 23.0 |
| `moo_constrained` | 2 目标 + 约束（对照） | cqehvi | **3/3** | 21.3 |
| `sphere2d` | 单目标（复孔路径） | bayes | **3/3** | 23.0 |

合计 **12/12**，缺项为空。

## 结论

1. 交付自查（前沿存在 / 逐批进展 / 复孔确认 / 可行全保真）在四种形态下都能满足，
   不是“约束场景专属”；
2. 观测数比预算少 0–3 次（保留的交付槽位），与 `MID_BUDGET.md` 的权衡一致；
3. 单目标场景走的是“复孔确认”分支，多目标走“前沿 + 进展”分支，两条分支都有实测覆盖。

## 复现

```
py -3 scripts/benchmark_delivery_evidence.py --seeds 0 1 2 --budget 24
```

## 复核（Round 291，budget 24，2 seeds × 4 形态）

| 形态 | 算法 | 交付自查 |
|---|---|---|
| moo_yield_cost（无约束 MOO） | parego | **2/2** |
| mf_moo（多保真 MOO） | mfparego | **2/2** |
| moo_constrained（约束 MOO，对照） | cqehvi | **2/2** |
| sphere2d（单目标） | bayes | **2/2** |

四种形态 8/8 通过交付自查，与归档的 12/12 一致；`missing` 列表为空。
