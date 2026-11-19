# 维度边界：cmaes 规则是"维度决定"还是"地貌决定"？

路由在 **8–14 维光滑连续**时切到 `cmaes`，但归档证据只有 **10 维**（rosenbrock10 / ackley10）。
本轮用同一套地形扫 6/8/10/12 维（`scripts/benchmark_dim_boundary.py`，budget 60、batch 6、3 seeds，
指标=原始 best score，越大越好）：

| 维度 | Rosenbrock bayes / cmaes | 胜者 | Ackley bayes / cmaes | 胜者 |
|---|---|---|---|---|
| 6 | −33.65 / **−12.02** | cmaes | **13.37** / 11.42 | bayes |
| 8 | −143.55 / **−9.85** | cmaes | **13.46** / 11.81 | bayes |
| 10 | −169.21 / **−21.43** | cmaes | 13.38 / **13.46** | cmaes（差 0.08，噪声级） |
| 12 | −323.54 / **−28.69** | cmaes | **13.38** / 12.72 | bayes |

## 结论

1. **窄谷型（Rosenbrock）从 6 维起就是 cmaes 更好**，而且差距随维度放大（21 → 294）；
   "8 维才切 cmaes" 对这类问题**偏保守**。
2. **多峰型（Ackley）在 6/8/12 维都是 bayes 更好**，10 维只是打平（差的 0.08 属噪声）；
   即原归档"10 维上 cmaes 也赢 ackley"的说法**没有稳定复现**。
3. 因此 **"维度"本身是弱信号**，真正的决定因素是地貌（窄谷 vs 多峰），而 router 的上下文里
   没有可用的地貌字段（只有 smoothness）。8 维规则应当理解为**"高维 GP 不可靠"的经验兜底**，
   而不是"8 维以上 cmaes 一定更好"。

## 处置（不改规则，改说明）

- 规则保持：≥8 维仍优先 `cmaes`（避免在高维上退回 GP）；
- `docs/ROUTING_POLICY.md` 增加该边界说明与两个反例；
- `docs/DEFAULTS.md` 的 cmaes 行标注"地貌依赖、非纯维度决定"；
- 登记表 `cmaes_high_dim` 增加该数据集，tradeoff 写明"6 维窄谷也可能该选 cmaes"；
- 场景 B2（6 维 → bayes）改为**按 Ackley 型标定**的说明（其证据只覆盖多峰地形）。

## 复现

```
py -3 scripts/benchmark_dim_boundary.py --dims 6 8 10 12 --budget 60 --seeds 0 1 2
```
