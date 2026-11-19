# 约束单目标的第二个数据集：硬可行域上 constrained 明显更好

`constrained_single_objective` 之前只靠 `solvent_constrained`（高基数分类 + 绑定上限）与带噪约束基准。
本轮补一个**连续、硬可行域**的数据集（`constraint_hard`：约 6% 可行），对照 `bayes`（不看约束）与
`random` 基线（`scripts/benchmark_constrained_second.py`，budget 30、batch 3、3 seeds）：

| 问题 | constrained | bayes | random |
|---|---|---|---|
| `constraint_hard`（6% 可行） | **92.97**（3/3 找到可行点） | 87.80（**仅 1/3**） | 78.11（3/3） |
| `constraint_easy`（约束不绑定） | 94.99 | 95.00 | 93.52 |

## 结论

1. **硬可行域上 `constrained` 比 `bayes` 高 5.2 分，而且 bayes 有 2/3 的运行根本没找到可行点** ——
   这比“目标值略差”严重得多：交付物可能直接不可用；
2. 约束不绑定时两者持平（94.99 vs 95.00）——规则只承诺“有硬约束/绑定约束时用 constrained”，
   并不声称无约束时更好；
3. 与既有证据（溶剂场景 90.39 vs 84.7）方向一致，且这次是连续空间，说明结论不依赖分类编码。

## 复现

```
py -3 scripts/benchmark_constrained_second.py --budget 30 --seeds 0 1 2
```
