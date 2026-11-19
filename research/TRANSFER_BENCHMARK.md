# 暖启动（transfer_from）收益验证

## 方法

`scripts/benchmark_transfer.py`，等总预算对比：

- **cold**：18 次评价，bayes 从零开始；
- **warm**：先跑 6 次 sobol 源战役，再用 `transfer_from(top_k=4)` 把最优 4 点作为先验，
  bayes 花剩余预算（总计仍 18 次评价）；
- 问题：`rosenbrock10`（10 维）与 `sphere2d`；3 seeds。

## 结果

| 问题 | cold best 均值 | warm best 均值 | 变化 |
|---|---|---|---|
| rosenbrock10 | −997.1 | **−631.7** | **+37%（更接近 0）** |
| sphere2d | −0.0021 | **−0.0016** | 略优 |

## 结论

- 同预算下，先做一小段低成本探索再把最优点迁移进主战役，**在困难 landscape 上显著更好**；
- 简单问题上差异小（两边都接近最优）；
- 该实验也暴露并修复了一个真实 bug：源战役的 `suggestion_id` 会与目标战役的 pending id
  冲突，导致 `tell` 把新结果当成“幂等重放”而跳过、pending 卡住。
  现在迁移时会把原 id 挪到 `original_suggestion_id`，不再参与目标 id 空间。

## 复现

```powershell
py -3 scripts/benchmark_transfer.py --seeds 0 1 2
```
## top_k 与源质量敏感性

`scripts/sweep_transfer.py`（rosenbrock10，总预算 24 = 8 源 + 16 主，3 seeds）
对比 good 源（真实标签）与 bad 源（标签打乱，模拟不相似战役）：

| top_k | good 源 best 均值 | bad 源 best 均值 |
|---|---|---|
| 1 | −429.4 | −369.2 |
| 2 | **−310.4** | −534.7 |
| 4 | −650.4 | −884.0 |
| 8 | −419.8 | −1190.2 |

结论：

- **bad 源负迁移随 top_k 单调恶化**（k=1 → k=8：−369 → −1190）；
- good 源收益不是单调的（k=2 最好），但即使 k 偏大也不会崩；
- 因此 `transfer_from` 在 `top_k > 4` 或未指定时返回 `warning`，CLI 帮助也提示
  “建议 1–3”；`fork` 默认 top_k 从 10 下调到 5。
## 源可信度自检

`transfer_from` 现在除 top_k 之外还会检查：

- **领域一致性**：`source.context.domain != target.context.domain` → 跨领域迁移警告；
- **源样本量**：源战役观测 < 5 条 → “先验统计价值有限”警告；
- 所有提示集中在返回值的 `warnings` 列表（`warning` 保留第一条，向后兼容）。

这些检查不阻止迁移（空间/目标一致性仍由硬校验保证），只提示风险，
让 AI/人决定是否继续。
## 多目标迁移：保留前沿折中

单目标迁移按主目标分数取 top_k；多目标若也这么做，会把 Pareto 前沿的另一端丢掉。
现在多目标迁移先取**非支配前沿**（按主目标分数排序），不足 top_k 再用剩余点补齐，
返回值里标注 `selection: "pareto_front"`（单目标为 `"primary_score"`）。

测试覆盖：源战役含 (yield 95, cost 15)、(80, 7)、被支配点 (70, 12)，
top_k=2 时导入前两者（cost 集合 {7, 15}），被支配点被排除。
## 先验可信度加权（prior_reliability）

坏源 + 大 top_k 的负迁移可以靠**抬高先验观测噪声**缓解：
`transfer_from(..., prior_reliability=r)` 给迁移点加噪声下限 `(1-r)·span/2`，
GP 仍能看到先验但不会让它压倒新证据。

实测（bad 源，rosenbrock10，3 seeds）：

| top_k | r=1.0 | r=0.5 | r=0.25 |
|---|---|---|---|
| 4 | −884.0 | −884.0 | **−697.8** |
| 8 | −1190.2 | **−1009.8** | −1009.8 |

默认 r=1.0（完全信任）；源可疑时建议 0.25–0.5。CLI：`import --prior-reliability`。
## 自动可信度评估（assess_transfer / transfer-audit）

`prior_reliability` 不必靠感觉：目标战役已有 ≥3 条自身观测后，
`Study.assess_transfer(source)` 会用目标代理预测源条件，并与源标签比较：

- 输出 Spearman 秩相关、归一化 MAE、`suggested_reliability`；
- 映射：rho ≥0.6 → 1.0；0.2–0.6 → 0.6；<0.2 → 0.3；无法估计 → None；
- CLI：`condopt transfer-audit --run <target> --from-run <source> --json`。

推荐流程：

```
1) target 先跑 3–6 次自身实验
2) transfer-audit 估计源可信度
3) transfer_from(prior_reliability=suggested)
```
## 迁移影响演练（simulate_transfer / transfer-impact）

真正的 `transfer_from` 之前，可以先用副本演练：

```
condopt transfer-impact --run <target> --from-run <source> --top-k 3 --json
```

- 在临时副本上执行迁移并重新 `recommend()`，返回 `before` / `after` / `changed` /
  `transferred` / `live_run_untouched`；
- **不改动 live run**（测试已断言 dataset 与 pending 不变）。

推荐决策顺序：`transfer-audit`（源可信度）→ `transfer-impact`（影响预览）→ `transfer_from`。

