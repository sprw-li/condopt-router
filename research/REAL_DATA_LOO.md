# 真实测量数据的留一验证（ART HTE，6 条实测点）

## 方法

- 数据：`examples/paper_campaigns/art_table2_real.csv`（Angew. Chem. Int. Ed. 2026，
  ART metallaphotoredox HTE，Table 2 Br 战役的 6 条实测条件/产率）。
- 空间：12 溶剂 × 10 光催化剂 × 5 Ni 前体 × 10 配体 × 5 胺 = 30,000 组合（真实水平）。
- 流程（`scripts/real_data_loo.py`）：
  1. 留一：每次用其余 5 条真实观测建立 study（`import_observations`）；
  2. `Study.predict(held_out)` 预测该点产率（冻结超参 GP，无代理 oracle）；
  3. 汇总 MAE、Spearman 秩相关、真实最优是否被预测为最优。
- 全程只用真实测量数据，不涉及任何合成函数。

## 结果

| 指标 | 值 |
|---|---|
| n | 6 |
| MAE（产率点） | **11.81** |
| Spearman（预测 vs 实测） | **−1.0** |
| 真实最优被预测为最优 | **否** |

实测/预测序列：

```
actual    : 99, 95, 91, 78, 77, 71
predicted : 82.4, 83.2, 84.0, 86.6, 86.8, 88.0
```

预测值几乎退化为均值（82–88），且排序与实测**反相关**。

## 解释

- 5 个训练点覆盖 30,000 组合的类别空间：GP 只能学到先验均值，类别间差异无法识别；
- 预测偏高/偏低的方向由 one-hot 距离主导，纯类别、无描述符时不可靠；
- 这不是 route/loop 的缺陷，而是**数据效率**问题：真实 HTE 需要
  - 更多先验点（论文自身用 30k 空间 + 主动学习），
  - 或类别描述符（分子量/logP/介电常数等）替代 one-hot（ALGORITHM_CATALOG 已列为已知缺口），
  - 或跨底物迁移（warm start / transfer）。

## 结论与后续

1. 诚实记录：**6 条真实数据不足以做代理预测**；任何“用 6 点预测 30k 空间”的说法都不可信。
2. 该实验同时验证了 `predict`/`import`/CSV 通路在真实数据上可用（含逗号字段、字号列）。
3. 后续优先级：
   - 类别描述符特征（把 one-hot 换成物理描述符）；
   - `transfer_from` 跨战役暖启动；
   - 更多真实测量片段（Table 2 全表或其他公开 HTE 数据）。
## 加入溶剂描述符后的复验

`examples/paper_campaigns/art_solvent_descriptors.json` 给出 12 种溶剂的
**近似**物性（MW / 介电常数 / logP；NBP、Polarclean、tert-amyl alcohol、2-MeTHF 为估计值，
文件内已标注，生产环境请换用权威表）。仅对 `solvent` 变量启用描述符（其余仍 one-hot）：

```powershell
py -3 scripts/real_data_loo.py --descriptors examples/paper_campaigns/art_solvent_descriptors.json
```

| 指标 | 无描述符 | 仅溶剂描述符 |
|---|---|---|
| feature_dim | 47 | 33 |
| MAE | 11.81 | 12.14 |
| Spearman | −1.00 | **−0.83** |
| 真实最优被预测为最优 | 否 | 否 |

结论：描述符让秩相关从 −1.0 改善到 −0.83（方向不再完全相反），但**仍然没有预测能力**。
根因不是特征表示，而是**样本量**：5 个训练点无法覆盖 30,000 组合，无论 one-hot 还是描述符。

后续要真正可用，需要：
1. 更多真实先验点（论文自身用主动学习跑满 30k 空间的一部分）；
2. 给全部 5 个变量配权威描述符（当前只有溶剂）；
3. 跨战役迁移（`transfer_from`）把历史 HTE 数据作为先验。

