# 迁移可信度评估（transfer-audit）能区分好坏源吗？

`Study.assess_transfer(source)`（CLI：`condopt transfer-audit --run 目标 --from-run 源`）
用目标已测数据拟合代理、预测源条件并与源标签比较（Spearman + 归一化 MAE），
给出建议的 `prior_reliability`。

本实验构造 4 类源，它们与目标**共享 space/objective 签名**（所以 `transfer_from` 会接受），
但预测能力不同（1 维问题 `y=-((x-0.3)^2)`，源 12 点、目标 6 点，seed 0–2）：

| 源类型 | 含义 | 建议 reliability（3 seeds） | Spearman 均值 |
|---|---|---|---|
| `consistent` | 同问题不同种子（真实） | **1.0 / 1.0 / 1.0** | 0.993 |
| `shifted` | 最优点被移动（相关性弱） | 0.3 / 0.3 / 0.3 | −0.552 |
| `shuffled` | 标签随机打乱 | 0.3 / 0.3 / **0.6** | 0.007 |
| `flipped` | 标签取负（反相关） | 0.3 / 0.3 / 0.3 | −0.993 |

## 结论

1. **一致性源被明确标成 1.0**，三类坏源全部落到 0.3–0.6 —— 评估器方向正确、区分度高；
2. **有一个告警案例**：打乱源在某个种子上给出 0.6（落在“中等可信”区间）。
   因此使用建议时应遵循：`1.0` 放心用；`0.5–0.6` 视为“可疑”，配 `top_k` 小一点；
   `≤0.3` 基本别用大 `top_k`；
3. 与既有收益证据互补：坏源按 0.25–0.5 折扣后，等预算收益从 −884 回升到 −698
   （见 `research/TRANSFER_BENCHMARK.md`），即“建议值可执行且确有保护作用”。

## 小样本会误判（真实数据上的反例）

在真实 ART Table 2 六点数据上（`examples/paper_campaigns/run_real_prior_transfer.py`，
目标 3 条 / 源 3 条，代理 oracle 明确标注）：

| 臂 | suggested_reliability | proxy_best | 是否找回真实最优（99.0） |
|---|---|---|---|
| cold s0 | 0.3 | 83.4 | 否 |
| cold s1 | 0.3 | 75.78 | 否 |
| warm s0 | 0.3 | **99.0** | **是** |
| warm s1 | 0.3 | **99.0** | **是** |

**迁移 2/2 成功，audit 却建议 0.3（“avoid”）**——因为可比条件只有 3 个，
秩相关在小样本下噪声极大。因此：

- `assess_transfer` 现在返回 `confidence`（可比条件 <5 → `low`）与 `sample_warning`；
- `reliability_grade` 在低置信度下给出 `low_confidence`：**建议值不可当硬约束**，
  先用小 `top_k`（1–2）试一次，或补目标观测后再评估；
- `condopt transfer-audit` 会打印该警告。

## 置信度：bootstrap 区间

`assess_transfer` 现在对秩相关做 200 次 bootstrap（固定种子，可复现），返回
`spearman_ci {low, high, level: 0.9}`；`confidence` 判定的条件是：

- 可比条件 <5，或
- bootstrap 90% 区间宽度 >1.0（关系不稳定）

满足任一即 `low`，`reliability_grade` 相应返回 `low_confidence`（建议值不可当硬约束）。
`condopt transfer-audit` 会打印区间与警告。

## 点估计 → bootstrap 中位数（更稳）

`suggested_reliability` 现在用 **bootstrap 中位数**（200 次重采样）而不是单点 Spearman：

- 4 类源分离效果不变（一致 1.0；shifted/shuffled/flipped 0.3–0.6）；
- 同时保留 `suggested_reliability_point`（单点估计）与 `spearman_robust`，便于对照；
- 真实 ART 案例仍是 0.3 —— 因为 3 点代理确实给出负相关，中位数救不回来；
  但它现在带 `confidence=low`，所以不会被当成判决（上一轮的教训）。

## 真实数据上的排序能力：没有（20 个切分）

`scripts/benchmark_real_audit_splits.py` 把 ART 六点做全部 C(6,3)=20 个源/目标切分，
每个切分跑一次 audit，并标记“源是否包含实测最优 99”：

| 指标 | 源含最优（10 个切分） | 源不含最优（10 个切分） |
|---|---|---|
| 建议 reliability | 0.44 | **0.65** |
| Spearman | −0.40 | −0.10 |
| 低置信度比例 | — | **1.00（全部 20 个）** |

**结论**：n=3 时 audit 不但没有排序能力，方向还略有反转（含最优的切分建议值更低）；
而 `confidence=low` 在 20/20 上全部触发——**保护机制按设计工作**：
小样本下不要相信建议值，先补观测。

## 复现

```powershell
py -3 scripts/benchmark_transfer_audit.py --seeds 0 1 2
```
