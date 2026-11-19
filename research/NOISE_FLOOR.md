# 噪声底：先估，再声称差异

工作区规则 #11：**单次测量不足以声称改进**。本项目的很多结论是"多臂比较"，
如果不同臂之间的差异落在 run-to-run 噪声内，就不能写成"更好/更差"，更不能据此改默认值。

`scripts/noise_floor.py` 把这件事变成一条命令，两种模式：

- `--mode floor`：同一配置跑 N 个种子，给出 sd / sem / 95% 半宽，以及**两个独立臂在同样 N 下
  可分辨的最小差异**（`t × sqrt(2/N) × sd`）与相对百分比；
- `--mode paired`：两个配置**同种子**跑，给出逐种子差、配对 t 统计量与结论（超出 / 噪声内）。
  配对设计能抵消种子效应，是描述符这类"同一 landscape、仅编码不同"比较的正确做法。

## 首个应用：描述符 vs one-hot 的可行 HV（5 seeds 配对）

```
py -3 scripts/noise_floor.py --mode paired \
  --problem moo_solvent_constrained --problem-b moo_solvent_constrained_desc \
  --algorithm cqehvi --budget 40 --seeds 0 1 2 3 4
```

| 指标 | 数值 |
|---|---|
| 逐种子差 | −3.00, −23.53, +0.85, +5.40, +2.13 |
| 平均差 | **−3.63（−0.34%）** |
| 差的 sd | 11.52 |
| sem | 5.15 |
| **t 统计量** | **−0.704**（临界 2.776） |
| 结论 | **在噪声内**，不足以声称描述符改变 HV |

**修正**：此前的 3 seeds 写法"MOO 可行 HV −1.4%（噪声内）"虽然结论方向对，但**"噪声内"当时是断言**；
现在有了测量：逐种子差的标准差约 **11.5 HV（≈1.1%）**，所以 3 seeds 只能分辨约 ±19 HV（≈1.8%）。
结论不变（HV 不变），但依据从"感觉在噪声内"升级为"配对 t 检验不显著"。

## 用法约定

- 声称"A 比 B 好/差"之前，先跑 `--mode paired`；`significant_95=false` 就只写"差异未超出噪声底"；
- 报告任何均值差异时同时给出种子数与 sd（`research/EVIDENCE_POLICY.md` §7 第 6 条）；
- 结构性变化（0 ↔ 非 0、成功/失败、可行点 0 → 非 0）不需要 t 检验，但要在文里说明它是结构性的。

## 等效性口径（TOST）："不显著"不等于"等价"

外部审查指出上面那句"`significant_95=false` ⇒ 在噪声内"是 **absence of evidence**：
不显著只说明"本设计没能检出差异"，它同时兼容"两臂确实一样"和"噪声太大、根本测不出来"。
所以 `--mode paired` 增加一条 TOST 等效性判据。**两个口径并存**：旧字段（`significant_95` /
`verdict` / `t_statistic` …）一个都没改，新结论单独放在新字段里（向后兼容）。

### 判据

- δ = `--equivalence-margin`，必须 > 0（否则报错退出，绝不静默退化）；
- 对逐种子差 d_i 做**单样本 t**，取均值差的 **90% 置信区间**（TOST 惯例：每侧 5%），
  `CI = mean_delta ± t(0.95, n−1) · sd_delta / sqrt(n)`；
- `equivalent=true` **当且仅当整个 CI 落在 (−δ, +δ) 内**；同时报出 `ci_low` / `ci_high` /
  `margin` / `mean_delta`，不只报 p 值；
- CI 整段落在 ±δ 之外 → `equivalent=false`（`ci_outside_margin=true`，方向明确）；
- CI 跨越 ±δ 边界 → `equivalent=false`（`ci_outside_margin=false`）即"不确定"：既不能声称等价，
  也没有有方向的差异——这正是旧口径会含糊过去的情形；
- n < 2 → `equivalent=null` 并在 `note` 里写"样本不足"，**不得当成等价**；
- JSON（含 `equivalent`）走 stdout 单行，人类可读摘要走 stderr，下游解析不受影响。

### δ 从哪里来

缺省 δ 不是拍出来的：取两配置各自**同输入重复测量**（多 seed 重跑）的**合并 sd**，即
`--mode floor` 的口径（`sd_a=10.1551, sd_b=3.9078`），输出里 `margin_source` 会写明来源与数值。
`--mode floor` 自身也报 `equivalence_margin_suggested`（= 同输入 sd）。也就是说 δ 的量纲就是
run-to-run 噪声本身：**在这个 δ 下判"等价"，等于要求差异小于噪声尺度**，比"不显著"严格得多。

### 实跑数字（2026-11-17）

```
py -3 scripts/noise_floor.py --mode floor --problem moo_solvent_constrained \
  --algorithm cqehvi --budget 40 --seeds 0 1 2 3 4
py -3 scripts/noise_floor.py --mode floor --problem moo_solvent_constrained_desc \
  --algorithm cqehvi --budget 40 --seeds 0 1 2 3 4
py -3 scripts/noise_floor.py --mode paired \
  --problem moo_solvent_constrained --problem-b moo_solvent_constrained_desc \
  --algorithm cqehvi --budget 40 --seeds 0 1 2 3 4
```

| 项 | 数值 |
|---|---|
| floor（one-hot） | mean 1072.9987、**sd 10.1551**、两臂可分辨最小差异 17.8293（1.662%） |
| floor（desc） | mean 1076.6296、**sd 3.9078**、两臂可分辨最小差异 6.861（0.637%） |
| 逐种子差 | −2.9957, −23.5333, +0.8476, +5.4017, +2.1254 |
| 均值差 | −3.6309（−0.337%），sd_delta 11.5249，t = −0.704 |
| **δ（缺省推导）** | **7.6941** = 合并 sd（sd_a=10.1551, sd_b=3.9078） |
| **90% CI** | **(−14.6193, +7.3576)** |
| **equivalent** | **false**（`ci_within_margin=false`、`ci_outside_margin=false`） |
| 结论 | **不确定**：CI 跨越 ±δ 边界 |

### 旧结论在新口径下怎么写

旧写法"描述符的 HV 差异在噪声底内"依据的是 t 检验不显著，属于 absence of evidence。
按新口径的实跑结果，**正确的表述是**：

> 描述符版与 one-hot 版的可行 HV 差异**未检出**（配对 t = −0.704 < 2.776，均值差 −3.63/−0.34%），
> 但在 δ = 7.6941（同输入噪声尺度）下 TOST **不能证明等价**：90% CI (−14.6193, +7.3576)
> 跨越 ±δ 边界，既不能说"两者一样"，也没有有方向的差异。

要在本次设计（5 seeds）下**确立等价**，δ 至少要 ≥ max(|ci_low|, |ci_high|) = **14.6193 HV
（≈1.36% of mean_a，约为自噪声 sd 的 1.9 倍）**，或者增加种子数把 CI 收窄到 ±δ 以内。
所以 `research/DESCRIPTOR_FEATURES.md` 等处"HV 差异在噪声底内"的措辞应读作"**未检出差异**"
（未检出 ≠ 已证明无差异），除非另跑一次把 δ 或种子数提到上述门槛。

回归测试：`tests/test_noise_floor_equivalence.py`（等价 / 不等价 / n<2 ⇒ null / δ≤0 报错 / 旧字段保留）。
