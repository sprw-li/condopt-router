# 适配器长跑演练（Long Run via Executor Adapter）

`scripts/adapter_long_run.py` 用 `run_campaign` 驱动 120 次预算的无人值守长跑，
仪器带噪声、每 11 次失败一次、4 个并行孔位，用来验证“长时间跑不出问题”。

```powershell
py -3 scripts/adapter_long_run.py --budget 120 --batch 6 --workers 4 --patience 100
py -3 scripts/adapter_long_run.py --budget 120 --batch 6 --workers 4 --force-budget
py -3 scripts/adapter_long_run.py --budget 120 --batch 6 --workers 4      # 默认停止语义
```

## 三次实测（`mf_constrained`，cost_budget=120）

| 配置 | 观测数 | 尝试/失败 | 成本 | 墙钟 | digest | 数据集 | run 目录 | 重复 id | best |
|---|---|---|---|---|---|---|---|---|---|
| `--patience 100`（用满预算） | **120** | 137 / 17 | 84.25 | 38.6 s | 0.61 s | 46.2 KB | 102.1 KB | 0 | 108.79 |
| `--force-budget`（停滞即 retune 续跑） | 81 | 90 / 9 | 59.25 | 14.2 s | 0.38 s | 31.2 KB | 77.5 KB | 0 | 104.13 |
| 默认（停滞即停） | 69 | 75 / 6 | 51.25 | 7.0 s | 0.30 s | 26.5 KB | 60.2 KB | 0 | 106.31 |

三次都：`duplicate_suggestion_ids=0`、`integrity_ok=true`、失败孔位未被计成观测。
仪器端 p95 延迟 2.5 ms（模拟）。

## 多目标 + 约束的长跑（`moo_constrained`，budget 40）

```powershell
py -3 scripts/adapter_long_run.py --problem moo_constrained --budget 40 --batch 4 --workers 4 --patience 100
```

`cqehvi`，40 条观测 / 43 次尝试（3 次失败全部补测）、0 重复 id、完整性 OK、
墙钟 8.5 s、run 目录 39 KB；可行比例最终 87.5%，可交付 HV 曲线 423.91 → **735.47**
（本次 raw HV 恰好也是 735.47，因为收尾前沿全部落在可行域内；前几轮并非如此——
见 `examples/multiobjective_executor.py` 的 143.03 vs 80.94）。

**并发陷阱（本轮实测发现并修复）**：失败注入原本用共享计数器 `calls % 11`，在 4 线程并行下
`calls += 1` 会竞态跳号，导致 11/22/33 永远命中不到 —— 表现为“40 次尝试 0 失败”。
改为加锁取号后，失败数正确（43 次尝试 / 3 次失败）。写并行执行器时，
凡是用共享状态做判断都要同步。

## 多保真 + 多目标的长跑（`mf_moo_constrained`，budget 40）

```powershell
py -3 scripts/adapter_long_run.py --problem mf_moo_constrained --budget 40 --batch 4 --workers 4 --patience 100
```

`mfparego`，40 条观测 / 43 次尝试（3 次失败补测）、0 重复 id、完整性 OK、
成本 19.0（多保真比全保真便宜一半）、墙钟 3.0 s、run 目录 37 KB；
可行比例 27.5%，**可交付 HV 277.52 → 467.17**（raw HV 693.81，说明大部分前沿点不可交付），
`n_feasible_full_fidelity = 6`。

### 指标一致性修复

首跑出现 **可行 HV(712) > raw HV(702)** 的悖论：raw 前沿只用全保真点（既有口径），
而可行 HV 把低保真点也算了进去，被保真偏差抬高。已新增
`Study.feasible_pareto_summary()`（先按保真口径取点、再过滤约束），
执行器与报告共用它；修复后同样场景得到 467 < 694 的合理关系。

## 4 目标 + 多保真的长跑（`mf_moo4`，budget 24）

```powershell
py -3 scripts/adapter_long_run.py --problem mf_moo4 --budget 24 --batch 4 --workers 4 --patience 100
```

`mfparego`，24 条观测 / 26 次尝试（2 次失败补测）、成本 14.5、0 重复 id、完整性 OK；
4 目标 HV 443407.79（量级与既有 `mf_moo4` 基准一致：515205 vs momfbo 507932）。
该问题**无约束**，因此 `feasible_hypervolume_curve` 为空——这也是一个需要覆盖的路径
（曾因 `constraints_fn is None` 直接抛 TypeError，现已 None-safe）。

## 结论

1. 适配器路径在 120 次规模下没有空转、没有重复 id、没有丢观测——失败 17 次全部
   留在 pending 并被补测（尝试 137 = 观测 120 + 失败 17）；
2. **默认停滞语义会提前停止**（69/120）；想用满预算有两条路：调大 `patience`
   （一次跑完）或在停滞时 `retune_for_exploration()` 续跑（`--force-budget`）。
   后者即使续 8 次也只到 81/120——说明该任务上“停滞后再探”的边际收益递减，
   不如一开始就把 `patience` 调大；
3. run 目录比基线 `long_run_drill` 大（102 KB vs 59 KB），增量来自
   `execution_log.jsonl`（每条约 300 B × 137），这是可接受的审计成本。

## 长跑对照表

<!-- adapter-table:start -->
| 配置 | 问题 | 算法 | 观测 | 尝试/失败 | 成本 | 墙钟(s) | 可行全保真 | 可交付 HV | raw HV | 重复id |
|---|---|---|---|---|---|---|---|---|---|---|
| 单目标 + MF + 约束 | `mf_constrained` | `mfbo` | 120 | 131/11 | 89.75 | 56.34 | 14 | — | — | 0 |
| 2 目标 + 约束 | `moo_constrained` | `cqehvi` | 40 | 43/3 | 40.0 | 6.66 | 35 | 489.81 → 735.34 | 735.34 | 0 |
| MF + 2 目标 + 约束 | `mf_moo_constrained` | `mfparego` | 40 | 43/3 | 23.5 | 2.7 | 9 | 277.52 → 678.44 | 700.97 | 0 |
| MF + 4 目标 | `mf_moo4` | `mfparego` | 24 | 26/2 | 14.25 | 1.15 | 11 | — | 527270.34 | 0 |
<!-- adapter-table:end -->

> 由 `scripts/build_adapter_table.py` 依据真实运行生成（含失败补测、指标口径检查）。
> 由 `scripts/build_adapter_table.py` 依据真实运行生成（含失败补测、指标口径检查）。
