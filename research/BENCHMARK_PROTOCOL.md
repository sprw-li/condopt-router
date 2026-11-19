# 基准协议（预注册）BENCHMARK_PROTOCOL

> 状态声明：**在本协议落地并被真正执行（默认 20 个种子、共享初始设计、IQM + 区间 + 效应量）之前，
> 仓库里的主 benchmark（`scripts/benchmark*.py`）只能称为「工程 smoke」，不能作为"某算法更好/更差"的证据。**
> 工程 smoke 的用途只有两个：算法是否还能跑、接口是否还通。

实现：`scripts/benchmark_protocol.py`　测试：`tests/test_benchmark_protocol.py`

---

## 1. 为什么需要预注册

外部审查指出的问题不是"数字不够好看"，而是**比较本身不成立**：

| 缺陷 | 后果 |
|---|---|
| 每个算法各自随机初始设计 | 算法差异里混着初始设计的好坏，A−B 无法归因给算法 |
| 种子数 3–4 | bootstrap 区间宽到几乎必然跨 0，任何"结论"都不可区分于波动 |
| 汇总只用算术均值 | 单个种子的发散会把均值整段拉走，看不出是多数种子还是离群点 |
| 没有配对与区间 | 报的是"均值差"，不是"差异的置信区间" |
| 没有噪声底 | 不知道多小的 Δ 属于同一配置的重复测量离散度 |

预注册的意思：**在看到结果之前就把问题、算法、种子、预算、初始设计、指标和判定规则固定下来**，
并且把实际用的种子列表写进输出。事后挑指标、挑种子、挑初始设计，都会让数字失去证据力。

## 2. 预注册的内容（一次性固定）

1. **问题**：从 `condopt.benchmarks.builtin_problems()` 里选，默认 `discrete_catalyst`
   （全离散、网格 12×81 可穷举、单目标、便宜）。
2. **算法集合**：默认 `sobol` + `tpe` 两个，显式列出，不事后替换。
3. **种子集合**：`--seeds N` 取 `range(seed_start, seed_start+N)`；**默认 N=20**，
   实际使用的种子列表原样写进 `results.json` 的 `seeds` 字段和 CSV 的 `seed` 列。
4. **预算**：`--budget`（默认 30），含共享初始设计在内；每个算法、每个种子都跑同样的预算。
5. **共享初始设计**：`--init`（默认 5）个点，由 `space.sample_sobol_base2(n, seed=seed)`
   派生，**同一 seed 下所有算法拿到同一组点**，并作为 `prior` 注入优化器后再开始自适应采样。
   每个算法的初始设计指纹（sha1 前 16 位）写进 CSV 的 `initial_design_sha` 列与 JSON，
   用来证明"跨算法同一起点"确实成立。
6. **指标**：按问题类型自动选
   - 单目标 → `best`：最终 best 分数，已按优化方向取符号（最小化问题乘 −1），
     因此协议里**永远是"越大越好"**；有约束的问题只在可行点上取 best，
     一个可行点都没有时记 `NaN`（聚合时整对剔除，不用不可行点冒充成绩）；
   - 多目标 → `hypervolume`：用 `condopt.pareto.summarise` 在固定参考点上算 HV。
7. **分析计划**：IQM + bootstrap 95% CI + 同 seed 配对效应量 + 噪声底（见 §4、§5）。

## 3. 怎么跑、产出什么

```bash
# 便宜档：几分钟内跑完，用于验证协议本身能跑通
py -3 scripts/benchmark_protocol.py --seeds 3 --budget 12

# 协议默认档（20 个种子，才够谈"证据"）
py -3 scripts/benchmark_protocol.py --seeds 20 --budget 30 --problem discrete_catalyst
```

产出（默认写到 `runs/benchmark_protocol/`）：

| 文件 | 内容 |
|---|---|
| `results.json` | 固定字段报告（见 §6） |
| `per_seed.csv` | per-seed 明细：`problem, metric, algorithm, seed, value, n_evals, run_kind, initial_design_sha, budget` |
| `summary.md` | 人类可读摘要（与 stdout 相同） |

`run_kind` 区分 `main`（进入汇总）与 `repeat`（噪声底重复跑，**不进入主汇总**）。

**空提议不截断预算**：采样型算法会去重，共享初始设计可能正好落在它的序列上，
某次 `ask` 返回空。协议把空提议记一次尝试并继续（序列会前进），而不是静默截断——
截断会让某个算法的 `n_evals` 悄悄小于 budget，跨算法比较随之失真。

## 4. 怎么判

设 A、B 为两个算法，`d_i = metric_A(seed_i) − metric_B(seed_i)`，只在同一 seed 上配对。

| 结论 | 条件 |
|---|---|
| `better` / `worse` | `d` 的 bootstrap 95% CI **不跨 0**，**且** `|mean(d)| > 噪声底` |
| `inconclusive` | CI 跨 0，或 `|mean(d)| ≤ 噪声底`，或出现 `NaN`/配对不足 |

**硬纪律**：CI 跨 0 时结论字段必须是 `inconclusive`，禁止写成"更好/更差"。
这条有测试（`tests/test_benchmark_protocol.py::test_verdict_inconclusive_when_ci_crosses_zero`）。
`inconclusive` 不是失败结论，而是正确结论：当前样本量回答不了这个问题。

CI 用 percentile bootstrap：对配对差值有放回重采样 `--bootstrap`（默认 2000）次求均值，
取 2.5% / 97.5% 分位；固定随机种子 `BOOTSTRAP_SEED = 20261117`，保证区间可复现
（否则"结论"会随重跑变化）。

## 5. 噪声底怎么估

**定义**：同一算法、同一 seed、完全相同配置重复跑一次，取 `|重复值 − 原值|`，
对所有算法取最大值，作为该次协议的噪声底（`noise_floor.value`），
并逐算法记录在 `noise_floor.per_algorithm`。默认只重复第一个种子（`--no-noise-floor` 可关闭）。

**注意它的边界**：本仓库的 benchmark 是确定性模拟器 + 固定 seed 的算法，
因此噪声底通常是 **0.0**——这只说明"协议层可复现"，**不代表实验室重复测量的噪声是 0**。
真实实验的噪声底必须用同一条件的重复测量估计（参见 `scripts/noise_floor.py`），
并在解释 Δ 之前先做对比。噪声底为 0 时，"CI 不跨 0" 就等于"确定性差异"，
但**仍然不自动等于"结论"**：样本量与问题覆盖面决定外推范围。

## 6. 输出字段契约

`results.json` 顶层必须含有以下固定字段（外部审查按名字取值）：

| 字段 | 含义 |
|---|---|
| `problem` | 问题名 |
| `metric` | `best` 或 `hypervolume` |
| `seeds` | 实际使用的种子列表 |
| `algorithms` | 算法名列表（顺序即配对比较的基线顺序） |
| `iqm` | 算法 → IQM |
| `ci_low` / `ci_high` | 算法 → bootstrap 95% CI 下/上界 |
| `n` | 算法 → 参与汇总的有效种子数（NaN 不计） |
| `noise_floor` | `{value, seed, replicates, per_algorithm, definition}` |
| `pairwise` | 配对效应量列表：`{baseline, algorithm, n_pairs, mean_diff, ci_low, ci_high, relative_pct, mean_a, mean_b, verdict, reason}` |

另有 `budget`、`initial_design_size`、`batch_size`、`initial_design`、`protocol`、
`metric_convention`、`generated_at`、`note` 作为审计上下文。
`relative_pct = 100 × mean_diff / |mean_b|`（B 的均值，配对子集上），B 均值≈0 时为 `null`。

## 7. 为什么用 IQM 而不是均值

- **算术均值的尾部敏感**：一个种子发散（例如某算法在高维 landscape 上崩到 −1e6），
  均值就被完全支配，读出来的"差异"其实只来自那一个种子。
- **中位数丢掉幅度**：中位数只关心排序，看不出"多数种子好了 5%"和"好了 0.1%"的区别。
- **IQM（四分位均值）**：排序后去掉上下各 25% 再取均值，既不被极端尾部支配，
  又保留主体分布的幅度信息。它是"截尾均值"的标准形式，配合 bootstrap CI 有明确的抽样语义。

`iqm()` 的实现与边界：`n=8, trim=0.25` → 去掉 2+2 个，取中间 4 个的均值；
点数太少（`n − 2k < 1`）时不截断，退化为均值；`NaN` 先剔除再计算。

## 8. 已知局限（诚实清单）

- **默认问题面太窄**：默认只跑 `discrete_catalyst`。协议可跑多目标/约束/多保真问题
  （`--problem` 任选 `builtin_problems()`），但**一个问题的结论不能外推到其它问题**。
- **确定性模拟器**：噪声底 0，不代表真实实验室的噪声底；真实场景必须单独估。
- **种子数**：20 个种子只够支持中等效应；报告里的区间宽度就是说这话的。
- **计算规模**：默认配置以"单机几分钟"为目标，不是大规模 campaign 的替代品。
- **不替代工程 smoke**：本协议验证"比较是否成立"，不替代"接口是否还能跑"的快速 smoke。

## 9. 验收

```bash
py -3 -m pytest -q tests/test_benchmark_protocol.py     # 必须 exit 0
py -3 scripts/benchmark_protocol.py --seeds 3 --budget 12
```

测试覆盖：① IQM 与 bootstrap CI 在已知合成数组上的正确性（不跑真优化）；
② CI 跨 0 → `inconclusive`；③ 共享初始设计在同一 seed 下跨算法一致；
④ 输出 JSON 固定字段齐全 + CSV/摘要落盘。

### 小规模示例（n=3，**不构成证据**，仅示范输出形态）

```
problem  : discrete_catalyst    metric: best（越大越好）
budget   : 12      共享初始设计: 5 点    batch: 1
seeds    : [0, 1, 2]
noise_floor: 0.00000（同配置重复跑，seed=0）

algorithm        n           IQM                    95% CI
sobol            3      75.25267      [73.92800, 76.84800]
tpe              3      74.98867      [74.87200, 75.11200]

sobol - tpe: Δ=0.26400 CI=[-0.94400, 1.86600] 相对=+0.35% n_pairs=3 -> inconclusive
```

注意：即使噪声底为 0，CI 依然跨 0 —— 3 个种子与一个便宜问题回答不了"哪个算法更好"。
这正是本协议存在的意义。
