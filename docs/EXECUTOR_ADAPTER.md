# 执行器适配（Executor Adapter）

`condopt` 负责“下一批做什么条件”，真实实验由你实现**一个方法**：

```python
from condopt.executor import ExecutionRequest, Executor, Measurement, run_campaign

class MyInstrument(Executor):
    def measure(self, request: ExecutionRequest) -> Measurement:
        y, constraints, cost = run_experiment(request.x, request.fidelity)
        return Measurement(
            id=request.id, y=y, fidelity=request.fidelity,
            constraints=constraints, cost=cost,
        )

report = run_campaign(study, MyInstrument())
```

## 复用旧战役：`prior_source`

```python
report = run_campaign(study, instrument, prior_source="runs/old_campaign")
```

执行器按已实测的**分阶段配方**处理先验（`research/TRANSFER_ADAPTIVE.md`）：先只搬 1 条，跑两批本地观测，
再由 `transfer-audit` 决定是否放大；结论（`widened` / `vetoed`）**只结算一次**（Round 288 实测复检
0 次翻案），事件序列在 `report["transfer_staged"]` 里可审计。自行导入过、或不想自动迁移时传
`prior_source=None` 或 `staged_transfer=False`。


## 契约

| 字段 | 含义 |
|---|---|
| `request.id` | 建议 id，回填时**必须原样带回**（幂等：重复回填会被忽略） |
| `request.x` | 条件（含分类/整数，已是合法配置） |
| `request.fidelity` | 保真档位；`cost` 应与之对应（成本预算按它记账） |
| `Measurement.y` | 目标值（多目标时给主目标，其它目标用 `tell` 的扩展字段） |
| `Measurement.constraints` | 约束名 → 实测值（约束上限在 objective 里声明） |
| `Measurement.failed=True` | 本次无法执行：建议**留在 pending** 以便补测，同时写入执行日志 |

## 失败处理（重要）

- 抛异常或返回 `failed=True`：`run_campaign` 会捕获/跳过该条，**不缩小预算**，
  并把 `note` 写进 `study.dir/execution_log.jsonl`；
- 全批失败时 `tell` 不会推进，下一轮 `next_action` 会重新给出同一批（可补测）；
- 报告里能看到 `attempts` / `failures` / `still_pending`，便于判断是仪器问题还是算法问题。

## 成本记账

成本预算（`Study.create(cost_budget=...)`）是**硬上限**：`run_campaign` 报告
`cost_spent` / `cost_budget`；当剩余成本不足以完成一次全保真确认时，study 会停止
（`stop_category="cost_exhausted"`）而不是空转——这是为无人值守循环专门修的边界。

## 一页式汇报：`condopt digest`

跑完（或跑到一半）用 `condopt digest --run <run_dir>` 生成单页摘要，包含：
预算与停止原因、执行日志与失败构成、最优/规格判定、推荐判定、多目标的可交付 HV 与逐批进展、
下一步行动与建议。长跑之后先看这一页，再决定是否深挖细节。

## 交付自查（报告里的打勾清单）

`Study.delivery_checklist()`（`condopt report` / `digest` 里显示为“交付自查:”）逐条核对
交付必须带的东西：多保真是否有全保真观测、约束场景是否有**可行的**最终保真观测、
多目标是否有前沿与逐批进展、单目标是否已复孔确认。缺哪条就标 `[缺 ]`——
交付前先用这份清单自查，而不是事后补证据。

## 失败原因分类

每次失败都会在执行日志里带上 `failure_kind`：`timeout` / `exception:<Type>` /
`instrument_error` / `other` / `abandoned`（重试耗尽被放弃）。
`run_campaign` 报告里有 `failure_kinds` 计数，`condopt.executor.failure_summary(run_dir)`
可以事后汇总整份日志——用来判断“是仪器问题还是算法问题”。

## 重试上限：不要让坏条件永远占着队列

失败的条件默认留在 pending 里等待补测；但无条件重试会一直占着队列（也让无人值守循环
更容易触发“连续空批”保护）。`run_campaign(max_retries=N)`（默认 2）给出上限：

- 同一个建议失败超过 N 次 → 通过 `Study.cancel_pending()` 放弃它，
  并在执行日志标 `abandoned: true`、在 `decisions.jsonl` 记 `cancel_pending`；
- 报告里的 `cancelled_after_retries` 说明因此放弃了几条；
- 想保留“无限补测”行为就设 `max_retries=-1`。

## 断点续跑与故障保护

- **断点续跑**：pending 保存在 study 目录里，直接对同一个 study 再跑一次
  `run_campaign` 即可——上一轮失败/未测完的条件会被优先补测（报告里的
  `resumed_pending` 会告诉你这次接手了多少条）；
- **故障保护**：`max_empty_batches`（默认 3）表示“连续多少批完全没有可用结果就停”，
  停止原因写进 `stopped_reason`（如“连续 3 批没有可用的实测结果（执行器故障？）”），
  避免仪器离线时无人值守循环空转到 `max_steps`。

## 多目标

`y` 装主目标，其余目标放进 `objectives`：

```python
return Measurement(
    id=request.id, y=yield_value, objectives={"cost": cost_value},
    constraints={"T_max": t}, cost=request.fidelity,
)
```

约束仍写在 `constraints` 里；多目标 + 约束时 router 会选 `cqehvi`/`mfparego`，
`run_campaign` 报告里的 `n_observations` 与 Pareto/可行性摘要都由 study 计算。
完整示例：`py -3 examples/multiobjective_executor.py`。

### 进展曲线

多目标 study 每批结束会把
`{round, n_front, hypervolume, n_feasible}`（有约束时再加 `hypervolume_feasible` 与 `feasible_fraction`）追加到
`study.dir/progress.jsonl`；`run_campaign` 报告里也有 `final_hypervolume` /
`final_n_front` / `progress_points`。`condopt report` / `digest` 会自动带出这条曲线，例如：

```
Pareto 前沿：4 个非支配点，HV=143，参考点=(71.9213, -15.0221)
  进展: 6 个记录点，HV 61.15410941032958 → 143.0257555027851（提升 4 次）
```

**约束下要看 `hypervolume_feasible`**：原始前沿可能包含不可行点，raw HV 会高估交付价值。
实测（约束 + 2 目标、23 次观测）：raw HV 143.0 vs feasible HV 80.9（可行率 78%）。

## 并行批与超时

```python
report = run_campaign(study, instrument, max_workers=4, timeout=120)
```

- `max_workers > 1`：同批条件并行测量（**执行器需线程安全**）；`study` 本身仍是串行调用，
  回填顺序不变，因此结果可复现；
- `timeout`（秒，逐条）：卡住的执行转成 `failed=True` + `note="timeout after …s"`，
  仍留在 pending 可补测，而不是拖死整个循环；
- 执行日志每条都带 `batch_elapsed`，便于判断仪器吞吐是否成为瓶颈。

## 停止不等于用尽预算

`stopped_reason` 可能不是“预算用尽”：默认停滞判定（连续 3 轮无显著改进）会提前停止。
示例 `--budget 60 --cost-budget 25` 实际上在 18 条观测后停下，原因写得很清楚：
`最近 3 轮无显著改进（49.298 → 49.298）`。想继续就调用
`study.retune_for_exploration()` 后再跑一轮 `run_campaign`（实测在噪声下优于直接停）。

## 长跑演练

```powershell
py -3 scripts/adapter_long_run.py --budget 120 --batch 6 --workers 4 --patience 100
```

实测 120 条观测、137 次尝试（17 次失败全部补测）、0 重复 id、run 目录 102 KB；
默认停滞语义下会在 69/120 停止——想用满预算就把 `patience` 调大（见
`research/ADAPTER_LONG_RUN.md` 的三次对照）。

## 长跑演练（多保真 + 多目标）

```powershell
py -3 scripts/adapter_long_run.py --problem mf_moo_constrained --budget 40 --batch 4 --workers 4 --patience 100
```

报告会同时给出 `feasible_hypervolume_curve`、`final_feasible_fraction` 与
`n_feasible_full_fidelity`（交付证据是否落在可行域）。

## 示例与测试

- 示例：`py -3 examples/executor_adapter.py`（约束 + 多保真 + 每 7 次失败一次）
- 测试：`tests/test_executor.py`（正常收敛、全失败保持 pending 并写日志、示例可跑）
