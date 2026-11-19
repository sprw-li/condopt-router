# 执行器接入（把 oracle 换成真实设备的最小改动）

优化内核与实验设备之间只有三个契约点：**取点（pending）→ 执行 → 回填（tell）**。
本文给出最小改动清单，任何机器人/人工流程都可以按此接入。

## 1. 契约文件

| 文件 | 方向 | 内容 |
|---|---|---|
| `pending.json` | 内核 → 执行器 | 待执行的建议（id / x / plate / well / fidelity） |
| `results_template.csv` | 内核 → 执行器 | 同上，带空的测量列，直接填 |
| 结果文件（CSV/JSON） | 执行器 → 内核 | `id` + `y` 或 `obj.*` / `con.*` 列 |

命令速查：

```powershell
py -3 scripts/run_condopt.py ask --run <dir> --n 4          # 取点
py -3 scripts/run_condopt.py pending --run <dir> --json     # 执行器轮询
py -3 scripts/run_condopt.py template --run <dir>           # 导出待填表
py -3 scripts/run_condopt.py tell --run <dir> --results results.csv --json
py -3 scripts/run_condopt.py doctor --run <dir>             # 回填后自检
```

## 2. 最小执行器（Python 模板）

```python
import json, time
from pathlib import Path
from condopt import Study

RUN = Path("runs/my_campaign")
study = Study.load(RUN)

while True:
    pending = study.pending                      # 待执行建议
    if not pending:
        action = study.next_action()
        if action["action"] == "stop":
            break
        if action["action"] == "ask":
            study.ask(action.get("ask_n", 4))
            continue
        if action["action"] == "confirm":
            study.confirm(action.get("confirm_n", 3))
            continue
        continue

    results = []
    for item in pending:
        try:
            value = robot.run(item["x"])             # ← 换成真实执行器
            results.append({"id": item["id"], "y": value})
        except Exception as exc:                     # 单孔失败：记录但不阻断
            print(f"well {item.get('well')} failed: {exc}")
    if results:
        study.tell(results)
    time.sleep(5)
```

关键点：

- **失败孔位直接跳过**：`tell` 只回填成功的 id，其余留在 pending，稍后补测重跑即可；
- **不要自己删除 pending**：`tell` 负责匹配与清理；
- **幂等**：同一个 id 重复提交会被忽略，不会产生重复观测。

## 3. 部分回填的语义

`tell` 现在返回 `still_pending`（剩余未回填 id），并在仍有 pending 时写入
`execution_partial` 审计记录：

```json
{"n": 1, "still_pending": ["r1-2", "r1-3"], "remaining": 5}
```

## 4. 测量元数据覆盖

执行器可以覆盖/补充执行信息：

| 字段 | 含义 |
|---|---|
| `plate` / `well` | 实际执行的板号与孔位（可不同于建议） |
| `fidelity` | 多保真档位（若算法支持） |
| `n_replicates` | 复孔数 |
| `objective_std` | 测量标准差（噪声感知算法会使用） |
| `cost` | 本次执行成本（成本预算会用） |

## 5. 无人值守前的自检

```powershell
py -3 scripts/run_condopt.py doctor --run <dir> --json   # ID 唯一 / pending 与 dataset 不重叠 / JSON 可解析
py -3 scripts/run_condopt.py transfer-audit --run <new> --from-run <old>   # 若用了迁移
```

## 6. 常见故障与处理

| 现象 | 处理 |
|---|---|
| 进程在批下发后崩溃 | 重跑同一命令：pending 仍在，先回填再继续（已测试覆盖） |
| 同一 id 重复提交 | 自动幂等跳过，不重复计数 |
| 结果缺测量列 | 该 id 保持 pending，`tell` 输出 `still_pending` 提示 |
| 磁盘上文件损坏 | `doctor` 非零退出并列出具体文件 |
| 成本超预算 | `next_action` 返回 stop；`ask` 返回空建议 |
## 7. 多保真 + 成本回写

多保真算法（`mfbo` / `momfbo`）的 pending 项带 `fidelity` 字段。执行器按档位选方法，
并把**实际成本**回写：

```python
fidelity = float(item.get("fidelity", 1.0) or 1.0)
results.append({
    "id": item["id"],
    "fidelity": fidelity,
    "y": measure(item["x"], fidelity),
    "cost": fidelity_cost(fidelity),      # 不填则默认 cost ≈ fidelity
})
```

成本预算是**硬上限**：

- `ask` 会按 `cost ≈ fidelity` 截断尾部的提议（审计 `cost_budget_trim`）；
- `confirm` 也会按剩余成本缩减（不足一次全保真时返回空）；
- 端到端示例：`py -3 examples/multifidelity_executor.py`（成本 12.0 时实测恰好花满 12.0）。

参考 `best()` 与 `recommend()` 默认**只用全保真观测**做交付建议；
`fidelity_summary()` 可检查各档位偏差。
### 可行率过低时的提醒

`step` 会报告 `observed_feasible_fraction`；当约束存在、观测数 ≥10 且可行率 <5% 时，
会给出 `encode_limits_as_bounds` 建议——把已知硬限制写进变量范围或固定因子，
而不是继续用约束采集硬找（实测 1/5 vs 5/5，见 `research/HARD_FEASIBILITY.md`）。

### 手写循环也要记录进展

多目标交付需要“逐批进展记录”。适配器会自动写；手写 ask/tell 循环请在每轮回填后调用
`study.record_progress()`（写入 `progress.jsonl`，内容含前沿点数、HV、可行比例），
否则 `delivery_checklist()` 会提示缺这条证据。

### 手写循环的步数上限要随预算缩放

复孔/探索批次是两次动作（ask + tell）。把循环写成固定的 `range(60)` 会在
120 次预算的战役里把它截断在 ~90 次观测处，进而错过收尾确认 —— 这是本项目实测踩过
的一次**基准脚本缺陷**（`benchmark_mid_budget.py`，Round 255 修复）。上限至少取
`4 × budget`，或直接用适配器 `run_campaign`（它按预算/成本自己判断何时停）。

### 适配器（推荐入口）

不想自己写 ask/tell 循环时，实现 `condopt.executor.Executor.measure` 后用
`run_campaign(study, executor)` 驱动即可；失败孔位、成本记账、执行日志都由它处理。
详见 [`EXECUTOR_ADAPTER.md`](EXECUTOR_ADAPTER.md)。

### 多保真 + 约束的注意点

当约束存在且尚无可行的全保真观测时，`confirm` 会确认“已观测的最优可行条件”
（`confirmation_basis="best_feasible_observed"`），避免昂贵复孔再次落在不可行区；
实测把"至少一条可行全保真"的比例从 0.40 提升到 1.00。

`step` 会报告 `n_full_fidelity` 与 `n_feasible_full_fidelity`；若全保真观测存在但
**没有一条满足约束**，仍会给出 `collect_feasible_full_fidelity` 建议作为兜底：
先对当前可行的低保真最优点做全保真确认，再谈交付。
示例：`py -3 examples/constrained_mf_executor.py`（成本硬上限 + 保真档位 + 约束回写）。

