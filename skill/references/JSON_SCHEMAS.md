# JSON 格式与校验

condopt 的输入输出都是 JSON，方便 Agent、UI、机器人调度直接消费。

## 1. `space.json` —— 搜索空间

```json
{"variables": [
  {"name": "T", "kind": "continuous", "low": 40, "high": 120, "unit": "degC",
   "description": "reaction temperature"},
  {"name": "conc", "kind": "log", "low": 0.02, "high": 1.0, "unit": "M"},
  {"name": "solvent", "kind": "categorical", "choices": ["MeCN", "EtOH", "water"]},
  {"name": "equiv", "kind": "integer", "low": 1, "high": 4},
  {"name": "inert", "kind": "bool"}
]}
```

字段：

| 字段 | 必填 | 说明 |
|---|---|---|
| `name` | 是 | 唯一变量名；建议英文短名 |
| `kind` | 是 | `continuous / integer / log / categorical / ordinal / bool` |
| `low`, `high` | 数值型必填 | 连续/整数/对数变量范围；`log` 要求 `low > 0`；`integer` 必须是整数 |
| `choices` | 分类必填 | 至少 2 个、互不相同；`bool` 可省略（默认 `["False","True"]`） |
| `unit` | 否 | 物理单位，强烈建议填 |
| `description` | 否 | 中文备注 |

校验：`SearchSpace.from_json(path)` 会拒绝重复变量名、非法范围、单类别分类等。

## 2. `context.json` —— router 输入

```json
{"budget_total": 24, "budget_remaining": 24, "batch_size": 4, "noise": "medium",
 "smoothness": "smooth", "n_objectives": 1, "n_constraints": 0, "multi_fidelity": false,
 "compute": "medium", "domain": "chemistry", "n_observations": 0, "notes": "自由备注"}
```

- `n_dims` 及各类变量计数可以不填：给 `space.json` 时由 `--space` 自动推导；
  只给 `context.json` 时必须填 `n_dims`。
- `budget_remaining` 省略时等于 `budget_total - n_observations`。
- 枚举取值：`noise ∈ {none,low,medium,high}`、`smoothness ∈ {smooth,mixed,rough}`、
  `compute ∈ {low,medium,high}`、`domain ∈ {chemistry,materials,process,generic}`。

## 3. `objective.json` —— 目标定义

```json
{"name": "yield", "direction": "max", "objectives": ["yield", "purity"], "directions": ["max", "max"], "constraints": ["T_max"]}
```

- `direction`：`max` 或 `min`；`condopt` 内部统一转成“越大越好”。`name` 是主目标（报告用）。
- `objectives` 多于 1 个 = 多目标声明；`directions` 给每个目标单独方向，省略时全部等于 `direction`。
  多目标会路由到 `parego` / `qehvi` / `qnehvi`，报告 Pareto 前沿 + 2D HV，而不是单一最优。
- **约束**：`constraints` 列约束名，`constraint_limits` 给阈值与方向，例如
  `{"constraints": ["T_max"], "constraint_limits": {"T_max": {"op": "<=", "value": 110.0}}}`；
  单目标有约束会路由到 `constrained`（先找可行点，再做带可行性的 EI）；
  **2 目标 + 约束**会路由到 `cqehvi`（可行域内 EHVI），此时每个回填观测都必须带 `constraints`。
- **多目标可复现口径**：`ref_point`（原始尺度，按目标顺序）固定 HV 参考点（qehvi/cqehvi 必需，
  否则 HV 只能在同一 run 内比较）；`objective_bounds` 给 ParEGO 的固定归一化窗口
  （`{"yield": {"low": 60, "high": 95}}`）。
- 命令行简写：`--objective yield:max`（等价于 `{"name":"yield","direction":"max"}`）；也可传 JSON 文件路径。
- Windows 注意：PowerShell 5.1 会吞掉命令行里的内联 JSON 引号，优先用简写或文件路径；
  读取文件时同时兼容带/不带 BOM 的 UTF-8。

### 3.1 `answers.json`（skill 问卷 → prepare）

`condopt prepare --answers answers.json --out workspace` 接受一份合并答案：

```json
{
  "objective": {"name": "yield", "objectives": ["yield", "cost"], "directions": ["max", "min"]},
  "variables": [{"name": "T", "kind": "continuous", "low": 60, "high": 115}],
  "budget": 24, "batch": 4, "noise": "medium", "smoothness": "smooth",
  "domain": "chemistry", "prior": 0, "constraints": 0, "notes": "..."
}
```

别名：`budget→budget_total`、`batch→batch_size`、`prior→n_observations`、`constraints→n_constraints`、
`fidelity→multi_fidelity`；`n_objectives` 从 `objective.objectives` 推导，不一致会报错。
生成 `space.json / context.json / objective.json` 并返回 router 推荐。

## 4. `results.json` —— 回收实测结果

两种写法都支持：

```json
[{"id": "r1-1", "y": 78.3, "source": "manual"},
 {"id": "r1-2", "y": 64.1, "constraints": {"T_max": 12.0}}]
```

或带完整条件（不依赖 id）：

```json
[{"x": {"T": 80.0, "conc": 0.2, "solvent": "MeCN", "equiv": 2, "inert": true}, "y": 78.3}]
```

多目标时给目标向量（不是 y）：

```json
[{"id": "r1-1", "objectives": {"yield": 78.3, "cost": 11.2}}]
```

**复孔/噪声**：`std`（或 `objective_std`）+ `k`（或 `n_replicates`）用于逐点估计 GP 噪声：

```json
[{"id": "r1-1", "y": 78.3, "std": 0.6, "k": 3}]
```

**多保真成本**：结果可带 `cost`（资源成本，缺省按 `cost=fidelity` 记账）；study 成本预算用
`condopt init --cost-budget 20`，之后 `condopt report` 的 summary 会给出 `cost_spent/remaining_cost`，
`mfbo` 会在预算不足时自动降档/裁剪，预算耗尽会明确报错。

**多保真**：`fidelity`（0–1）记录本次评价的保真度；不填则沿用建议批次里的 fidelity：

```json
[{"id": "r1-2", "y": 75.1, "fidelity": 0.5}]
```

**孔板/批次**：`ask` 会给每条建议自动打 `plate`（默认 `P1`、`P2`…）+ `well`（A1…H12）+ `batch`，
`tell` 时自动写进观测 `meta`。实际执行换板/重排孔位时在结果里覆写：

```json
[{"id": "r1-1", "y": 78.3, "plate": "P2", "well": "C3", "batch": 7}]
```

不写则沿用建议里的 `plate/well/batch`。板规格默认 8×12=96 孔，可用 `meta.plate_rows/plate_columns`
调整；同一块板孔位用尽会**报错**（提示换板），不会静默生成越界孔位。批次效应由
`report` / `summary()["plates"]` 给出（逐板 n/均值/极差 + 单因素 ANOVA 检验），只在板间差异
显著 **且** 相对均值 ≥2% 时警告；警告是诊断信息，不会自动校正数据。

| 字段 | 说明 |
|---|---|
| `id` | 对应 `suggestions.csv` 的建议 id，可省略（省略时必须给 `x`） |
| `x` | 完整条件字典（必须通过空间校验） |
| `y` | 主目标值（单目标必填） |
| `objectives` | 多目标/附加指标字典 |
| `constraints` | 约束测量值 |
| `fidelity` | 保真度，默认 1.0 |
| `cost` | 本次实验成本 |
| `plate` / `well` / `batch` | 实际执行位置（覆写建议里的值） |
| `source` | `manual / robot / simulation / benchmark` |
| `meta` | 任意附加信息（操作者、仪器等；`plate/well/batch` 也存这里） |

也接受 JSONL（每行一个对象）。**必须来自真实测量。**

历史数据（本轮 study 之前做过的实验）用 `condopt import --run R --results prior.json` 导入：
计入预算、可暖启动，但不记入本轮 round。文件读取兼容带/不带 BOM 的 UTF-8。

## 5. `route` 输出（`--json`）

```json
{
  "algorithm": "bayes",
  "params": {"kernel": "matern52", "alpha": 0.0001, "n_initial": 7, "acquisition": "ei", "...": "..."},
  "confidence": "high",
  "rationale": ["维度 6 ≤ 12，GP 代理可靠", "..."],
  "warnings": [],
  "alternatives": [{"algorithm": "sobol", "score": 48.0, "why": "..."}],
  "candidate_scores": {"bayes": 80.0, "sobol": 48.0, "lhs": 30.0, "random": 15.0, "grid": 0.0},
  "design": {"algorithm": "sobol", "n": 7, "main": "bayes", "remaining": 17},
  "plan": [{"phase": "initial_design", "algorithm": "sobol", "n": 7},
           {"phase": "main_loop", "algorithm": "bayes", "n": 17}],
  "context": {"...": "..."}
}
```

## 6. Study 目录

```
run/
  problem.json      space + objective + seed + budget + meta
  config.json       algorithm + params + context + batch_size + auto_route + patience
  dataset.jsonl     已测观测（append-only；`meta.plate/plate_well/batch` 记执行位置）
  pending.json      当前待测建议（含 id / batch / plate / well / algorithm / x）
  rounds.json       每轮记录（n / best / algorithm），用于停滞检测
  decisions.jsonl   initial_route / auto_reroute / override，含 reason
  opt_state.json    序列算法计数器（sobol 已生成数 / grid 索引）
  suggestions.csv   最新待测建议表（id, batch, algorithm, fidelity, plate, well, 变量列）
```

可复现性：`problem.json + dataset.jsonl` 足以重建整个优化历史；算法内部数据不持久化，
每次 `ask` 都从数据集重建，避免状态文件与数据不一致。

## spec（目标窗口/规格，可选）

`answers.json` 里用 `spec` 替代 `objective`：

```json
{
  "variables": [...],
  "spec": {
    "goals": [
      {"metric": "yield", "mode": "target", "target": 80, "tolerance": 2, "scale": 5},
      {"metric": "purity", "mode": "range", "low": 98, "high": 100},
      {"metric": "cost", "mode": "min", "weight": 0.5}
    ],
    "constraints": {"temp": {"op": "<=", "value": 110}}
  }
}
```

- `mode`: `max` / `min` / `target` / `range`；
- `target` 无 `tolerance` 时是软偏好（不参与 in-spec 判定）；
- `prepare` 会写 `spec.json`；`init --spec spec.json` 自动生成 `spec_score` 单目标。

`tell` 可以用原始指标回填：

```json
[{"id": "r1-1", "objectives": {"yield": 80.2, "purity": 99.1}, "constraints": {"temp": 105}}]
```

`recommend --run <run> --json` 返回：

```json
{
  "run_dir": "...", "condition": {...}, "score": -0.12, "best_score": -0.10,
  "objectives": {...}, "constraints": {...},
  "spec": {"in_spec": false, "goals": {...}}, "spec_raw": {"in_spec": true, "goals": {...}},
  "confirmation": {"n": 3, "mean": -0.117, "std": 0.064},
  "should_stop": true, "stop_reason": "达到目标分数 ...",
  "selection_basis": "plate_adjusted | pareto_knee | pareto_knee_plate_adjusted | pareto_spec | pareto_spec_plate_adjusted | raw",
  "raw_best": {"condition": {...}, "score": 21.0, "plate": "P2"},
  "plate_effects": {"detected": true, "effects": {"P1": -3.9, "P2": 3.9}}
}
```

`selection_basis=plate_adjusted` 表示检测到批次效应后按随机效应校正分数选点；`raw_best` 保留原始最高分供审计。

- `score`：报告用分数；有确认复孔时用**复孔均值**，`best_score` 是原始最优观测值；
- `spec`：`null`（没建 spec）或逐项符合性；`confirmation`：`null` 或复孔 n/mean/std/scores；
- `should_stop`/`stop_reason`：四种停止条件（预算/目标分数/平台期/最大轮次）的判定与原因。

## 7. `step` 输出（自动循环）

`step --run <run> --json` 把 pending、`should_stop`、`recommend`、`confirm` 合并成一个
机器可读的循环决策；skill 每轮 `tell` 之后调用它，按 `action` 行动：

```json
{
  "action": "confirm",
  "reason": "收尾前确认最优条件（3 次复孔）：达到目标分数 8（当前 9）",
  "should_stop": true,
  "stop_reason": "达到目标分数 8（当前 9）",
  "used": 4,
  "remaining": 6,
  "pending": [],
  "confirm_n": 3,
  "ask_n": null,
  "recommendation": {"...": "同 recommend 输出"},
  "batch_effect": {"detected": false, "message": "未检测到显著批次/孔板效应（...）"},
  "next_command": "condopt confirm --run ... --n 3",
  "diagnostics": {
    "batch_effect": {"...": "同 plate_summary"},
    "plate_effect": {"detected": false, "effects": {}},
    "plate_effect_detected": false,
    "selection_basis": "raw",
  "fidelity_filter": "full",
  "n_low_fidelity_excluded": 1,
    "selection_reason": "按原始分数选最优条件",
    "knee_weights": {"yield": 2.0},
    "constraints_supported": true,
    "multi_fidelity_context": false,
    "multi_fidelity_supported": true,
    "fidelity_levels": ["0.5", "1"],
    "fidelity_paired_n": 2,
    "fidelity_normalized_bias": 0.86,
    "n_full_fidelity": 0,
    "n_descriptor_vars": 1,
    "cost_spent": 2.0,
    "remaining_cost": 8.0,
    "forked_from": null,
    "fork_spec_mode": null,
    "spec_in_spec_count": 3,
    "spec_in_spec_adjusted": 4,
    "spec_check": {"raw_in_spec": true, "adjusted_in_spec": false,
                   "changed_by_plate": true},
    "spec_review": {"suggested_spec": {"goals": [{"metric": "yield", "target": 60.0}]},
                    "metrics": {"yield": {"observed": [50.0, 60.0],
                                           "target_within_observed": false,
                                           "suggested_target": 60.0}}},
    "recommendation_spec": {"in_spec": true},
    "spec_calibration_ready": true,
    "spec_calibration": {"report": {"metrics": {"yield": {"n": 5, "suggested_scale": 1.2}}},
                          "suggested_goals": [{"metric": "yield", "mode": "target", "tolerance": 2.4}]}
  },
  "suggested_actions": ["replicate_on_new_plate", "calibrate_spec", "calibrate_by_plate"],
  "suggested_action_reasons": {"calibrate_spec": "已有 5 条观测，可用历史校准"}
}
```

| `action` | 何时出现 | 下一步 |
|---|---|---|
| `tell` | `pending` 非空 | 回填实测结果后重新 `step` |
| `confirm` | 目标/平台期已到或预算只剩 ≤ `confirm_n` 次，且最优条件还没确认过 | `confirm --n <confirm_n>` → `tell` → 再 `step` |
| `ask` | 还能继续、且无上述情况 | `ask --n <batch> [--plate P]` |
| `stop` | 预算耗尽/目标达到/平台期/最大轮次，且无需再确认 | `recommend --json` 输出结论并收尾 |

- `meta.auto_confirm=false` 关闭“收尾前确认”这一步（直接 `stop`）；`meta.confirm_n` 设复孔数（默认 3）。
- `batch_effect` 是每轮都会返回的批次效应摘要；`detected=true` 时要向用户指出板间差异。
- `diagnostics` 汇总批次/板效应、推荐依据、spec 是否可校准；`suggested_actions` 给出可选动作
  （如换板复孔、按板效应校正、`calibrate_spec`），**只是建议，不自动执行**。
- `next_command` 只是建议命令，参数仍以实际 study 为准；`action=tell` 时结果文件由用户提供。


