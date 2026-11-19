# 决策层：规格、停止、确认、推荐

> 条件优化不是“找一个最大值”，而是在目标窗口、硬约束、噪声、预算和放大要求下，
> 用最少实验做出可执行、可复现、可转移的决策。这个目录把“决策层”做成 Study 的一等公民。

## 1. Spec：目标窗口 + 约束

`condopt.spec.Spec` 用「目标」描述真实需求，不再强迫用户把一切压成“越大越好”：

```python
from condopt import Goal, Spec

spec = Spec(
    goals=(
        Goal("E_MPa", mode="target", target=4.5, tolerance=0.4, scale=2.0),
        Goal("UTS_MPa", mode="target", target=6.8, tolerance=0.6, scale=2.5),
        Goal("elongation_pct", mode="target", target=95.0, tolerance=7.0, scale=25.0),
    ),
    constraints={
        "cure_time_s": {"op": "<=", "value": 10.0},
        "delta_T_C": {"op": "<=", "value": 70.0},
    },
)
```

支持的 goal 模式：

| mode | 含义 | 得分 |
|---|---|---|
| `max` | 越大越好 | 归一化值 |
| `min` | 越小越好 | 负归一化值 |
| `target` | 命中目标 ± `tolerance` | `-|v-target|/scale`；无 tolerance 视为软偏好 |
| `range` | 落在 `[low, high]` 内 | 区间内 0，区间外负距离 |

`Spec.scalar_objective()` 生成单目标 `Objective`（名字 `spec_score`，方向 max，约束进入
`constraint_limits`），因此现有算法/路由无需改动；`Spec.score(values)` 是加权和，
`Spec.compliance(values)` 给出每个 goal/constraint 是否 in-spec。

## 2. Study 集成

`Study.create(..., spec=...)`：

- 自动构造 `objective = spec.scalar_objective()`；
- `tell`/`import` 时可用原始指标：
  `{"objectives": {"E_MPa": 4.4, ...}, "constraints": {"cure_time_s": 6.8, ...}}`，
  Study 自动计算 `y = spec.score(...)`，原始指标保留在 `objectives` 里；
- `summary()["spec"]` 给出 in-spec 数量、最优观测的逐项符合性；
- `recommend()` 给出可执行条件、规格符合性、确认复孔统计、停止判断；
- `should_stop()`：预算/目标分数/平台期/最大轮次四种停止条件；**目标分数判定优先用确认复孔均值**，
避免一次幸运测量触发提前停止（确认均值未达目标则继续优化）；
- `confirm(n)`：对**推荐条件**做 n 次确认复孔（`purpose="confirmation"`；检测到批次效应时
  推荐条件可能不是 raw best，复孔应跟随推荐条件），
  结果进入推荐里的 mean/std；**推荐分数优先使用确认均值，避免“幸运噪声”被当结论**；
- `next_action()`（CLI `condopt step`）：把 pending / `should_stop` / `recommend` / `confirm` 合成
  一个 `action ∈ {tell, confirm, ask, stop}` 决策（含 `next_command`、`batch_effect`），
  是 skill 自动循环的唯一入口；收尾前会自动要求确认最优条件（`meta.auto_confirm=false` 可关）。

## 3. CLI

```powershell
# 1) 规格进 prepare（answers.json 里放 "spec": {...}）
py -3 scripts/run_condopt.py prepare --answers answers.json --out runs/prepared --json

# 2) 用 spec 建 study（objective 自动由 spec 生成）
py -3 scripts/run_condopt.py init --run runs/spec_demo --space runs/prepared/space.json --spec runs/prepared/spec.json --budget 32 --batch 4 --json

# 3) 自动循环：每轮 tell 后问引擎下一步（tell/confirm/ask/stop）
py -3 scripts/run_condopt.py step --run runs/spec_demo --json
py -3 scripts/run_condopt.py report --run runs/spec_demo

# 4) 给推荐 + 做确认复孔（step 返回 action=confirm 时）
py -3 scripts/run_condopt.py recommend --run runs/spec_demo --json
py -3 scripts/run_condopt.py confirm --run runs/spec_demo --n 3 --json
```

## 4. 演示

```powershell
py -3 examples/spec_decision_demo.py
```

输出：`examples/spec_decision_demo_results.json`，含算法、spec、in-spec、推荐条件、
确认复孔 mean/std、停止原因；run 目录有 `dashboard.html`。

注意：演示 oracle 是合成的（脚本内注明），目的是展示决策层，不是化学结论。

## 5. 边界

- `spec` 仍是“加权标量化 + 约束”；真正的多目标 Pareto 由 `parego`/`qehvi`/`qnehvi` 提供，
  **2 目标 + 硬约束**由 `cqehvi` 提供（可行域内 EHVI）；`spec` 与多目标前沿暂不自动合并
  （需要用户用 `objectives` 而不是 `spec` 建 study）；
- `target` 无 tolerance 时只作为软偏好，不参与 in-spec 判定；
- 确认复孔只在 `confirm` / `step(action=confirm)` 后发生；`step` 只给建议，最终是否停/确认
  由调用方决定（用户叫停优先），引擎不替人做业务决策；
- 批次效应只做**诊断与警告**，不自动用板均值做校正（校正会改变数据口径，需要用户确认）；
- 规格得分依赖权重与 scale，权重需要领域知识或后续校准。

## 6. 下一步

1. ✅ 多目标 + 约束联合采集：`cqehvi`（可行域内 EHVI + 两阶段 PoF）已实现，
   验收见 `docs/ALGORITHM_CATALOG.md` §5.1 与 `tests/test_cqehvi.py`；
2. ✅ 批量/孔板效应：`ask` 给每条建议打 `plate/well/batch`（`--plate` 可续测同一块板并避让已用孔位），
   `tell` 写入观测 `meta` 且允许实际执行时覆写；`summary()["plates"]` / `report` 给出逐板 n/均值/极差
   与单因素 ANOVA **批次效应警告**（板间差异显著且相对 ≥2% 时提示同批比较、把 plate 当随机效应/特征），
   验收见 `tests/test_plates.py`；
3. ✅ 把 `recommend`、`confirm`、`should_stop` 接进 skill 自动循环：`Study.next_action()` / `condopt step`
   输出 `tell/confirm/ask/stop` 决策（含 `next_command` 与 `batch_effect`），SKILL.md 第 5 步与
   `references/JSON_SCHEMAS.md` §7 同步，验收见 `tests/test_loop.py`；
4. 用真实/半真实战役校准 spec 的 tolerance/scale。

## 7. 批次/孔板校正推荐（v0.13）

`recommend()` 默认先诊断批次效应（单因素 ANOVA + 相对板间差 ≥2%），检测到后：

- 用 one-way random-effects 的 BLUP 估计各板位移，得到每个观测的 **plate-adjusted score**；
- 按调整后分数选推荐条件，而不是 raw best；
- 返回值同时给 `selection_basis`（`raw` / `plate_adjusted`）、`raw_best`、`plate_effects`，
  审计时能看出“为什么没选那个最高分”。

v0.15 起：检测到批次效应时，**每个 spec 指标/约束也按各自的板效应校正**后再算 in-spec；
返回值里 `spec` 是校正后的符合性，`spec_raw` 保留原始符合性，避免“板间偏移导致 in-spec 误判”。

标量推荐在**各指标板效应可用**时，直接用校正后的指标值重算 spec_score（weighted_sum 或
worst_case 都适用），而不是只平移主目标分数；无逐指标效应时退回主目标平移。

边界：效应只是估计且不自动改写数据；板数/每板样本不足时如实返回“样本不足”，不会硬校正。

## 8. 规格校准（v0.14）

初始 `tolerance`/`scale` 常常是拍的。`Spec.calibrate(records)` 用历史测量值给出建议：

- `scale`：1.4826×MAD（稳健尺度；不足时退回范围/std）；
- `target` 的 `tolerance`：默认 `2 × scale`（约 95% 观测带）；
- `range` 的上下界、goal 模式、权重保持不动（不猜边界）；
- 样本 < `min_samples`（默认 3）时保持原值，并在 report 里标低置信。

CLI：

```powershell
py -3 scripts/run_condopt.py calibrate-spec --run <run> --out runs/<run>/spec_calibrated.json --json
```

输出新 spec + 每个 metric 的样本数/建议 scale/tolerance/置信度；**不自动替换 study 的 spec**。

有板/批次分组时，用 `calibrate-spec --by-plate`（或 `Spec.calibrate_by_group`）：
以各组内稳健尺度的中位数作为 pooled scale，板间系统偏移不会把 tolerance 撑大；
报告同时给 `global_scale` / `global_tolerance` 供对比，看板显示“scale x（全局 y）”。

若 target 本身不可达，用 `suggest-spec`：

```powershell
py -3 scripts/run_condopt.py suggest-spec --run <run> --out spec_suggested.json --json
```

它会先校准 scale/tolerance，再把落在观测范围外的 target 移到最近观测边界、把无点的 range
改成观测范围；返回 `changed` 列表，**不自动替换 study**。

v0.23 起可在下一次 `prepare` 时直接复用历史校准：
`answers.json` 加 `"spec_calibrate_from": "<old-run>"`（可选 `spec_calibrate_min_samples` /
`spec_calibrate_tolerance_sigmas` / `spec_calibrate_by_plate`），prepare 会写出**校准后的 spec.json** 和
`spec_calibration.json` 报告；原 `spec` 作为无样本时的回退。

## 9. report 的推荐块（v0.17）

`calibrate-spec` 会把报告写入 `run/spec_calibration.json`，dashboard 增加“规格校准”卡片；
`condopt digest` 生成单页摘要（report + 推荐判定 + 下一步行动 + 规格校准），`--format html` 可输出自包含 HTML；
适合直接发给同事；`condopt step` 把下一步行动写入 `run/step.json`，dashboard 增加“下一步行动”卡片；
CLI decision 会把结果写入 `run/decision.json`，dashboard 增加“推荐判定”卡片（selection_basis、
板效应、规格符合性、explain 理由链）。

CLI `condopt decision --run <run> --json` 直接输出该判定（轻量，不渲染整份 report）；
加 `--explain` 会附理由链（观测数、推荐依据、板效应、规格判定、确认复孔、停止条件）。

`summary()["decision"]`（无 spec 的多目标也有）给出推荐点的 selection_basis / score /
`selection_reason` 人类可读理由、in_spec / raw_in_spec / plate_effect_detected / spec_check /
knee_weights / forked_from；看板“推荐判定”卡片显示推荐理由。`summary()["spec"]["recommended"]` 复用同一份结果。

`summary()["spec"]["recommended"]` 同步给出推荐点的 selection_basis / score /
in_spec / raw_in_spec，dashboard 不必再单独调 recommend。

`report()` 不再只打印 raw best：它会输出 `推荐条件（selection_basis）`，包含板效应校正后的条件、
确认复孔 mean/std、规格符合性、`raw_best` 对照、以及是否可用 `calibrate-spec`。
这样人/agent 读报告时不会把“原始最高分”误当推荐。

## 9. step 的诊断与建议动作（v0.16）

`next_action()` / `condopt step --json` 除 `action` 外还返回：

- `diagnostics`：`batch_effect`、`plate_effect`、`selection_basis`、`recommendation_spec`、
  `plate_effect_detected`、`spec_calibration_ready`、`knee_weights`、`spec_in_spec_count`、
  `spec_in_spec_adjusted`（板校正后 in-spec 数）、`spec_check`、`multi_fidelity`、
  `multi_fidelity_context` / `multi_fidelity_supported`；上下文要多保真但算法不支持时
  建议 `switch_to_multi_fidelity`；`cost_spent` / `remaining_cost`；成本预算用尽（remaining_cost ≤ 0）时 next_action 直接 stop；
  另有 `fidelity_levels` / `fidelity_paired_n` / `fidelity_normalized_bias`：
  同一条件多次测量时估计低保真偏差，归一化偏差 ≥50% 且配对 ≥2 时建议
  `review_fidelity_channel`；
  多保真预算不足最便宜档位时 ask 返回空建议（不抛异常）；预算 <1 时建议
  `review_cost_budget`；
  手动 pin 的算法若未声明 constraints 能力（如 mfbo），给 `constraints_supported=false` 并建议
  `switch_to_constraint_aware`（override 到 constrained/cqehvi）
  （原始 vs 板校正后的 in-spec 对比）；样本足够时还附 `spec_calibration`
  （建议 scale/tolerance 的报告）；若 ≥5 个观测仍 0 个 in-spec，`suggested_actions` 会给出
  `review_spec`（规格可能不可达或写错）；`suggested_action_reasons` 为每个建议动作给出理由。
  触发 review_spec 时，`spec_review` 直接附 `suggested_spec`（target/range 修正建议；有板时按板内校准）。
- `suggested_actions`：如 `tell`、`replicate_on_new_plate`、`use_plate_adjusted_recommendation`、
  `calibrate_spec`、`ask_next_batch`。

这样 agent 不只知道“下一步 tell/ask/stop”，还知道**为什么**以及有哪些可选动作（换板复孔、
按板效应校正、校准规格）。建议动作只是建议，不自动执行。

## 10.1 论文体系两阶段验收（v0.24）

`examples/paper_campaigns/run_spec_decision.py`：热固性体系先按原始 target 跑（in_spec=0，
触发 review_spec/spec_review），再用 `suggest_spec` 的校准结果重跑，得到 in_spec=true 的推荐，
并完成确认复孔。说明“规格不可达 -> 校准/建议 -> 重跑”的闭环可用。

## 9.1 实测结果导入格式（v0.29）

`condopt tell --results` / `import --results` 支持 JSON / JSONL / **CSV**：

- 固定列：`id`、`y`、`fidelity`、`plate`、`well`、`source`、`cost`；
- 前缀列：`x.<name>`、`obj.<name>`、`con.<name>`；
- 也可用 `objectives` / `constraints` 列承载 JSON 对象；
- 无前缀列在没有 `obj./con.` 时按条件变量解析；若列名等于目标/约束/spec 指标名
（如 `yield`、`temp`），tell/import 会自动归位到目标或约束，不会当成条件变量。


分隔符自动嗅探（逗号 / 分号 / Tab），`.csv` / `.tsv` / `.txt` 均可；
CSV/TSV 让“仪器导出的表格 → Study”少一步转换；`condopt template --run <run>`
为当前 pending 生成 `results_template.csv`（id/plate/well[/fidelity]/x.*/y|obj.*/con.*，多保真 study 自动带 fidelity 列）；
`condopt decisions --run <run> [--action X] [--limit N]` 查看审计日志（router / override / transfer / knee_weights / fork 等）；
`condopt pareto --run <run>` 导出 `pareto_front.csv`（任意目标数，含 x/obj/con）；
`condopt export --run <run>` 反向导出 `dataset_export.csv`（含 plate/well/fidelity/cost/y），
实验室填完直接 `tell --results` 回填。

## 10. 端到端验收（v0.18）

`examples/full_campaign_demo.py` 把整条决策链串起来跑：

- Spec 目标窗口 + 硬约束；
- 合成 oracle 带**板间系统偏差**；
- `next_action()` 驱动 tell → ask/confirm/stop；
- 自动为确认复孔**预留预算**（避免最后一批把预算吃光）；
- 板效应检测 → `selection_basis=plate_adjusted`；
- 确认复孔 → 用均值判定停止；
- 最后输出 report / recommend / `calibrate-spec` 建议。

`step` 新增 `ask_n`：剩余预算不足以跑默认 batch 时，`ask --n ask_n` 而不是溢出报错；
确认复孔预算会被预留。测试：`tests/test_campaign_e2e.py` 跑完整循环并断言出现 confirm、stop、
复孔统计与推荐块。

## 11.0 暖启动实证（v0.32）

等总预算（18 次评价）：先跑 6 次 sobol 源战役，再 `transfer_from(top_k=4)` 把最优 4 点
迁入 bayes 主战役。rosenbrock10 best 从 −997.1 提升到 **−631.7（+37%）**，
sphere2d 从 −0.0021 到 −0.0016。详见 `research/TRANSFER_BENCHMARK.md`。

源质量差时负迁移随 top_k 单调恶化（实测 k=1 −369 → k=8 −1190），因此 `transfer_from` 在 top_k>4/未指定时给出 warning，CLI 建议 1–3、fork 默认 5。
多目标迁移按 `selection="pareto_front"` 优先保留非支配折中点（不足 top_k 再补齐）；
决策顺序：`transfer-audit`（可信度）→ `transfer-impact`（副本预览推荐变化）→ `transfer_from`；
`Study.assess_transfer(source)` / `condopt transfer-audit` 用目标已测数据估计源可信度并
给出 suggested_reliability；`transfer_from(prior_reliability=r)` 可抬高先验噪声（坏源 top_k=4 时 r=0.25 把 −884 改善到 −698）；
`transfer_from` 还会做源可信度自检：领域不一致或源观测 <5 条时给出 warnings；
迁移时会剥离源 `suggestion_id`（保留为 `original_suggestion_id`），
避免与目标 pending id 冲突导致 `tell` 误判为幂等重放、pending 卡住。

## 11. Warm start / 迁移（v0.19）

同一空间、同一目标（含约束）的历史 study 可以作为先验导入：

```powershell
py -3 scripts/run_condopt.py import --run <target> --from-run <source> --top-k 10 --reason "同一底物历史批次"
```

- 严格校验 **search space / objective / constraint limits** 一致，否则拒绝迁移；
- 默认只迁移全保真观测；`top_k` 只保留得分最高的 k 条（防负迁移的粗筛）；
- 导入观测 `source="transfer"`、`meta.transferred_from` 留痕，并写 `transfer_import` 决策；
- 迁移观测计入新 study 的 budget（warm start 不是免费数据）。

边界：这是“共享同一条件空间”的迁移；跨底物/跨设备真正的外推仍需分层/多任务模型，
`top_k` 只是保守的负迁移防护，不能保证正迁移。

## 12. 多目标推荐（v0.20）

`recommend()` 对多目标 study 不再按主目标 y 选点，而是：

- 取当前 **Pareto 前沿**（`pareto_summary`）；
- 用 compromise programming 选 **knee**：各目标归一化后离 utopia (1,…,1) 最近的点；
- 返回 `selection_basis="pareto_knee"`、`pareto.n_front/front`、推荐点的原始目标向量；
- `confirm()` 跟随这个 knee 条件做复孔；`report()` 的推荐块同样显示 knee。

这样 `qehvi`/`parego`/`cqehvi` 跑出来的 Pareto 前沿不会在收尾时被“主目标最高但被支配”的点替代。

多目标返回的 `spec` 是板校正后的符合性，`spec_raw` 是原始观测的符合性；
report 会在两者不同时打印 `规格判定: 原始 ✓/✗ / 板校正后 ✓/✗`。

v0.21：检测到板效应时，先按各目标的板效应校正再重算 Pareto 前沿并选 knee，
`selection_basis="pareto_knee_plate_adjusted"`，同时返回 `objectives_adjusted` 与
`plate_effects.metrics`。

v0.22：若 study 有 spec，优先在前沿上选 **in-spec 且 spec score 最高**的点；
`selection_basis="pareto_spec"`（有板效应时 `pareto_spec_plate_adjusted`），
没有 in-spec 点时退回 knee。这样“Pareto 最优”与“规格命中”不会互相打架。

knee 的 utopia 距离支持按目标加权：
`Study.set_knee_weights({"yield": 10, "cost": 1})` 或
`condopt knee-weights --run <run> --weights yield=10,cost=1`；
权重写入 meta 并记 `knee_weights` 决策，重载后仍生效。

## 11.1 从历史 fork 新 campaign（v0.25）

`step` 建议 `review_spec` 后，不需要手工拼 `prepare`：

```powershell
py -3 scripts/run_condopt.py fork --run <old> --new-run <new> --spec-mode suggested --top-k 10
```

- `--spec-mode calibrated`：只校准 scale/tolerance；
- `--spec-mode suggested`：校准 + 把越界 target 移到观测边界；
- 新建 study 继承 space / context / budget / batch_size / seed / meta（并记 `forked_from`）；
- `--top-k` 把旧 study 的 top-k 观测 warm start 进新 study（计入预算）；
- 可用 `--spec-file` 直接指定外部建议 spec（优先于 `--spec-mode`）；
- `--by-plate`：按板内尺度做 calibrated/suggested（忽略板间偏移）；
- 可用 `--budget` / `--batch-size` / `--seed` 覆盖继承值；
- 写出 `fork_report.json`（changed 列表）与 `spec_fork.json`；不覆盖已有目录；
- 新 study 的 `report()` 顶部会显示 `来源: fork from ...` 与规格改动。

### 12.1.1 论文体系组合验收（v0.28）

`examples/paper_campaigns/run_combined_decision.py`：热固性真实组件/约束 + 合成板偏差 +
2 目标 + spec，跑出 `selection_basis=pareto_spec_plate_adjusted`、板效应检测、7 点前沿、
2 次确认复孔；规格不可达时如实 in_spec=false。结果见
`thermoset_combined_decision_results.json`。

## 12.1 组合验收（v0.26）

`tests/test_multiobjective.py::test_combined_moo_constraints_plate_and_spec` 覆盖四者叠加：

- 2 目标（yield/cost）+ 硬约束（T_max ≤ 78）；
- 板间系统偏差（P1 高 yield、P2 低 yield）；
- spec（yield target 88 ± 1.5）；
- 推荐结果 `selection_basis="pareto_spec_plate_adjusted"`，即：
  板效应校正 -> Pareto 前沿 -> spec 优先 -> 确认复孔跟随。

这条测试是当前决策层最完整的回归用例。

## 13. Spec 聚合方式（v0.27）

`Spec.aggregation` 支持两种：

- `weighted_sum`（默认）：各 goal 按 weight 加权平均，允许“一好遮百丑”；
- `worst_case`：取最弱 goal 的分数（短板主导），适合“每个指标都必须达标”的稳健规格。

`worst_case` 下 weight 不参与评分，标量化目标仍是 `spec_score`，算法/路由不受影响；
`spec_review` / `suggest_spec` / `fork` 都保留 aggregation 并沿用同一语义。

`pareto_summary()` 同样只用全保真点计算前沿与 HV，并给出
`n_low_fidelity_excluded`；多保真 study 的 Pareto 推荐只使用全保真点（`fidelity_filter: full`），低保真有偏点不参与 knee/spec 选择，与 `Study.best` 的全保真口径一致；
若无任何全保真点则退回全部点并在 `step` 建议 `collect_full_fidelity`。
多维超体积：`pareto.summarise` / `Study.pareto_summary()` 现对 2-4 目标计算 HV——
2 目标用闭式 `hypervolume_2d`，3-4 目标用递归切片 `hypervolume_nd`（front 规模 ≤ 数十点时开销可接受）；
参考点优先 `Objective.ref_point`，否则用观测 nadir 下移 5%。

## 代理预测（what-if）

`Study.predict(config)` / `condopt predict --run <run> --config '{"T": 80}' [--json]`：
用已记录观测拟合冻结超参 GP，返回每个目标的 `mean/std`（原始单位）、
每个约束的 `P(feasible)` 与合并可行性概率，以及有 spec 时的预测 spec 分数。
用途：在占用实验资源前先做 what-if 复核，或给人一个可解释的预期区间。

## 停止语义（step.stop_category）

`next_action()` 的 stop 现在带机器可读分类与剩余预算：

| `stop_category` | 触发 |
|---|---|
| `budget_exhausted` | 预算用尽 |
| `cost_exhausted` | 成本预算用尽 |
| `target_reached` | 达到目标分数（确认复孔均值优先） |
| `max_rounds` | 达到 `meta["max_rounds"]` |
| `stagnation` | 最近 patience 轮无显著改进 |

诊断同时给出 `stop_reason`（原文）与 `remaining_evals`。当因**停滞**停止但仍有预算时，
建议动作会追加 `consider_exploration_restart`，其建议文案基于实测（rosenbrock10，
budget 60）：**先调参重调** `retune_for_exploration()` −205.4 > 原样继续 −319.8 >
探索重启 −373.7。CLI：`condopt retune --run <run>`，见 `NEGATIVE_RESULTS.md` 第 9 条。

`patience` 敏感性（budget 60，3 seeds）见 `research/STAGNATION_TUNING.md`：
patience=1 太早停止（−1511），=3 最优（重调后 −205），≥5 几乎不触发（−306）；
因此 `Study.create` 把 patience<2 夹紧到 2 并记录 `meta["patience_clamped_from"]`。
该结论已在第二个 landscape（ackley10）复核：patience=3 的 tune 最好（−4.745），且各 patience 下 tune 均不劣于 stop。

长跑演练 `scripts/long_run_drill.py --force-budget` 就是模拟“操作者选择继续”：
实测 mixed_reaction budget 120 时 7 次强制续跑跑满，ask p95 0.39 s、digest 1.1 s、
run 目录 59 KB、无重复 id、完整性 OK。

## 目录自检（doctor）

`Study.check_integrity()` / `condopt doctor --run <run>` 检查：
suggestion_id 唯一、pending 与 dataset 不重叠、`opt_state.json` 的算法与
`config.json` 一致、problem/config/rounds JSON 可解析；有问题时 CLI 非零退出。
适合无人值守循环开始前与结束后各跑一次。

