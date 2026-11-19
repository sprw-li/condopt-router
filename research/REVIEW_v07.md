# REVIEW_v07 — 对 v0.6.1（qEHVI NaN-ref / qNEHVI 告警 / mf_suzuki / router 权重）的审查

> 审查者：algo-research ｜ 日期：2026-11-05 ｜ 快照：18:08（`qehvi.py` 8.3KB、`qnehvi.py` 6.9KB、`mfbo.py` 12.7KB、`router.py` 35.4KB、`cli.py` 16.7KB、`benchmarks.py` 23.0KB）
> 格式：**`文件:行` → 问题 → 文献依据 → 修复建议 → 优先级**。P0 = 正确性/结果不可信；P1 = 显著影响样本效率或与宣称不符；P2 = 工程质量。
> **本轮实测**（本机 CPU、`auto_route=False`、`moo_yield_cost`/`mf_suzuki`）：见 §4。这些数字也用于定 `QNEHVI_MF_ACCEPTANCE.md` 的阈值。

---

## 0. 总结

| 编号 | 位置 | 一句话 | 优先级 |
|---|---|---|---|
| P0-1 | `study.py:306-308` + `study.py:92`（`manual_override` 只由 `override()` 设置） | `Study.create(algorithm="mfbo")` **不会钉住**算法：只要 `auto_route=True`（默认），第一次 `ask` 就被 router 改掉。**实测请求 mfbo，pending 里是 `sobol`** —— 用户以为在跑多保真/多目标算法，实际跑的是别的 | **P0** |
| P0-2 | `benchmarks.py:130-142` + 报告/验收口径 | `moo_yield_cost` 的"reference front"是**真值**，但 run 的 HV 常用**带噪观测**计算 → HV ratio 可以 **>1**。实测 qEHVI noisy：observed HV ratio `[1.0213, 0.95]`，true HV ratio `[0.9538, 0.9231]`；qNEHVI：observed `[0.9969, 0.9613]`，true `[0.9565, 0.9234]`。用观测值做验收会奖励"幸运噪声" | **P0** |
| P1-1 | `router.py:801-806` | 权重是**乘在总分上**：负分算法（如 MOO 场景的 bayes=-165）乘 1.5 会变成 -247.5（更负 = 更被排斥，与"提高权重"直觉相反）；权重 0 会把分数清零，可能**排到其他负分算法前面**。实测 `{"bayes":0}` → 得分 0，router 改选 sobol | P1 |
| P1-2 | `router.py:801` | `ranked = sorted(...)` 后立刻被 `:806` 重排覆盖 → 死代码；`confidence` 用乘权后的分差，而 `margin` 语义未重新标定 | P2 |
| P1-3 | `study.py:306-308`（与 P0-1 同源） | `manual_override` 只在 `override()` 里置 True；`Study.load` 也不恢复该标志 → **重启后 override 丢失**（v0.1 P0-4 的残留） | P1 |
| P1-4 | `mfbo.py:149-166` + `study.py` 预算语义 | `ask_with_fidelity` 一次返回整批，`Study` 的 `remaining()` 只按**次数**；实测成本超支：预算 cost=24 实际花 25.25（+5.2%）。多保真场景"预算"应按成本计 | P1 |
| P1-5 | `mfbo.py:174-191` | 初始设计按 `levels[index % len]` 轮转 → 实测 fidelity 序列 `[0.25,0.5,1.0,0.25,0.5,1.0]`，初始就花全保真成本（规格建议初始全用最低档）；`best` 仍用混合保真度的 `max(self._y)`（v0.6 P1 未改） | P1 |
| P1-6 | `benchmarks.py:191-208` | `cost_aware_best` 的 `best_value_per_cost = max(value / spent)` 不是可比较指标（`spent` 单调增、raw value 未归一化，会偏向早期点）；`best_full_fidelity` 是对的，但它只取 fidelity=1 | P1 |
| P1-7 | `benchmarks.py:178-188` | `mf_suzuki_problem()` 的 `fn` 固定 `fidelity=1.0`，`BenchmarkProblem` 没有 fidelity 字段 → 通用 benchmark runner 调 `problem.evaluate(config)` 只会跑全保真；多保真评估必须走专用协议（`MF_BENCHMARK.md`） | P1 |
| P2-1 | `qehvi.py:104-116` | v0.6 P0-1 已修（只用完整观测推导 ref、无完整观测抛错）✓；但 ref 仍在"首次 `_model_ask`"冻结（≥3 完整观测），预算紧时会由 3 个点决定（v0.6 P1-3 未改） | P2 |
| P2-2 | `qnehvi.py:99-135` | 无噪声信息 warning 已加，实测**能触发**（`RuntimeWarning: ... nested MC reduces to the noiseless qEHVI`）✓；但只在 `_model_ask` 里触发（一次 campaign 可能刷多条），建议只警告一次并写 decisions | P2 |
| P2-3 | `router.py:40-52` | 权重校验只有"名称已知 + ≥0"；无上限（1e6 会让分数爆炸）、不校验 NaN/inf（`float('nan')` 可通过 `>=0` 的 False → 报错？`nan>=0` 为 False，会被 `factor < 0` 判断为 False → 接受！需 `math.isfinite`） | P1 |
| P2-4 | `cli.py:128-142,354,365` | `--weights` 只挂在 `route`/`init`；`ask`/`tell`/`override` 加载 study 后不会重新应用权重 → **同一 study 的后续 ask 与 init 时的路由权重可能不同**（取决于进程） | P2 |
| P2-5 | `qehvi.py:119-176` | v0.6 的 P1（每 ask 两次 fit）已修（复用 `models[0]`）✓；`_candidate_pool` 的 local 中心仍来自主目标 `self._y`（v0.5 P1-2 未改） | P2 |

**总体**：v0.6.1 把 qEHVI 的两个错误路径修干净了（NaN ref、无完整观测、重复 fit），qNEHVI 的告警也实测生效，`mf_suzuki` 基准与成本指标可用。**本轮最大的两个问题是"静默改算法"（P0-1）与"用带噪观测算 HV"（P0-2）**——都不是算法数学问题，而是试验口径问题，会直接让"验收通过但实际没跑目标算法"。

---

## 1. P0 详述

### P0-1 `study.py:306-308` — 显式 `algorithm=` 被 auto_route 静默覆盖

```python
algorithm, params = self.algorithm, dict(self.params)
if self.auto_route and not self.manual_override:
    rec = route_round(ctx, self._recent_best_scores(), patience=self.patience)
    if rec.algorithm != algorithm:
        self.log_decision(action="auto_reroute", ...)
        algorithm, params = rec.algorithm, rec.params
```

`Study.create(algorithm="mfbo", auto_route=True)`（默认）→ 第一次 ask 直接 reroute。**实测**（`mixed_reaction`、budget 8、batch 2）：

```
requested=mfbo, pending algorithms=['sobol', 'sobol']
```

后果：①用户/agent 指定算法无效（对 `qehvi/qnehvi/mfbo/constrained` 等新算法尤其致命，因为 router 的规则可能还没覆盖它们或评分偏低）；②`auto_reroute` 会写 decisions，但用户通常不会读；③多保真/Fidelity 链路在 ask 时被换成不含 fidelity 的算法，`supports_fidelity` 分支根本不执行。

**文献依据**：R9（Optuna 的 `study.sampler` 显式指定即生效；路由是附加能力而非强制）、R8（SMAC3 Facade 的场景选择是显式的）、`REVIEW_v01.md` P0-4（override 被 auto_route 覆盖，当时修了 override 但没修 create 的显式 algorithm）。

**修复（二选一，推荐 A）**

```python
# A) Study.create 时：显式算法 = 用户意图 → 钉住（等价于 override）
if algorithm.strip().lower() != "auto":
    manual_override = True            # 新增初始化参数，__init__ 里 self.manual_override = bool(manual_override)

# B) 保持 auto 语义，但首个 ask 的 reroute 必须显式记录"与你指定的不同"并在 report 顶部警告
```

同时：`Study.load` 从 `config.json` 恢复 `manual_override`；CLI `init --algorithm` 传入时打印"pinned/auto-routed"状态。

**验收**：`test_explicit_algorithm_survives_first_ask`（requested mfbo → pending algorithm=="mfbo"）；`test_auto_still_reroutes_when_algorithm_auto`；`test_override_survives_reload`。

### P0-2 `benchmarks.py:130-142` + HV 报告口径 — 用带噪观测算 HV

`moo_reference_front()` 是真值前沿；而 `pareto.summarise(dataset)` 用 `observation.objectives`（**带噪观测**）算 HV。噪声下 `max` 会被"幸运观测"抬高：实测 qEHVI noisy（σ=1.5、2 seeds）：

```
observed HV ratio = [1.0213, 0.9500]      ← 可 >1（说明"比参考前沿还好"）
true     HV ratio = [0.9538, 0.9231]      ← 用真实目标重算
qNEHVI   observed  = [0.9969, 0.9613]
          true      = [0.9565, 0.9234]
```

**文献依据**：R42（HV 是前沿质量的度量，必须基于"真实目标值"或一致口径）、R32（benchmark 报告必须可复现/可比）、R12（噪声多目标的评估应报告"真实前沿"或概率前沿）。

**修复**

1. **评估器保存真值**：`Observation.meta["true_objectives"]`（合成/emulator 才有；真实实验没有 → 用重复测量均值）；
2. `pareto.summarise(..., use_true=True)` 或 benchmark 层统一选 `true_objectives` 优先；
3. `Study.report` 的 HV 明确标注"基于观测值（含噪声）"，跨 run 比较时用 `true`；
4. 验收脚本（`qNEHVI_ACCEPTANCE.md`）必须用 true HV 阈值（≥0.90），不要用 observed（会误判通过）。

**验收**：`test_benchmark_hv_uses_true_objectives`；`test_hv_ratio_bounded_by_reference`（用真值时 ≤1+1e-9）。

---

## 2. P1 详述

### P1-1 `router.py:801-806` — 权重乘在总分上的语义问题

```python
for name in scores:
    factor = _ALGORITHM_WEIGHTS.get(name)
    if factor is not None and factor != 1.0:
        scores[name] = round(scores[name] * factor, 2)
```
- 负分算法（`moo_unsupported=-200` 已通过能力过滤排除，但如 parego 在单目标场景 -165 仍可能出现）：`-165 × 1.5 = -247.5` → 提高因子反而**更被排斥**；
- `factor=0` → 分数 0，可能**高于**其他负分算法（实测 `{"bayes":0}` 时 bayes=0、parego=-165 → 0 > -165）；
- 权重作用在 `round(score,2)` 之后 → 双重取整，校准不可微。

**建议语义**（与 `ROUTER_SCORETERM.md` 对齐但更简单）：**加性**校准 `score' = score × factor` 仅用于正分算法；或改为 `score' = score + log(factor)·scale`（softmax 风格）；或干脆等 ScoreTerm 化后按 term 加权（推荐，长期）。短期内至少：`factor=0` 解释为"禁用该算法"（明确文档），负数分数只做加性惩罚。

**依据**：R8（场景先验应可解释）、R24（权重需可解释/可审计）。

### P1-3 `study.py:92,306-308` — `manual_override` 不持久化

`config.json` 不写 `manual_override`；`Study.load` 不恢复 → override 后重启进程，下一次 ask 又回到 auto_route。修复：写入 config 并在 load 时恢复；`resume_auto_route()` 已有，审计链完整。

### P1-4 `mfbo.py` + Study 预算 — 成本超支

实测：budget ≈ cost 24，实际花 25.25（+5.2%）；原因是 `ask(2)` 不看剩余成本、`Study.remaining()` 只计次数。修复：`Study` 增加 `remaining_cost()`（用 `Objective.fidelity.costs` 与 `cost_spent`）；`mfbo.ask_with_fidelity` 内部按成本截断（若下一档超出剩余成本，降档到剩余可支付的最低档）。

### P1-5 `mfbo.py:174-191,203` — 初始设计含全保真 + `best` 混保真

实测初始 fidelity：`[0.25, 0.5, 1.0, 0.25, 0.5, 1.0]`。修复：初始设计全用 `min(levels)`（另加 1–2 个全保真验证点）；`best = max(y[fidelity==max_level])`（无全保真时用 bias 校正）。

### P1-6 `benchmarks.py:191-208` — `cost_aware_best` 指标定义

`best_value_per_cost = max(value/spent)` 会把"早期小成本点"算成很高（分母小），且 raw value 与成本不可比。修复：提供
```
def cost_aware_best(observations, budget) -> float:
    # 按观测顺序累计成本，返回累计成本 ≤ budget 时的最优真值
```
`best_full_fidelity` 保留；报告另给"达到 best 的成本"（cost-to-threshold）。

### P1-7 `benchmarks.py:178-188` — `mf_suzuki_problem` 的 `fn` 忽略 fidelity

通用 runner（`run_algorithm`）调 `problem.evaluate(config)` → 只得到全保真值；多保真必须用 `ask_with_fidelity` + `mf_suzuki(config, s)` + `mf_suzuki_cost(s)` 的专用协议。修复：给 `BenchmarkProblem` 加 `fn_fidelity` 与 `cost_fn`，或把 `mf_suzuki` 拆成独立 `MultiFidelityTask`（避免通用 runner 误用）。

### P2-3 `router.py:40-52` — 权重校验缺 `math.isfinite`

`nan` 的比较：`nan < 0 → False`，`nan >= 0 → False`，因此 `if factor < 0: raise` 不会拦住 NaN；`float('nan')` 会进入 `_ALGORITHM_WEIGHTS`，`score * nan = nan` → 排序用 nan 会得到不可预测结果（`sorted` 对 nan 不稳定）。修复：

```python
import math
factor = float(value)
if not math.isfinite(factor) or factor < 0:
    raise ValueError(...)
if factor > 10.0:
    warnings.warn(f"extreme router weight {factor} for {name}")
```

### P2-4 `cli.py` — `--weights` 只在 route/init 应用

`ask/tell/report/override` 从磁盘加载 study 时不会重新应用权重 → 同一 study 在不同命令间路由可能不同（`ask` 使用进程内权重=空 → 默认）。修复：所有会触发 `route` 的命令都接受 `--weights`，或把权重写入 `config.json`（推荐：init 时把生效权重快照进 config，`ask` 从 config 读）。

---

## 3. P2（择机）

- **P2-1** `qehvi.py:104-116`：ref 冻结时机（≥3 完整观测即冻结）→ 建议与 ParEGO 一致（≥`max(3,2m)` 才冻结）。
- **P2-2** `qnehvi.py:126-135`：warning 每次 `_model_ask` 都发 → 用 `self._noise_warned` 只发一次并写 decisions。
- **P2-5** `qehvi.py:117`：候选池 local 中心用主目标（v0.5 P1-2 未改）。
- **P2-6** `benchmarks.py`：`moo_reference_front` 的 `time` 网格用 `linspace(5,120)`，而问题 space 是 `time∈[5,120]` 连续——与 `BENCHMARK_TIER_CD.md` 早期规格（log time 20–180）不一致；不影响正确性，但**跨文档口径需统一**并写进注释。
- **P2-7** `qehvi.py:168-188`：state 只存 ref；`_Y` 依赖 prior 重建（可接受），但若 restart 时 prior 顺序变化，ref 冻结值可能不同（写 decisions 记录 ref 来源）。

---

## 4. 实测证据（本机，`auto_route=False`，budget 24 / batch 4 / 2 seeds）

| 运行 | observed HV ratio | true HV ratio | 备注 |
|---|---|---|---|
| qEHVI 无噪 | 0.9466 / 0.9426 | 同 observed | 与 Lead 报的 ≈0.95 一致 |
| qEHVI 噪声 σ=1.5 | 1.0213 / 0.9500 | **0.9538 / 0.9231** | observed 可 >1（P0-2 证据） |
| qNEHVI 无噪 | 0.9464 / 0.9555 | 同 observed | 触发 no-noise warning ✓ |
| qNEHVI 噪声 σ=1.5（n_base=32,n_y=64） | 0.9969 / 0.9613 | **0.9565 / 0.9234** | true HV 略优于 qEHVI（均值 +0.0015） |

| 多保真（cost 预算 24，2 seeds） | best full-fidelity | 实际成本 |
|---|---|---|
| `mfbo`（levels 0.25/0.5/1.0，min_full 0.15） | **103.46 / 104.65** | 25.25（+5.2%） |
| `bayes`（全保真，24 次） | 98.92 / 93.24 | 24.00 |

| 权重机制 | 结果 |
|---|---|
| 空权重 vs 默认 | 12 个算法分数字典完全相同 ✓ |
| `{"bayes": 1.5}`（CLI 文件） | bayes 80→120 ✓ |
| `{"bayes": 0.0}` | bayes 0；router 改选 sobol（负分算法仍 -165）→ P1-1 |
| 未知算法 / 负权重 | 均 `ValueError` ✓ |
| NaN 权重 | **未被拦截**（待修，P2-3） |

---

## 5. 修复顺序

1. **P0-1**（显式算法钉住）——影响所有"指定算法"的实验与验收。
2. **P0-2**（HV 用真值）——影响基准/验收结论。
3. **P1-3**（override 持久化）、**P2-3**（NaN 权重）——小改动、防事故。
4. **P1-4/P1-5**（mfbo 成本预算、初始设计、best 口径）。
5. **P1-1/P2-4**（权重语义与 CLI 一致性）→ 等 ScoreTerm 化时一并解决。
6. P2 清理。

**回归门槛**：123 tests 全绿 + 上述新增单测；`qNEHVI_ACCEPTANCE.md` 用 true-HV 阈值；`mfbo` 在 cost 24 时 best_full / cost 不超过全保真 BO -5%（当前实测 +8.4% 均值，门槛可设 +5%）。

---

## 6. 参考

R8（SMAC3 Facade：场景/算法选择显式化）、R9（Optuna sampler 显式生效）、R12（qNEHVI/噪声多目标）、R21（qEHVI）、R42（HV/固定 ref/真值口径）、R32（benchmark 统计协议）、R41（BOCA：保真度与成本）、R24（规则可解释）。
