# UI_PARETO_COST — 看板 Pareto/成本视图的组件与验收

> 维护者：algo-research ｜ v1（2026-11-05，Round 13）
> 对应：`NEXT_PRIORITIES_V3.md` §3（UI 规格）、`REVIEW_v12.md`（成本字段进 summary）、`render.py`（v0.10.2 渲染层）。
> 文献：R42（HV/固定 ref/真值口径）、R12（概率 Pareto）、R41（成本/多保真）、R32（统计口径）、R54（评估报告规范）。
> 依赖（先决条件）：① 成本闭环（v0.10.2 已具备 `cost_spent/remaining_cost`）；② records/验收用 true-HV（否则 UI 把观测噪声可视化）；③ weights 版本入库（router 决策可追溯）。

---

## 1. 数据契约（只读，不重算）

看板只消费 `Study.summary()`（v0.10.2 已有）：

```json
{
  "algorithm": "qehvi", "params": {...},
  "used": 32, "remaining": 16, "pending": 0,
  "best": {"x": {...}, "y": 98.2, "objectives": {...}},
  "pareto": {"n_front": 12, "front": [...], "hypervolume": 0.94,
             "reference_point": [60, 30]},
  "cost_spent": 18.5, "remaining_cost": 5.5,
  "router": {"next": {...}, "weights": {"version": "router-weights-2026-11-08", "sha256": "..."}}
}
```

**原则**：UI 不重新拟合 GP、不重算 HV（避免口径分叉）；所有派生量（`P_nd`、`E[HVC]`）由后端 `pareto`/`study` 提供或由后端 MC 预计算后放进 summary。

---

## 2. 组件

### 2.1 Pareto 表（`render_pareto_table`）

| 列 | 来源 | 说明 |
|---|---|---|
| 条件 | `summary.pareto.front[i].x` | 人类可读（含单位） |
| 目标均值 ± std | `objectives`/`objective_std` | 有复孔时显示 std |
| `P_nd` | 后端 MC（M=128） | 概率非支配；噪声场景必选 |
| `E[HVC]` | 后端 MC | 期望超体积贡献 |
| `P(feasible)` | 后端（约束时） | 无约束时隐藏列 |
| 来源 | `meta.algorithm`/`suggestion_id` | 可追溯到哪一轮 |

- 排序：默认 `E[HVC]` 降序；可切 `P_nd`；页脚显示 "ref=(60,30) 来自 `Objective.ref_point`（固定）"；
- 无 `ref_point`：显示 banner "HV 不可跨 run 比较（数据推导 ref）"，数值改灰或不显示；
- 单目标：**隐藏整个 Pareto 区**，只显示 best 表。

### 2.2 成本视图（`render_cost_panel`）

- **成本曲线**：x=累计成本、y=`cost_aware_best`（全保真 best），叠加 `n_evals` 口径的次级线（虚线）；
- **指标卡**：`cost_spent / cost_budget / remaining_cost`、各 fidelity 次数（`0.25: 8, 0.5: 3, 1.0: 2`）；
- **超支/未设预算**：`cost_budget` 未设置 → 隐藏成本轴、显示"次数口径"；`cost_spent > cost_budget` → 红色 banner；
- **fidelity 标注**：每个观测点按 fidelity 着色（低=浅、全=深），tooltip 显示 `cost`。

### 2.3 口径与可追溯区（`render_meta_footer`）

- ref 来源、true/observed 标注（合成/emulator 有 true；真实实验没有 → 显示"observed only"）；
- weights 版本 + sha256（来自 `summary.router.weights`）；
- 生成时间、library version、seed、`auto_route/manual_override` 状态（审计）。

### 2.4 导出（`export_dashboard`）

- `pareto.csv`（含 P_nd/E[HVC]/P(feasible)）、`cost_curve.csv`、`dashboard.html`（自包含，可存档）；
- 文件名含 `weights.version` 与时间戳，便于实验记录引用。

---

## 3. 空/异常状态（必须显式）

| 状态 | UI 行为 |
|---|---|
| 无观测 | 显示"暂无数据" + 引导 `ask` |
| 单目标 | 隐藏 Pareto 区；显示 best 表 |
| 无 `ref_point` | 显示"不可比"banner；不显示 HV 数值 |
| 无 `cost_budget` | 隐藏成本轴；显示"次数口径" |
| `pending > 0` | Pareto 表标注"含未回测建议"并置灰（不能把待测点当观测） |
| 旧 study（v0.10.1 前 cost=0） | 成本区显示 ⚠️ "历史观测成本缺失" |
| 约束全不可行 | 显示"无可行点" + 最近可行方向建议（不显示假 Pareto） |

---

## 4. 验收

```python
def test_pareto_table_uses_fixed_ref():
    # 同一 dataset：render 的 HV 与 pareto.summarise 一致；ref 来自 objective.ref_point
def test_no_ref_banner():
    # objective.ref_point 为空 → banner 存在且 HV 数值不渲染
def test_single_objective_hides_pareto():
def test_cost_panel_hidden_without_budget():
    # meta 无 cost_budget → 成本轴隐藏，显示次数口径
def test_cost_panel_shows_overspend_banner():
    # cost_spent > cost_budget → 红色 banner + 数值
def test_pending_excluded_from_front():
    # pending 点不出现在 Pareto 前沿（或标注"待测"）
def test_export_contains_weights_version(tmp_path):
    # dashboard.html/pareto.csv 的 footer 含 weights.version + sha256
def test_true_vs_observed_labeled():
    # 合成/emulator：显示 true 与 observed；真实数据：仅 observed
```

**性能**：`n_front ≤ 50`、`n_obs ≤ 500` 时渲染 <200 ms；Pareto 表分页，默认 20 行。

---

## 5. 与 skill/agent 的接口

- `report --json` 已返回 summary；看板可直接消费其 JSON（不新增接口）；
- skill 在解释结果时引用 `pareto.reference_point` 与 `cost_spent/cost_budget`，不要自行重算；
- LLM 读取 summary 时只允许引用其中的数值（禁止"估算 HV"）。

---

## 6. 参考

R42（HV/ref/真值）、R12（概率 Pareto）、R41（成本/多保真）、R32（统计口径）、R54（评估报告）、`NEXT_PRIORITIES_V3.md` §3、`render.py`（现有渲染层）。
