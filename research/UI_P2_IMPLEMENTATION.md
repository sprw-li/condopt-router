# UI_P2_IMPLEMENTATION — `render.py` 上 Pareto/成本视图的最小接线

> 维护者：algo-research ｜ v1（2026-11-05，Round 14）
> 对应：`UI_PARETO_COST.md`（Round 13 规格）、`NEXT_PRIORITIES_V3.md` §3。
> 目标：**不改算法层**，只在 `render.py` 增加两个区块 + 空状态 + 6 条验收；数据全部来自 `Study.summary()`。

---

## 1. 数据契约（只读）

```python
summary = study.summary()      # v0.10.3
# 已有字段（本 UI 需要的）：
#   algorithm, params, used, remaining, pending, best,
#   pareto: {n_front, front[], hypervolume, reference_point},
#   cost_spent: float, remaining_cost: float|None,
#   router: {next, weights?}
```

**规则**：UI 不调用 `pareto.summarise`、不重算 HV；`P_nd`/`E[HVC]` 若需要，后端在 `pareto_summary()` 里预计算（M=128）后加入 summary（下一轮）；MVP 先只显示已有点级信息（目标值、std、fidelity、cost、ref 来源）。

---

## 2. 最小接线（render.py，约 +120 行）

### 2.1 入口分派

```python
def render_report(study, *, json=False):
    summary = study.summary()
    sections = [
        _header_section(summary),
        _best_section(summary),
        _pareto_section(summary),      # NEW（单目标返回空）
        _cost_section(summary),        # NEW（无 cost_budget 时降级为次数口径）
        _router_section(summary),
        _footer_section(summary),      # ref 来源 / true-vs-observed / weights 版本
    ]
    return "<html>…" + "\n".join(s for s in sections if s) + "</html>"
```

### 2.2 `_pareto_section(summary)`

```python
def _pareto_section(summary):
    if summary["pareto"]["n_front"] == 0:
        return ""                                   # 单目标/无观测：隐藏
    ref = summary["pareto"].get("reference_point")
    ref_from_objective = summary.get("ref_point_source") == "objective"
    rows = []
    for i, item in enumerate(summary["pareto"]["front"][:20]):
        objs = item["objectives"]
        rows.append(
            "<tr>"
            f"<td>{_fmt_config(item['x'])}</td>"
            f"<td>{_fmt_obj(objs)}</td>"
            f"<td>{_fmt(item.get('objective_std'))}</td>"
            f"<td>{item.get('meta', {}).get('algorithm', '-')}</td>"
            "</tr>")
    banner = "" if ref_from_objective else (
        '<div class="warn">参考点由数据推导（非 Objective.ref_point）：'
        'HV 不可跨 run 比较</div>')
    return f"""
<section id="pareto">
  <h2>Pareto 前沿（{summary['pareto']['n_front']} 点，显示前 20）</h2>
  {banner}
  <p class="muted">ref = {ref}（{'Objective.ref_point' if ref_from_objective else '数据推导'}）</p>
  <table><thead><tr><th>条件</th><th>目标</th><th>std</th><th>来源</th></tr></thead>
  <tbody>{''.join(rows)}</tbody></table>
</section>"""
```

### 2.3 `_cost_section(summary)`

```python
def _cost_section(summary):
    spent = summary.get("cost_spent")
    if spent is None:
        return ""
    remaining = summary.get("remaining_cost")
    if remaining is None:
        # 无 cost_budget：只显示次数口径 + 提示
        return (f'<section id="cost"><h2>成本</h2>'
                f'<p>已花费 {spent:.3g}（未设置 cost_budget：成本口径不可用于跨 run 比较）</p>'
                f'<p class="muted">如需成本预算：<code>condopt init --cost-budget N</code></p></section>')
    budget = spent + remaining
    over = spent > budget + 1e-12
    bar = f'<div class="bar"><span style="width:{min(100, 100*spent/max(budget,1e-9)):.0f}%"></span></div>'
    warn = '<div class="warn">成本超支</div>' if over else ""
    return (f'<section id="cost"><h2>成本</h2>{warn}{bar}'
            f'<p>已花 {spent:.3g} / 预算 {budget:.3g}（剩余 {remaining:.3g}）</p>'
            f'<p class="muted">成本模型见 skill 文档（cost=fidelity^exponent）</p></section>')
```

### 2.4 `_footer_section(summary)`

```python
def _footer_section(summary):
    weight = (summary.get("router") or {}).get("weights") or {}
    return (f"<footer><p>ref={summary['pareto'].get('reference_point')} · "
            f"对齐口径={'ref_point' if summary.get('ref_point_source')=='objective' else '数据推导'} · "
            f"weights={weight.get('version','default')} "
            f"({weight.get('sha256','-')[:8]}) · observed-only</p></footer>")
```

---

## 3. 空/异常状态（最小实现）

| 状态 | 返回 |
|---|---|
| 无观测 | `_pareto_section` 返回 `""`；`_best_section` 显示"暂无数据" |
| 单目标 | `_pareto_section` 返回 `""`（`n_front` 为 0 或 `n_objectives==1`） |
| 无 `ref_point` | Pareto 区显示"不可比"banner（`ref_point_source != "objective"`） |
| 无 `cost_budget` | 成本区显示 spent + 提示 `--cost-budget`，不显示预算/进度条 |
| `pending > 0` | Pareto 表末尾加 `<tr class="pending">` 标注"含 N 个待测建议"（不混入前沿） |
| 旧 study cost=0 | 成本区加 ⚠️ "历史观测成本缺失（v0.10.1 前）" |

---

## 4. 验收（6 条）

```python
def test_pareto_section_hidden_for_single_objective():
    # study(algorithm="bayes", 1 objective) → "id=\"pareto\"" not in html
def test_pareto_banner_when_ref_derived():
    # objective.ref_point 空 → "不可跨 run 比较" in html
def test_cost_section_prompts_cost_budget_when_absent():
    # 无 cost_budget → html 含 "--cost-budget"，不含进度条
def test_cost_section_shows_overspend():
    # 构造 spent > budget → "成本超支" in html
def test_footer_contains_weights_version_and_ref():
    # html 含 weights.version（或 default）与 ref 数值
def test_render_does_not_recompute_hv(monkeypatch):
    # monkeypatch pareto.hypervolume_2d 抛错 → render 仍成功（证明 UI 不重算 HV）
```

**性能**：`n_front ≤ 50`、`n_obs ≤ 500` 时 render <200 ms；Pareto 表默认 20 行 + "显示全部" 折叠。

---

## 5. 与下一轮（P_nd/E[HVC]）的衔接

- 后端在 `pareto_summary()` 中加 `probability_nondominated`（M=128 MC，`MULTI_OBJECTIVE_AND_NOISE.md` §5），summary 加字段；
- 前端只在表里增加两列，不改数据契约；
- 概率列必须有 `mc_samples` 与 `model_version` footer，保证可复现。

---

## 6. 参考

R42（HV/ref/真值）、R12（概率 Pareto）、R41（成本/多保真）、R32（统计）、`UI_PARETO_COST.md`、`NEXT_PRIORITIES_V3.md` §3。
