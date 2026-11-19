# UI_P2_APPLY_PLAN — render.py Pareto/成本视图接线顺序与验收

> 维护者：algo-research ｜ v1（2026-11-05，Round 15）
> 对应：`UI_P2_IMPLEMENTATION.md`（代码骨架）、`UI_PARETO_COST.md`（规格）。
> 前置：v0.10.4（`Study.summary()` 含 `cost_spent/remaining_cost`）；建议在 records v2（true-HV 口径）之后上线，避免把 observed HV 当验收口径。

---

## 0. 准备

```bash
py -3 -m pytest -q                       # 基线 148 passed
git rev-parse HEAD                       # 回滚点 R0
py -3 -m condopt report --study runs/demo  # 先看现有 HTML 结构，确认插入点
```

---

## 1. 接线顺序（单文件 `render.py`）

| 步 | 内容 | 检查 |
|---|---|---|
| 1 | `_pareto_section(summary)` | 无观测/单目标时返回 `""` |
| 2 | `_cost_section(summary)` | 无 `cost_budget` 时降级为次数口径 + 提示 |
| 3 | `_footer_section(summary)` | ref 来源、true/observed、weights 版本 |
| 4 | `render_report` 的 section 列表插入 3 个新函数（Pareto 在 best 后、成本在 Pareto 后） | 顺序不影响旧 section |
| 5 | 共用格式化 helper：`_fmt_config/_fmt_obj/_fmt` | 复用现有 render 的 helper，避免重复 |

**注意**：`summary` 是唯一数据源；**禁止**在 render 里调用 `pareto.summarise`/`hypervolume_2d`（验收里有 monkeypatch 测试专门防这个）。

---

## 2. 空/异常状态实现顺序

1. `_pareto_section`：`n_front==0` → `""`；`n_objectives==1` → `""`（双保险）；
2. `ref_point_source != "objective"` → banner "不可跨 run 比较"；
3. `_cost_section`：`remaining_cost is None` → 次数口径 + `--cost-budget` 提示；
4. `spent > budget` → "成本超支" 红条；
5. `pending > 0` → Pareto 表尾注 "含 N 个待测建议"，不混入前沿；
6. 旧 study（有 `used>0` 但 `cost_spent==0`）→ ⚠️ "历史观测成本缺失"。

---

## 3. 验收命令与预期

```bash
# 3.1 单元测试（6 条，见 UI_P2_IMPLEMENTATION.md §4）
py -3 -m pytest -q tests/test_render_dashboard.py -m fast_ci --durations=10
# 预期：6 passed，<1 s（纯字符串渲染，无 GP）

# 3.2 手工冒烟：单目标
py -3 -m condopt init --space configs/space.example.json --objective configs/objective.example.json --algorithm bayes --budget 8 --dir /tmp/ui_single
py -3 - <<'PY'
from condopt.study import Study
s = Study.load("/tmp/ui_single")
print(s.report())        # 期望：无 pareto section；有 cost 提示（无 --cost-budget）
PY

# 3.3 手工冒烟：多目标 + 成本预算
py -3 -m condopt init --space configs/moo.space.json --objective configs/moo.objective.json \
    --algorithm qehvi --budget 16 --cost-budget 16 --dir /tmp/ui_moo
py -3 - <<'PY'
from condopt.study import Study
s = Study.load("/tmp/ui_moo")
html = s.report()
assert "Pareto" in html and "成本" in html
assert "成本超支" not in html
print("OK", len(html))
PY
```

**预期**：单目标无 Pareto 区；无 `--cost-budget` 时出现提示；多目标出现前沿表 + ref banner（若 ref 来自 objective 则无 banner）；成本区显示 spent/budget/剩余。

---

## 4. 全量回归与提交

```bash
py -3 -m pytest -q                          # 期望 148 + 6
py -3 -m pytest -q -m fast_ci --durations=10
git diff --stat                             # 期望 render.py + tests/test_render_dashboard.py
```
**提交信息**：
```
feat(render): Pareto + cost panels (read-only from Study.summary)

- _pareto_section/_cost_section/_footer_section with empty states
- no HV recomputation in UI (monkeypatch test enforces)
- ref source / observed-vs-true / weights version in footer
```

---

## 5. 回滚点

| 点 | 操作 |
|---|---|
| R0 | `git checkout R0 -- src/condopt/render.py` |
| R1 | 若 render_report 结构破坏旧测试：`git checkout -- src/condopt/render.py tests/test_render_dashboard.py` |

---

## 6. 风险点

1. **不要重算 HV**：UI 只展示 `summary.pareto.hypervolume`；若要 `P_nd/E[HVC]`，由后端 `pareto_summary()` 预计算（下一轮）。
2. **HTML 注入**：配置/目标名来自用户数据 → 必须 `html.escape`（现有 render 已有 helper，复用）。
3. **大表性能**：`n_front>20` 分页；`n_obs>500` 只渲染前沿与 Top-k。
4. **Jinja/内联模板**：跟随现有 render 的模板方式，不引入新依赖。

---

## 7. 参考

R42（HV/ref/真值）、R12（概率 Pareto）、R41（成本/多保真）、R32（统计）、`UI_P2_IMPLEMENTATION.md`、`UI_PARETO_COST.md`。
