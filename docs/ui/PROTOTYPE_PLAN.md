# 最小可视化原型计划（静态优先，服务次之）

- 版本：v0.3（2026-11-05）｜task-2 设计 + **task-7 落地 P0+P1** + **task-11 落地 P2（多目标 Pareto / 参数重要性 / 只读服务）**
- 关联：[FEASIBILITY.md](FEASIBILITY.md) §4（推荐架构）、[INPUT_LINES_SPEC.md](INPUT_LINES_SPEC.md)（表单字段）、[AI_HARD_INTEGRATION.md](AI_HARD_INTEGRATION.md)（决策日志）
- 结论：**P0+P1+P2 已实现**——`src/condopt/render.py`（`render_study()`）、`scripts/render.py`、`scripts/serve.py`；产出单文件离线 `dashboard.html`（含多目标 Pareto、参数重要性）+ 可选 `convergence.png`，本地只读服务只绑 `127.0.0.1` 且仅 GET。可交互版式示意见 `prototype/dashboard_mock.html`（mock 数据，零依赖）。

---

## 0. 一页计划

| 项 | 决定 |
|---|---|
| 数据源 | `runs/<study_id>/`：`problem.json` + `config.json` + `dataset.jsonl` + `rounds.json` + `pending.json` + `decisions.jsonl`（+ 空 pending 时回退 `suggestions.csv`） |
| 图表 | 已实现：① 收敛曲线（SVG + PNG）② 空间覆盖（前两维投影，已测/待测样式区分）③ 参数重要性（n≥20，数值 |ρ| / 分类 η²，不足 20 显示占位）④ 建议批次表 ⑤ router/override 决策时间线 ⑥ **多目标 Pareto 区块**（非支配点表 + 2D 散点/阶梯前沿 + HV/参考点；单目标隐藏） |
| 技术 | matplotlib 3.11.2（Agg，PNG 可选）+ 标准库；HTML 手写 SVG/CSS，数据内联，无 JS 依赖；服务用 stdlib `http.server` |
| 实现状态 | ✅ `render.py` + `scripts/render.py` + `scripts/serve.py` + `tests/test_render.py`（8 项）；`runs/demo` 18.0 KB / `runs/demo_moo` 25.6 KB；pytest **102 passed** |
| 顺序 | S0 数据 ✅ → S1 metrics ✅（并入 renderer）→ S2 PNG ✅ → S3 单文件 HTML ✅ → **S4 本地只读服务 ✅** → S5 验收 |
| 不做 | TUI 图表、plotly、Jupyter、前端框架、数据库、WebSocket |
| 关键约束 | 渲染逻辑**不依赖 sklearn**；`scripts/*.py` 直接按路径加载 `render.py`、绕过 `condopt/__init__`（后者经 algorithms 导入 sklearn ~3 s）；`file://` 下不 fetch，数据内联；单文件 ≤500 KB（实测 18–26 KB） |

### 0.1 已实现（task-7，2026-11-05；P0+P1）

| 项 | 内容 |
|---|---|
| API | `condopt.render.render_study(run_dir, out_dir=None, make_png=True) -> dict`；只读 run 目录，返回路径 + 计数 + warnings |
| CLI | `py -3 scripts/render.py --run runs\demo [--out DIR] [--no-png]`（默认写到 run 目录） |
| 产物 | `dashboard.html`（UTF-8 无 BOM、无 CDN/外部请求、内联 SVG）；`convergence.png`（Agg，`plt.close`，失败不阻塞 HTML） |
| 数据源 | `problem.json / config.json / dataset.jsonl / rounds.json / pending.json / decisions.jsonl`；pending 为空时用 `suggestions.csv`（标注「上一批，可能已 tell」） |
| 不伪造 | `pending.json` 无 `predicted_*` → 表格显示「—」；收敛图不画预测带 |
| 测试 | `tests/test_render.py` 4 项：离线无 http、无 BOM、关键区块、override 理由、PNG 存在、`--no-png`/自定义输出、空 study、缺失目录 |
| 已知降级 | 覆盖只投影前两维；无 pending 时显示上一批 CSV；不存在参数交互（只读） |

### 0.2 已实现（P2，task-11，2026-11-05）

| 项 | 内容 |
|---|---|
| 只读服务 | `scripts/serve.py`：`py -3 scripts/serve.py --run runs\<dir> [--port 8765] [--out DIR] [--png] [--verbose]`；只绑 `127.0.0.1`；`GET /`（每次请求重渲染 + 注入 10s 自动刷新）、`GET /api/state`、`GET /convergence.png`（仅 `--png`）；**非 GET 一律 405**；路径遍历 400；未知路径 404；**不写 run 目录**（渲染输出写 `--out` 或临时目录） |
| 多目标 Pareto | 仅当 `objective.objectives` 长度 > 1：非支配点表（条件 + 各目标原始值 + 方向箭头 + 来源）、2D 目标散点（被支配灰 / 非支配蓝）+ 阶梯前沿 SVG、HV 与参考点；参考点口径**优先 `Objective.ref_point`，否则观测 nadir 下方 5%**，与 `condopt.pareto.summarise` 逐位一致（测试断言相等）；目标数 > 2 只出表不画图 |
| 参数重要性 | n≥20：数值变量 `|Spearman ρ|`、分类/序数/bool `η²`（组间方差比），按效应降序取前 8；n<20 显示「样本不足（<20）」；不 import sklearn、不伪造数值 |
| 覆盖图 | 已测实心点 vs 待测空心虚线点（P1 已实现，P2 在多目标 study 下同样生效） |
| 测试 | `tests/test_render.py` 新增 4 项：多目标 Pareto 区块 + HV 与 Study 一致；20 样本重要性；<20 样本占位；serve 只读（GET 200 / /api/state / POST/PUT/DELETE/HEAD 405 / 遍历 400 / 404 / run 目录不被写） |
| 实测 | `runs/demo` dashboard 18,471 B / PNG 27,012 B；`runs/demo_moo` dashboard **26,222 B** / PNG 29,162 B；serve `GET /` 约 26 KB（demo_moo，含注入的 refresh meta）、`/api/state` JSON；pytest **102 passed** |
| 已知降级 | HV 仅 2 目标计算（>2 只出非支配表）；重要性是相关/方差近似而非置换重要性；serve 无鉴权（仅本机 loopback，无写接口）；无 SSE/WebSocket（纯 meta refresh） |

---

## 1. 数据契约（先定死，再写渲染）

### 1.1 目录结构（**已实现**，`src/condopt/study.py`；`runs/demo` 是现成样例）

| 文件 | 状态 | 内容 | 渲染层是否读 |
|---|---|---|---|
| `problem.json` | ✅ | `space`（变量 name/kind/low/high/choices/unit/description）、`objective`（name/direction/objectives/**directions/ref_point**/constraints）、`seed`、`budget`、`meta` | 读（轴标签/单位/目标方向/预算/HV 参考点） |
| `config.json` | ✅ | `algorithm`、`params`（经 `AlgorithmSpec` 校验）、`context`（router 特征：noise/smoothness/compute/domain/n_dims/…）、`batch_size`、`auto_route`、`patience`、`manual_override` | 读（header/参数展示） |
| `dataset.jsonl` | ✅（`Dataset.append_jsonl`） | 每行 `Observation.to_dict()`：`x, y, objectives, constraints, fidelity, cost, t, source, meta`；多目标用 `objectives` | 读（收敛/覆盖/重要性/Pareto/预算） |
| `pending.json` | ✅ | 最近一次 `ask` 的待测批次（含 `id/batch/algorithm/x`） | 读（批次表首选；为空回退 CSV） |
| `rounds.json` | ✅ | 每轮 `{round, t, n, algorithm, best}`；`best` 已按 `direction` 转成 signed | 读（每轮最优曲线） |
| `suggestions.csv` | ✅ | `id,batch,algorithm,<变量列>`；**只保留最新一批** | 读（pending 为空时的兜底） |
| `decisions.jsonl` | ✅ | `{t, action, algorithm, params, reason, payload}`；`action` 现有 `initial_route/auto_reroute/override/resume_auto_route` | 读（时间线） |
| `opt_state.json` | ✅ | 优化器内部状态（断点续跑） | 不读 |
| `metrics.json` | ❌ 不生成 | 直接从源文件算并内联进 HTML，无需中间缓存 |
| `figures/*.png` | ✅ | 只生成 `convergence.png`；其余图用内联 SVG |
| `dashboard.html` | ✅ | 单文件，数据内联（实测 18.0–25.6 KB） |

> 渲染层只读 6 个文件：`problem.json / config.json / dataset.jsonl / rounds.json / pending.json / decisions.jsonl`（`suggestions.csv` 仅兜底）。**不读 `opt_state.json`**。

### 1.2 渲染层用到的字段（v1 只依赖这些，缺了就降级）

| 数据 | 字段 | 用途 |
|---|---|---|
| `problem.json` | `space.variables[].{name,kind,low,high,choices,unit}` | 表格列名/单位、坐标、固定条件校验 |
| | `objective.{name,direction,objectives,directions,ref_point}` | 主目标 y 轴、signed 转换、多目标方向、HV 参考点 |
| | `budget` | 进度条、剩余 |
| `config.json` | `algorithm`, `params`, `batch_size`, `context.{noise,smoothness,compute,domain,n_dims}`, `patience` | header、参数卡、停滞阈值 |
| `dataset.jsonl` | `x`, `y`, `objectives`, `source`, `t`, `meta.algorithm` | 散点、best-so-far、覆盖、重要性、Pareto |
| `rounds.json` | `round`, `best`, `n`, `algorithm` | 每轮 best 阶梯线、轮次轴、算法切换标注 |
| `pending.json` | `id`, `batch`, `algorithm`, `x` | 建议批次表（待测）；无预测列 |
| `suggestions.csv` | `id`, `batch`, `algorithm`, 变量列 | pending 为空时的兜底表 |
| `decisions.jsonl` | `t`, `action`, `algorithm`, `params`, `reason`, `payload` | 决策时间线（最近 12 条） |

> **v1 明确没有的东西**：`predicted_mean/predicted_std/acquisition`（`Study.ask()` 只返回点，不返回代理预测）。因此批次表不显示预测列，收敛图不画预测带；这两项列为 v2 的算法侧改动（见 §7）。

### 1.3 `metrics.json`（渲染层缓存，纯派生；示例对应 `runs/demo`）

```json
{"study": "runs/demo", "algorithm": "bayes", "params": {"acquisition": "ei", "kernel": "matern52"},
 "budget": {"total": 24, "used": 24, "left": 0, "rounds": 6, "batch_size": 4},
 "best": {"y": 101.06, "round": 5, "x": {"T": 78.96, "time": 31.49, "conc": 0.265, "solvent": "MeCN", "equiv": 2, "inert": true}},
 "curve": [{"round": 1, "best": 93.79, "n": 4}, {"round": 2, "best": 94.07, "n": 4}],
 "coverage": {"pair": ["T", "time"], "nx": 10, "ny": 10, "occupied": 37, "empty": 63,
              "expected": 24, "note": "空网格 63%，建议增加探索"},
 "importance": [{"var": "T", "rho": -0.62, "n": 24}, {"var": "solvent", "group": true, "rho": 0.21, "n": 24}],
 "flags": []}
```

> `metrics.json` 只是缓存，**永远可以从 6 个源文件重建**；不把它当唯一真源。

---

## 2. 图表规格（每张图：解决什么问题、画什么、怎么降级）

### 2.1 收敛曲线 `convergence.png`

| 项 | 内容 |
|---|---|
| 解决 | 优化是否在收敛？还有没有空间？ |
| X 轴 | 主：轮次（`rounds.json` 的 `round`，1..6）；副：累计评估次数（`dataset.jsonl` 顺序） |
| Y 轴 | 目标值（按 `objective.direction` 转为 signed score，或显示原始 + 方向注释） |
| 系列 | ① 原始观测散点（按评估序，灰，size 8）② **每轮 best 阶梯线**（`rounds.json`，主色，粗） |
| 标注 | 当前最优配置摘要（2 个变量 + y）；预算进度「已用 24/24，6 轮」；`decisions.jsonl` 中 `auto_reroute` 处画竖虚线并标算法名 |
| 数据 | `rounds.best` + `dataset.y` |
| 实现 | matplotlib：scatter + step，约 60 行 |
| 降级 | 无 `rounds.json` → 用 dataset 顺序累积 best 近似；观测 <2 → 占位文本 |
| 静态 HTML | 内联 SVG（polyline + circle + text），约 80 行 JS 生成 |
| 常见误读防护 | 不要画「每轮最优」当收敛（会假装改进）；必须 best-so-far；y 轴不从 0 强制开始，但需在标题写范围 |

### 2.2 搜索空间覆盖（内联 SVG，已实现）

| 项 | 内容 |
|---|---|
| 解决 | 还有哪些条件区间没被试过？待测建议是否落在新区域？ |
| 实现（v1） | 取**前两个变量**做二维投影散点：连续/整数按原始区间归一化，`log` 按 log10 归一化，分类/序数/布尔按类别序号；坐标轴刻度显示原始值（含单位） |
| 已测 vs 待测 | 已测 = 实心蓝点（`dataset.jsonl`）；待测 = 空心橙色虚线圆（`pending.json`）；图例显示两者数量 |
| 网格统计 | 前两维 10×10 网格命中数（如 `20/100`）写在图例与说明里，**白格=完全未覆盖** |
| 数据 | `dataset.jsonl.x` + `pending.json.x` + `problem.json.space` |
| 降级 | 变量 <1 → 区块显示「没有可投影的变量」；观测 0 → 仍显示待测点；维度 >2 时只画前两维并注明 |
| 误读防护 | 标题/说明必须写「仅反映前两维，其他维度未展开」 |

### 2.3 参数重要性（内联 SVG 条形图，已实现）

| 项 | 内容 |
|---|---|
| 解决 | 哪个条件在推动主目标？（样本 ≥20 才给数值） |
| v1 指标 | 数值变量：与 signed 主目标的 **|Spearman ρ|**（并列取平均秩）；分类/序数/布尔变量：**η²**（组间方差比）；两者都在 0–1 |
| 表达 | 水平条形图，按效应降序取前 8；每行标注指标名、数值、n |
| 数据 | `dataset.jsonl.x/y`（多目标时用主目标） |
| 实现 | 纯 Python 排序/秩/方差，约 120 行；**不 import sklearn** |
| 降级 | n<20 → 显示「样本不足（<20），暂不计算参数重要性（当前 n=X）」；全部缺失/常量 → 显示「没有可分析的变量」 |
| 误读防护 | 固定文案「相关 ≠ 因果，仅反映已评价区域；数值=|Spearman ρ|，分类=η²」 |
| v2（未做） | 置换重要性（需 sklearn GP 或代理模型），作为独立 flag |

### 2.6 多目标 Pareto 区块（内联 SVG，已实现）

| 项 | 内容 |
|---|---|
| 触发 | `problem.json.objective.objectives` 长度 > 1；单目标时整块不输出 |
| 非支配点表 | 条件列（带单位）+ 每个目标原始值（表头带 ↑/↓ 方向）+ 来源；最多显示前 30 个，注明总数 |
| 2D 图 | 前两个目标散点：被支配灰点、非支配蓝点；**阶梯前沿虚线**；参考点只在落在可视域内时画 ×，否则图内提示「参考点在图外」 |
| HV 口径 | 优先 `Objective.ref_point`（原始值，按方向转 signed），否则观测 signed nadir 下方 5%（与 `condopt.pareto.summarise` 完全一致；测试断言 HV 与 `Study.pareto_summary()` 相等） |
| 目标数 > 2 | 只出非支配点表 + 说明「不画二维前沿图」，HV 不计算（显示 —） |
| 数据 | `dataset.jsonl.objectives`（缺任一目标值的观测跳过，并在说明中给跳过数） |
| 常见误读防护 | 图注写清「越好越右/越好越上」；HV 参考点口径写进说明；单目标 study 绝不显示该区块 |

### 2.4 建议批次表（HTML 表格，不出 PNG）

| 列 | 内容 |
|---|---|
| # | 批次内序号（1..batch_size） |
| 条件列 | 每个变量一列，**带单位**（`degC`、`h`、`eq`、`M`）；分类显示原值 |
| 预测均值/σ/采集值 | **v1 不可用**（`suggestions.csv` 无此列），显示「—」并在表下注明「预测列需算法侧输出（v2）」；不得用假数据填充 |
| 与已测点最近距离 | 单位立方体欧氏距离（numpy 计算），<0.05 标「重复风险」 |
| 来源 | `algorithm` 列（如 `bayes`）+ `decisions.jsonl` 最近一条路由的 action |
| 操作 | 「复制整批 TSV」按钮（纯前端 clipboard，无后端） |

> 批次表是**实验执行者唯一真正要用的东西**，优先级高于参数重要性；列宽/单位/复制必须做对。

### 2.5 「为什么这么选」决策时间线（HTML 列表，不出 PNG）

- 数据：`decisions.jsonl` 最近 10 条（当前 schema：`t/action/algorithm/params/reason/payload`）；
- 每行：时间、`action` 徽章（`initial_route`/`auto_reroute`/`override`，未来扩展 `ai_proposal`/`human_approve`）、算法名、`reason` 摘要；`payload` 可折叠（含 `confidence`/`warnings`/`previous`）；
- 无 `decisions.jsonl` → 显示「审计未启用」，整个区块隐藏。

---

## 3. 技术方案与关键取舍

| 决策 | 选项 | 结论 | 理由 |
|---|---|---|---|
| 图表渲染 | 内联 SVG 为主 + 可选 `convergence.png` | **SVG 进 HTML（无图片 base64）、PNG 作为旁路交付物** | 单文件更小（18–26 KB）、文字可选中；PNG 便于贴到聊天/报告 |
| 前端 | 原生 CSS，无 JS | 手写 | 无框架、无 CDN、离线；服务端注入 meta refresh 即可刷新 |
| 数据传递 | 全部内联进 HTML | 必须 | `file://` 下 fetch 本地 JSON 被 CORS 拦 |
| 服务端 | stdlib `http.server`（P2 已实现） | `scripts/serve.py`：ThreadingHTTPServer，绑 127.0.0.1，GET only | 无需安装；`/` 每请求重渲染，`/api/state` 返回计数；无 POST/PUT/DELETE |
| 刷新 | 10s meta refresh | S4 | 简单可靠；不要 WebSocket（标准库无） |
| 字体 | matplotlib 中文 YaHei 回退，SVG 走 CSS 字体栈 | 显式设置 | Windows 有 YaHei；避免方块；PNG 与 SVG 均设置 |
| 进程 | `render` 一次性；`serve` 常驻只读 | S2/S3 一次性；S4 常驻 | serve 不持有大数据集，每次请求重读文件 |

### 3.1 渲染骨架（已实现，`src/condopt/render.py` 约 1600 行）

```python
# src/condopt/render.py（实际结构）
def load_run(run_dir) -> dict:                  # 只读 6 个源文件；坏行/缺文件容错
def build_html(info) -> tuple[str, dict]:       # 内联 CSS + SVG；返回 (html, stats)
    #   _convergence_series / _convergence_svg
    #   _coverage / _coverage_svg（已测 vs 待测）
    #   _importance_summary / _importance_section（n>=20）
    #   _pareto_summary / _pareto_svg / _pareto_section（多目标）
    #   _batch_table_html / _decision_html / _details_html
def render_study(run_dir, out_dir=None, make_png=True) -> dict:
    #   写 dashboard.html（UTF-8 无 BOM）+ 可选 convergence.png（matplotlib Agg）
```

- 纯函数可单测（`tests/test_render.py`）；渲染层**不 import sklearn**；
- 命令行：`py -3 scripts/render.py --run runs\<id> [--out DIR] [--no-png]`；
- 服务：`py -3 scripts/serve.py --run runs\<id> [--port 8765] [--out DIR] [--png]`；
- PNG 失败只写 warning，不影响 HTML 生成；单目标/多目标共用同一渲染入口。

---

## 4. 阶段计划与验收

### S0：真实数据（0.2 人日，已具备）

- `runs/demo` 由 `py -3 scripts/demo.py` 生成：6 变量（T/time/conc/solvent/equiv/inert）、24 次评估、6 轮、batch=4、bayes；`dataset.jsonl` 24 行、`rounds.json` 6 条、`suggestions.csv` 最新 4 行、`decisions.jsonl` 1 条（initial_route）。
- 需要更多数据：改 `scripts/demo.py` 的 `budget`/轮数，或直接 `condopt ask/tell` 续跑（`runs/demo` 可复用）。
- 验收：`Dataset.from_jsonl("runs/demo/dataset.jsonl")` 可读、`best_score` 有值、`rounds.json` 每轮有 best。
- 目的：渲染层从第一天就吃**真实 schema**，避免 mock 字段漂移。

### S1：JSONL → HTML/metrics（0.5 人日）✅ 已并入 renderer

- 无独立 `compute_metrics` 模块：`load_run()` + `build_html()` 内完成 shaping（收敛/覆盖/重要性/Pareto/批次表/决策），并有单测覆盖。
- 验收：`render_study(..., make_png=False)` 在 `runs/demo` 上 **11 ms**（同进程）；不 import sklearn。

### S2：静态 PNG（1 人日）✅ 已实现（task-7）

- `draw_convergence/coverage/importance` 三函数；中文字体设置；Agg。
- 实测目标：3 图合计 ≤2s（当前单图 0.54s，需控制 dpi=110、figsize 7×4）。
- 验收：`figures/*.png` 生成且非空；无 matplotlib GUI 依赖；`figures/` 缺失时自动建。

### S3：单文件 HTML（1.5 人日）✅ 已实现（task-7）

- 模板：header（run 目录 / 算法 / 参数 / 目标方向 / 预算进度 / 轮次）、卡片（收敛、覆盖、重要性、Pareto、批次表、决策时间线、数据摘要）、footer；
- 数据内联、内联 SVG（**无图片 base64、无外部请求**）；
- 交互：只读展示、卡片折叠；**不做表单提交**（表单在 P3）；
- 验收（FEASIBILITY §6 A2–A5 + 新增）：
  - `file://` 打开无报错；断网正常；
  - 24 点 → 全流程 ≤3s；
  - `dashboard.html` ≤500 KB（实测 18.0–25.6 KB）；
  - 观测 <20 时重要性卡片显示占位而非空图；
  - 中文/英文混排不乱码（UTF-8 meta + 无 BOM）。

### S4：本地只读服务（1 人日）✅ 已实现（task-11）

- `py -3 scripts/serve.py --run runs\<id> [--port 8765] [--out DIR] [--png]`；**只绑 127.0.0.1**；
- 端点：`GET /`（每次请求重渲染 HTML，注入 10s meta refresh）、`GET /api/state`（JSON 计数）、`GET /convergence.png`（仅 `--png`）；
- **只接受 GET**：POST/PUT/DELETE/PATCH/OPTIONS/HEAD → 405；路径遍历 → 400；未知路径 → 404；
- 写操作只发生在渲染输出目录（`--out` 或临时目录），**run 目录全程只读**；
- 验收：`GET /` 200；`POST /` 405；`/../problem.json` 400（原始 socket）/404（浏览器归一化后）；run 目录无新文件；多目标 study 同样可服务。

### S5：集成与验收（0.5 人日）✅

- 串起来：每次 `condopt tell` 后跑 `py -3 scripts/render.py --run runs\<id>`；需要自动刷新时用 `scripts/serve.py`；
- 归档：复制 `dashboard.html` + `dashboard` 所需的源 JSONL（`problem/config/dataset/rounds/pending/decisions`）到交付目录；
- 已跑：FEASIBILITY §6 A1–A10 + 多目标/重要性/服务用例（见 §6 与 `tests/test_render.py`）。

---

## 5. 原型示意（本目录已交付）

`prototype/dashboard_mock.html`：

- **单文件、212 行、零依赖、无网络**，双击即可打开；
- mock 数据（40 点收敛序列、10×10 覆盖、8 行批次、6 条决策），字段类型与真实文件一致（变量名/study id 为示意值，真实数据见 `runs/demo`）；
- 演示四件事：① 档位切换（quick/normal/thorough 改变预算/批大小与图表刻度；**注意：档位预设是提案，当前实现只有 budget/batch_size/patience**）② 输入线校验（批大小×轮数 > 预算时红字 E005）③「AI 建议 UCB + 理由」接受/拒绝后写入模拟审计时间线 ④ 建议批次表复制 TSV；
- 它不是最终渲染器，只用于：

  1. 让用户 5 分钟内看到最终形态并给反馈（比读文档快）；
  2. 给 S3 的 HTML 模板提供布局/配色/交互基线；
  3. 作为 FEASIBILITY 形态结论的「可点开的证据」。

> 打开方式：资源管理器双击，或 `start docs\ui\prototype\dashboard_mock.html`；无需 Python/服务。渲染截图：`prototype/dashboard_mock_preview.png`（Edge headless 实测，1400×900）。

---

## 6. 实测记录（随实现回填）

| 指标 | 目标 | 实测（Windows 11 / Python 3.14.8） |
|---|---|---|
| HTML 生成（renderer 内部） | <100 ms | **11 ms**（纯 stdlib；同进程、import 已就绪） |
| CLI 冷启动 `--no-png` | — | **1.3 s**（py launcher ~0.7 s + 脚本；已绕过 condopt 包导入） |
| 全流程 render（含 PNG） | ≤3 s | 同进程复跑 **0.4–0.7 s**；冷进程约 6.8 s（matplotlib + PIL 首次导入，一次性成本） |
| `runs/demo` dashboard | ≤500 KB | **18,471 B（18.0 KB）**；PNG 27,012 B |
| `runs/demo_moo` dashboard | ≤500 KB | **26,222 B（25.6 KB）**；PNG 29,162 B；13 个非支配点 / HV=664.3 |
| 多目标 HV 一致性 | 与 Study 一致 | 测试断言 `Study.pareto_summary()["hypervolume"] == render hypervolume`（参考点口径一致） |
| 重要性 | n≥20 出数值 | demo/demo_moo n=24 均输出（如 T |ρ|=0.76）；n<20 测试显示「样本不足（<20）」 |
| serve `GET /` | 200 + 刷新 | **200**，26,123 B（demo_moo），含 `http-equiv="refresh"`；`POST` → **405**；遍历 → 400/404 |
| serve 只读 | 不写 run 目录 | 渲染只写 `--out`/临时目录；测试断言 run 目录无 `dashboard.html` |
| 首屏打开（Edge `file://`） | <1 s | **<0.5 s**（headless 截图验证，无控制台错误） |
| 依赖 | 不 import sklearn | ✅ `render.py` 只用标准库（matplotlib 仅 PNG 时懒加载）；`scripts/*.py` 绕过包初始化 |
| 测试 | 全绿 | **102 passed**（`tests/test_render.py` 8 项：P0/P1 4 + P2 4） |

---

## 7. 与其它文档/实现的依赖

| 依赖 | 状态 | 若缺失的降级 |
|---|---|---|
| `problem.json` / `config.json`（Study ✅） | ✅ 已实现 | 无（必须）；header 可以显示算法/参数/预算 |
| `dataset.jsonl`（Study ✅） | ✅ 已实现 | 无（必须） |
| `rounds.json`（Study ✅） | ✅ 已实现 | 无 rounds → 用 dataset 累积 best 近似；轮次轴退化为评估序号 |
| `pending.json`（Study ✅） | ✅ 已实现 | 为空时回退 `suggestions.csv`（标注「上一批，可能已 tell」） |
| `suggestions.csv`（Study ✅） | ✅ 已实现（仅最新一批） | 无 pending 时作为兜底表；历史批次需从 `decisions.jsonl`/dataset 重建 |
| `decisions.jsonl`（Study ✅） | ✅ 已实现（简单 schema） | 决策时间线只显示 `action/algorithm/reason`；AI/HITL 字段等扩展 |
| `scripts/render.py` | ✅ 已实现（`--run/--out/--no-png`） | 无（必须） |
| `scripts/serve.py` | ✅ 已实现（GET only、127.0.0.1、`--port/--out/--png`） | 无（P2 必须） |
| 预测列（mean/std/acq） | ❌ 未实现 | 批次表预测列显示「—」；收敛图不画预测带（不伪造） |
| 档位实现（quick/normal/thorough） | ❌ 未实现（只有 budget/batch_size/patience） | 档位控件只读展示；表单校验仍可按 INPUT_LINES_SPEC 实现 |
| 固定条件（L12） | ❌ 未实现 | 批次表无固定条件列；header 提示「固定条件未配置」 |
| 多目标 Pareto（任务侧） | ✅ 已实现（`condopt.pareto` + `Study.pareto_summary`） | 渲染侧兼容无 `objectives` 的旧观测（跳过并计数） |
| 参数重要性（任务侧） | ✅ 渲染侧近似实现 | 不依赖额外算法输出；n<20 显示占位 |
| `condopt render` / `serve` 子命令 | ❌ 未接入 `cli.py`（按 task-7/task-11 约束不改 cli.py） | 用 `scripts/render.py`、`scripts/serve.py`；后续可 1 行注册进 CLI |

---

## 8. 参考

- 形态与成本：[FEASIBILITY.md](FEASIBILITY.md)
- 41 条线（表单字段/错误码）：[INPUT_LINES_SPEC.md](INPUT_LINES_SPEC.md)
- 审计与 AI 闸门：[AI_HARD_INTEGRATION.md](AI_HARD_INTEGRATION.md)
- 交互示意：`prototype/dashboard_mock.html`
