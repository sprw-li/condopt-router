# 可视化界面可行性评估（condopt / Project6）

- 版本：v0.1（2026-11-05）
- 任务：共享任务板 task-2
- 适用范围：把 `condopt`（ask/tell 条件优化算法库 + skill router）的可视化从 0 做到可用
- 事实基础：本机实测（第 3 节）+ 当前 `src/condopt` 已实现代码（`base.py` 的 `AlgorithmSpec/ParamSpec`、`registry.py`、`space.py`、`dataset.py`、`algorithms/{sampling,bayes}.py`）

---

## 0. 结论（先看这里）

**推荐形态：CLI 内核 + 零依赖静态 HTML 看板（主形态）+ 可选本地 stdlib Web 只读服务（增强形态）。**

一句话：**执行只走 `condopt` CLI，界面只读 `runs/<study_id>/` 里的 JSONL/PNG，渲染层不允许产生新状态。**

| 决策 | 内容 | 理由 |
|---|---|---|
| 主形态 | 单文件 `dashboard.html`（数据内联，双击用 Edge 打开） | 零新依赖、离线可归档、可直接进 DSH GUI 文件卡片；实测环境缺 plotly/fastapi/streamlit，装包违背前提 |
| 增强形态 | `condopt serve`（stdlib `http.server`，只绑 127.0.0.1，只读） | 支持轮询刷新；不需要任何框架，~120 行 |
| 入口形态 | DSH Web GUI 里以「文件/卡片/链接」打开看板 | 实测未授权 GET 返回 401，**不能 iframe 嵌入 DSH 页面**，只能当外部入口 |
| 不推荐 | 桌面 GUI、Jupyter、TUI、IDE 插件、聊天内嵌控制台 | 见第 2 节评分与第 2.3 节逐项结论 |
| 首版工期 | **3 人日**（P0 静态 PNG 0.5d + P1 单页 HTML 1.5d + P2 本地服务 1d） | 复用 matplotlib（已装）、Dataset JSONL（已实现）、`ParamSpec` schema（已实现，可自动生成表单） |
| 关键风险 | file:// 下 `fetch()` 本地 JSON 被 CORS 拦、DSH 401、matplotlib 首图 0.66s、中文路径/编码 | 见第 5 节，均有确定缓解手段 |

**可执行建议：本周先做 P0+P1（2 人日），产出「一次真实 condopt campaign 的 dashboard.html」，不写任何服务端；跑通再决定要不要 P2。**

---

## 1. 需求与硬约束

### 1.1 使用场景（按频率排序）

| 场景 | 用户 | 频率 | 界面需要提供 |
|---|---|---|---|
| 看「下一批该做哪些条件实验」 | 实验执行者 | 每轮（分钟级） | 建议批次表（可复制、带单位）、本轮预测均值/方差、与已测点的距离 |
| 看「优化有没有在收敛」 | 算法研究者 | 每轮 | 收敛曲线（观测散点 + best-so-far）、预算消耗 |
| 看「空间是否被覆盖」 | 算法研究者 | 每 3–5 轮 | 2D 投影散点 / 占用热图 |
| 看「哪个条件在起作用」 | 研究者 / AI orchestrator | 每 10 轮 | 参数重要性（样本 ≥20 才显示） |
| 看「为什么这轮这么选」 | 审计 / 合作者 | 按需 | router 决策 + AI 覆写理由 + 人审记录 |
| 把一次 study 归档给别人 | 合作者 | 一次性 | 单文件 HTML，离线可开 |

### 1.2 硬约束（本机实测，不可协商）

1. Windows 11（build 26200），Python 通过 `py -3` → **3.14.8**（`py -3`）。
2. 已装：numpy 2.5.4、scipy 1.18.1、scikit-learn 1.9.1、pandas 3.0.6、matplotlib 3.11.2、PyYAML 6.0.3、rich。
   未装：**plotly、fastapi、streamlit、jupyter、textual、PySide6、pydantic**（逐个 `import` 实测 `ModuleNotFoundError`）。
3. 本轮任务明文「不安装包」→ 一切方案必须**只用标准库 + 上述已装包**。
4. 本机有 DSH Web GUI（`http://127.0.0.1:59104`），但未带会话请求 `GET /` 返回 **HTTP 401**（实测）。
5. 一次 study 预算 8–20000 次评估、时长分钟到数天 → 看板必须支持「增量刷新」，不能每次全量重算 GP。
6. 产物必须可离线、可归档、可审计（合作者/审稿/实验室留档），不能依赖网络 CDN。
7. 已有代码事实：数据真源是 `Dataset` 的 JSONL（`Observation.to_dict()`），算法参数 schema 是 `AlgorithmSpec.params` 的 `ParamSpec` 列表 → **界面应由 schema 生成，而不是手写字段**。

---

## 2. 七种形态对比

### 2.1 评分矩阵（1=最差，5=最好；加权总分 5 分制）

权重：落地成本 25%、复用现有栈 15%、交互能力 20%、可视化表达 20%、可维护/可移植 10%、可归档/可嵌入 10%。

| 形态 | 落地成本(.25) | 复用(.15) | 交互(.20) | 表达(.20) | 维护(.10) | 归档/嵌入(.10) | **加权总分** |
|---|---|---|---|---|---|---|---|
| 静态 HTML 报告（零服务） | 5 | 5 | 2 | 5 | 5 | 4 | **4.40** |
| 本地 Web（stdlib 只读看板） | 4 | 4 | 4 | 5 | 5 | 5 | **4.35** |
| 聊天内嵌（DSH GUI 卡片/链接） | 5 | 5 | 2 | 3 | 4 | 5 | **4.05** |
| 纯 CLI | 5 | 5 | 1 | 1 | 5 | 5 | **3.60** |
| 桌面 GUI（Tkinter / PySide6） | 3 | 2 | 4 | 3 | 3 | 1 | **2.90** |
| Jupyter Notebook | 2 | 3 | 3 | 4 | 3 | 1 | **2.65** |
| TUI（终端面板） | 2 | 1 | 2 | 1 | 3 | 4 | **1.95** |
| IDE 插件（VS Code） | 1 | 1 | 3 | 2 | 2 | 2 | **1.75** |

### 2.2 落地成本明细（人日 = 一个熟悉本仓库的人，在本机，不含算法开发）

| 形态 | 技术栈 | 新增依赖 | 首版人日 | 实测启动/首屏 | 维护面 | 结论 |
|---|---|---|---|---|---|---|
| 静态 HTML | matplotlib 3.11.2 + 标准库 + 手写 SVG/CSS | 0 | **1.5–2** | 生成 <2s；Edge 打开 <0.5s | 单文件模板 + 渲染脚本 | ✅ 主形态 |
| 本地 Web | `http.server`/`wsgiref` + 原生 JS | 0 | **+1–1.5** | 服务启动 <0.2s；首包 0.6s | 再 +1 个端点、1 个进程 | ✅ 增强 |
| 桌面 GUI | Tkinter 9.0（已可用）/ PySide6（未装） | Tkinter 0；PySide6 约 200MB+ | 8–12（Tk）/ 12–18（Qt） | Tk 首次建窗 0.9s | 布局/DPI/字体/打包（PyInstaller） | ❌ v1 不做 |
| Jupyter | 需装 notebook/jupyterlab（约 300–500MB） | 有 | 3–4（装环境另计） | kernel 启动 2–5s | notebook 与仓库两套版本 | ❌ 不交付，仅分析用 |
| TUI | rich（已装）/ textual（未装）/ Windows 无 curses | 0（rich） | 4–6 | 秒开 | 中文宽度、ANSI、日志混排 | ❌ 价值低于 CLI 摘要 |
| 聊天内嵌 | DSH GUI 文件卡片/链接 + 不嵌 iframe | 0 | 0.5 | 即开 | 依赖 DSH 版本 | ✅ 只做入口 |
| IDE 插件 | VS Code API + Node 22.23.4 构建链 | 有（Node 已在，但需扩展工程） | 10–15 | 秒开 | Webview/打包/多 IDE | ❌ 不做 |

> 注：Jupyter 实际未安装（`import jupyter` 失败），pandas 3.0.6 虽在但本方案不依赖它。

### 2.3 逐项结论

- **静态 HTML**：完全命中「离线/归档/零依赖/不装包」。缺点无实时交互，但本轮用户要的是「可控参数 + 看结果」，不是多人协同。**首要交付**。
- **本地 Web**：只读服务可以解决「跑完一轮要手动重开文件」的痛点（页面 5s 轮询 `/api/state`）。**但所有写操作仍走 CLI**，看板不提供「提交实验」入口，避免把执行权和界面耦合。次优先。
- **聊天内嵌（DSH）**：DSH GUI 是现成入口（用户已经开着 59104），但（a）未授权请求 401，外部页面无法反向嵌入 DSH；（b）DSH 页面里套本地文件依赖 file 访问策略。可行做法只有：**把 `dashboard.html` 当交付物，用户从 DSH 卡片/文件管理器打开；或聊天里回一个 `file:///.../dashboard.html` 链接**。成本 0.5 人日，值得做。
- **桌面 GUI**：Tkinter 在本机可用（Tk 9.0，屏幕 1600×1000，首窗 0.9s），但做「表单 + 4 张图 + 表格」要比 HTML 多 3–5 倍代码，且 PySide6 未装、打包体积/杀软/DPI 都要踩坑。**v1 明确不做**。
- **Jupyter**：环境未装；且 notebook 会把「研究分析」和「交付看板」混在一起，版本/路径漂移。**只在算法研究阶段由研究者自行使用**。
- **TUI**：rich 已装，但 Windows 终端中文宽度与 ANSI 兼容性差，图表表达力弱。**用已实现的 `condopt report`（文本摘要：预算/当前最优/下一轮建议）替代**。
- **IDE 插件**：Node/pnpm 都在，但为一个内部工具维护 VS Code 扩展是纯投入无回报。**不做**。

---

## 3. 本机环境实测数据（2026-11-05）

| 项目 | 实测值 | 对方案的直接影响 |
|---|---|---|
| Python | 3.14.8（`py -3`），npm 侧 Node v22.23.4 / pnpm 12.16.1 | 只用标准库即可运行 |
| numpy / scipy 导入 | 317 ms / 50 ms | 渲染进程可放心用 |
| **scikit-learn 导入** | **1639 ms**（1.9.1） | ⚠️ 看板渲染**不要** import sklearn；参数重要性用 numpy 相关替代 |
| pandas 导入 | 0 ms（已被 sklearn 预热）/ 独立约 300 ms | 非必需，不引入 |
| matplotlib 导入 + Agg 首图 | 153 ms + 660 ms（41 KB PNG） | 4 图约 2.6s，可接受；首图慢是字体缓存 |
| matplotlib Agg 稳定出图 | **542 ms/图**（7×4 in，110 dpi） | 每轮刷新只重画 1–2 张，避免全量 |
| Tkinter | 可用，Tk 9.0，屏幕 1600×1000，建窗 899 ms | 桌面 GUI 可行但非首选 |
| stdlib `http.server` 本地 GET | 595 ms（loopback 首包，含 urllib 开销） | 只读看板刷新够用 |
| DSH GUI `GET http://127.0.0.1:59104` | **HTTP 401**（未授权） | 不能做 iframe/跨页嵌入；只能当外部入口 |
| plotly / fastapi / streamlit / jupyter / textual / PySide6 / pydantic | **均未安装** | 任何方案不得依赖 |
| 默认浏览器 | Edge（`MSEdgeHTM`） | `dashboard.html` 双击即开 |
| 现有实现测试 | `py -3 -m pytest -q` → **37 passed**（space/algorithms/router/study/CLI） | 渲染层需新增同类测试；执行内核可信 |
| 现有样例数据 | `runs/demo`：`dataset.jsonl` 24 行 / `rounds.json` 6 轮 / `suggestions.csv` 4 行 | P0 可直接用，不需要先造数据 |

---

## 4. 推荐架构与分阶段

### 4.1 三层架构（产物即接口）

```
┌───────────────────────────── 执行层（唯一写者） ─────────────────────────────┐
│  condopt CLI（已实现：spec / route / init / ask / tell / report / override） │
│  拟新增：render / serve / audit verify（本方案要补的 3 个子命令）            │
│  依赖：registry + AlgorithmSpec/ParamSpec + Dataset JSONL + router（硬规则） │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │ 写
                                   ▼
┌───────────────────────── 产物层（唯一真源，只追加） ─────────────────────────┐
│ runs/<study_id>/（Study 已实现这套命名，见 src/condopt/study.py）            │
│   problem.json          # space/objective/seed/budget/meta                    │
│   config.json           # algorithm/params/context/batch_size/auto_route/patience │
│   dataset.jsonl         # Dataset.append_jsonl（已实现）                      │
│   pending.json          # 待测建议（一次 ask 的批次）                         │
│   rounds.json           # 每轮 best/n/algorithm（收敛曲线直接可用）           │
│   suggestions.csv       # 最新一批建议（人/机器人友好；暂无预测列）           │
│   decisions.jsonl       # router 决策与 override（含 reason）                 │
│   opt_state.json        # 优化器内部状态（断点续跑）                          │
│   metrics.json          # 渲染层缓存（拟新增，派生数据）                      │
│   figures/*.png         # matplotlib 静态图（拟新增）                         │
│   dashboard.html        # 单文件，数据内联（拟新增）                          │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │ 只读
                                   ▼
┌───────────────────────────── 渲染层（无状态） ───────────────────────────────┐
│  P1 静态单文件 HTML（PNG/SVG + 内联 JSON）  ← 主                           │
│  P2 `condopt serve`（stdlib，127.0.0.1，只读 /api/state，5s 轮询）← 增强     │
│  P4 DSH GUI 文件卡片/链接（不嵌 iframe）                                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

设计要点：

1. **界面字段来自 schema**：`registry.describe_all()` 返回的 `AlgorithmSpec.to_dict()` 已含 `name/type/default/choices/low/high/description/unit`，表单直接按它渲染，避免「文档写一套、代码是另一套」。见 `INPUT_LINES_SPEC.md` 第 3 节。
2. **渲染层不 import sklearn**（省 1.6s），只 import numpy + matplotlib；如果不需要 PNG 而用 SVG，连 matplotlib 都可以不进渲染进程。
3. **单文件看板数据内联**：`file://` 页面里 `fetch('metrics.json')` 会被 CORS 拦截，因此把 JSON 塞进 `<script type="application/json">`。代价是文件变大（4 图 + 80 点 < 500 KB），可接受。
4. **DSH 集成只做入口**：不建议、也无法 iframe；把 HTML 作为 artifact 交给用户，或在聊天里给 `file:///` 链接。

### 4.2 分阶段（每阶段可独立验收）

| 阶段 | 交付 | 人日 | 依赖 | 验收 |
|---|---|---|---|---|
| P0 | `condopt render --run runs/<id>`：读 `dataset.jsonl` + `rounds.json` → 2 张 PNG（收敛/覆盖）放进 `figures/` | 0.5 | 已有 numpy/matplotlib/JSONL（`runs/demo` 有 24 条可直接用） | 用 24 条真实观测出图，不 import sklearn |
| P1 | `dashboard.html`（4 图 + 建议批次表 + 决策日志） | 1.5 | P0 | Edge 双击可开，全程离线，<500KB |
| P2 | `condopt serve`（只读、127.0.0.1:8765、5s 轮询、无 POST） | 1.0 | P1 | 一轮新观测后 ≤5s 页面自更新；断网无影响 |
| P3（可选） | 表单化参数页 + 「生成命令并复制」（仍由用户执行 CLI） | 2–3 | P1 + INPUT_LINES_SPEC | 41 条线按档位生成配置，校验规则前端可复现 |
| P4（可选） | DSH GUI 卡片/链接 | 0.5 | P1 | 从 DSH 打开 HTML，无 401 依赖 |

**总建议：先 P0+P1（2 人日），P2 视使用频率再上；P3/P4 不进 v1。**

### 4.3 明确「不做」清单

| 不做 | 理由 |
|---|---|
| Electron / PySide6 桌面应用 | 未装、200MB+ 依赖、打包与杀软风险，收益低于 HTML |
| Jupyter widget | 环境未装；notebook 不是交付物 |
| TUI 图表 | 表达力弱，用 CLI 摘要替代 |
| 聊天里直接「点按钮跑实验」 | 把执行权和 UI 耦合，审计与复现都变难；执行必须走 CLI |
| 实时多人协同编辑 | 无此需求，成本极高 |
| 前端组件库 / CDN | 离线要求 + 单文件交付，手写 CSS/JS |
| iframe 嵌入 DSH GUI | 实测 401 + 跨源/文件策略，不可行 |

---

## 5. 风险与缓解

| # | 风险 | 概率 | 影响 | 触发信号 | 缓解 |
|---|---|---|---|---|---|
| R1 | `file://` 页面 `fetch()` 本地 JSON 被 CORS 拦 | 高（必现） | 看板空白 | 浏览器控制台 `Not allowed to load local resource` | 数据内联进 HTML；JSON 作为 `<script type="application/json">` |
| R2 | DSH GUI 401/跨源 | 已确认 | 无法内嵌 | 未授权 GET 返回 401 | 只做文件卡片/链接；DSH 内打开本地 HTML 由用户手动确认 |
| R3 | matplotlib 首图 0.66s + 字体缓存缺失 | 中 | 首轮慢 | 冷启动耗时 >2s | 首次 `render` 预热；进程常驻时不重复 import |
| R4 | 看板进程 import sklearn 拖慢 1.6s | 中 | 启动慢/内存 | `top` 看到 sklearn | 渲染层只 import numpy/matplotlib；重要性用 Spearman |ρ| |
| R5 | 中文路径 / 编码（Windows 默认 GBK 终端） | 中 | 文件打不开/乱码 | `UnicodeDecodeError`、乱码 | 所有 JSON/HTML 显式 `encoding="utf-8"`；CLI 打印用 `sys.stdout.reconfigure(encoding="utf-8")`；文件名禁中文 |
| R6 | 长跑 campaign 看板数据膨胀（20000 点） | 低 | 页面卡顿 | >5000 观测 | `metrics.json` 预降采样；曲线只画 best-so-far + 每轮代表点 |
| R7 | 暴露实验控制导致误操作 | 中 | 浪费实验成本 | 有人要求 UI 直接提交 | P1/P2 只读；P3 也只生成命令不执行；预算硬上限见 INPUT_LINES_SPEC |
| R8 | 单文件 HTML 体积失控 | 中 | 邮件/聊天发不出 | >2MB | PNG 用 110 dpi；4 图总预算 ≤150KB；超限改 SVG |
| R9 | 图表结论误导（早期参数重要性） | 高 | 错误决策 | 样本 <20 就显示重要性 | 明确规则：观测 <20 不画重要性、<8 不画覆盖热图，显示「样本不足」 |

---

## 6. 验收清单（可直接执行）

| # | 检查 | 通过标准 |
|---|---|---|
| A1 | 用 `py -3 scripts/demo.py` 生成 `runs/demo`（24 点 / 6 轮，合成目标） | `dataset.jsonl` 24 行、`rounds.json` 6 条，每行含 `x`、`y`、`source` |
| A2 | `condopt render --run runs/demo` | 生成 `figures/convergence.png`、`coverage.png`、`dashboard.html` |
| A3 | 双击 `dashboard.html`（Edge，断网） | 4 图 + 表格 + 决策日志全部显示，控制台无网络错误 |
| A4 | 文件体积 | `dashboard.html` ≤ 500 KB，单文件不含外部 `<script src>` |
| A5 | 渲染耗时 | 60 点 → 全流程 ≤3s；只刷 1 图 ≤1s |
| A6 | 依赖 | `py -3 -c "import condopt... render"` 无 sklearn/plotly/fastapi 导入 |
| A7 | 增量 | 追加 8 条观测后重新 render，收敛曲线末端更新，无需重跑算法 |
| A8 | 样本不足降级 | 观测 <20：重要性图显示「样本不足（<20）」占位；<8：覆盖图同样降级 |
| A9 | 安全 | 页面无 `POST`、无外部请求、无 eval 注入；所有文本 HTML 转义 |
| A10 | 编码 | 中文标签在 Edge/Chrome 正常；PowerShell 管道输出 UTF-8 无乱码 |

---

## 7. 与现有实现的接口核对（2026-11-05 实测，`src/condopt` 已落地）

> 结论：**执行内核、路由、Study 持久化、CLI、审计骨架都已存在；缺的只有「渲染」与「AI 闸门扩展」。** 因此 P1 的输入是现成 JSONL，不需要等后端。

| 能力 | 现状 | 看板怎么用 | 还缺什么 |
|---|---|---|---|
| CLI | ✅ `spec/route/init/ask/tell/report/override`（`src/condopt/cli.py`，支持 `--json`） | 数据生产；看板不调用 | 新增 `render`（P0/P1）、`serve`（P2） |
| 空间/目标 | ✅ `problem.json`：`space`、`objective`（name/direction/objectives/constraints）、`seed`、`budget`、`meta` | header、轴标签、单位 | 固定条件（L12）无处存放 |
| 策略/超参 | ✅ `config.json`：`algorithm`、`params`（经 `AlgorithmSpec` 校验）、`context`（router 特征）、`batch_size`、`auto_route`、`patience` | header 显示算法/参数；表单 schema | 41 条线尚未 flatten 成独立字段（见 INPUT_LINES_SPEC §1.5） |
| 观测数据 | ✅ `dataset.jsonl`（`Observation.to_dict()`：x/y/objectives/constraints/fidelity/cost/t/source/meta） | 收敛、覆盖、重要性 | 无（已足够） |
| 轮次 | ✅ `rounds.json`：`round/t/n/algorithm/best` | 收敛曲线的「每轮 best」；停滞检测 | 无（已足够） |
| 建议批次 | ✅ `suggestions.csv`：`id,batch,algorithm,<变量列>`；`pending.json` 含同一批 | 批次表、复制 TSV | **缺 `predicted_mean/predicted_std/acquisition`**（需算法侧输出，或 v1 留空） |
| 审计 | ✅ `decisions.jsonl`：`t/action/algorithm/params/reason/payload`；`Study.override()` 强制要求 reason | 「为什么这么选」时间线 | AI 提案、HITL、哈希链（AI_HARD_INTEGRATION §4 的扩展） |
| 状态 | ✅ `opt_state.json`、`pending.json` | 无需展示（可选 footer） | — |
| 测试 | ✅ 37 passed（`py -3 -m pytest -q`） | — | 渲染层无测试（P0/P1 要补） |
| DSH | ⚠️ 401 未授权 | 只做文件卡片/链接 | 无（不嵌 iframe） |

最小改动清单（按优先级）：

1. `src/condopt/render.py` + `condopt render --run runs/<id> [--html-only] [--open]`（P0/P1，唯一必须新增的代码）；
2. `Study` 增加 `predicted_mean/predicted_std/acquisition` 写入 `suggestions.csv`（可选，提升批次表价值）；
3. 固定条件（L12）字段：`problem.json` 增 `context: {...}`，suggestions 输出时 merge（不放进 `SearchSpace`）；
4. `decisions.jsonl` 扩展字段（向后兼容：新字段可选，旧消费者忽略）；
5. `condopt serve`（P2，可选）。

---

## 8. 参考（项目内）

- 参数与档位规格：[INPUT_LINES_SPEC.md](INPUT_LINES_SPEC.md)
- AI 覆写/审计/回退：[AI_HARD_INTEGRATION.md](AI_HARD_INTEGRATION.md)
- 原型计划与数据契约：[PROTOTYPE_PLAN.md](PROTOTYPE_PLAN.md)
- 可交互示意：`docs/ui/prototype/dashboard_mock.html`（单文件、零依赖、mock 数据）
