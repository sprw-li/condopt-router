# 文献扫描管线（可复用）

用于把「大量论文摘要」变成「可审计的 router 经验」，而不是一次性脚本。
它同时是一套**可重复运行的数据管线**：配置驱动、分阶段、缓存/断点续跑、每阶段写 manifest。

## 为什么需要它

router 的规则不能只靠手拍。项目里已有：
- `scripts/benchmark.py`：合成基准上的算法对照；
- `scripts/calibrate_router.py`：在基准矩阵上做 LOPO 权重校准。

文献管线补的是第三类证据：**真实论文里人们如何选择优化方法**（BO / TPE / DoE / 多目标 / 约束 / 多保真 / 自动化平台）。
输出是 router 的“经验先验”，需要再用基准/代理战役验证，不能直接当结论。

## 管线阶段

```
fetch  ->  Europe PMC (默认 2024-2026、开放获取) -> raw/<query>.json -> corpus.jsonl
extract ->  规则抽取标题/摘要特征 -> papers.jsonl + papers.csv
chunk   ->  按 chunk_size 切分 -> chunks/chunk_XX.jsonl（给子代理批量阅读）
report  ->  统计 + AGGREGATE.md + experience.json
```

所有阶段都幂等/可续跑：

```powershell
py -3 scripts/literature_scan.py fetch   --config configs/literature_scan.json
py -3 scripts/literature_scan.py extract --config configs/literature_scan.json
py -3 scripts/literature_scan.py chunk   --config configs/literature_scan.json
py -3 scripts/literature_scan.py report  --config configs/literature_scan.json
# 或一次跑完：
py -3 scripts/literature_scan.py all     --config configs/literature_scan.json
```

- `fetch` 默认复用 `raw/<slug>.json`，`--refresh` 强制重拉；
- `--queries` 可临时覆盖配置里的查询；
- 每个阶段写 `fetch_manifest.json / extract_manifest.json / chunks/index.json`，
  记录参数、时间、计数，便于复现与审计。

## 配置

`configs/literature_scan.json`：

| 字段 | 含义 |
|---|---|
| `output_dir` | 输出目录（默认 `research/literature_scan`） |
| `since` / `until` | 出版日期窗口 |
| `open_access` | 只取开放获取 |
| `page_size` / `max_per_query` | 每个查询最多抓多少条（默认 100 / 60） |
| `sleep_seconds` | 查询间隔（礼貌限速） |
| `chunk_size` | 每个子代理处理的论文数（默认 40） |
| `queries` | 检索式列表；换领域只改这里 |

## 产出

```
research/literature_scan/
├─ raw/*.json                 # 每个 query 的原始 API 结果（缓存）
├─ corpus.jsonl               # 去重后的论文元数据 + 摘要
├─ papers.jsonl / papers.csv  # 规则抽取后的结构化字段
├─ chunks/chunk_XX.jsonl      # 子代理输入（40 篇/块）
├─ labels/labels_XX.jsonl     # 子代理逐篇路由标签
├─ lessons/lesson_XX.json     # 子代理经验/模式
├─ AGGREGATE.md               # 规则抽取统计
├─ ROUTER_EXPERIENCE.md       # 标签聚合 + 候选规则
└─ router_experience.json     # 机器可读版本
```

## 子代理批量阅读

1. `chunk` 生成 `chunks/*.jsonl` 后，用 workflow/子代理逐块阅读：每篇输出一条标签
   （`space_type`、`objectives`、`constraints`、`noisy`、`automation`、`literature_method`、
   `router_recommendation`、`confidence`、`evidence`），并写本块 `lesson_XX.json`。
2. 聚合：

```powershell
py -3 scripts/literature_experience_report.py --input research/literature_scan
```

输出 `ROUTER_EXPERIENCE.md`：标签分布、方法一致性、空间类型 × 推荐算法交叉表、
候选规则（带证据条数与置信度）与子代理经验。

## 本轮实际结果（2026-11）

- 40 条查询，去重 **1357 篇**（2024–2026 开放获取，摘要覆盖 88%）
- 34 个 subagent × 40 篇 → **356 条标签** / 34 份 lesson
- 高/中置信 216 条用于权重校准：标签一致率 0.352 → **0.497**（留出 20% 为 0.4595）
- 候选权重：`configs/router_weights_literature.json`（非默认）
- 经验报告：[`research/literature_scan/ROUTER_EXPERIENCE.md`](../research/literature_scan/ROUTER_EXPERIENCE.md)
- 落地改动与证据：[`research/literature_scan/APPLIED_CHANGES.md`](../research/literature_scan/APPLIED_CHANGES.md)

## 边界

- 只用标题/摘要，很多论文不写清变量数、目标数、噪声；因此字段允许 `null`，并在
  `confidence` 里标低/中/高。
- 抽取是规则 + LLM 判断，不是人工复核；只能作为 **router 先验**。
- 改变 router 默认前必须过：
  1. `py -3 -m pytest -q`；
  2. `scripts/benchmark.py` / `scripts/calibrate_router.py` 的 LOPO 对照；
  3. 真实或代理战役再验证（例如 `examples/paper_campaigns/`）。

## 路由校准中的多保真 + 约束指标

`collect_records` 对 multi_fidelity 问题走 `_run_multifidelity`；若问题同时带硬约束，
使用 `best_feasible_full_fidelity`，metric_name = `cost_aware_feasible_best`。
`BenchmarkProblem.fidelity_fn` 提供保真度感知评估（mf_suzuki / mf_constrained）；
collect_records 会依据 problem 的 `multi_fidelity` tag 自动把 router 上下文设为多保真，
避免校准把多保真问题当成单保真来路由。

