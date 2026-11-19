# 会话状态（Round 310 结束 / 目标轮 312/320）

> 新会话唯一上下文来源：只写结论与路径，细节在仓库里（都能用命令复现）。
> 项目：`condopt` —— 实验条件优化“前置技术”（AI 全自动实验室），agent 用 skill `condopt-router` 选算法取样。

## 目标

用户原话：**“保持继续，不要停止，按照你的想法改进，进行循环。”**
每轮固定闭环：**改代码/文档 → 加测试 → `py -3 scripts/release_check.py`（19 门禁）→ 全绿 → 追加 `research/ralph/LOG.md`**。
验收：19/19 门禁 + `pytest` 全绿 + LOG 有本轮结论。

## 发布

- 公开仓库：**https://github.com/sprw-li/condopt-router**（336 文件 / 3.57 MB；文献原始语料不入库，可由 `scripts/literature_scan.py` 重生成）。
- 发布验收口径：**必须在新克隆里跑 `py -3 -m pytest -q` 全绿**（本地全绿不算数）。
- 去本机化（Round 310）：跟踪文件与**全部历史**已无本机盘符路径（40 → 0）；
  共享技能库/eval 脚本路径改为可发现 + 环境变量覆盖；净化脚本见 `_work/sanitize_for_public.py`。

## 已完成

- **规模**（Round 308 实测）：18 算法、**19 门禁**、`pytest` **585 passed**、**22 路由场景**、
  192 上下文审计（违规 0）、**47/47 决策检查**、**25 条可重跑证据声明**、12 条登记决策、**14 条负结果**。
- **能力**：算法库/路由器/决策层（`prepare` 六段建议：算法·停止·可行率·先验·交付口径·噪声处置）/
  执行器适配器/多目标·多保真·约束/迁移/交付自查/skill（五条文档流程有可执行 smoke）。
- **最有价值的实测结论**（均可重跑，详见 `docs/DEFAULTS.md`、`research/README.md`）：
  - 真实 ART 六点暖启动 **12/12 命中 vs 冷启动 0/12**（目标侧是代理 oracle）；
  - **Round 277 重大修复**：导入先验会污染平台期判定 → 暖启动在 16–19/30 次提前停；
    修复后好源暖启动从 **−31.5 → +6.0~+9.2**（`research/TRANSFER_TOP_K.md`）；
  - 暖启动收益**取决于预算口径**：纯次数 +6~+9；硬成本预算 −5（`research/TRANSFER_COST.md`）；
    坏源负迁移随 top_k 放大 → 策略按 audit 分级；
  - **噪声处置地形依赖**：混合/离散型 EI 最稳、UCB 更差；光滑低维 hartmann3 UCB 更好 →
    只保留方向一致的“复孔+小批量+基线臂”（`research/NOISE_POLICY.md`）；
  - 硬可行域 constrained **92.97 vs bayes 87.80**（bayes 仅 1/3 找到可行点，`research/CONSTRAINED_SECOND_DATASET.md`）；
  - `budget_first` 40–120 更优（92.9/98.8/100.6/102.2 vs 71.8/65.6/76.9/65.6）且交付自查全通过；
  - 极小预算保留代理优于纯空间填充；cmaes 规则是地貌依赖而非纯维度。
- **修掉的真实缺陷**：状态持久化遮蔽 resume（mfbo/momfbo/ehvi_nd）、迁移观测继承 `purpose`、
  超时阻塞循环、停滞中断不写日志、`transfer_from(top_k=0)` 静默导入、mfbo 忽略 `objective_std`、
  交付自查判定对象错配、基准脚本写死 `range(60)` 截断、**先验污染平台期（Round 277）**。

## 优化循环（loop-optimize-loop 协议 + council）

- **绑定**：gate = `py -3 scripts/release_check.py`（必须全绿，现 **19/19**）；
  metric = `py -3 scripts/backlog_metric.py`（未结待办数，越低越好）。
- 进度：基线 **9** → A6 门禁诚实性 → **8** → A5 qnehvi baseline/逐目标噪声 → **7**。
- council（`council_review`）：默认模型组合连续 2 次“阶段失败”，改用可用模型后成功；
  它给的 2 条“清单外缺陷”**经本地复现全部证伪**（qnehvi `_Y`/`_noise_std` 对齐正常；
  `export` 的 `con.*` 列有值，空列是 MOO 的 `y`）。教训：无源码访问的评审只算**推断**。
- 进度更新：A6 → 8、A5 → 7、**A10 → 6**（连续/对数变量边界校验 + 显式 clip 留痕）。
- **门禁（干净 cp936 环境、无 PYTHONUTF8）19/19 通过**，pytest 全绿，verify_evidence 25/25。
- 集成期修掉 2 个真实缺陷：子进程 UTF-8 解码（scripts 消费端/生产端 + cli.main）、EVIDENCE_AUDIT 旧 claim 名。
- **未结 0 项**：A4/A7/A8/A9/A11/A12 已于 Round 308 全部完成（workflow 扇出 + Lead 集成），
  每项均有验收命令与测试（见 `docs/IMPROVEMENT_BACKLOG.md` 的“已修”行与 `_work/optimize-loop/ledger.tsv`）。

## 外部审查与同类项目扫描（2026-11-11，重要）

- 桌面审查报告 外部独立审查报告（非公开） 与 5 份独立角度评审
  + 78 个 GitHub 项目的扫描结果，已分别落成：
  `research/EXTERNAL_REVIEW_RESPONSE.md`、`research/EXTERNAL_BENCHMARK_SCAN.md`、
  `docs/IMPROVEMENT_BACKLOG.md`（**A1–A12 全部已修、未结 0 项**；仅剩 4 项“不建议做”并附理由；每条都带验收命令）。
- **已修并带回归测试**（`tests/test_review_regressions.py` 等）：
  1. 路由“报告 tpe、实际 20 个 Sobol 点、TPE 零次”→ `_clamp_initial_design`；
  2. 目标判据方向（min + target 误报达标）；
  3. 热启动宽限期标记不持久（CLI 单进程/条）→ `warm_start.json`；
  4. **cmaes 在 Study 路径完全不自适应**（旧 opt_state 快照覆盖 + Study 从不 tell）→
     `replay_history_on_load` + 载入后重放历史；实测批次均值 0.49→0.41；
  5. 决策矩阵区分 FAIL 与 BLOCKED（环境不可写临时目录）；
  6. 版本漂移（README v0.31 → 读 pyproject）、ARCHITECTURE 的 tell 描述；
  7. 算法命名夸大（tpe/qehvi/qnehvi/cmaes）→ catalog §14 边界表 + README 限定词；
  8. 场景矩阵键冲突漏检 → 行级匹配。
- **backlog A1/A2/A3 已修**：`init` 存在性守卫（`tests/test_init_guard.py` 4 passed）；momfbo incumbent 权重一致性；**归档数字门禁**（`docs/GOLDEN_NUMBERS.json` + `scripts/check_golden_numbers.py --self-test`，端到端负对照实测
  clean=0→corrupted=1→restored=0）。
- 桌面报告里的 P0“门禁不全绿”经复核**不成立**（失败的两条规则恰是唯一需要可写临时目录的两条，
  受控实验可复现；本仓库可写环境下 19/19、`pytest` 全绿）。

## 待办

- [ ] 继续循环（用户要求不停）。优先做**能改变默认值决策**的实测。
- [ ] `research/TRANSFER_MOO.md` 已按修复后重测（+0.73% HV、前沿 +21%）；`TRANSFER_COST` 结论不变（−5）。
- [ ] 仍未复核的旧数字：`research/PRIOR_RELIABILITY.md`（测于平台期修复前，收益可能被低估）。
- [ ] 加默认值/规则时必须**同时**：`docs/DEFAULTS.md` 一行 + `docs/DEFAULT_EVIDENCE.json` 一条
      （≥2 数据集或写清 tradeoff）+ `scripts/verify_evidence.py` 声明（有可重跑脚本时）+ 决策矩阵断言。

## 关键文件路径

- `scripts/release_check.py` — 19 项门禁入口（每轮必跑）；门禁清单由 `scripts/build_status.py` 自动读取。
- `src/condopt/study.py` — 核心：`next_action`/`delivery_checklist`/`_recent_best_scores`（本地 best）
  /`_stagnation_guard_rounds`/`transfer_from`/`default_confirm_n`。
- `src/condopt/prepare.py` — 六段开工建议（含 `noise_advice`、`delivery_tradeoff`）。
- `src/condopt/router.py` — 路由评分 + 建模下限 + `capability_relaxed` 诊断。
- 文档：`docs/DEFAULTS.md`、`docs/DECISIONS.md`、`docs/STATUS.md`（生成物）、`docs/DEFAULT_EVIDENCE.json`。
- 证据：`research/README.md`（自动索引）、`research/EVIDENCE_POLICY.md`、`research/NEGATIVE_RESULTS.md`。
- 脚本：`scripts/check_evidence_policy.py`（含 benchmark hygiene + doc counts + 文档计数）、
  `verify_evidence.py`、`decision_matrix.py`、`build_release_notes.py`、`verify_skill_workflow.py`。
- 迁移相关：`scripts/sweep_transfer_top_k.py`、`benchmark_transfer_moo.py`、`benchmark_transfer_cost.py`。

## 已知坑

- 改了生成物来源（DEFAULTS/研究笔记/门禁清单）必须重跑对应 `build_*.py`，否则 `status_snapshot`、
  `release_notes`、`research_index` 相关测试会红。
- 新增测试后必须同步 `README.md` 的「**测试**：N 个单测」与 `tests/ 单测（N）`。
- 新加声明要同步 `research/EVIDENCE_AUDIT.md` 行与条数，否则 `test_evidence_audit_doc_lists_claims` 红。
- 文档里别写死会漂的数字（门禁数/声明数/算法数）——已有 `check_doc_counts.py` 拦截；生成物用实时计数。
- PowerShell 里不要用 here-string 写补丁；用临时 `py -3` 脚本再删除（`patch` 断言失败时文件不会被写）。
- 真实数据只有 ART Table 2 六点，其余合成/代理 oracle；结论必须标注。
- 不要 git 提交/推送；不要申请沙箱升级（approvals 已禁用）；`<agent-library>` 可能被并发改写。

## 下一步第一个动作

在项目根目录执行：

```
py -3 scripts/release_check.py        # 确认 19/19
py -3 -m pytest -q                    # 确认 484 passed
```

然后做：**用修复后的代码重测 `research/PRIOR_RELIABILITY.md`**（`scripts/benchmark_prior_reliability.py`
若不存在则参照 `sweep_transfer_top_k.py` 写一个），确认“reliability 不敏感”的旧结论是否也是
平台期 bug 造成的假象；若收益出现，更新登记表 `transfer_reliability_grading` 与 DEFAULTS，
并按固定流程收尾（加测试 → 登记 → LOG → 门禁全绿）。
