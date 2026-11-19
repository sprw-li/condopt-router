# 改进待办（外部审查 + 同类项目扫描汇总，按优先级）

> 每条都必须有**可执行的验证**（命令或测试路径）。没有验证方式的条目视为不存在。
> 来源：`research/EXTERNAL_REVIEW_RESPONSE.md`、`research/EXTERNAL_BENCHMARK_SCAN.md`、
> 5 份独立角度评审、外部独立审查报告（非公开）。

## 已修（有回归测试）

| 项 | 验证 |
|---|---|
| 路由「报告 tpe、实际全跑 Sobol」 | `tests/test_review_regressions.py::test_initial_design_leaves_room_for_the_reported_algorithm` |
| 目标判据方向错（min + target 误报达标） | `…::test_target_score_stop_respects_a_minimised_objective` |
| 热启动宽限期标记不持久 | `…::test_warm_start_grace_marker_survives_a_reload` |
| **cmaes 在 Study 路径完全不自适应** | `…::test_cmaes_actually_adapts_inside_a_study` |
| 决策矩阵无法区分「失败」与「环境跑不了」 | `py -3 scripts/decision_matrix.py`（BLOCKED 标记） |
| 版本漂移（README v0.31 vs pyproject 0.1.0） | `tests/test_release.py::test_readme_version_matches_the_package_metadata` |
| 算法命名夸大（tpe/qehvi/qnehvi/cmaes） | `tests/test_algorithm_naming.py` |
| 场景矩阵键冲突漏检 | `tests/test_scenarios.py::test_scenarios_doc_lists_every_scenario` |

## P0（影响结论可信度，下一步优先）

| # | 待办 | 验证方式 | 完成判据 |
|---|---|---|---|
| ~~A1~~ **已修** | `init` 存在性守卫：默认 `--if-exists=error`（退出码 2）；`keep` 保留数据、写 `campaign_reset` 决策并置暖启动宽限期；`reset` 丢弃配置 | `py -3 -m pytest tests/test_init_guard.py -q` | **exit 0（4 passed）** |
| ~~A2~~ **已修** | `momfbo` 的 EI incumbent 改用本轮 Dirichlet 权重（`_best_scalar(weights)`） | `py -3 -m pytest tests/test_momfbo_weights.py -q` | **exit 0（2 passed）**；momfbo 666.22→669.88，mfparego 仍 670.72（差 0.12%，噪声内） |
| ~~A3~~ **已修** | 归档数字登记 `docs/GOLDEN_NUMBERS.json` + 门禁 `scripts/check_golden_numbers.py`（presence / `--self-test` 内存改坏自检 / `--claims` 执行关联 claim） | `py -3 scripts/check_golden_numbers.py --self-test` | **exit 0**；端到端负对照 `py -3 scripts/golden_negative_control.py` 实测 clean=0 → **corrupted=1** → restored=0 |
| ~~A4~~ **已修** | `noise_floor.py --mode paired` 增加 **TOST**（`--equivalence-margin δ`，缺省 δ = 两臂同输入重复测量的合并 sd）：90% CI 整段落在 (−δ,+δ) 才 `equivalent=true`；跨边界=`false` 且 `equivalence_verdict="不确定"`；n<2 或推不出 δ ⇒ `equivalent=null`；δ≤0 双重拒绝。旧字段全部保留 | `py -3 -m pytest -q tests/test_noise_floor_equivalence.py` | **exit 0（15 passed）** |
| ~~A5~~ **已修** | `qnehvi` baseline 改为在 `X_baseline` 上取**模型后验**（两个目标各一次 GP 采样）；`Observation.objectives_std` 支持**逐目标**噪声（标量仍可用） | `py -3 -m pytest tests/test_qnehvi_baseline.py -q` | **exit 0（3 passed）**：结构断言 baseline 必须在 X 上采样两次；逐目标噪声不再相同 |
| ~~A6~~ **已修** | 门禁显式 `status`（ok/fail/skipped/external_drift）；缺 eval 脚本记 **skipped 且 ok=False**；`--skip-slow` 打印 **PARTIAL RUN** 并退出码 **3**（`--allow-partial` 才回 0） | `py -3 -m pytest tests/test_release_gate_honesty.py -q` | **exit 0（4 passed）** |

## P1（工程质量，影响可用性）

| # | 待办 | 验证方式 |
|---|---|---|
| ~~A7~~ **已修** | `errors.py` 定义错误族与退出码（2 输入 / 3 状态 / 4 环境）；CLI 已知异常 → stderr **单行 JSON** `{ok:false,error:{type,message,hint}}`，`--debug` 才附 traceback；`init` 缺 `--space`=2、`ask` 未 init=3、`tell` 未知 id=3、缺文件=4 | `py -3 -m pytest -q tests/test_cli_errors.py` | **exit 0（含 4+ 子进程用例）** |
| ~~A8~~ **已修** | 纯 stdlib JSON Schema 子集（`schema.py`）：未知键报错并用 difflib 给**最近候选键**（`constaint` → `constraints`），缺必填/类型错都报错；合法输入行为不变 | `py -3 -m pytest -q tests/test_answers_schema.py` | **exit 0** |
| ~~A9~~ **已修** | `Observation` 加 `schema_version`/`actor`（`CONDOPT_ACTOR`）；`log_decision` 加 `seq`/`config_hash`/`actor`；`decisions()` 为旧行补默认值（`config_hash=None`、`actor="unknown"`，**不拿今天的值冒充**）；`audit_trail()` 汇总 | `py -3 -m pytest -q tests/test_audit_fields.py` | **exit 0（15 passed）**；旧 JSONL 可读 |
| ~~A10~~ **已修** | 连续/对数变量**校验边界**（此前只查有限性）；越界 `tell` 默认**拒绝**并给出可选做法；`meta.clip_out_of_bounds`/`init --clip-out-of-bounds` 才夹边界，且写 `bounds_clipped` + `decisions.jsonl`；一 ulp 往返（`decode` 出 4.999999999999999）不误判 | `py -3 -m pytest tests/test_bounds_guard.py -q` | **exit 0（5 passed）** |
| ~~A11~~ **已修** | `scripts/benchmark_protocol.py`：共享初始设计（同 seed 跨算法)sha 相同）、显式种子集、per-seed CSV、**IQM + bootstrap 95% CI**、配对效应量、噪声底；CI 跨 0 ⇒ `inconclusive`（不许写“更好/更差”） | `py -3 -m pytest -q tests/test_benchmark_protocol.py` + `py -3 scripts/benchmark_protocol.py --seeds 3 --budget 12` | **exit 0（13 passed）**；实跑 20 seeds 也判 `inconclusive` |
| ~~A12~~ **已修** | `feasibility.py` 三态 `TRUE/FALSE/UNKNOWN`：未测到/非有限/无限值 ⇒ **UNKNOWN**（不算可行）；总体任一 FALSE ⇒ FALSE；`feasible_count` 报 `known_fraction` 与逐约束缺失数；`study._is_feasible`、`benchmarks`、`calibration` 全部改用同一口径；`prepare` 在缺失比例高时给“先补测”建议 | `py -3 -m pytest -q tests/test_feasibility_semantics.py` | **exit 0（8 passed）** |

## 不建议做（已论证）

- Hyperband bracket 调度：需要单调可提升的 resource 维度，而 `_cost` 接受任意 `fidelity_costs`
  且不校验单调（`research/EXTERNAL_BENCHMARK_SCAN.md` §4）。
- LLM 编排前端：应在 provenance/actor 字段（A9）之后做。
- 一次性迁移到 SQLAlchemy/Ax SQA 存储：先定 `schema_version` 与并发语义（A9）。
- auto-sklearn 式元特征冷启动：真实数据只有 ART 六点，统计上不可用。
