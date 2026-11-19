# ARCHITECTURE_PATTERNS — 条件优化库的程序架构模式

> 维护者：algo-research ｜ v1（2026-11-05）
> 目标：把 `ALGORITHM_SURVEY.md` 的算法落成**可插拔、可持久化、可复现、可被 agent 调用**的程序结构。
> 素材来源：`GITHUB_SCAN.md` 中 Ax / Optuna / SMAC3 / BayBE / Emukit / Nevergrad / Olympus 的架构对照（见各节标注）。
> 本文出现的代码块是**接口草案**（伪 Python，非可运行代码），用于对齐命名与职责，不是最终实现。

---

## 0. 分层总览

```
┌──────────────────────────────────────────────────────────┐
│  agent / skill router（本项目外层）                        │
│  - 读问题描述 → 选算法 + 配置（JSON）→ 调 library           │
└───────────────────────────┬──────────────────────────────┘
                            │ CampaignConfig (JSON, 可持久化)
┌───────────────────────────▼──────────────────────────────┐
│  campaign/   Campaign 编排层                              │
│  - 状态机: created → running → done/failed                │
│  - ask(n) / tell(results) / recommend() / history         │
│  - pending 管理、去重、预算计数、持久化                    │
└───────┬───────────────┬───────────────┬──────────────────┘
        │               │               │
┌───────▼──────┐ ┌──────▼───────┐ ┌─────▼────────────┐
│ sampler/     │ │ surrogate/   │ │ acquisition/     │
│ 算法注册表    │ │ GP / 森林 /  │ │ EI/qEI/qNEHVI/   │
│ BO/TPE/CMA/  │ │ KDE 模型     │ │ PoF/UCB          │
│ 随机/LHS     │ │              │ │                  │
└───────┬──────┘ └──────┬───────┘ └─────┬────────────┘
        │               │               │
┌───────▼───────────────▼───────────────▼──────────────────┐
│  space/  空间层（唯一真理源）                              │
│  Parameter(encode/decode/validate/sample) + Constraint    │
│  + SearchSpace(采样/枚举/投影/序列化)                      │
└───────────────────────────┬──────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────┐
│  storage/ + benchmark/ + telemetry/                       │
│  RunHistory(SQLite/JSONL) / emulator / 指标 / 日志         │
└──────────────────────────────────────────────────────────┘
```

**一句话原则**：**算法只看得见 `[0,1]^d` 的张量和 `(X, y)` 历史；一切"化学语义"（单位、类别、约束、物质描述符）都在 space 层。**

---

## 1. Optimizer 的 ask/tell 接口（核心契约）

### 1.1 为什么是 ask/tell

- **实验是异步的**：推荐 8 个点，实验台 3 天后才回；期间要能继续推荐（把 pending 当"未知 y"）。
- **执行与决策解耦**：机器人/人工/模拟器都能接同一个算法。
- **可回放**：只要记录 `(ask 的输出, tell 的输入)` 序列 + RNG seed，就能完整复现。
- 参照：Optuna `study.ask()/tell()`、Nevergrad `ask()/tell()`、Emukit loop、Trieste `AskTellOptimizer`。

### 1.2 接口草案

```python
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence
import numpy as np

@dataclass
class Trial:
    trial_id: int
    params: dict[str, Any]        # 原始空间参数（化学语义），如 {"temperature": 85.0, "solvent": "EtOH"}
    encoded: np.ndarray           # [0,1]^d，算法内部用
    status: str                   # pending | running | complete | failed | pruned
    values: list[float] | None = None      # 目标值（单目标 list 长度 1；多目标 list 长度 m）
    constraints: np.ndarray | None = None  # 约束值 c(x)，约定 c <= 0 可行
    fidelity: float | None = None
    cost: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)  # 重复次数、批次、操作员、时间戳

class Sampler(Protocol):
    """算法层唯一需要实现的接口。"""
    name: str
    capabilities: "Capabilities"

    def setup(self, space: "SearchSpace", config: dict[str, Any],
              rng: np.random.Generator, history: "RunHistory") -> None: ...

    def ask(self, n: int) -> list[np.ndarray]:
        """返回 n 个 encode 后的候选点 [0,1]^d；必须与 pending 点去重。"""

    def tell(self, trials: Sequence[Trial]) -> None:
        """接收已完成（或失败）的 trial，更新内部模型/状态。"""

    def state_dict(self) -> dict[str, Any]:
        """算法内部状态（模型超参、协方差矩阵、KDE 等），用于持久化/续跑。"""

    def load_state_dict(self, state: dict[str, Any]) -> None: ...
```

**契约细节（必须写进文档，否则实现会漂移）**

1. `ask(n)` **可以少于 n**（例如剩余预算/可行域耗尽），调用方必须处理。
2. `ask` 返回的点必须满足 `space.is_valid(x)`；**合法性由 space 层保证，算法不负责投影**（算法可以在内部松弛，但出口要 decode→validate→re-encode）。
3. `tell` 必须**幂等**（重复 tell 同一 trial 不改变结果）——网络/实验数据回传会重试。
4. `ask`/`tell` 之间允许交错（异步）；采样器要能处理"pending 但未回"的点（放进模型时用伪观测或在采集函数中惩罚）。
5. 任何随机性都来自注入的 `rng`；**禁止模块级 `np.random`**。

### 1.3 与"批处理"的结合

```python
# Campaign 内部循环（伪代码）
while budget_left > 0:
    n = min(batch_size, budget_left)
    X_enc = sampler.ask(n)                  # 可能包含 pending 惩罚
    trials = [make_trial(x, space.decode(x)) for x in X_enc]
    yield trials                            # 交给执行器（实验台/模拟器）
    results = yield from executor.run(trials)   # 可能异步
    sampler.tell(results)
    history.add(results)
```

- **pending 处理**：`ask` 之后、`tell` 之前，trial 状态为 `pending`；采样器内部维护 pending 集合，用"去重半径 + 采集惩罚"避免重复推荐。
- **批量语义**：`ask(n)` 是"一次性给 n 个"，不是"n 次 ask(1)"；BO 采样器需要 mini-batch 生成（qEI/TS/local-penalization），TPE/CMA 采样器逐点生成 + constant liar。

---

## 2. 搜索空间抽象（唯一真理源）

### 2.1 参数对象

```python
class Parameter(Protocol):
    name: str
    def encode(self, value: Any) -> np.ndarray: ...       # → [0,1]^k
    def decode(self, z: np.ndarray) -> Any: ...            # [0,1]^k → 原始值
    def validate(self, value: Any) -> bool: ...
    def sample(self, rng, n: int) -> list[Any]: ...        # 空间填充/随机
    def grid(self) -> list[Any] | None: ...                # 可枚举时返回全部取值（离散参数）
    def to_json(self) -> dict: ...                         # 序列化（campaign 配置）
```

**内置参数类型**（对照 BayBE/Nevergrad）：

| 类型 | 参数 | encode | 备注 |
|---|---|---|---|
| `FloatParameter` | `low, high, log: bool` | `(x-low)/(high-low)` 或 `(log x - log low)/(log high - log low)` | `log=True` 用于跨数量级（浓度/速率） |
| `IntegerParameter` | `low, high, step=1` | 同 float（+取整） | decode 时 `clip(round(...))` |
| `CategoricalParameter` | `categories: list` | one-hot（k 维）或索引（1 维） | k>10 建议外部传"描述符矩阵" |
| `OrdinalParameter` | `levels: list` | 索引/累积 | 有序分类（温度档位） |
| `SubstanceParameter` | `substances + descriptor_fn` | 描述符向量（k 维连续） | 化学物质专属：SMILES/名称 → 物性描述符 |
| `TaskParameter` | `values: list` | one-hot/索引，**不可优化** | 上下文变量（批次/设备/原料），用于条件建模 |
| `ForbiddenConstraint` | 参数组合黑名单 | — | 解码后校验 |
| `LinearConstraint` | `A, b`（可选 bounds） | — | 投影/拒绝 |
| `ConditionalParameter` | `parent, parent_value, child` | 未激活时给哨兵值 | 树状空间 |

### 2.2 SearchSpace

```python
class SearchSpace:
    parameters: list[Parameter]
    constraints: list[Constraint]

    @property
    def dim(self) -> int: ...                 # encode 后的总维度
    def encode(self, params: dict) -> np.ndarray: ...
    def decode(self, z: np.ndarray) -> dict: ...
    def is_valid(self, z: np.ndarray) -> bool: ...       # 含约束
    def project(self, z: np.ndarray) -> np.ndarray: ...  # 投影到可行域（重要！）
    def sample(self, rng, n: int, method="sobol") -> np.ndarray: ...
    def enumerate(self, max_points=100_000) -> np.ndarray | None:
        """全离散且规模可控时返回全部候选（采集优化用枚举，不用梯度）。"""
    def to_json(self) -> dict: ...
    @classmethod
    def from_json(cls, cfg: dict) -> "SearchSpace": ...
```

**关键设计决策**

1. **编码后的维度可能 ≠ 参数个数**：one-hot 会膨胀。要在 space 层暴露 `dim`，算法不得假设 `dim == len(parameters)`。
2. **`enumerate` 是"混合空间的秘密武器"**：BayBE 的 `SubspaceDiscretizer` 思路——把连续维离散化（如 20 档）+ 分类维枚举，得到候选池，采集函数直接打分。**默认执行路径优先枚举，梯度优化只作为连续子域的加速**。
3. **`project` 必须存在**：没有投影，约束空间的 BO 采样会大量浪费在不可行点。
4. **空间 JSON 化**：`CampaignConfig = {"space": space.to_json(), "sampler": {...}, "objective": {...}}`，这是 agent 传参、续跑、审计的基础。
5. **条件空间**：`ConditionalParameter` 的编码要"未激活维固定哨兵 + 算法侧 mask"；更稳的是"按分支选子空间/子模型"（复杂，v2 再做）。

### 2.3 目标/约束定义

```python
@dataclass
class Objective:
    name: str
    direction: str            # "max" | "min"
    log: bool = False         # 对目标取 log（乘性噪声）
    transform: Callable | None = None   # 如 yield → 收率归一化

@dataclass
class Constraints:
    specs: list[str]          # 约束函数名（执行器侧实现）
    noise_aware: bool = False # 是否用 Letham 式带噪约束建模
```

- 多目标 = `objectives: list[Objective]`；约束 = `constraints`；保真度 = `fidelity: FidelitySpec`（name, levels, cost mapping）。
- **观测记录用"长表"**：`(trial_id, metric_name, value, std, n_replicates, fidelity, timestamp)`，避免宽表 schema 变更。

---

## 3. 算法注册表与参数 Schema

### 3.1 注册表设计

```python
@dataclass
class Capabilities:
    supports_batch: bool = False
    supports_constraints: bool = False
    supports_multi_objective: bool = False
    supports_multi_fidelity: bool = False
    supports_categorical: bool = False
    supports_conditional_space: bool = False
    needs_gradient: bool = False
    max_dim_soft: int = 20          # 超过则建议换算法（router 用）
    min_observations: int = 0
    cost_profile: str = "medium"    # cheap | medium | expensive（每次 suggest 的 CPU 成本）

@dataclass
class AlgorithmSpec:
    name: str                       # "bo_gp", "tpe", "cma_es", "sobol", "random", "parego"
    cls: type[Sampler]
    param_schema: dict              # JSON Schema（见 3.2）
    defaults: dict
    capabilities: Capabilities
    version: str
```

- 用**装饰器注册**：`@register("bo_gp", schema=..., capabilities=...)`，`registry.get("bo_gp")(**cfg)`。
- **能力过滤**发生在 router：先 `capabilities` 硬过滤（例如有约束 → 排除不支持约束的算法），再打分排序。
- **算法别名/组合**：`"auto"` 走 `Portfolio` 策略（同时/分段跑多算法）；`"bo_gp"` 内部按 batch 自动切 qEI/TS。

### 3.2 参数 Schema（JSON Schema + 默认值合并）

```json
{
  "type": "object",
  "properties": {
    "kernel":   {"enum": ["matern52", "matern32", "rbf"], "default": "matern52"},
    "acquisition": {"enum": ["logEI", "qEI", "qNEI", "UCB", "TS"], "default": "logEI"},
    "n_initial": {"type": "integer", "minimum": 2, "default": null},
    "num_restarts": {"type": "integer", "minimum": 1, "default": 10},
    "raw_samples": {"type": "integer", "minimum": 64, "default": 512},
    "mc_samples": {"type": "integer", "minimum": 16, "default": 256},
    "noise_mode": {"enum": ["learn", "fixed", "replicate"], "default": "learn"}
  },
  "additionalProperties": false
}
```

**规则**：

1. **每个算法一个 schema**，用 `jsonschema` 校验后再实例化；未知字段直接报错（防止 agent 传错参数静默忽略）。
2. **`default: null` 表示"由系统按空间/预算推导"**（如 `n_initial = max(5, min(20, 2*(d_eff+1)))`），推导逻辑写在一个 `resolve_defaults(space, budget, noise)` 函数里，**可单测**。
3. **版本化**：`AlgorithmSpec.version` + config 里记录 `library_version`；升级时做 config 迁移（BayBE 的 `from_config` 是范例）。
4. **能力/参数的交叉校验**：如 `acquisition=qNEI` 需要 `noise_mode != "fixed"`、`supports_batch` 要求 `q>1`；在 `validate_config()` 里集中检查。

---

## 4. 批量化与异步（真实实验流水线的必备语义）

### 4.1 三种运行模式

| 模式 | 语义 | 实现要点 |
|---|---|---|
| 同步串行 | ask(1) → 做完 → tell | 最简单，便于调试 |
| 同步批量 | ask(q) → 全做完 → tell(q) | qEI/TS；实验台一次跑 q 个孔位 |
| **异步** | ask(q) 后陆续 tell，任何时候可 ask | pending 集合、部分 tell、模型可增量更新 |

### 4.2 异步的实现清单

1. **pending 点进入采集函数**：作为"伪观测"（constant liar / Kriging believer）或直接加"局部惩罚"（距离惩罚项）。
2. **去重**：`min_dist(x, pending ∪ history) < ε` 时重新采样；`ε` 按空间尺度设（连续维归一化后 0.02–0.05）。
3. **状态机**：`pending → running → complete | failed | pruned`；`failed` 的 trial 要记录原因（不可行/设备故障/中止），**失败点也进 history**（至少进约束模型，避免重复失败）。
4. **并发安全**：两个 worker 同时 `ask` 会重复推荐 → 用 storage 层锁/事务（Optuna RDB 的做法），或在 campaign 层串行化 suggest。
5. **超时与重试**：`tell` 支持部分结果（如只回了 3/8 个点）。

### 4.3 批量策略的选择逻辑（router 可编码）

```
if batch_size <= 1: 串行 EI/qNEI
elif sampler 支持 batch and budget_is_tight: qEI 贪心（quality）
elif sampler 支持 batch and batch_size > 8: TS 或 local-penalization
else: 逐点 ask + 距离约束
```

---

## 5. 持久化与可复现

### 5.1 需要持久化的四类东西

| 对象 | 内容 | 载体建议 |
|---|---|---|
| Campaign 配置 | space JSON、目标定义、算法配置、版本号 | `campaign.json`（人可读、可 diff） |
| RunHistory | 每条 trial 的参数/结果/状态/时间戳/seed | SQLite（`trials` 表 + `observations` 表）或 JSONL（append-only） |
| 算法状态 | 模型超参、GP 训练数据索引、CMA 协方差、KDE 带宽、RNG 状态 | `state_{n}.pkl` / `state.json`（体积大时存二进制） |
| 遥测 | 每轮耗时、采集优化统计、失败原因 | JSONL（便于分析） |

### 5.2 复现三要素

1. **RNG 分层**：`rng_campaign` 派生出 `rng_sampler` / `rng_acq` / `rng_candidate`，每一层单独 seed 并记录。验证方式：同 config + 同 seed + 同 tell 序列 → ask 序列**逐位一致**。
2. **确定性开关**：`deterministic=True` 时禁用多线程/GPU 非确定性算子（numpy BLAS 线程数、torch 的 `use_deterministic_algorithms`），并记录。
3. **数据快照**：`state_{n}` 与 history 的 `n` 对齐；支持"从第 k 轮 fork 出另一个算法"（历史重放）。

### 5.3 代码骨架

```python
class RunHistory:
    def __init__(self, path): ...           # sqlite3 连接 + schema 迁移
    def add(self, trial: Trial) -> None: ...
    def update(self, trial_id: int, **fields) -> None: ...
    def complete(self) -> list[Trial]: ...  # status == complete
    def pending(self) -> list[Trial]: ...
    def best(self, objective: str, direction: str) -> Trial | None: ...
    def to_dataframe(self) -> "pd.DataFrame": ...
    def replay(self, seed: int, until: int) -> "ReplayIterator": ...  # 复现/调试

class Campaign:
    def __init__(self, config: CampaignConfig, storage: RunHistory): ...
    def ask(self, n: int) -> list[Trial]: ...
    def tell(self, results: list[Observation]) -> None: ...
    def recommend(self, n: int = 1) -> list[dict]: ...
    def status(self) -> dict: ...           # 给 agent/router 读的状态摘要
    def save(self) -> None: ...
    @classmethod
    def load(cls, path: str) -> "Campaign": ...
```

- **`status()` 是给 LLM router 的接口**：返回 `{n_obs, best, best_x, pending, improvement_history, algorithm_health: {...}, suggestions: [...]}`——**结构化、短、可序列化**，不要让 LLM 去解析原始数据表。

---

## 6. Benchmark 与评估架构

### 6.1 三层评估对象

```python
class Evaluator(Protocol):
    """可替换的评估器：模拟器 / emulator 数据集 / 真实实验台。"""
    def evaluate(self, trials: list[Trial]) -> list[Trial]: ...
    def cost(self, trials: list[Trial]) -> float: ...

# 层次
# 1) SyntheticEvaluator: Branin/Hartmann/...（带噪声、带约束、带 fidelity）
# 2) EmulatorEvaluator: Olympus 数据集（HPLC/Suzuki/SnAr/...）[R18]
# 3) LiveEvaluator: 真实实验（人工/机器人），异步
```

### 6.2 基准协议

- **固定**：空间、初始设计（相同 seed 的 Sobol）、总评估数、噪声模型、ref point（多目标）。
- **每次运行**记录：`best_so_far[n]`、`cumulative_cost[n]`、`n_feasible`、算法内部统计（GP 超参、采集优化 restart 成功率）、wall-clock。
- **统计**：≥20 seeds；报告 **IQM + bootstrap 95% CI**（比均值±std 稳健，是 HPO 社区惯例）；成对比较用 Wilcoxon/配对 bootstrap。
- **成本感知指标**：`best-so-far per unit cost`（实验里成本可能差 100 倍，纯评估次数会误导）。
- **回归门槛**（CI 里必须的 gate）：新算法在 ≥70% 基准上不劣于随机基线；否则不进 router 候选集。

### 6.3 目录结构建议

```
benchmark/
  tasks/            # 每个任务一个 yaml/json：space, objective, noise, evaluator 类型
  evaluators/       # synthetic.py, olympus.py, live.py
  protocols.py      # 运行/记录/统计的公共逻辑
  report.py         # 生成表格/图（best-so-far 曲线、IQM 表）
  registry.py       # 任务注册（name → Task）
```

---

## 7. 与 agent / skill router 的集成

### 7.1 边界划分（照 ChemOS 的 planner/executor 分离）

| 层 | 职责 | 不做什么 |
|---|---|---|
| 算法库（本项目核心） | 空间、建模、采集、推荐、历史、benchmark | 不碰硬件、不做安全判断、不调 LLM |
| skill / router | 读需求 → 选算法+配置 → 调库 → 解释结果 → 失败诊断 | 不实现数值算法、不直接读写实验数据 |
| 执行器（实验台/机器人/人工） | 执行实验、回传结果 | 不做优化决策 |

### 7.2 router 调库的 JSON 契约（草案）

```json
{
  "request_id": "c-2026-11-05-001",
  "space": { "parameters": [ ... ], "constraints": [ ... ] },
  "objectives": [{"name": "yield", "direction": "max", "log": false}],
  "budget": {"n_trials": 40, "batch_size": 8, "cost_limit": 5000},
  "algorithm": {"name": "bo_gp", "params": {"acquisition": "qNEI", "n_initial": 10}},
  "history": [{"params": {...}, "values": [0.71], "std": [0.03]}, ...],
  "options": {"seed": 42, "deterministic": true}
}
```

响应（推荐）：

```json
{
  "request_id": "...",
  "recommendations": [
    {"params": {"temperature": 85.0, "solvent": "EtOH"}, "predictions": [{"mean": 0.78, "std": 0.05}],
     "predicted_feasibility": 0.97, "rationale": "high EI + feasible"}
  ],
  "diagnostics": {"model_fit_r2": 0.91, "lengthscales": [0.4, 0.9], "acq_restart_agreement": 0.83},
  "suggested_next": {"algorithm": "bo_gp", "params": {...}},
  "warnings": []
}
```

**要点**：router 需要的是 `diagnostics`（健康指标）而不是原始后验矩阵；`suggested_next` 让算法自检后主动建议切换（如"GP 高维不稳，建议 TuRBO/CMA"）。

### 7.3 LLM 在架构中的位置（约束）

1. **只做决策点上的选择/诊断**（选算法、选空间变换、解释结果），**不进数值内循环**。
2. 输出必须过 **JSON Schema 校验 + 白名单**（算法名、参数范围）。
3. 失败/超时 → 回退到规则层默认（`router/rules.py` 永远是第一道）。
4. 所有 prompt/输出写进 `telemetry/llm_calls.jsonl`（可审计、可复现）。

---

## 8. 目录与模块划分（建议的 Python 包结构）

```
src/optlib/
  space/
    parameter.py      # Float/Integer/Categorical/Ordinal/Substance/Task
    space.py          # SearchSpace: encode/decode/validate/project/enumerate/sample
    constraints.py    # 线性/禁止组合/条件激活
    transform.py      # log、标准化、one-hot 工具
  surrogate/
    gp.py             # 自研 GP（numpy/scipy）或 torch 封装
    forest.py         # 随机森林代理（SMAC 式）
    kde.py            # TPE 的 Parzen 估计
  acquisition/
    ei.py             # EI / logEI / qEI
    nei.py            # qNEI（baseline 集）
    uhvi.py           # qNEHVI（v2）
    ucb.py, ts.py, pof.py
  optimizer/
    candidate_pool.py # 候选池枚举 + 打分（默认路径）
    lbfgs.py          # 多起点梯度优化
    local_penalty.py  # 批量局部惩罚
  sampler/
    bo_gp.py, tpe.py, cma_es.py, random_search.py, sobol.py, parego.py
    portfolio.py      # 多算法组合/切换
  campaign/
    campaign.py, history.py, storage.py, config.py
  benchmark/
    tasks/, evaluators/, protocols.py, report.py
  router/
    rules.py          # 确定性规则（第一道）
    llm_router.py     # 可选 LLM 层
  telemetry/
    logger.py, metrics.py
tests/
  test_space_roundtrip.py, test_determinism.py, test_acq_values.py, test_benchmark_smoke.py
```

---

## 9. 工程约定清单（写进 CONTRIBUTING 的硬规则）

1. **所有随机性来自注入的 `rng`**；测试里用固定 seed 断言"两次 ask 逐位一致"。
2. **所有数值常量进 config/constants 模块**，禁止魔法数（尤其噪声/先验/阈值）。
3. **任何算法必须通过同一组 benchmark smoke test** 才能注册（synthetic × 3 + 混合空间 × 1，各 10 次评估不报错且 best 有改进）。
4. **禁止算法直接读过 `space.parameters` 的化学语义**；算法只见 `dim` 与 `encode` 后的张量。
5. **每个 `ask` 结果必须可 decode 回合法参数**（往返测试 `decode(encode(x)) == x`）。
6. **失败/异常必须落盘**（trial 记录 status=failed + error），不能吞掉。
7. **版本号 + schema 版本**：`CampaignConfig.schema_version`，升级时写迁移函数。
8. **文档即契约**：接口文档与 `protocols.py` 保持同步，CI 检查（如 `mypy`/`pytest` 冒烟）。

---

## 10. 参考实现对照（速查）

| 模式 | 本项目 | 参考 |
|---|---|---|
| ask/tell | `Campaign.ask/tell` + `Sampler` | Optuna、Nevergrad、Trieste `AskTellOptimizer` |
| 空间抽象 | `Parameter`/`SearchSpace` | BayBE、Nevergrad `parametrization`、skopt `Space` |
| 算法注册表 | `registry` + `Capabilities` | Nevergrad `optimizers`、SMAC3 Facade |
| 生成策略图 | `Portfolio` / 阶段切换 | Ax `GenerationStrategy`（v2 借鉴） |
| 批量 | qEI/TS/local-penalty | BoTorch `optimize_acqf` |
| 持久化 | SQLite + config JSON | Optuna `RDBStorage`、BayBE `Campaign` JSON |
| 预算/intensifier | `BudgetAllocator` | SMAC3 `Intensifier`、Optuna `Pruner` |
| benchmark | `Evaluator` + 协议 | Olympus `Planner`/`Benchmark`、pymoo `Problem` |
