# 开工简报（Prepare Brief）

> 本文件由 `scripts/build_prepare_brief.py` 依据真实的 `prepare.write_problem` 输出生成，
> 因此与代码行为一致（`tests/test_prepare_brief.py` 校验）。

## 三段建议

| 字段 | 作用 | 决策依据 |
|---|---|---|
| `recommendation` | 选哪个算法、用什么参数 | `docs/SCENARIOS.md` 的 19 个场景断言 + 证据声明 |
| `stopping` | 停滞优先还是预算优先 | 120 次预算实测：默认口径在约 57% 处停止（69/120） |
| `feasibility` | 可行率预警与处置建议 | 0.4% 可行域实测：约束搜索 1/5 vs 写进变量范围 5/5 |
| `prior` | 有先验数据时怎么暖启动 | 坏源 top_k 1→8 从 −369 恶化到 −1190；可靠度降到 0.25–0.5 回升到 −698 |
| `noise_plan` | 噪声处置：采集函数/alpha/批量/复孔/是否加基线臂 | 高噪声 → ucb + 大 alpha + 复孔 3 + 基线臂 |
| `deliverable` | 交付指标与必须包含的证据 | 约束下 raw HV 会高估（143.0 vs 80.9）；多保真必须给全保真证据 |

## 场景 A：小预算 + 单约束（默认口径）

```json
{
  "algorithm": "constrained",
  "params": {
    "kernel": "rbf",
    "length_scale": 0.3,
    "ard": true,
    "alpha": 0.0001,
    "n_initial": 9,
    "n_candidates": 1024,
    "acquisition": "ei",
    "kappa": 2.0,
    "xi": 0.01,
    "batch_strategy": "local_penalty",
    "local_frac": 0.4,
    "local_sigma": 0.15,
    "mc_draws": 128,
    "constraint_margin": 0.0,
    "feasibility_threshold": 0.02
  },
  "stopping": {
    "recommended": "stagnation_first",
    "why": "预算 24：实测默认停滞口径会在约 57% 处停止（69/120），预算有限时建议保留提前停止，把省下的次数留给下一轮实验",
    "how_to_enable": "Study.create(..., stop_on_stagnation=False) 或 condopt init --budget-first"
  },
  "feasibility": {
    "risk": "low",
    "estimated_feasible_fraction": null,
    "basis": "1 条约束 / 6 维，未发现明显窄域信号",
    "advice": "约束机制适用于“可行域较大、只是局部越界”的情形；若实测可行率偏低，step 会给出 encode_limits_as_bounds 建议"
  },
  "prior": {
    "has_prior": false,
    "advice": "无先验数据；正常从零开始采样"
  },
  "deliverable": {
    "metric": "feasible_best_value",
    "definition": "满足全部约束的最优目标值",
    "confirmation_plan": {
      "n": 3,
      "share_of_budget": 0.125,
      "advice": "收尾会自动做 3 次确认复孔（约占预算 12%）"
    }
  },
  "noise_plan": {
    "level": "medium",
    "replicates": 2,
    "baseline_arm": false
  }
}
```

## 场景 B：大预算（>=60，建议 budget_first）

```json
{
  "algorithm": "constrained",
  "params": {
    "kernel": "rbf",
    "length_scale": 0.3,
    "ard": true,
    "alpha": 0.0001,
    "n_initial": 10,
    "n_candidates": 1024,
    "acquisition": "ei",
    "kappa": 2.0,
    "xi": 0.01,
    "batch_strategy": "local_penalty",
    "local_frac": 0.4,
    "local_sigma": 0.15,
    "mc_draws": 128,
    "constraint_margin": 0.0,
    "feasibility_threshold": 0.02
  },
  "stopping": {
    "recommended": "budget_first",
    "why": "预算 120：实测默认停滞口径会在约 57% 处停止（69/120），预算较大时建议跑满，以获得更好的最优值与更充分的设计空间覆盖",
    "how_to_enable": "Study.create(..., stop_on_stagnation=False) 或 condopt init --budget-first"
  },
  "feasibility": {
    "risk": "low",
    "estimated_feasible_fraction": null,
    "basis": "1 条约束 / 6 维，未发现明显窄域信号",
    "advice": "约束机制适用于“可行域较大、只是局部越界”的情形；若实测可行率偏低，step 会给出 encode_limits_as_bounds 建议"
  },
  "prior": {
    "has_prior": false,
    "advice": "无先验数据；正常从零开始采样"
  },
  "deliverable": {
    "metric": "feasible_best_value",
    "definition": "满足全部约束的最优目标值",
    "confirmation_plan": {
      "n": 3,
      "share_of_budget": 0.025,
      "advice": "收尾会自动做 3 次确认复孔（约占预算 2%）"
    }
  },
  "noise_plan": {
    "level": "medium",
    "replicates": 2,
    "baseline_arm": false
  }
}
```

## 场景 D：有先验数据（暖启动建议）

```json
{
  "algorithm": "constrained",
  "params": {
    "kernel": "rbf",
    "length_scale": 0.3,
    "ard": true,
    "alpha": 0.0001,
    "n_initial": 10,
    "n_candidates": 1024,
    "acquisition": "ei",
    "kappa": 2.0,
    "xi": 0.01,
    "batch_strategy": "local_penalty",
    "local_frac": 0.4,
    "local_sigma": 0.15,
    "mc_draws": 128,
    "constraint_margin": 0.0,
    "feasibility_threshold": 0.02
  },
  "stopping": {
    "recommended": "stagnation_first",
    "why": "预算 40：实测默认停滞口径会在约 57% 处停止（69/120），预算有限时建议保留提前停止，把省下的次数留给下一轮实验",
    "how_to_enable": "Study.create(..., stop_on_stagnation=False) 或 condopt init --budget-first"
  },
  "feasibility": {
    "risk": "low",
    "estimated_feasible_fraction": null,
    "basis": "1 条约束 / 6 维，未发现明显窄域信号",
    "advice": "约束机制适用于“可行域较大、只是局部越界”的情形；若实测可行率偏低，step 会给出 encode_limits_as_bounds 建议"
  },
  "prior": {
    "has_prior": true,
    "advice": "用 transfer_from 暖启动而不是把先验拼进本次数据：先跑 condopt transfer-audit <源 run> 判断来源可信度（一致源会建议 1.0，打乱源 0.3–0.6），再按建议设置 --prior-reliability；top_k 建议 1–3"
  },
  "deliverable": {
    "metric": "feasible_best_value",
    "definition": "满足全部约束的最优目标值",
    "confirmation_plan": {
      "n": 3,
      "share_of_budget": 0.075,
      "advice": "收尾会自动做 3 次确认复孔（约占预算 8%）"
    }
  },
  "noise_plan": {
    "level": "medium",
    "replicates": 2,
    "baseline_arm": false
  }
}
```

## 场景 C：窄可行域（3 条约束 / 少维数）

```json
{
  "algorithm": "constrained",
  "params": {
    "kernel": "rbf",
    "length_scale": 0.3,
    "ard": true,
    "alpha": 0.0001,
    "n_initial": 4,
    "n_candidates": 1024,
    "acquisition": "ei",
    "kappa": 2.0,
    "xi": 0.01,
    "batch_strategy": "local_penalty",
    "local_frac": 0.4,
    "local_sigma": 0.15,
    "mc_draws": 128,
    "constraint_margin": 0.0,
    "feasibility_threshold": 0.02
  },
  "stopping": {
    "recommended": "stagnation_first",
    "why": "预算 40：实测默认停滞口径会在约 57% 处停止（69/120），预算有限时建议保留提前停止，把省下的次数留给下一轮实验",
    "how_to_enable": "Study.create(..., stop_on_stagnation=False) 或 condopt init --budget-first"
  },
  "feasibility": {
    "risk": "high",
    "estimated_feasible_fraction": 0.05,
    "basis": "3 条约束落在 3 维空间上（结构性推测）",
    "advice": "建议把已知硬限制写进变量范围或固定因子（真实工艺规程本来就该锁死），不要指望约束采集去“找”可行域"
  },
  "prior": {
    "has_prior": false,
    "advice": "无先验数据；正常从零开始采样"
  },
  "deliverable": {
    "metric": "feasible_best_value",
    "definition": "满足全部约束的最优目标值",
    "confirmation_plan": {
      "n": 3,
      "share_of_budget": 0.075,
      "advice": "收尾会自动做 3 次确认复孔（约占预算 8%）"
    }
  },
  "noise_plan": {
    "level": "low",
    "replicates": 1,
    "baseline_arm": false
  }
}
```

## Agent 用法

1. `condopt prepare --answers answers.json` 拿到三段建议；
2. 按 `stopping.recommended` 决定 `condopt init --budget-first` 与否；
3. `feasibility.risk=high` 时先把硬限制写进变量范围再开工；
3b. `prior.has_prior=true` 时先跑 `condopt transfer-audit`，再按建议设 `--prior-reliability`；
4. 跑完用 `condopt digest` 汇报（含停止原因与下一步建议）。

复现：`py -3 scripts/build_prepare_brief.py`。
