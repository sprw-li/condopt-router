"""Generate docs/PREPARE_BRIEF.md from the real ``prepare`` output.

Keeping the brief generated (instead of hand-written) means the advice an agent
reads can never drift from the code that produces it.

Run:
    py -3 scripts/build_prepare_brief.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from condopt.prepare import template, write_problem  # noqa: E402

OUT_ROOT = ROOT / "runs" / "_prepare_brief"
DOC = ROOT / "docs" / "PREPARE_BRIEF.md"

CASES = (
    {
        "key": "A",
        "title": "小预算 + 单约束（默认口径）",
        "answers": {
            "budget": 24,
            "batch": 4,
            "noise": "medium",
            "constraints": 1,
            "objective": {
                "name": "yield",
                "direction": "max",
                "constraints": ["T_max"],
                "constraint_limits": {"T_max": {"op": "<=", "value": 90}},
            },
        },
    },
    {
        "key": "B",
        "title": "大预算（>=60，建议 budget_first）",
        "answers": {
            "budget": 120,
            "batch": 6,
            "noise": "medium",
            "constraints": 1,
            "objective": {
                "name": "yield",
                "direction": "max",
                "constraints": ["T_max"],
                "constraint_limits": {"T_max": {"op": "<=", "value": 90}},
            },
        },
    },
    {
        "key": "D",
        "title": "有先验数据（暖启动建议）",
        "answers": {
            "budget": 40,
            "batch": 4,
            "noise": "medium",
            "prior": 6,
            "constraints": 1,
            "objective": {
                "name": "yield",
                "direction": "max",
                "constraints": ["T_max"],
                "constraint_limits": {"T_max": {"op": "<=", "value": 90}},
            },
        },
    },
    {
        "key": "C",
        "title": "窄可行域（3 条约束 / 少维数）",
        "answers": {
            "budget": 40,
            "batch": 4,
            "noise": "low",
            "constraints": 3,
            "objective": {
                "name": "yield",
                "direction": "max",
                "constraints": ["T_max", "time_max", "conc_max"],
                "constraint_limits": {
                    "T_max": {"op": "<=", "value": 80},
                    "time_max": {"op": "<=", "value": 60},
                    "conc_max": {"op": "<=", "value": 0.5},
                },
            },
            "variables": [
                {"name": "T", "kind": "continuous", "low": 40, "high": 120},
                {"name": "time", "kind": "continuous", "low": 5, "high": 120},
                {"name": "conc", "kind": "log", "low": 0.02, "high": 1.0},
            ],
        },
    },
)


def build_case(case: dict) -> dict:
    answers = template()
    answers.update(case["answers"])
    if "variables" in case["answers"]:
        answers["variables"] = case["answers"]["variables"]
    out_dir = OUT_ROOT / case["key"]
    shutil.rmtree(out_dir, ignore_errors=True)
    result = write_problem(answers, out_dir)
    return {
        "key": case["key"],
        "title": case["title"],
        "algorithm": result["recommendation"]["algorithm"],
        "params": result["recommendation"].get("params") or {},
        "stopping": result["stopping"],
        "feasibility": result["feasibility"],
        "prior": result["prior"],
        "deliverable": result["deliverable"],
        "noise_plan": result["noise_plan"],
    }


def render(cases: list[dict]) -> str:
    lines = [
        "# 开工简报（Prepare Brief）",
        "",
        "> 本文件由 `scripts/build_prepare_brief.py` 依据真实的 `prepare.write_problem` 输出生成，",
        "> 因此与代码行为一致（`tests/test_prepare_brief.py` 校验）。",
        "",
        "## 三段建议",
        "",
        "| 字段 | 作用 | 决策依据 |",
        "|---|---|---|",
        "| `recommendation` | 选哪个算法、用什么参数 | `docs/SCENARIOS.md` 的 19 个场景断言 + 证据声明 |",
        "| `stopping` | 停滞优先还是预算优先 | 120 次预算实测：默认口径在约 57% 处停止（69/120） |",
        "| `feasibility` | 可行率预警与处置建议 | 0.4% 可行域实测：约束搜索 1/5 vs 写进变量范围 5/5 |",
        "| `prior` | 有先验数据时怎么暖启动 | 坏源 top_k 1→8 从 −369 恶化到 −1190；可靠度降到 0.25–0.5 回升到 −698 |",
        "| `noise_plan` | 噪声处置：采集函数/alpha/批量/复孔/是否加基线臂 | 高噪声 → ucb + 大 alpha + 复孔 3 + 基线臂 |",
        "| `deliverable` | 交付指标与必须包含的证据 | 约束下 raw HV 会高估（143.0 vs 80.9）；多保真必须给全保真证据 |",
        "",
    ]
    for case in cases:
        stopping = case["stopping"]
        feasibility = case["feasibility"]
        lines += [
            f"## 场景 {case['key']}：{case['title']}",
            "",
            "```json",
            json.dumps(
                {
                    "algorithm": case["algorithm"],
                    "params": case["params"],
                    "stopping": {
                        "recommended": stopping["recommended"],
                        "why": stopping["why"],
                        "how_to_enable": stopping["how_to_enable"],
                    },
                    "feasibility": {
                        "risk": feasibility["risk"],
                        "estimated_feasible_fraction": feasibility.get(
                            "estimated_feasible_fraction"
                        ),
                        "basis": feasibility.get("basis"),
                        "advice": feasibility.get("advice"),
                    },
                    "prior": {
                        "has_prior": case["prior"].get("has_prior"),
                        "advice": case["prior"].get("advice"),
                    },
                    "deliverable": {
                        "metric": case["deliverable"]["metric"],
                        "definition": case["deliverable"]["definition"],
                        "confirmation_plan": case["deliverable"]["confirmation_plan"],
                    },
                    "noise_plan": {
                        "level": case["noise_plan"]["level"],
                        "replicates": case["noise_plan"]["replicates"],
                        "baseline_arm": case["noise_plan"]["baseline_arm"],
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            "```",
            "",
        ]
    lines += [
        "## Agent 用法",
        "",
        "1. `condopt prepare --answers answers.json` 拿到三段建议；",
        "2. 按 `stopping.recommended` 决定 `condopt init --budget-first` 与否；",
        "3. `feasibility.risk=high` 时先把硬限制写进变量范围再开工；",
        "3b. `prior.has_prior=true` 时先跑 `condopt transfer-audit`，再按建议设 `--prior-reliability`；",
        "4. 跑完用 `condopt digest` 汇报（含停止原因与下一步建议）。",
        "",
        "复现：`py -3 scripts/build_prepare_brief.py`。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    cases = [build_case(case) for case in CASES]
    DOC.write_text(render(cases), encoding="utf-8")
    print(f"wrote {DOC}")
    for case in cases:
        print(
            f"  {case['key']}: {case['algorithm']} | "
            f"stopping={case['stopping']['recommended']} | "
            f"feasibility={case['feasibility']['risk']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())