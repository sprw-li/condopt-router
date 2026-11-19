"""End-to-end demo: optimize a synthetic reaction-condition problem.

The "lab" is a deterministic synthetic yield surface -- the demo exists to
exercise the whole loop (route -> ask -> tell -> reroute -> report), not to
claim chemical accuracy.  Run:

    py -3 scripts/demo.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from condopt import Objective, SearchSpace, Study  # noqa: E402


SPACE = SearchSpace.from_dict(
    [
        {"name": "T", "kind": "continuous", "low": 40.0, "high": 120.0, "unit": "degC"},
        {"name": "time", "kind": "continuous", "low": 5.0, "high": 120.0, "unit": "min"},
        {"name": "conc", "kind": "log", "low": 0.02, "high": 1.0, "unit": "M"},
        {"name": "solvent", "kind": "categorical", "choices": ["MeCN", "EtOH", "water", "THF"]},
        {"name": "equiv", "kind": "integer", "low": 1, "high": 4},
        {"name": "inert", "kind": "bool"},
    ]
)


def synthetic_yield(config: dict) -> float:
    """A smooth-ish, noisy-free stand-in for a real yield measurement."""
    t = (config["T"] - 82.0) / 18.0
    dur = (config["time"] - 55.0) / 25.0
    conc = (math.log10(config["conc"]) + 0.8) / 0.9
    solvent_bonus = {"MeCN": 4.0, "EtOH": 1.0, "water": -3.0, "THF": 2.0}[config["solvent"]]
    equiv_bonus = -0.8 * (config["equiv"] - 2) ** 2
    inert_bonus = 2.0 if config["inert"] else 0.0
    return (
        70.0
        + 12.0 * math.exp(-(t**2))
        + 8.0 * math.exp(-(dur**2))
        + 6.0 * math.exp(-(conc**2))
        + solvent_bonus
        + equiv_bonus
        + inert_bonus
    )


def main() -> int:
    run_dir = Path(__file__).resolve().parents[1] / "runs" / "demo"
    if run_dir.exists():
        import shutil

        shutil.rmtree(run_dir)

    objective = Objective(name="yield", direction="max")
    study = Study.create(
        run_dir,
        SPACE,
        objective,
        budget=24,
        batch_size=4,
        seed=0,
    )
    print("== 初始路由 ==")
    print(f"算法: {study.algorithm}")
    print(json.dumps(study.params, ensure_ascii=False, indent=2))
    print(f"初始设计: {study.context.to_dict().get('budget_total')} 次预算，批量 {study.batch_size}")

    best_so_far = -math.inf
    for round_index in range(6):
        pending = study.ask(4)
        results = []
        for item in pending:
            value = synthetic_yield(item["x"])
            results.append({"id": item["id"], "y": value, "source": "simulation"})
        study.tell(results)
        best = study.best()
        assert best is not None
        best_so_far = max(best_so_far, best.y)
        print(
            f"round {round_index + 1}: 建议 {len(pending)} 条，"
            f"本批最优 {max(r['y'] for r in results):.2f}，累计最优 {best_so_far:.2f}"
        )

    print()
    print(study.report())
    print()
    print("记录文件：", study.dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
