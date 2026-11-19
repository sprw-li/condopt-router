"""End-to-end multi-objective demo: ParEGO on a yield-vs-cost trade-off.

    py -3 scripts/demo_moo.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from condopt import Study  # noqa: E402
from condopt.benchmarks import builtin_problems  # noqa: E402


def main() -> int:
    problem = builtin_problems()["moo_yield_cost"]
    run_dir = Path(__file__).resolve().parents[1] / "runs" / "demo_moo"
    if run_dir.exists():
        import shutil

        shutil.rmtree(run_dir)

    study = Study.create(
        run_dir,
        problem.space,
        problem.objective,
        budget=24,
        batch_size=4,
        seed=0,
    )
    print("== 多目标路由 ==")
    print(f"算法: {study.algorithm}")
    print(f"目标: {study.objective.objectives}  方向: {study.objective.directions}")

    for round_index in range(6):
        pending = study.ask(4)
        results = []
        for item in pending:
            objectives = problem.objectives_fn(item["x"])
            results.append({"id": item["id"], "objectives": objectives, "source": "simulation"})
        study.tell(results)
        pareto = study.pareto_summary()
        print(
            f"round {round_index + 1}: 建议 {len(pending)} 条，"
            f"当前前沿 {pareto['n_front']} 点，HV={pareto['hypervolume']}"
        )

    print()
    print(study.report())
    print()
    print("记录文件：", study.dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
