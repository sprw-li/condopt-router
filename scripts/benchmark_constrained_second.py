"""Second dataset for the "constrained single objective -> constrained" decision.

The registry entry was backed by ``solvent_constrained`` (high-cardinality categorical)
and noisy-constraint runs.  This adds a **hard, continuous** feasible region
(``constraint_hard``: ~6% feasible, binding constraint) so the rule is not only
supported on categorical/threshold problems.

Arms: ``constrained`` (feasibility-aware EI), ``bayes`` (ignores constraints),
``random`` (model-free baseline).  Metric: ``best_feasible_score`` from the runner.

Run:
    py -3 scripts/benchmark_constrained_second.py --budget 30 --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from condopt.benchmarks import builtin_problems, run_algorithm  # noqa: E402

OUT = ROOT / "runs" / "benchmark_constrained_second"
PROBLEMS = ("constraint_hard", "constraint_easy")
ALGORITHMS = ("constrained", "bayes", "random")


def run_arm(problem_name: str, algorithm: str, budget: int, batch: int, seed: int) -> dict:
    problem = builtin_problems()[problem_name]
    result = run_algorithm(
        problem, algorithm, budget=budget, batch_size=batch, seed=seed
    )
    return {
        "problem": problem_name,
        "algorithm": algorithm,
        "budget": budget,
        "seed": seed,
        "best_feasible": (
            None if result.best_feasible_score is None else round(float(result.best_feasible_score), 4)
        ),
        "best_raw": round(float(result.best_score), 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", nargs="+", default=list(PROBLEMS))
    parser.add_argument("--budget", type=int, default=30)
    parser.add_argument("--batch", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--out", default=str(OUT / "results.csv"))
    args = parser.parse_args()

    shutil.rmtree(OUT, ignore_errors=True)
    rows: list[dict] = []
    for problem_name in args.problems:
        for algorithm in ALGORITHMS:
            for seed in args.seeds:
                rows.append(run_arm(problem_name, algorithm, args.budget, args.batch, seed))
        for algorithm in ALGORITHMS:
            arm = [
                row for row in rows
                if row["problem"] == problem_name
                and row["algorithm"] == algorithm
                and row["best_feasible"] is not None
            ]
            print(json.dumps({
                "problem": problem_name,
                "algorithm": algorithm,
                "n_feasible": len(arm),
                "best_feasible_mean": (
                    round(statistics.mean(row["best_feasible"] for row in arm), 4)
                    if arm else None
                ),
            }, ensure_ascii=False))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())