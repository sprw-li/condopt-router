"""High-cardinality categorical + descriptors + a binding constraint.

Compares one-hot encoding with physical descriptors for GP-based and
distribution-based optimizers on ``solvent_constrained`` / ``..._desc``
(12 solvent levels, T <= 80 degC while the unconstrained optimum sits at T~95).

Run:
    py -3 scripts/benchmark_solvent_constrained.py --budget 60 --seeds 0 1 2
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

from condopt.benchmarks import builtin_problems  # noqa: E402
from condopt.study import Study  # noqa: E402

OUT = ROOT / "runs" / "benchmark_solvent_constrained"


def run_one(
    algorithm: str,
    problem_name: str,
    budget: int = 60,
    batch_size: int = 4,
    seed: int = 0,
) -> dict:
    problem = builtin_problems()[problem_name]
    run_dir = OUT / f"{algorithm}_{problem_name}_{seed}"
    shutil.rmtree(run_dir, ignore_errors=True)
    study = Study.create(
        run_dir, problem.space, problem.objective,
        algorithm=algorithm, budget=budget, batch_size=batch_size, seed=seed,
    )
    while study.remaining() > 0:
        pending = study.ask(min(batch_size, study.remaining()))
        study.tell([
            {
                "id": item["id"],
                "y": problem.fn(item["x"]),
                "constraints": problem.constraints_fn(item["x"]),
            }
            for item in pending
        ])
    feasible = [
        float(problem.fn(obs.x))
        for obs in study.dataset.observations
        if float(obs.constraints.get("T_max", 1e9)) <= 80.0
    ]
    best = max(feasible) if feasible else None
    return {
        "algorithm": algorithm,
        "problem": problem_name,
        "seed": seed,
        "n_feasible": len(feasible),
        "feasible_rate": round(len(feasible) / max(1, len(study.dataset.observations)), 3),
        "best_feasible": None if best is None else round(best, 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithms", nargs="+", default=["bayes", "constrained", "tpe"])
    parser.add_argument("--budget", type=int, default=60)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--out", default=str(OUT / "results.csv"))
    args = parser.parse_args()

    rows = []
    for problem_name in ("solvent_constrained", "solvent_constrained_desc"):
        for algorithm in args.algorithms:
            for seed in args.seeds:
                rows.append(run_one(algorithm, problem_name, args.budget, args.batch, seed))
            subset = [
                row for row in rows
                if row["problem"] == problem_name and row["algorithm"] == algorithm
            ]
            print(json.dumps({
                "problem": problem_name,
                "algorithm": algorithm,
                "n": len(subset),
                "best_feasible_mean": round(
                    statistics.mean(row["best_feasible"] for row in subset), 4
                ),
                "feasible_rate_mean": round(
                    statistics.mean(row["feasible_rate"] for row in subset), 3
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
