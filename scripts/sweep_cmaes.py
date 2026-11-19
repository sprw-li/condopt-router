"""CMA-ES parameter sensitivity sweep on the 10-D benchmarks.

Grid over sigma0 / population_size / c_mu on ``rosenbrock10`` + ``ackley10``;
prints mean best_score per setting so defaults can be set from evidence.

Run:
    py -3 scripts/sweep_cmaes.py --budget 60 --seeds 0 1
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from condopt.benchmarks import builtin_problems, run_algorithm  # noqa: E402

OUT = ROOT / "runs" / "sweep_cmaes"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=60)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--sigma0", type=float, nargs="+", default=[0.15, 0.3, 0.5])
    parser.add_argument("--population", type=int, nargs="+", default=[0, 8, 16])
    parser.add_argument("--c-mu", type=float, nargs="+", default=[0.2, 0.3, 0.5])
    parser.add_argument("--problems", nargs="+", default=["rosenbrock10", "ackley10"])
    parser.add_argument("--out", default=str(OUT / "results.csv"))
    args = parser.parse_args()

    rows = []
    for sigma0, population, c_mu in itertools.product(args.sigma0, args.population, args.c_mu):
        params = {"sigma0": sigma0, "population_size": population, "c_mu": c_mu}
        for problem_name in args.problems:
            problem = builtin_problems()[problem_name]
            scores = []
            for seed in args.seeds:
                result = run_algorithm(
                    problem, "cmaes", budget=args.budget, batch_size=args.batch,
                    seed=seed, params=params,
                )
                scores.append(float(result.best_score))
            rows.append({
                "problem": problem_name,
                "sigma0": sigma0,
                "population_size": population,
                "c_mu": c_mu,
                "n": len(scores),
                "best_score_mean": round(statistics.mean(scores), 4),
            })
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for problem_name in args.problems:
        subset = [row for row in rows if row["problem"] == problem_name]
        subset.sort(key=lambda row: row["best_score_mean"], reverse=True)
        print(f"\n{problem_name}: top 5")
        for row in subset[:5]:
            print(json.dumps(row, ensure_ascii=False))
    print(f"\nCSV: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())