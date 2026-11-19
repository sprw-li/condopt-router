"""High-dimensional continuous benchmark: CMA-ES vs sampling / GP baselines.

Problems: ``rosenbrock10`` (smooth narrow valley) and ``ackley10`` (multimodal),
both 10-D continuous on the unit cube.  Metric: best-so-far score after the same
budget (higher is better; optimum 0).

Run:
    py -3 scripts/benchmark_highdim.py --budget 80 --batch 8 --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from condopt.benchmarks import builtin_problems, run_algorithm  # noqa: E402

OUT = ROOT / "runs" / "benchmark_highdim"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=80)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--algorithms", nargs="+", default=["cmaes", "sobol", "random", "bayes", "tpe"]
    )
    parser.add_argument("--problems", nargs="+", default=["rosenbrock10", "ackley10"])
    parser.add_argument("--out", default=str(OUT / "results.csv"))
    args = parser.parse_args()

    rows = []
    for problem_name in args.problems:
        problem = builtin_problems()[problem_name]
        for algorithm in args.algorithms:
            for seed in args.seeds:
                result = run_algorithm(
                    problem, algorithm, budget=args.budget, batch_size=args.batch, seed=seed
                )
                rows.append({
                    "problem": problem_name,
                    "algorithm": algorithm,
                    "seed": seed,
                    "best_score": round(float(result.best_score), 4),
                    "regret": round(float(problem.regret(result.best_score)), 6)
                    if problem.optimum is not None
                    else None,
                })
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("summary (mean best_score, lower regret is better)")
    for problem_name in args.problems:
        for algorithm in args.algorithms:
            subset = [
                row for row in rows
                if row["problem"] == problem_name and row["algorithm"] == algorithm
            ]
            print(json.dumps({
                "problem": problem_name,
                "algorithm": algorithm,
                "n": len(subset),
                "best_score_mean": round(statistics.mean(r["best_score"] for r in subset), 4),
            }, ensure_ascii=False))
    print(f"CSV: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())