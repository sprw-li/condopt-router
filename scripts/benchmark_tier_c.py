"""Tier C: high-cardinality categorical solvent, one-hot vs descriptors.

Same landscape, same algorithm, same budget; the only difference is whether the
12 solvent levels are encoded as one-hot or with their polarity/protic
descriptors.  Both tables come from the synthetic problem definition, so this
isolates the feature-encoding effect (no real chemistry claims).

Run:
    py -3 scripts/benchmark_tier_c.py --budget 20 --seeds 0 1 2 3 4
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

OUT = ROOT / "runs" / "benchmark_tier_c"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=20)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--algorithms", nargs="+", default=["bayes", "tpe"])
    args = parser.parse_args()

    rows = []
    for problem_name in ("high_cardinality_solvent", "high_cardinality_solvent_desc"):
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
                    "regret": round(float(problem.regret(result.best_score)), 6),
                })
    out = OUT / "results.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("summary (mean best_score / mean regret)")
    for problem_name in ("high_cardinality_solvent", "high_cardinality_solvent_desc"):
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
                "regret_mean": round(statistics.mean(r["regret"] for r in subset), 6),
            }, ensure_ascii=False))
    print(f"CSV: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())