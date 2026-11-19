"""mfbo parameter sweep on the constrained multi-fidelity benchmark.

Metric: best feasible full-fidelity yield at a fixed cost budget (higher is
better).  Grid over cost_exponent / min_full_fraction / acquisition.

Run:
    py -3 scripts/sweep_mfbo.py --budget 20 --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import itertools
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

spec = importlib.util.spec_from_file_location(
    "benchmark_mf_constrained", ROOT / "scripts" / "benchmark_mf_constrained.py"
)
benchmark = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(benchmark)

OUT = ROOT / "runs" / "sweep_mfbo"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=float, default=20.0)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--cost-exponent", type=float, nargs="+", default=[0.5, 1.0, 1.5])
    parser.add_argument("--min-full-fraction", type=float, nargs="+", default=[0.0, 0.25, 0.5])
    parser.add_argument("--acquisition", nargs="+", default=["logei", "ei", "ucb"])
    parser.add_argument("--out", default=str(OUT / "results.csv"))
    args = parser.parse_args()

    rows = []
    for cost_exponent, min_full, acquisition in itertools.product(
        args.cost_exponent, args.min_full_fraction, args.acquisition
    ):
        params = {
            "cost_exponent": cost_exponent,
            "min_full_fraction": min_full,
            "acquisition": acquisition,
        }
        scores = []
        for seed in args.seeds:
            result = benchmark.run(
                "mfbo", budget=args.budget, batch_size=args.batch, seed=seed, params=params
            )
            if result["best_feasible_full_yield"] is not None:
                scores.append(float(result["best_feasible_full_yield"]))
        rows.append({
            "cost_exponent": cost_exponent,
            "min_full_fraction": min_full,
            "acquisition": acquisition,
            "n": len(scores),
            "best_feasible_yield_mean": round(statistics.mean(scores), 3) if scores else None,
        })
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    ranked = [row for row in rows if row["best_feasible_yield_mean"] is not None]
    ranked.sort(key=lambda row: row["best_feasible_yield_mean"], reverse=True)
    for row in ranked[:8]:
        print(json.dumps(row, ensure_ascii=False))
    print(f"CSV: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())