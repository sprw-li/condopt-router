"""momfbo parameter sweep on the multi-objective multi-fidelity benchmark.

Metric: full-fidelity hypervolume of the observed front; also reports how many
seeds finish without any full-fidelity observation (a failure mode).

Run:
    py -3 scripts/sweep_momfbo.py --budget 20 --seeds 0 1
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
    "benchmark_momfbo", ROOT / "scripts" / "benchmark_momfbo.py"
)
benchmark = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(benchmark)

OUT = ROOT / "runs" / "sweep_momfbo"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=float, default=20.0)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--acquisition", nargs="+", default=["ei", "logei"])
    parser.add_argument("--min-full-fraction", type=float, nargs="+", default=[0.0, 0.25])
    parser.add_argument("--cost-exponent", type=float, nargs="+", default=[0.5, 1.0, 1.5])
    parser.add_argument("--out", default=str(OUT / "results.csv"))
    args = parser.parse_args()

    rows = []
    for acquisition, min_full, cost_exponent in itertools.product(
        args.acquisition, args.min_full_fraction, args.cost_exponent
    ):
        params = {
            "acquisition": acquisition,
            "min_full_fraction": min_full,
            "cost_exponent": cost_exponent,
        }
        hvs = []
        failures = 0
        for seed in args.seeds:
            result = benchmark.run(
                "momfbo", budget=args.budget, batch_size=args.batch, seed=seed, params=params
            )
            if result["n_full_fidelity"] == 0:
                failures += 1
            hvs.append(float(result["hv_full_fidelity"]))
        rows.append({
            "acquisition": acquisition,
            "min_full_fraction": min_full,
            "cost_exponent": cost_exponent,
            "n": len(hvs),
            "no_full_fidelity_runs": failures,
            "hv_mean": round(statistics.mean(hvs), 3),
        })
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    ranked = sorted(rows, key=lambda row: (row["no_full_fidelity_runs"], -row["hv_mean"]))
    for row in ranked[:8]:
        print(json.dumps(row, ensure_ascii=False))
    print(f"CSV: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())