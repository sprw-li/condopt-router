"""How sensitive is the campaign to the stagnation-detection patience?

For each ``patience`` value the drill compares accepting the plateau
(``stop``) with retuning and continuing (``tune``) at the same total budget.

Run:
    py -3 scripts/sweep_patience.py --budget 60 --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location(
    "benchmark_stagnation", ROOT / "scripts" / "benchmark_stagnation.py"
)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)

OUT = ROOT / "runs" / "sweep_patience"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", default="rosenbrock10")
    parser.add_argument("--budget", type=int, default=60)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--patience", type=int, nargs="+", default=[1, 2, 3, 5, 8])
    parser.add_argument("--out", default=str(OUT / "results.csv"))
    args = parser.parse_args()

    rows = []
    for patience in args.patience:
        for arm in ("stop", "tune"):
            for seed in args.seeds:
                row = _mod.run_arm(
                    arm, args.problem, args.budget, args.batch, seed,
                    patience=patience,
                )
                rows.append(row)
            subset = [r for r in rows if r["patience"] == patience and r["arm"] == arm]
            print(json.dumps({
                "patience": patience,
                "arm": arm,
                "n": len(subset),
                "best_mean": round(statistics.mean(r["best_score"] for r in subset), 4),
                "obs_mean": round(
                    statistics.mean(r["n_observations"] for r in subset), 1
                ),
                "retunes_mean": round(
                    statistics.mean(r["retunes"] for r in subset), 2
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
