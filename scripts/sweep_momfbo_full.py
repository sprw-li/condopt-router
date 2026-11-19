"""Parameter sweep for the multi-fidelity multi-objective stack.

Varies ``momfbo.min_full_fraction`` and ``mfparego.rho / weight_alpha`` on
``mf_moo_constrained`` (two objectives + binding constraint + fidelity bias).

Run:
    py -3 scripts/sweep_momfbo_full.py --seeds 0 1 2
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
    "benchmark_momfbo", ROOT / "scripts" / "benchmark_momfbo.py"
)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)

OUT = ROOT / "runs" / "sweep_momfbo_full"


def evaluate(
    algorithm: str,
    params: dict,
    seeds: list[int],
    budget: float,
    batch: int,
    problem: str,
) -> dict:
    hv, feasible = [], []
    for seed in seeds:
        row = _mod.run(
            algorithm, budget=budget, batch_size=batch, seed=seed,
            params=params, problem_name=problem,
        )
        hv.append(row["hv_feasible_full"])
        feasible.append(row["n_feasible_full"])
    return {
        "algorithm": algorithm,
        "params": params,
        "n": len(seeds),
        "hv_feasible_mean": round(statistics.mean(hv), 4),
        "feasible_full_mean": round(statistics.mean(feasible), 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", default="mf_moo_constrained")
    parser.add_argument("--budget", type=float, default=40.0)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--out", default=str(OUT / "results.csv"))
    args = parser.parse_args()

    rows = []
    grid: list[tuple[str, dict]] = []
    for fraction in (0.25, 0.4, 0.6, 0.8):
        grid.append(("momfbo", {"min_full_fraction": fraction}))
    for rho in (0.02, 0.05, 0.15):
        grid.append(("mfparego", {"rho": rho}))
    for alpha in (0.5, 1.0, 2.0):
        grid.append(("mfparego", {"weight_alpha": alpha}))

    for algorithm, params in grid:
        row = evaluate(
            algorithm, params, args.seeds, args.budget, args.batch, args.problem
        )
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["algorithm", "params", "n", "hv_feasible_mean", "feasible_full_mean"]
        )
        writer.writeheader()
        writer.writerows(rows)
    best = max(rows, key=lambda row: row["hv_feasible_mean"])
    print(f"best: {best['algorithm']} {best['params']} -> {best['hv_feasible_mean']}")
    print(f"CSV: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())