"""Does per-point noise propagation help ParEGO under measurement noise?

Problem: ``moo_yield_cost`` with Gaussian noise injected into both objectives.
Every observation records ``objective_std=sigma``; the only difference between
arms is parego's ``noise_aware`` flag.  Metric: hypervolume of the *true*
objective vectors at all proposed conditions.

Run:
    py -3 scripts/benchmark_noise.py --budget 30 --sigma 3 --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from condopt.benchmarks import builtin_problems  # noqa: E402
from condopt.dataset import Observation  # noqa: E402
from condopt.pareto import hypervolume_2d  # noqa: E402
from condopt.registry import build  # noqa: E402

REFERENCE = (60.0, -30.0)


def run(
    noise_aware: bool,
    sigma: float = 3.0,
    budget: int = 30,
    batch_size: int = 2,
    seed: int = 0,
) -> dict:
    problem = builtin_problems()["moo_yield_cost"]
    optimizer = build(
        "parego",
        problem.space,
        problem.objective,
        params={"noise_aware": noise_aware},
        seed=seed,
    )
    rng = np.random.default_rng(seed + 12345)
    true_vectors: list[list[float]] = []
    done = 0
    while done < budget:
        n = min(batch_size, budget - done)
        configs = optimizer.ask(n)
        batch = []
        for config in configs:
            truth = problem.objectives_fn(config)
            noisy = {
                name: float(value) + float(rng.normal(0.0, sigma))
                for name, value in truth.items()
            }
            true_vectors.append([float(truth["yield"]), -float(truth["cost"])])
            batch.append(Observation(
                x=config, objectives=noisy, objective_std=sigma, cost=1.0
            ))
        optimizer.tell(batch)
        done += len(batch)
    hv = float(hypervolume_2d(true_vectors, REFERENCE))
    return {
        "noise_aware": noise_aware,
        "sigma": sigma,
        "seed": seed,
        "budget": budget,
        "hv_true": round(hv, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=30)
    parser.add_argument("--sigma", type=float, default=3.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--out", default=str(ROOT / "runs" / "benchmark_noise" / "results.csv"))
    args = parser.parse_args()

    rows = []
    for noise_aware in (True, False):
        for seed in args.seeds:
            row = run(noise_aware, sigma=args.sigma, budget=args.budget, seed=seed)
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("\nsummary")
    for flag in (True, False):
        subset = [row["hv_true"] for row in rows if row["noise_aware"] is flag]
        print(json.dumps({
            "noise_aware": flag,
            "n": len(subset),
            "hv_true_mean": round(statistics.mean(subset), 3),
        }, ensure_ascii=False))
    print(f"CSV: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())