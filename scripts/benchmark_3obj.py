"""3-objective + constraint benchmark: qehvi_nd vs parego.

Metric: hypervolume of the *true* feasible objective vectors at all proposed
conditions (3D recursive HV).  Problem: ``moo3_constrained``.

Run:
    py -3 scripts/benchmark_3obj.py --budget 40 --batch 4 --seeds 0 1
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
from condopt.pareto import hypervolume_nd  # noqa: E402
from condopt.registry import build  # noqa: E402

OUT = ROOT / "runs" / "benchmark_3obj"


def run(
    algorithm: str,
    budget: int = 40,
    batch_size: int = 4,
    seed: int = 0,
    params: dict | None = None,
    problem_name: str = "moo3_constrained",
    noise_std: float = 0.0,
) -> dict:
    import time

    import numpy as _np

    rng = _np.random.default_rng(seed + 991)
    started = time.perf_counter()
    problem = builtin_problems()[problem_name]
    optimizer = build(
        algorithm, problem.space, problem.objective, params=params or {}, seed=seed
    )
    true_vectors: list[list[float]] = []
    feasible_vectors: list[list[float]] = []
    done = 0
    while done < budget:
        n = min(batch_size, budget - done)
        configs = optimizer.ask(n)
        batch = []
        signs = [
            1.0 if direction == "max" else -1.0
            for direction in problem.objective.directions
        ]
        for config in configs:
            truth = problem.objectives_fn(config)
            constraints = problem.constraints_fn(config) if problem.constraints_fn else {}
            signed = [
                float(truth[name]) * sign
                for name, sign in zip(problem.objective.objectives, signs)
            ]
            true_vectors.append(signed)
            feasible = True
            limits = problem.objective.constraint_limits or {}
            for name in problem.objective.constraints:
                value = constraints.get(name)
                if value is None:
                    feasible = False
                    break
                limit = limits.get(name) or {}
                threshold = float(limit.get("value", 0.0))
                op = str(limit.get("op", "<="))
                if (op == "<=" and float(value) > threshold + 1e-9) or (
                    op == ">=" and float(value) < threshold - 1e-9
                ):
                    feasible = False
                    break
            if feasible:
                feasible_vectors.append(signed)
            observed = (
                {
                    name: float(value) + float(rng.normal(0.0, noise_std))
                    for name, value in truth.items()
                }
                if noise_std > 0 else dict(truth)
            )
            batch.append(Observation(
                x=config, objectives=observed, constraints=constraints,
                objective_std=noise_std, cost=1.0,
            ))
        optimizer.tell(batch)
        done += len(batch)
    signs = [
        1.0 if direction == "max" else -1.0
        for direction in problem.objective.directions
    ]
    if problem.objective.ref_point:
        reference = [
            float(value) * sign
            for value, sign in zip(problem.objective.ref_point, signs)
        ]
    else:
        nadir = [min(vector[index] for vector in true_vectors)
                 for index in range(problem.objective.n_objectives)]
        span = [
            max(vector[index] for vector in true_vectors) - nadir[index]
            for index in range(problem.objective.n_objectives)
        ]
        reference = [nadir[index] - 0.05 * max(span[index], 1e-9)
                     for index in range(problem.objective.n_objectives)]
    hv_feasible = hypervolume_nd(feasible_vectors, reference) if feasible_vectors else 0.0
    hv_all = hypervolume_nd(true_vectors, reference)
    return {
        "algorithm": algorithm,
        "seed": seed,
        "problem": problem.name,
        "budget": budget,
        "n_feasible": len(feasible_vectors),
        "hv_feasible": round(float(hv_feasible), 4),
        "hv_all": round(float(hv_all), 4),
        "noise_std": noise_std,
        "elapsed_s": round(time.perf_counter() - started, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=40)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--algorithms", nargs="+", default=["qehvi_nd", "parego"])
    parser.add_argument("--problem", default="moo3_constrained")
    parser.add_argument("--mc-draws", type=int, default=None)
    parser.add_argument("--mc-prefilter", type=int, default=None)
    parser.add_argument("--n-candidates", type=int, default=None)
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--out", default=str(OUT / "results.csv"))
    args = parser.parse_args()

    params = {}
    if args.mc_draws is not None:
        params["mc_draws"] = args.mc_draws
    if args.mc_prefilter is not None:
        params["mc_prefilter"] = args.mc_prefilter
    if args.n_candidates is not None:
        params["n_candidates"] = args.n_candidates

    rows = []
    for algorithm in args.algorithms:
        for seed in args.seeds:
            row = run(
                algorithm, budget=args.budget, batch_size=args.batch, seed=seed,
                params=params, problem_name=args.problem, noise_std=args.noise_std,
            )
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("\nsummary")
    for algorithm in args.algorithms:
        subset = [row for row in rows if row["algorithm"] == algorithm]
        print(json.dumps({
            "algorithm": algorithm,
            "n": len(subset),
            "hv_feasible_mean": round(statistics.mean(r["hv_feasible"] for r in subset), 3),
            "n_feasible_mean": round(statistics.mean(r["n_feasible"] for r in subset), 2),
        }, ensure_ascii=False))
    print(f"CSV: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())