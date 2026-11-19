"""Compare multi-objective multi-fidelity BO (momfbo / mfparego / parego).

Problems:
* ``mf_moo`` (max yield / min cost, cheap runs biased low on yield),
* ``mf_constrained`` (same two objectives plus hard constraints).

Metric: hypervolume of the observed *full-fidelity* objective vectors under the
same resource budget (feasible points only when the problem has constraints).

Run:
    py -3 scripts/benchmark_momfbo.py --problem mf_constrained --budget 40 --seeds 0 1 2
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

from condopt.benchmarks import builtin_problems  # noqa: E402
from condopt.dataset import Observation  # noqa: E402
from condopt.pareto import hypervolume_nd  # noqa: E402
from condopt.registry import build  # noqa: E402

OUT = ROOT / "runs" / "benchmark_momfbo"


def _signs(problem) -> list[float]:
    return [
        1.0 if direction == "max" else -1.0
        for direction in problem.objective.directions
    ]


def _signed(problem, objectives: dict) -> list[float]:
    return [
        float(objectives[name]) * sign
        for name, sign in zip(problem.objective.objectives, _signs(problem))
    ]


def _reference(problem, observations: list[Observation]) -> list[float]:
    if problem.objective.ref_point:
        return [
            float(value) * sign
            for value, sign in zip(problem.objective.ref_point, _signs(problem))
        ]
    if not observations:
        return [0.0] * problem.objective.n_objectives
    matrix = [_signed(problem, obs.objectives) for obs in observations]
    import numpy as np

    low = np.min(np.asarray(matrix), axis=0)
    return (low - 1.0).tolist()


def _feasible(problem, observation: Observation) -> bool:
    if problem.constraints_fn is None:
        return True
    values = observation.constraints or {}
    limits = problem.objective.constraint_limits or {}
    for name in problem.objective.constraints:
        value = values.get(name)
        if value is None:
            return False
        limit = limits.get(name) or {}
        threshold = float(limit.get("value", 0.0))
        op = str(limit.get("op", "<="))
        if op == "<=" and float(value) > threshold + 1e-9:
            return False
        if op == ">=" and float(value) < threshold - 1e-9:
            return False
    return True


def _hypervolume(problem, observations: list[Observation]) -> float:
    signed = [_signed(problem, obs.objectives) for obs in observations]
    if not signed:
        return 0.0
    return float(hypervolume_nd(signed, _reference(problem, observations)))


def run(
    algorithm: str,
    budget: float = 20.0,
    batch_size: int = 2,
    seed: int = 0,
    params: dict | None = None,
    problem_name: str = "mf_moo",
) -> dict:
    problem = builtin_problems()[problem_name]
    optimizer = build(algorithm, problem.space, problem.objective, params=params or {}, seed=seed)
    observations: list[Observation] = []
    spent = 0.0
    while spent < budget - 1e-9:
        if getattr(optimizer, "supports_fidelity", False):
            pairs = optimizer.ask_with_fidelity(batch_size, cost_remaining=budget - spent)
        else:
            pairs = [(config, 1.0) for config in optimizer.ask(batch_size)]
        batch = []
        for config, fidelity in pairs:
            if spent + float(fidelity) > budget + 1e-9:
                continue
            if problem.objectives_fidelity_fn is None:
                # single-fidelity MOO problem: every measurement is a full-fidelity one
                objectives = problem.objectives_fn(config)
            else:
                objectives = problem.objectives_fidelity_fn(config, float(fidelity))
            constraints = (
                problem.constraints_fn(config) if problem.constraints_fn is not None else {}
            )
            batch.append(Observation(
                x=config, objectives=objectives, constraints=constraints,
                fidelity=float(fidelity), cost=float(fidelity),
            ))
            spent += float(fidelity)
        if not batch:
            break
        optimizer.tell(batch)
        observations.extend(batch)
    full = [obs for obs in observations if obs.fidelity >= 1.0 - 1e-9]
    feasible_full = [obs for obs in full if _feasible(problem, obs)]
    hv_full = _hypervolume(problem, full)
    hv_feasible = _hypervolume(problem, feasible_full)
    best_yield = max((obs.objectives["yield"] for obs in full), default=None)
    return {
        "algorithm": algorithm,
        "problem": problem_name,
        "seed": seed,
        "budget": budget,
        "n_observations": len(observations),
        "cost_spent": round(spent, 4),
        "n_full_fidelity": len(full),
        "n_feasible_full": len(feasible_full),
        "hv_full_fidelity": round(hv_full, 4),
        "hv_feasible_full": round(hv_feasible, 4),
        "hv_all_levels": round(_hypervolume(problem, observations), 4),
        "best_full_yield": None if best_yield is None else round(float(best_yield), 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=float, default=20.0)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--algorithms", nargs="+", default=["momfbo", "parego"])
    parser.add_argument(
        "--problem",
        default="mf_moo",
        choices=[
            "mf_moo",
            "mf_moo3",
            "mf_moo4",
            "mf_constrained",
            "mf_moo_constrained",
            "moo_solvent_constrained",
                "moo_solvent_constrained_desc",
"mf_solvent",
            "mf_solvent_desc",
        ],
    )
    parser.add_argument("--out", default=str(OUT / "results.csv"))
    parser.add_argument("--min-full-fraction", type=float, default=None)
    parser.add_argument("--fidelity-levels", default=None)
    args = parser.parse_args()

    params = {}
    if args.min_full_fraction is not None:
        params["min_full_fraction"] = args.min_full_fraction
    if args.fidelity_levels is not None:
        params["fidelity_levels"] = args.fidelity_levels

    rows = []
    for algorithm in args.algorithms:
        for seed in args.seeds:
            row = run(
                algorithm, budget=args.budget, batch_size=args.batch, seed=seed,
                params=params, problem_name=args.problem,
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
            "problem": args.problem,
            "n": len(subset),
            "hv_feasible_full_mean": round(
                statistics.mean(row["hv_feasible_full"] for row in subset), 4
            ),
            "hv_full_fidelity_mean": round(
                statistics.mean(row["hv_full_fidelity"] for row in subset), 4
            ),
            "n_feasible_full_mean": round(
                statistics.mean(row["n_feasible_full"] for row in subset), 3
            ),
            "cost_mean": round(statistics.mean(row["cost_spent"] for row in subset), 4),
        }, ensure_ascii=False))
    print(f"CSV: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
