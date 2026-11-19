"""Does the feasibility-aware full-fidelity slot help under constraints?

Compares ``mfbo`` with ``feasibility_aware_full`` on/off on ``mf_constrained``
(T <= 80 degC, cheap runs biased low), reporting feasible full-fidelity evidence.

Run:
    py -3 scripts/benchmark_feasible_full.py --budget 24 --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from condopt.benchmarks import builtin_problems  # noqa: E402
from condopt.router import OptimizationContext  # noqa: E402
from condopt.study import Study  # noqa: E402

OUT = ROOT / "runs" / "benchmark_feasible_full"


def run_one(
    aware: bool,
    threshold: float = 0.0,
    two_phase: bool = True,
    problem_name: str = "mf_constrained",
    budget: int = 24,
    batch_size: int = 3,
    cost_budget: float = 10.0,
    seed: int = 0,
) -> dict:
    problem = builtin_problems()[problem_name]
    run_dir = OUT / f"aware_{int(aware)}_t{threshold}_{seed}"
    shutil.rmtree(run_dir, ignore_errors=True)
    context = OptimizationContext.from_space(
        problem.space, budget_total=budget, batch_size=batch_size,
        multi_fidelity=True, n_constraints=1, noise="medium",
    )
    study = Study.create(
        run_dir, problem.space, problem.objective,
        algorithm="mfbo",
        params={
            "feasibility_aware_full": aware,
            "feasibility_full_threshold": threshold,
            "two_phase_feasibility": two_phase,
        },
        budget=budget, batch_size=batch_size, cost_budget=cost_budget,
        context=context, seed=seed,
    )
    steps = 0
    while steps < 40:
        steps += 1
        action = study.next_action()
        kind = action["action"]
        if kind == "ask":
            study.ask(int(action.get("ask_n") or batch_size))
        elif kind == "confirm":
            study.confirm(int(action.get("confirm_n") or 3))
        elif kind == "tell":
            study.tell([
                {
                    "id": item["id"],
                    "fidelity": float(item.get("fidelity", 1.0) or 1.0),
                    "y": float(problem.fidelity_fn(item["x"], float(item.get("fidelity", 1.0) or 1.0))),
                    "constraints": problem.constraints_fn(item["x"]),
                    "cost": float(item.get("fidelity", 1.0) or 1.0),
                }
                for item in study.pending
            ])
        elif kind == "stop":
            break
    observations = study.dataset.observations
    full = [obs for obs in observations if obs.fidelity >= 1.0 - 1e-9]
    feasible_full = [obs for obs in full if float(obs.constraints.get("T_max", 1e9)) <= 80.0]
    return {
        "feasibility_aware_full": aware,
        "feasibility_full_threshold": threshold,
        "two_phase_feasibility": two_phase,
        "seed": seed,
        "n_observations": len(observations),
        "n_full": len(full),
        "n_feasible_full": len(feasible_full),
        "cost_spent": round(study.cost_spent(), 3),
        "best_feasible_true": (
            None if not feasible_full else round(max(float(obs.y) for obs in feasible_full), 4)
        ),
        "has_feasible_full": bool(feasible_full),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=24)
    parser.add_argument("--batch", type=int, default=3)
    parser.add_argument("--cost-budget", type=float, default=10.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.0])
    parser.add_argument("--compare-two-phase", action="store_true")
    parser.add_argument("--out", default=str(OUT / "results.csv"))
    args = parser.parse_args()

    rows = []
    arms = [(False, 0.0, True)] + [
        (True, threshold, two_phase)
        for threshold in args.thresholds
        for two_phase in ([True, False] if args.compare_two_phase else [True])
    ]
    for aware, threshold, two_phase in arms:
        for seed in args.seeds:
            rows.append(run_one(
                aware, threshold, two_phase, budget=args.budget, batch_size=args.batch,
                cost_budget=args.cost_budget, seed=seed,
            ))
        subset = [
            row for row in rows
            if row["feasibility_aware_full"] is aware
            and row["feasibility_full_threshold"] == threshold
            and row["two_phase_feasibility"] is two_phase
        ]
        print(json.dumps({
            "feasibility_aware_full": aware,
            "feasibility_full_threshold": threshold,
            "two_phase_feasibility": two_phase,
            "n": len(subset),
            "n_full_mean": round(statistics.mean(row["n_full"] for row in subset), 2),
            "n_feasible_full_mean": round(
                statistics.mean(row["n_feasible_full"] for row in subset), 2
            ),
            "has_feasible_full_rate": round(
                statistics.mean(1.0 if row["has_feasible_full"] else 0.0 for row in subset), 2
            ),
            "best_feasible_mean": round(
                statistics.mean(
                    row["best_feasible_true"]
                    for row in subset
                    if row["best_feasible_true"] is not None
                ), 3
            ) if any(row["best_feasible_true"] is not None for row in subset) else None,
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
