"""Low-feasibility stress test: does two-phase feasibility search earn its keep?

Constructs a narrow feasible region (~0.4% of the space) on the mf_constrained
search space and compares ``two_phase_feasibility`` on/off: how quickly a
feasible point is found and how much feasible full-fidelity evidence is produced.

Run:
    py -3 scripts/benchmark_hard_feasibility.py --budget 30 --seeds 0 1 2 3 4
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

from condopt.base import Objective  # noqa: E402
from condopt.benchmarks import builtin_problems  # noqa: E402
from condopt.router import OptimizationContext  # noqa: E402
from condopt.study import Study  # noqa: E402

OUT = ROOT / "runs" / "benchmark_hard_feasibility"

# Narrow box: T <= 70, 40 <= time <= 70, solvent in {DMF, MeCN}, base_eq == 2
# Roughly 0.4% of the mixed space is feasible.
T_MAX = 70.0
TIME_RANGE = (40.0, 70.0)
SOLVENTS = ("DMF", "MeCN")


def make_objective() -> Objective:
    return Objective(
        name="yield",
        direction="max",
        constraints=("T_max", "time_ok", "solvent_ok", "base_ok"),
        constraint_limits={
            name: {"op": "<=", "value": 0.0}
            for name in ("T_max", "time_ok", "solvent_ok", "base_ok")
        },
    )


def constraints_fn(config: dict) -> dict:
    """Every constraint is expressed as a violation magnitude (feasible when <= 0)."""
    time_value = float(config["time"])
    midpoint = 0.5 * (TIME_RANGE[0] + TIME_RANGE[1])
    half_width = 0.5 * (TIME_RANGE[1] - TIME_RANGE[0])
    return {
        "T_max": float(config["T"]) - T_MAX,
        "time_ok": abs(time_value - midpoint) - half_width,
        "solvent_ok": 0.0 if config["solvent"] in SOLVENTS else 1.0,
        "base_ok": 0.0 if int(config.get("base_eq", 2)) == 2 else 1.0,
    }


def yield_fn(config: dict, fidelity: float = 1.0) -> float:
    """Local oracle so both arms share one landscape (the benchmark's own
    fidelity function needs the ``base_eq`` variable, which the bounds-encoded
    arm fixes by removing it)."""
    import math

    solvent_bonus = {"DMF": 6.0, "MeCN": 2.0}.get(config["solvent"], 0.0)
    ligand_bonus = {"L2": 4.0, "L4": 3.0}.get(config.get("ligand", "L1"), 0.0)
    value = (
        60.0
        + 20.0 * math.exp(-(((float(config["T"]) - 65.0) / 20.0) ** 2))
        + 10.0 * math.exp(-(((float(config["time"]) - 55.0) / 25.0) ** 2))
        + solvent_bonus
        + ligand_bonus
    )
    return value - 14.0 * ((1.0 - float(fidelity)) ** 1.5)


def is_feasible(config: dict) -> bool:
    return all(value <= 0.0 for value in constraints_fn(config).values())


def restricted_space():
    """Same chemistry with the narrow box baked into the variable bounds."""
    from condopt.space import SearchSpace

    return SearchSpace.from_dict([
        {"name": "T", "kind": "continuous", "low": 40.0, "high": T_MAX, "unit": "degC"},
        {"name": "time", "kind": "log", "low": TIME_RANGE[0], "high": TIME_RANGE[1]},
        {"name": "cat_loading", "kind": "log", "low": 0.1, "high": 5.0},
        {"name": "solvent", "kind": "categorical", "choices": list(SOLVENTS)},
        {"name": "ligand", "kind": "categorical",
         "choices": ["L1", "L2", "L3", "L4", "L5", "L6"]},
    ])


def run_one(
    two_phase: bool,
    budget: int = 30,
    batch_size: int = 3,
    seed: int = 0,
    bounds_encoded: bool = False,
) -> dict:
    problem = builtin_problems()["mf_constrained"]
    run_dir = OUT / f"tp{int(two_phase)}_b{int(bounds_encoded)}_{seed}"
    shutil.rmtree(run_dir, ignore_errors=True)
    space = restricted_space() if bounds_encoded else problem.space
    context = OptimizationContext.from_space(
        space, budget_total=budget, batch_size=batch_size,
        multi_fidelity=True, n_constraints=0 if bounds_encoded else 4, noise="medium",
    )
    study = Study.create(
        run_dir,
        space,
        Objective(name="yield", direction="max") if bounds_encoded else make_objective(),
        algorithm="mfbo",
        params={"two_phase_feasibility": two_phase},
        budget=budget,
        batch_size=batch_size,
        cost_budget=budget,
        context=context,
        seed=seed,
    )
    for _ in range(max(120, 8 * budget)):
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
                    "y": float(yield_fn(item["x"], float(item.get("fidelity", 1.0) or 1.0))),
                    "constraints": constraints_fn(item["x"]),
                    "cost": float(item.get("fidelity", 1.0) or 1.0),
                }
                for item in study.pending
            ])
        elif kind == "stop":
            break

    observations = study.dataset.observations
    feasible = [obs for obs in observations if is_feasible(obs.x)]
    feasible_full = [obs for obs in feasible if obs.fidelity >= 1.0 - 1e-9]
    first_index = next(
        (index for index, obs in enumerate(observations) if is_feasible(obs.x)), None
    )
    return {
        "two_phase_feasibility": two_phase,
        "bounds_encoded": bounds_encoded,
        "seed": seed,
        "n_observations": len(observations),
        "n_feasible": len(feasible),
        "n_feasible_full": len(feasible_full),
        "found_feasible": bool(feasible),
        "first_feasible_index": first_index,
        "best_feasible_true": (
            None if not feasible else round(max(float(obs.y) for obs in feasible), 4)
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=30)
    parser.add_argument("--batch", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--out", default=str(OUT / "results.csv"))
    args = parser.parse_args()

    rows = []
    arms = [(False, False), (True, False), (True, True)]
    for two_phase, bounds_encoded in arms:
        for seed in args.seeds:
            rows.append(run_one(two_phase, args.budget, args.batch, seed, bounds_encoded))
        subset = [
            row for row in rows
            if row["two_phase_feasibility"] is two_phase
            and row["bounds_encoded"] is bounds_encoded
        ]
        found_indices = [
            row["first_feasible_index"]
            for row in subset
            if row["first_feasible_index"] is not None
        ]
        print(json.dumps({
            "two_phase_feasibility": two_phase,
            "bounds_encoded": bounds_encoded,
            "n": len(subset),
            "found_rate": round(
                statistics.mean(1.0 if row["found_feasible"] else 0.0 for row in subset), 2
            ),
            "n_feasible_mean": round(
                statistics.mean(row["n_feasible"] for row in subset), 2
            ),
            "n_feasible_full_mean": round(
                statistics.mean(row["n_feasible_full"] for row in subset), 2
            ),
            "median_first_feasible_index": (
                None if not found_indices else statistics.median(found_indices)
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