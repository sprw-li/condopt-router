"""What should an unattended campaign do when it stalls?

Three arms on the same budget:

* ``stop``     — accept the plateau (stop early);
* ``continue`` — keep asking with the same algorithm (operator override);
* ``tune``     — keep the algorithm but shift an exploration-related parameter
  (acquisition mode or ``kappa``) before continuing;
* ``explore``  — inject one model-free batch (``sobol``) and then resume auto-routing.

Run:
    py -3 scripts/benchmark_stagnation.py --budget 60 --seeds 0 1 2
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
from condopt.study import Study  # noqa: E402

OUT = ROOT / "runs" / "benchmark_stagnation"


def _tell(study: Study, problem) -> None:
    """Feed measured outcomes back, honouring fidelity and constraints."""
    results = []
    for item in study.pending:
        x = item["x"]
        fidelity = float(item.get("fidelity", 1.0))
        fidelity_fn = getattr(problem, "fidelity_fn", None)
        y = (
            float(fidelity_fn(x, fidelity))
            if callable(fidelity_fn)
            else float(problem.fn(x))
        )
        payload: dict = {
            "id": item["id"], "y": y, "fidelity": fidelity, "cost": fidelity,
        }
        constraints = getattr(problem, "constraints_fn", None)
        if callable(constraints):
            payload["constraints"] = constraints(x)
        results.append(payload)
    study.tell(results)


def run_arm(
    arm: str,
    problem_name: str = "rosenbrock10",
    budget: int = 60,
    batch_size: int = 4,
    seed: int = 0,
    max_steps: int = 200,
    patience: int = 3,
) -> dict:
    problem = builtin_problems()[problem_name]
    run_dir = OUT / f"{arm}_{problem_name}_{seed}"
    shutil.rmtree(run_dir, ignore_errors=True)
    from condopt.router import OptimizationContext

    context = OptimizationContext.from_space(
        problem.space,
        budget_total=budget,
        batch_size=batch_size,
        multi_fidelity=callable(getattr(problem, "fidelity_fn", None)),
        n_objectives=problem.objective.n_objectives,
        n_constraints=len(getattr(problem.objective, "constraints", ()) or ()),
        noise="medium",
    )
    study = Study.create(
        run_dir, problem.space, problem.objective,
        algorithm="auto", budget=budget, batch_size=batch_size, seed=seed,
        patience=patience, context=context,
    )
    restarts = 0
    retunes = 0
    stopped_at: int | None = None
    restart_waiting = False
    for _ in range(max_steps):
        action = study.next_action()
        kind = action["action"]
        if kind == "tell":
            _tell(study, problem)
            if restart_waiting:
                # exploration batch finished: hand control back to the router
                study.resume_auto_route()
                restart_waiting = False
        elif kind == "ask":
            n = int(action.get("ask_n") or batch_size)
            study.ask(max(1, min(n, study.remaining())))
        elif kind == "confirm":
            study.confirm(int(action.get("confirm_n") or 3))
        elif kind == "stop":
            category = (action.get("diagnostics") or {}).get("stop_category")
            if arm == "stop" or category != "stagnation" or study.remaining() <= 0:
                stopped_at = len(study.dataset.observations)
                break
            if arm == "continue":
                study.ask(max(1, min(batch_size, study.remaining())))
            elif arm == "tune":
                retunes += 1
                study.retune_for_exploration(
                    reason=f"stagnation retune #{retunes}"
                )
                study.ask(max(1, min(batch_size, study.remaining())))
            else:  # explore
                restarts += 1
                study.override("sobol", reason=f"stagnation restart #{restarts}")
                study.ask(max(1, min(batch_size, study.remaining())))
                restart_waiting = True
        if study.remaining() <= 0 and not study.pending:
            break
    constraints = tuple(getattr(problem.objective, "constraints", ()) or ())
    if constraints:
        feasible = [
            obs for obs in study.dataset.observations if study._is_feasible(obs)
        ]
        best_score = max((float(obs.y) for obs in feasible), default=None)
    else:
        best = study.best()
        best_score = None if best is None else float(best.y)
    return {
        "arm": arm,
        "problem": problem_name,
        "seed": seed,
        "patience": patience,
        "n_observations": len(study.dataset.observations),
        "restarts": restarts,
        "retunes": retunes,
        "stopped_early_at": stopped_at,
        "best_score": None if best_score is None else round(best_score, 4),
        "final_algorithm": study.algorithm,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", nargs="+", default=["rosenbrock10", "mixed_reaction"])
    parser.add_argument("--budget", type=int, default=60)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--arms", nargs="+", default=["stop", "continue", "tune", "explore"])
    parser.add_argument("--out", default=str(OUT / "results.csv"))
    args = parser.parse_args()

    rows = []
    for problem_name in args.problems:
        for arm in args.arms:
            for seed in args.seeds:
                row = run_arm(
                    arm, problem_name, args.budget, args.batch, seed
                )
                rows.append(row)
                print(json.dumps(row, ensure_ascii=False))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("\nsummary (mean best; higher is better)")
    for problem_name in args.problems:
        for arm in args.arms:
            subset = [row for row in rows if row["problem"] == problem_name and row["arm"] == arm]
            print(json.dumps({
                "problem": problem_name,
                "arm": arm,
                "n": len(subset),
                "best_mean": round(statistics.mean(row["best_score"] for row in subset), 4),
                "obs_mean": round(statistics.mean(row["n_observations"] for row in subset), 1),
                "restarts_mean": round(statistics.mean(row["restarts"] for row in subset), 2),
                "retunes_mean": round(statistics.mean(row["retunes"] for row in subset), 2),
            }, ensure_ascii=False))
    print(f"CSV: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())