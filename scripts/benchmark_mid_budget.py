"""Is ``budget_first`` worth it at mid budgets (10-40)?

``prepare.stopping_advice`` switches to ``budget_first`` at budget >= 60, derived
from a 120-evaluation experiment (``research/ADAPTER_LONG_RUN.md``: stagnation-first
stopped at 69/120).  The mid band was never measured directly.  This compares both
policies on a multi-fidelity constrained problem and a smooth 2-D problem.

Run:
    py -3 scripts/benchmark_mid_budget.py --budgets 12 20 40 --seeds 0 1 2 3 4
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

OUT = ROOT / "runs" / "benchmark_mid_budget"
PROBLEMS = ("mf_constrained", "sphere2d")


def _evaluate(problem, item) -> dict:
    x = item["x"]
    fidelity = float(item.get("fidelity", 1.0))
    payload: dict = {"id": item["id"], "fidelity": fidelity, "cost": fidelity}
    if problem.objective.is_multi_objective:
        fidelity_fn = getattr(problem, "objectives_fidelity_fn", None)
        values = (
            fidelity_fn(x, fidelity)
            if callable(fidelity_fn)
            else problem.objectives_fn(x)
        )
        payload["objectives"] = values
        payload["y"] = float(list(values.values())[0])
    else:
        fidelity_fn = getattr(problem, "fidelity_fn", None)
        payload["y"] = (
            float(fidelity_fn(x, fidelity)) if callable(fidelity_fn) else float(problem.fn(x))
        )
    constraints = getattr(problem, "constraints_fn", None)
    if callable(constraints):
        payload["constraints"] = constraints(x)
    return payload


def run_arm(problem_name: str, budget_first: bool, budget: int, seed: int) -> dict:
    problem = builtin_problems()[problem_name]
    run_dir = OUT / f"{problem_name}_{'budget' if budget_first else 'stag'}_b{budget}_s{seed}"
    shutil.rmtree(run_dir, ignore_errors=True)
    context = OptimizationContext.from_space(
        problem.space,
        budget_total=budget,
        batch_size=3,
        multi_fidelity=callable(getattr(problem, "fidelity_fn", None)),
        n_objectives=problem.objective.n_objectives,
        n_constraints=len(getattr(problem.objective, "constraints", ()) or ()),
        noise="medium",
    )
    study = Study.create(
        run_dir, problem.space, problem.objective, algorithm="auto",
        budget=budget, batch_size=3, context=context, seed=seed,
        stop_on_stagnation=not budget_first,
    )
    stops: list[str] = []
    # step cap must scale with the budget: a fixed 60 silently truncated budget-120
    # campaigns at ~90 observations (ask+tell = two steps per batch)
    for _ in range(max(60, 4 * budget)):
        action = study.next_action()
        kind = action["action"]
        if kind == "stop":
            stops.append(action.get("stop_reason") or "")
            break
        if kind == "ask":
            study.ask(int(action.get("ask_n") or 3))
        elif kind == "confirm":
            study.confirm(int(action.get("confirm_n") or 3))
        else:
            study.tell([_evaluate(problem, item) for item in study.pending])
            # Hand-written loops must record the trajectory themselves, otherwise the
            # delivery checklist reports "逐批进展记录" missing (this script used to
            # report delivery_ok=0 for that reason, not because of the policy).
            study.record_progress()
    constraints = tuple(getattr(problem.objective, "constraints", ()) or ())
    metric_name = "best_feasible"
    if problem.objective.is_multi_objective:
        summary = (
            study.feasible_pareto_summary() if constraints else study.pareto_summary()
        )
        best = summary.get("hypervolume")
        metric_name = "feasible_hv" if constraints else "hv"
    elif constraints:
        feasible = [obs for obs in study.dataset.observations if study._is_feasible(obs)]
        best = max((float(obs.y) for obs in feasible), default=None)
    else:
        incumbent = study.best()
        best = None if incumbent is None else float(incumbent.y)
    checklist = study.delivery_checklist()
    return {
        "problem": problem_name,
        "policy": "budget_first" if budget_first else "stagnation_first",
        "budget": budget,
        "seed": seed,
        "metric": metric_name,
        "n_observations": len(study.dataset.observations),
        "best": None if best is None else round(best, 4),
        "delivery_ok": bool(checklist) and all(item["ok"] for item in checklist),
        "stop_reason": stops[0] if stops else "",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", nargs="+", default=list(PROBLEMS))
    parser.add_argument("--budgets", type=int, nargs="+", default=[12, 20, 40])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--out", default=str(OUT / "results.csv"))
    args = parser.parse_args()

    shutil.rmtree(OUT, ignore_errors=True)
    rows: list[dict] = []
    for problem_name in args.problems:
        for budget in args.budgets:
            for budget_first in (False, True):
                for seed in args.seeds:
                    rows.append(run_arm(problem_name, budget_first, budget, seed))
            for budget_first in (False, True):
                arm = [
                    row for row in rows
                    if row["problem"] == problem_name
                    and row["budget"] == budget
                    and row["policy"] == ("budget_first" if budget_first else "stagnation_first")
                ]
                values = [row["best"] for row in arm if row["best"] is not None]
                print(json.dumps({
                    "problem": problem_name,
                    "budget": budget,
                    "policy": arm[0]["policy"],
                    "n": len(arm),
                    "n_obs_mean": round(statistics.mean(row["n_observations"] for row in arm), 1),
                    "best_mean": round(statistics.mean(values), 4) if values else None,
                    "delivery_ok": sum(row["delivery_ok"] for row in arm),
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