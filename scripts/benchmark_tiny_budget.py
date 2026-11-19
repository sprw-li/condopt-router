"""Does space filling beat a surrogate at a tiny budget?

The router now falls back to space filling when ``budget <=`` a model's fitting
floor (see ``docs/ROUTING_POLICY.md``).  This measures that decision on the two
contexts that used to keep a model-based recommendation:

* ``moo_constrained``  -- 2 objectives + 1 constraint (was ``parego``/``cqehvi``);
* ``mf_constrained``   -- single objective + 1 constraint + multi fidelity (``mfbo``).

Run:
    py -3 scripts/benchmark_tiny_budget.py --budgets 4 6 8 --seeds 0 1 2 3
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

OUT = ROOT / "runs" / "benchmark_tiny_budget"

ARMS = {
    "moo_constrained": ("sobol", "lhs", "parego", "cqehvi"),
    "mf_constrained": ("sobol", "lhs", "mfbo", "constrained"),
}


def _measure(problem, x, fidelity: float) -> dict:
    objective = problem.objective
    payload: dict = {"fidelity": fidelity}
    if objective.is_multi_objective:
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
        y = (
            float(fidelity_fn(x, fidelity))
            if callable(fidelity_fn)
            else float(problem.fn(x))
        )
        payload["y"] = y
    constraints = getattr(problem, "constraints_fn", None)
    if callable(constraints):
        payload["constraints"] = constraints(x)
    payload["cost"] = float(fidelity)
    return payload


def run_arm(problem_name: str, algorithm: str, budget: int, seed: int) -> dict:
    problem = builtin_problems()[problem_name]
    run_dir = OUT / f"{problem_name}_{algorithm}_b{budget}_s{seed}"
    shutil.rmtree(run_dir, ignore_errors=True)
    context = OptimizationContext.from_space(
        problem.space,
        budget_total=budget,
        batch_size=2,
        multi_fidelity=callable(getattr(problem, "fidelity_fn", None)) and "mf" in problem_name,
        n_objectives=problem.objective.n_objectives,
        n_constraints=len(getattr(problem.objective, "constraints", ()) or ()),
        noise="medium",
    )
    study = Study.create(
        run_dir, problem.space, problem.objective, algorithm=algorithm,
        budget=budget, batch_size=2, context=context, seed=seed,
        meta={"auto_confirm": False},  # compare exploration only
    )
    for _ in range(max(60, 8 * budget)):
        action = study.next_action()
        if action["action"] == "stop":
            break
        if action["action"] == "ask":
            study.ask(int(action.get("ask_n") or 2))
        else:
            study.tell([
                {"id": item["id"], **_measure(problem, item["x"], item.get("fidelity", 1.0))}
                for item in study.pending
            ])
    summary = (
        study.feasible_pareto_summary()
        if problem.objective.is_multi_objective
        else {"hypervolume": None}
    )
    constraints = tuple(getattr(problem.objective, "constraints", ()) or ())
    if constraints:
        feasible = [
            obs for obs in study.dataset.observations if study._is_feasible(obs)
        ]
        best = max((float(obs.y) for obs in feasible), default=None)
    else:
        incumbent = study.best()
        best = None if incumbent is None else float(incumbent.y)
    return {
        "problem": problem_name,
        "algorithm": algorithm,
        "budget": budget,
        "seed": seed,
        "n_observations": len(study.dataset.observations),
        "n_feasible": sum(
            1 for obs in study.dataset.observations if study._is_feasible(obs)
        ),
        "best_feasible": None if best is None else round(best, 4),
        "feasible_hv": (
            None if summary.get("hypervolume") is None else round(summary["hypervolume"], 4)
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budgets", type=int, nargs="+", default=[4, 6, 8])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--out", default=str(OUT / "results.csv"))
    args = parser.parse_args()

    shutil.rmtree(OUT, ignore_errors=True)
    rows: list[dict] = []
    for problem_name, algorithms in ARMS.items():
        for budget in args.budgets:
            for algorithm in algorithms:
                for seed in args.seeds:
                    rows.append(run_arm(problem_name, algorithm, budget, seed))
            subset = [
                row for row in rows
                if row["problem"] == problem_name and row["budget"] == budget
            ]
            for algorithm in algorithms:
                arm = [row for row in subset if row["algorithm"] == algorithm]
                feasible = [row["n_feasible"] for row in arm]
                metric = (
                    [row["feasible_hv"] for row in arm if row["feasible_hv"] is not None]
                    if problem_name.startswith("moo")
                    else [row["best_feasible"] for row in arm if row["best_feasible"] is not None]
                )
                print(json.dumps({
                    "problem": problem_name,
                    "budget": budget,
                    "algorithm": algorithm,
                    "n": len(arm),
                    "feasible_mean": round(statistics.mean(feasible), 2) if feasible else None,
                    "metric_mean": round(statistics.mean(metric), 4) if metric else None,
                    "metric_hits": len(metric),
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