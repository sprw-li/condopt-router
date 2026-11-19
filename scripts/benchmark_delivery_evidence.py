"""Is the delivery checklist satisfiable outside constrained scenarios?

``delivery_evidence_policy`` was registered from constrained runs (feasible HV vs raw
HV, feasible full-fidelity evidence).  This measures the checklist pass rate on four
shapes, through the executor adapter (so the real campaign path is exercised):

* ``moo_yield_cost``   -- 2 objectives, unconstrained, single fidelity;
* ``mf_moo``           -- 2 objectives, unconstrained, multi fidelity;
* ``moo_constrained``  -- 2 objectives + constraints (control);
* ``sphere2d``         -- single objective (confirmation-replicate path).

Run:
    py -3 scripts/benchmark_delivery_evidence.py --seeds 0 1 2 --budget 24
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
from condopt.executor import ExecutionRequest, Executor, Measurement, run_campaign  # noqa: E402
from condopt.router import OptimizationContext  # noqa: E402
from condopt.study import Study  # noqa: E402

OUT = ROOT / "runs" / "benchmark_delivery_evidence"
PROBLEMS = ("moo_yield_cost", "mf_moo", "moo_constrained", "sphere2d")


class SimulatedInstrument(Executor):
    """Deterministic synthetic instrument (no noise, no failures)."""

    def __init__(self, problem) -> None:
        self.problem = problem

    def measure(self, request: ExecutionRequest) -> Measurement:
        x = request.x
        fidelity = float(request.fidelity)
        fidelity_fn = getattr(self.problem, "fidelity_fn", None)
        objectives_fn = getattr(self.problem, "objectives_fidelity_fn", None)
        payload: dict = {"id": request.id, "fidelity": fidelity, "cost": fidelity}
        if self.problem.objective.is_multi_objective:
            values = (
                objectives_fn(x, fidelity)
                if callable(objectives_fn)
                else self.problem.objectives_fn(x)
            )
            payload["objectives"] = values
            payload["y"] = float(list(values.values())[0])
        else:
            payload["y"] = (
                float(fidelity_fn(x, fidelity)) if callable(fidelity_fn) else float(self.problem.fn(x))
            )
        constraints = getattr(self.problem, "constraints_fn", None)
        if callable(constraints):
            payload["constraints"] = constraints(x)
        return Measurement(**payload)


def run_seed(problem_name: str, seed: int, budget: int) -> dict:
    problem = builtin_problems()[problem_name]
    run_dir = OUT / f"{problem_name}_s{seed}"
    shutil.rmtree(run_dir, ignore_errors=True)
    context = OptimizationContext.from_space(
        problem.space,
        budget_total=budget,
        batch_size=4,
        multi_fidelity=callable(getattr(problem, "fidelity_fn", None)),
        n_objectives=problem.objective.n_objectives,
        n_constraints=len(getattr(problem.objective, "constraints", ()) or ()),
        noise="medium",
    )
    study = Study.create(
        run_dir, problem.space, problem.objective, algorithm="auto",
        budget=budget, batch_size=4, context=context, seed=seed,
    )
    report = run_campaign(
        study, SimulatedInstrument(problem), max_steps=60, max_workers=1,
        log_path=run_dir / "execution_log.jsonl",
    )
    checklist = study.delivery_checklist()
    return {
        "problem": problem_name,
        "seed": seed,
        "algorithm": study.algorithm,
        "n_observations": len(study.dataset.observations),
        "delivery_ok": bool(checklist) and all(item["ok"] for item in checklist),
        "missing": ";".join(item["item"] for item in checklist if not item["ok"]),
        "items": len(checklist),
        "stop_reason": report.get("stopped_reason") or "",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--budget", type=int, default=24)
    parser.add_argument("--out", default=str(OUT / "results.csv"))
    args = parser.parse_args()

    shutil.rmtree(OUT, ignore_errors=True)
    rows = [
        run_seed(problem, seed, args.budget)
        for problem in PROBLEMS
        for seed in args.seeds
    ]
    for problem in PROBLEMS:
        arm = [row for row in rows if row["problem"] == problem]
        print(json.dumps({
            "problem": problem,
            "n": len(arm),
            "delivery_ok": sum(row["delivery_ok"] for row in arm),
            "obs_mean": round(statistics.mean(row["n_observations"] for row in arm), 1),
            "algorithms": sorted({row["algorithm"] for row in arm}),
            "missing": sorted({row["missing"] for row in arm if row["missing"]}),
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