"""Multi-objective + constrained campaign through the executor adapter.

Objectives: ``yield`` (max) and ``cost`` (min) with a ``T_max <= 80`` constraint,
measured by a simulated instrument in 4 parallel wells.

Run:
    py -3 examples/multiobjective_executor.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from condopt.benchmarks import builtin_problems  # noqa: E402
from condopt.base import Objective  # noqa: E402
from condopt.executor import ExecutionRequest, Executor, Measurement, run_campaign  # noqa: E402
from condopt.router import OptimizationContext  # noqa: E402
from condopt.study import Study  # noqa: E402

OUT = ROOT / "runs" / "multiobjective_executor"


class TwoMetricInstrument(Executor):
    """Measures yield, cost and the temperature constraint for one condition."""

    def __init__(self, problem, noise_std: float = 1.0, seed: int = 0) -> None:
        self.problem = problem
        self.noise_std = noise_std
        self.rng = __import__("random").Random(seed)
        self.calls = 0

    def measure(self, request: ExecutionRequest) -> Measurement:
        self.calls += 1
        # objectives_fn returns {"yield": ..., "cost": ...} for this benchmark
        values = self.problem.objectives_fn(request.x)
        primary = float(values["yield"])
        secondary = float(values["cost"])
        return Measurement(
            id=request.id,
            y=primary + self.rng.gauss(0.0, self.noise_std),
            objectives={"cost": secondary + self.rng.gauss(0.0, self.noise_std)},
            fidelity=request.fidelity,
            constraints=self.problem.constraints_fn(request.x),
            cost=request.fidelity,
        )


def run(seed: int = 0, budget: int = 24, batch_size: int = 4, workers: int = 4) -> dict:
    shutil.rmtree(OUT, ignore_errors=True)
    problem = builtin_problems()["moo_constrained"]
    objective = Objective(
        name="yield",
        direction="max",
        objectives=("yield", "cost"),
        directions=("max", "min"),
        constraints=("T_max", "time_max"),
        constraint_limits={
            "T_max": {"op": "<=", "value": 78.0},
            "time_max": {"op": "<=", "value": 50.0},
        },
    )
    context = OptimizationContext.from_space(
        problem.space, budget_total=budget, batch_size=batch_size,
        n_objectives=2, n_constraints=2, noise="medium",
    )
    study = Study.create(
        OUT, problem.space, objective, algorithm="auto",
        budget=budget, batch_size=batch_size, context=context, seed=seed,
    )
    report = run_campaign(study, TwoMetricInstrument(problem, seed=seed), max_workers=workers)
    observations = study.dataset.observations
    feasible = [
        obs for obs in observations
        if float(obs.constraints.get("T_max", 1e9)) <= 78.0
        and float(obs.constraints.get("time_max", 1e9)) <= 50.0
    ]
    summary = study.pareto_summary()
    progress_rows = study.progress_trajectory()
    feasible_volumes = [
        row.get("hypervolume_feasible")
        for row in progress_rows
        if row.get("hypervolume_feasible") is not None
    ]
    report.update({
        "algorithm": study.algorithm,
        "n_feasible": len(feasible),
        "n_objective_vectors": sum(1 for obs in observations if obs.objectives),
        "pareto_size": summary.get("n_front") or summary.get("front_size"),
        "hypervolume": summary.get("hypervolume"),
        "feasible_hypervolume": feasible_volumes[-1] if feasible_volumes else None,
        "progress_rows": (
            len((OUT / "progress.jsonl").read_text(encoding="utf-8").splitlines())
            if (OUT / "progress.jsonl").exists()
            else 0
        ),
        "fidelity_summary": (
            study.fidelity_summary() if hasattr(study, "fidelity_summary") else None
        ),
    })
    return report


def main() -> int:
    report = run()
    (OUT / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        key: report[key]
        for key in (
            "algorithm", "n_observations", "n_feasible", "n_objective_vectors",
            "pareto_size", "hypervolume", "feasible_hypervolume",
            "final_hypervolume", "progress_points",
            "cost_spent", "still_pending", "stopped_reason",
            "integrity_ok", "ok",
        )
    }, ensure_ascii=False))
    return 0 if report["ok"] and report["n_objective_vectors"] == report["n_observations"] else 1


if __name__ == "__main__":
    raise SystemExit(main())