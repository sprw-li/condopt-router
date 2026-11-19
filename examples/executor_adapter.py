"""End-to-end executor adapter demo: constraint + multi-fidelity + instrument failures.

Implements ``Executor.measure`` for a simulated chemist (with occasional failed
runs) and drives a full campaign through ``run_campaign``.

Run:
    py -3 examples/executor_adapter.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from condopt.benchmarks import builtin_problems  # noqa: E402
from condopt.executor import ExecutionRequest, Executor, Measurement, run_campaign  # noqa: E402
from condopt.router import OptimizationContext  # noqa: E402
from condopt.study import Study  # noqa: E402

OUT = ROOT / "runs" / "executor_adapter"


class SimulatedChemist(Executor):
    """Runs one condition; every 7th request 'fails' (simulating an instrument error)."""

    def __init__(self, problem, noise_std: float = 2.0, fail_every: int = 7) -> None:
        self.problem = problem
        self.noise_std = noise_std
        self.fail_every = fail_every
        self.calls = 0
        self.rng = __import__("random").Random(0)

    def measure(self, request: ExecutionRequest) -> Measurement:
        self.calls += 1
        if self.fail_every and self.calls % self.fail_every == 0:
            return Measurement(
                id=request.id, fidelity=request.fidelity, failed=True,
                note="instrument error (simulated)",
            )
        y = float(self.problem.fidelity_fn(request.x, request.fidelity))
        y += self.rng.gauss(0.0, self.noise_std)
        return Measurement(
            id=request.id,
            y=y,
            fidelity=request.fidelity,
            constraints=self.problem.constraints_fn(request.x),
            cost=request.fidelity,
        )


# ``algorithm="mfbo"`` keeps the demo deterministic; ``"auto"`` lets the router
# decide (at a tiny budget it advises space filling -- see docs/ROUTING_POLICY.md).
def run(
    seed: int = 0,
    budget: int = 24,
    cost_budget: float = 10.0,
    max_workers: int = 1,
    timeout: float | None = None,

    algorithm: str = "mfbo",) -> dict:
    shutil.rmtree(OUT, ignore_errors=True)
    problem = builtin_problems()["mf_constrained"]
    context = OptimizationContext.from_space(
        problem.space, budget_total=budget, batch_size=3,
        multi_fidelity=True, n_constraints=1, noise="medium",
    )
    study = Study.create(
        OUT, problem.space, problem.objective, algorithm=algorithm,
        budget=budget, batch_size=3, cost_budget=cost_budget, context=context, seed=seed,
    )
    executor = SimulatedChemist(problem)
    import time as _time

    started = _time.perf_counter()
    report = run_campaign(
        study, executor, log_path=OUT / "execution_log.jsonl",
        max_workers=max_workers, timeout=timeout,
    )
    report["wall_seconds"] = round(_time.perf_counter() - started, 2)
    report["executor_calls"] = executor.calls
    report["algorithm"] = study.algorithm
    report["pending_after"] = len(study.pending)
    return report


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budget", type=int, default=24)
    parser.add_argument("--cost-budget", type=float, default=10.0)
    parser.add_argument("--fail-every", type=int, default=7)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=None)
    args = parser.parse_args()
    report = run(
        seed=args.seed, budget=args.budget, cost_budget=args.cost_budget,
        max_workers=args.max_workers, timeout=args.timeout,
    )
    report["fail_every"] = args.fail_every
    (OUT / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        key: report[key]
        for key in (
            "algorithm", "steps", "attempts", "failures", "n_observations",
            "still_pending", "cost_spent", "cost_budget", "stopped_reason",
            "integrity_ok", "ok",
        )
    }, ensure_ascii=False))
    print(f"execution log: {report['execution_log']}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())