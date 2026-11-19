"""Long unattended run through the executor adapter (perf + integrity evidence).

Mirrors ``scripts/long_run_drill.py`` but drives the campaign with ``run_campaign``
(parallel workers, simulated instrument failures) instead of a hand-written loop.

Run:
    py -3 scripts/adapter_long_run.py --budget 120 --batch 6 --workers 4
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from condopt.benchmarks import builtin_problems  # noqa: E402
from condopt.executor import ExecutionRequest, Executor, Measurement, run_campaign  # noqa: E402
from condopt.router import OptimizationContext  # noqa: E402
from condopt.study import Study  # noqa: E402

OUT = ROOT / "runs" / "adapter_long_run"


class NoisyChemist(Executor):
    """Also records per-call latency so we can report a p95 for the instrument."""

    def __init__(self, problem, noise_std: float = 1.5, fail_every: int = 11, seed: int = 0):
        self.problem = problem
        self.noise_std = noise_std
        self.fail_every = fail_every
        self.rng = random.Random(seed)
        self.calls = 0
        # ``max_workers > 1`` calls measure() from several threads; an unsynchronised
        # counter can skip the exact multiples used for failure injection.
        self._lock = __import__("threading").Lock()
        self.latencies: list[float] = []
        self.multi_objective = bool(
            getattr(problem.objective, "is_multi_objective", False)
        )

    def _constraints(self, config: dict) -> dict:
        """Some benchmarks (mf_moo4) declare no constraints -> constraints_fn is None."""
        fn = getattr(self.problem, "constraints_fn", None)
        return dict(fn(config)) if callable(fn) else {}

    def measure(self, request: ExecutionRequest) -> Measurement:
        started = time.perf_counter()
        with self._lock:
            self.calls += 1
            call_index = self.calls
        time.sleep(0.002)  # pretend the instrument takes a moment
        self.latencies.append(time.perf_counter() - started)
        if self.fail_every and call_index % self.fail_every == 0:
            return Measurement(
                id=request.id, fidelity=request.fidelity, failed=True, note="instrument error"
            )
        if self.multi_objective:
            # Multi-fidelity MOO problems expose objectives_fidelity_fn(config, s);
            # the plain objectives_fn is full-fidelity only.
            fidelity_objectives = getattr(self.problem, "objectives_fidelity_fn", None)
            if callable(fidelity_objectives):
                values = fidelity_objectives(request.x, request.fidelity)
            else:
                values = self.problem.objectives_fn(request.x)
            primary = float(values.get("yield", next(iter(values.values()))))
            others = {
                name: float(value)
                for name, value in values.items()
                if name != "yield"
            }
            return Measurement(
                id=request.id,
                y=primary + self.rng.gauss(0.0, self.noise_std),
                objectives={
                    name: value + self.rng.gauss(0.0, self.noise_std)
                    for name, value in others.items()
                },
                fidelity=request.fidelity,
                constraints=self._constraints(request.x),
                cost=request.fidelity,
            )
        y = float(self.problem.fidelity_fn(request.x, request.fidelity))
        y += self.rng.gauss(0.0, self.noise_std)
        return Measurement(
            id=request.id, y=y, fidelity=request.fidelity,
            constraints=self._constraints(request.x), cost=request.fidelity,
        )


def _dir_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def run(
    budget: int = 120,
    batch_size: int = 6,
    workers: int = 4,
    seed: int = 0,
    force_budget: bool = False,
    patience: int | None = None,
    problem_name: str = "mf_constrained",
) -> dict:
    shutil.rmtree(OUT, ignore_errors=True)
    problem = builtin_problems()[problem_name]
    multi_objective = bool(getattr(problem.objective, "is_multi_objective", False))
    multi_fidelity = hasattr(problem, "fidelity_fn") and "mf" in problem_name
    constraints = len(getattr(problem.objective, "constraints", ()) or ())
    context = OptimizationContext.from_space(
        problem.space, budget_total=budget, batch_size=batch_size,
        multi_fidelity=multi_fidelity,
        n_objectives=problem.objective.n_objectives,
        n_constraints=constraints,
        noise="medium",
    )
    create_kwargs = {}
    if patience is not None:
        create_kwargs["patience"] = patience
    study = Study.create(
        OUT, problem.space, problem.objective, algorithm="auto",
        budget=budget, batch_size=batch_size, cost_budget=budget, context=context, seed=seed,
        **create_kwargs,
    )
    executor = NoisyChemist(problem, seed=seed)
    started = time.perf_counter()
    segments = []
    continuations = 0
    report = run_campaign(
        study, executor, max_workers=workers, log_path=OUT / "execution_log.jsonl"
    )
    segments.append(report)
    # A stagnation stop is not "budget exhausted": keep going by re-tuning, which
    # measured better than stopping (see research/STAGNATION_TUNING.md).
    while force_budget and continuations < 8:
        if len(study.dataset.observations) >= budget or study.remaining() <= 0:
            break
        study.retune_for_exploration()
        report = run_campaign(
            study, executor, max_workers=workers, log_path=OUT / "execution_log.jsonl"
        )
        segments.append(report)
        continuations += 1
    report["wall_seconds"] = round(time.perf_counter() - started, 2)
    report["continuations"] = continuations
    report["attempts"] = sum(segment["attempts"] for segment in segments)
    report["failures"] = sum(segment["failures"] for segment in segments)
    report["steps"] = sum(segment["steps"] for segment in segments)

    # "digest" is the report + recommendation + next-step bundle (CLI: condopt digest).
    digest_started = time.perf_counter()
    (OUT / "digest.md").write_text(study.report(), encoding="utf-8")
    study.decision(explain=True)
    study.next_action()
    report["digest_seconds"] = round(time.perf_counter() - digest_started, 3)

    observations = study.dataset.observations
    ids = [obs.meta.get("suggestion_id") for obs in observations]
    report["duplicate_suggestion_ids"] = len(ids) - len(set(ids))
    report["integration_ok"] = report["integrity_ok"]
    report["best_score"] = round(float(study.best().y), 4) if study.best() else None
    report["dataset_kb"] = round(
        (OUT / "dataset.jsonl").stat().st_size / 1024, 1
    ) if (OUT / "dataset.jsonl").exists() else 0.0
    constraints = tuple(getattr(problem.objective, "constraints", ()) or ())
    limits = getattr(problem.objective, "constraint_limits", {}) or {}
    feasible_full = 0
    for observation in study.dataset.observations:
        if float(observation.fidelity) < 1.0 - 1e-9:
            continue
        ok = True
        for name in constraints:
            limit = limits.get(name) or {}
            value = (observation.constraints or {}).get(name)
            if value is None or float(value) > float(limit.get("value", 0.0)) + 1e-9:
                ok = False
                break
        feasible_full += int(ok)
    report["n_feasible_full_fidelity"] = feasible_full
    report["problem"] = problem_name
    report["multi_fidelity"] = multi_fidelity
    if multi_objective:
        trajectory = study.progress_trajectory()
        feasible = [
            row["hypervolume_feasible"]
            for row in trajectory
            if row.get("hypervolume_feasible") is not None
        ]
        report["problem"] = problem_name
        report["progress_points"] = len(trajectory)
        report["final_hypervolume"] = trajectory[-1]["hypervolume"] if trajectory else None
        report["feasible_hypervolume_curve"] = (
            [round(feasible[0], 4), round(feasible[-1], 4)] if feasible else None
        )
        report["final_feasible_fraction"] = (
            trajectory[-1].get("feasible_fraction") if trajectory else None
        )
    report["run_dir_kb"] = round(_dir_size(OUT) / 1024, 1)
    if executor.latencies:
        ordered = sorted(executor.latencies)
        index = min(len(ordered) - 1, int(0.95 * len(ordered)))
        report["instrument_p95_seconds"] = round(ordered[index], 4)
        report["instrument_mean_seconds"] = round(statistics.mean(ordered), 4)
    report["algorithm"] = study.algorithm
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=120)
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--force-budget", action="store_true",
        help="停滞停止后 retune 继续，直到评估预算用尽（对齐 long_run_drill 的强制预算）",
    )
    parser.add_argument(
        "--problem", default="mf_constrained",
        help="基准问题（默认 mf_constrained；moo_constrained 走多目标 + 约束）",
    )
    parser.add_argument(
        "--patience", type=int, default=None,
        help="停滞判定轮数（默认 3）；设大可在一次运行里用满预算",
    )
    args = parser.parse_args()
    report = run(
        args.budget, args.batch, args.workers, args.seed,
        force_budget=args.force_budget, patience=args.patience,
        problem_name=args.problem,
    )
    (OUT / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    keys = (
        "algorithm", "n_observations", "steps", "attempts", "failures",
            "still_pending", "cost_spent", "wall_seconds", "digest_seconds",
            "instrument_p95_seconds", "dataset_kb", "run_dir_kb",
            "duplicate_suggestion_ids", "integrity_ok", "best_score", "stopped_reason",
            "continuations", "n_feasible_full_fidelity",
    )
    optional = ("final_hypervolume", "feasible_hypervolume_curve", "final_feasible_fraction")
    print(json.dumps({
        **{key: report[key] for key in keys},
        **{key: report[key] for key in optional if key in report},
    }, ensure_ascii=False))
    return 0 if report["integrity_ok"] and report["duplicate_suggestion_ids"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())