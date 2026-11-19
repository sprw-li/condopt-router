"""Does importing low-fidelity priors pay off under a *cost* budget?

``transfer_from`` charges the target for the imported observations.  A
low-fidelity source point is cheap in the source campaign, but it still occupies
a target slot.  This measures the three plausible policies on a multi-fidelity
single-objective problem with a hard cost budget:

* ``none``      – no transfer (baseline);
* ``full``      – import the best full-fidelity priors only;
* ``with_low``  – import full- and low-fidelity priors.

Run:
    py -3 scripts/benchmark_transfer_cost.py --seeds 0 1 2 --budget 40
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import shutil
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from condopt.router import OptimizationContext  # noqa: E402
from condopt.executor import run_campaign  # noqa: E402
from condopt.study import Study  # noqa: E402

OUT = ROOT / "runs" / "benchmark_transfer_cost"


def _long_run_module():
    spec = importlib.util.spec_from_file_location(
        "adapter_long_run", ROOT / "scripts" / "adapter_long_run.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _problems(module):
    from condopt.benchmarks import builtin_problems

    return builtin_problems()


def _context(problem, budget: int, batch: int):
    return OptimizationContext.from_space(
        problem.space,
        budget_total=budget,
        batch_size=batch,
        multi_fidelity=hasattr(problem, "fidelity_fn"),
        n_objectives=problem.objective.n_objectives,
        n_constraints=len(getattr(problem.objective, "constraints", ()) or ()),
        noise="medium",
    )


def build_source(module, problem, source_dir: Path, seed: int, budget: int) -> Study:
    """A cheap multi-fidelity source campaign (mostly low fidelity)."""
    shutil.rmtree(source_dir, ignore_errors=True)
    context = _context(problem, budget, 4)
    study = Study.create(
        source_dir, problem.space, problem.objective, algorithm="mfbo",
        budget=budget, batch_size=4, cost_budget=budget, context=context, seed=seed,
    )
    executor = module.NoisyChemist(problem, seed=seed)
    run_campaign(
        study, executor, max_steps=40, max_workers=1,
        log_path=source_dir / "execution_log.jsonl",
    )
    return study


def run_arm(
    module, problem, source: Study, arm: str, seed: int, budget: int
) -> dict:
    run_dir = OUT / f"{arm}_s{seed}"
    shutil.rmtree(run_dir, ignore_errors=True)
    context = _context(problem, budget, 4)
    target = Study.create(
        run_dir, problem.space, problem.objective, algorithm="mfbo",
        budget=budget, batch_size=4, cost_budget=budget, context=context, seed=seed,
    )
    transferred = 0
    transfer_report = None
    if arm != "none":
        transfer_report = target.transfer_from(
            source.dir,
            top_k=3,
            prior_reliability=0.7,
            include_low_fidelity=(arm == "with_low"),
            reason=f"cost-budget {arm}",
        )
        transferred = int(transfer_report["imported"])
    executor = module.NoisyChemist(problem, seed=seed)
    run_campaign(
        target, executor, max_steps=40, max_workers=1,
        log_path=run_dir / "execution_log.jsonl",
    )
    best = target.best()
    observations = target.dataset.observations
    full = [obs for obs in observations if float(obs.fidelity) >= 1.0 - 1e-9]
    return {
        "arm": arm,
        "seed": seed,
        "transferred": transferred,
        "cost_charged": None if not transfer_report else transfer_report.get("cost_charged"),
        "budget_total": None if not transfer_report else transfer_report.get("budget_total"),
        "remaining_after_transfer": (
            None if not transfer_report else transfer_report.get("remaining_after_transfer")
        ),
        "n_observations": len(observations),
        "n_full_fidelity": len(full),
        "n_local_full_fidelity": len([obs for obs in full if not obs.meta.get("imported")]),
        "cost_spent": round(float(target.cost_spent()), 4),
        "best": None if best is None else round(float(best.y), 4),
        "checklist_ok": all(item["ok"] for item in target.delivery_checklist()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--budget", type=int, default=40)
    parser.add_argument("--problem", default="mf_constrained")
    parser.add_argument("--out", default=str(OUT / "results.csv"))
    args = parser.parse_args()

    module = _long_run_module()
    problems = _problems(module)
    problem = problems[args.problem]
    shutil.rmtree(OUT, ignore_errors=True)

    rows = []
    for seed in args.seeds:
        source = build_source(
            module, problem, OUT / f"source_s{seed}", seed=seed + 100, budget=args.budget
        )
        for arm in ("none", "full", "with_low"):
            rows.append(run_arm(module, problem, source, arm, seed, args.budget))

    for arm in ("none", "full", "with_low"):
        subset = [row for row in rows if row["arm"] == arm]
        print(json.dumps({
            "arm": arm,
            "n": len(subset),
            "best_mean": round(statistics.mean(row["best"] for row in subset), 4),
            "cost_mean": round(statistics.mean(row["cost_spent"] for row in subset), 3),
            "local_full_mean": round(
                statistics.mean(row["n_local_full_fidelity"] for row in subset), 2
            ),
            "imported_mean": round(
                statistics.mean(row["transferred"] for row in subset), 2
            ),
            "checklist_ok": sum(row["checklist_ok"] for row in subset),
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