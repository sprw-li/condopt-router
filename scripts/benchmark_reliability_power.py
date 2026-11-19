"""A *powered* prior-reliability test: import 5+ full-fidelity priors.

The old band benchmark could only import one prior (multi-fidelity sources spend most
of their budget at low fidelity), so ``prior_reliability`` could not show any effect.
This builds a **single-fidelity** source with many full-fidelity points, then imports
``top_k=5`` at different reliability levels into an identical target.

Run:
    py -3 scripts/benchmark_reliability_power.py --seeds 0 1 2
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

OUT = ROOT / "runs" / "benchmark_reliability_power"


def _context(problem, budget: int, batch: int, multi_fidelity: bool) -> OptimizationContext:
    return OptimizationContext.from_space(
        problem.space,
        budget_total=budget,
        batch_size=batch,
        multi_fidelity=multi_fidelity,
        n_objectives=problem.objective.n_objectives,
        n_constraints=len(getattr(problem.objective, "constraints", ()) or ()),
        noise="medium",
    )


def _evaluate(problem, item, multi_fidelity: bool) -> dict:
    x = item["x"]
    fidelity = float(item.get("fidelity", 1.0)) if multi_fidelity else 1.0
    fidelity_fn = getattr(problem, "fidelity_fn", None)
    y = float(fidelity_fn(x, fidelity)) if callable(fidelity_fn) and multi_fidelity else float(problem.fn(x))
    payload: dict = {"id": item["id"], "y": y, "fidelity": fidelity, "cost": fidelity}
    constraints = getattr(problem, "constraints_fn", None)
    if callable(constraints):
        payload["constraints"] = constraints(x)
    return payload


def _drive(problem, study: Study, multi_fidelity: bool) -> None:
    for _ in range(max(120, 8 * int(getattr(study, "budget", 30) or 30))):
        action = study.next_action()
        if action["action"] == "stop":
            break
        if action["action"] == "ask":
            study.ask(int(action.get("ask_n") or 3))
        elif action["action"] == "confirm":
            study.confirm(int(action.get("confirm_n") or 3))
        else:
            study.tell([_evaluate(problem, item, multi_fidelity) for item in study.pending])
            study.record_progress()


def build_source(problem, budget: int, seed: int, multi_fidelity: bool) -> Study:
    run_dir = OUT / f"source_s{seed}"
    shutil.rmtree(run_dir, ignore_errors=True)
    study = Study.create(
        run_dir, problem.space, problem.objective, algorithm="auto",
        budget=budget, batch_size=3,
        context=_context(problem, budget, 3, multi_fidelity), seed=seed + 100,
    )
    _drive(problem, study, multi_fidelity)
    full = [obs for obs in study.dataset.observations if float(obs.fidelity) >= 1.0]
    print(json.dumps({"source_seed": seed, "n_obs": len(study.dataset.observations),
                      "n_full_fidelity": len(full)}, ensure_ascii=False))
    return study


def run_arm(
    problem, source: Study, reliability: float | None, seed: int, budget: int,
    multi_fidelity: bool, top_k: int,
) -> dict:
    run_dir = OUT / f"arm_{'cold' if reliability is None else f'r{reliability}'}_s{seed}"
    shutil.rmtree(run_dir, ignore_errors=True)
    target = Study.create(
        run_dir, problem.space, problem.objective, algorithm="auto",
        budget=budget, batch_size=3,
        context=_context(problem, budget, 3, multi_fidelity), seed=seed,
    )
    imported = 0
    if reliability is not None:
        imported = int(
            target.transfer_from(
                source.dir, top_k=top_k, prior_reliability=reliability,
                reason="reliability power test",
            )["imported"]
        )
    _drive(problem, target, multi_fidelity)
    constraints = tuple(getattr(problem.objective, "constraints", ()) or ())
    if constraints:
        feasible = [obs for obs in target.dataset.observations if target._is_feasible(obs)]
        best = max((float(obs.y) for obs in feasible), default=None)
    else:
        incumbent = target.best()
        best = None if incumbent is None else float(incumbent.y)
    return {
        "arm": "cold" if reliability is None else f"r{reliability}",
        "reliability": reliability,
        "seed": seed,
        "imported": imported,
        "best": None if best is None else round(best, 4),
        "n_obs": len(target.dataset.observations),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", default="mf_constrained")
    parser.add_argument("--source-budget", type=int, default=25)
    parser.add_argument("--budget", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--reliabilities", type=float, nargs="+", default=[1.0, 0.7, 0.4])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--source-multi-fidelity", action="store_true")
    parser.add_argument("--out", default=str(OUT / "results.csv"))
    args = parser.parse_args()

    problem = builtin_problems()[args.problem]
    shutil.rmtree(OUT, ignore_errors=True)
    target_multi_fidelity = callable(getattr(problem, "fidelity_fn", None))
    rows: list[dict] = []
    for seed in args.seeds:
        source = build_source(problem, args.source_budget, seed, args.source_multi_fidelity)
        rows.append(
            run_arm(problem, source, None, seed, args.budget, target_multi_fidelity, args.top_k)
        )
        for reliability in args.reliabilities:
            rows.append(
                run_arm(
                    problem, source, reliability, seed, args.budget,
                    target_multi_fidelity, args.top_k,
                )
            )

    print(json.dumps({"target_multi_fidelity": target_multi_fidelity}, ensure_ascii=False))
    arms = ["cold"] + [f"r{value}" for value in args.reliabilities]
    cold = statistics.mean(
        row["best"] for row in rows if row["arm"] == "cold" and row["best"] is not None
    )
    for arm in arms:
        subset = [row for row in rows if row["arm"] == arm and row["best"] is not None]
        print(json.dumps({
            "arm": arm,
            "n": len(subset),
            "best_mean": round(statistics.mean(row["best"] for row in subset), 4),
            "delta_vs_cold": round(statistics.mean(row["best"] for row in subset) - cold, 4),
            "imported_mean": round(statistics.mean(row["imported"] for row in subset), 2),
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