"""Is the recommended prior count (top_k 1-3) actually the sweet spot?

The transfer warning says "top_k 1-3; the damage grows with top_k on a bad source".
This sweeps top_k on a *good* source (same problem, different seed) to check that
the recommendation is not just conservative.

Run:
    py -3 scripts/sweep_transfer_top_k.py --top-ks 0 1 2 3 5 8 --seeds 0 1 2
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

OUT = ROOT / "runs" / "sweep_transfer_top_k"


def _context(problem, budget: int, batch: int) -> OptimizationContext:
    return OptimizationContext.from_space(
        problem.space,
        budget_total=budget,
        batch_size=batch,
        multi_fidelity=callable(getattr(problem, "fidelity_fn", None)),
        n_objectives=problem.objective.n_objectives,
        n_constraints=len(getattr(problem.objective, "constraints", ()) or ()),
        noise="medium",
    )


def _evaluate(problem, item) -> dict:
    x = item["x"]
    fidelity = float(item.get("fidelity", 1.0))
    fidelity_fn = getattr(problem, "fidelity_fn", None)
    y = float(fidelity_fn(x, fidelity)) if callable(fidelity_fn) else float(problem.fn(x))
    payload: dict = {"id": item["id"], "y": y, "fidelity": fidelity, "cost": fidelity}
    constraints = getattr(problem, "constraints_fn", None)
    if callable(constraints):
        payload["constraints"] = constraints(x)
    return payload


def _drive(problem, study: Study) -> None:
    for _ in range(400):
        action = study.next_action()
        kind = action["action"]
        if kind == "stop":
            break
        if kind == "ask":
            study.ask(int(action.get("ask_n") or 3))
        elif kind == "confirm":
            study.confirm(int(action.get("confirm_n") or 3))
        else:
            study.tell([_evaluate(problem, item) for item in study.pending])
            study.record_progress()


def build_source(problem, seed: int, budget: int) -> Study:
    run_dir = OUT / f"source_s{seed}"
    shutil.rmtree(run_dir, ignore_errors=True)
    study = Study.create(
        run_dir, problem.space, problem.objective, algorithm="auto",
        budget=budget, batch_size=3, context=_context(problem, budget, 3), seed=seed + 100,
    )
    _drive(problem, study)
    return study


def run_arm(
    problem, source: Study, top_k: int, seed: int, budget: int, reliability: float,
    include_low: bool,
) -> dict:
    run_dir = OUT / f"k{top_k}_s{seed}"
    shutil.rmtree(run_dir, ignore_errors=True)
    target = Study.create(
        run_dir, problem.space, problem.objective, algorithm="auto",
        budget=budget, batch_size=3, context=_context(problem, budget, 3), seed=seed,
    )
    imported = 0
    if top_k > 0:
        report = target.transfer_from(
            source.dir, top_k=top_k, prior_reliability=reliability,
            include_low_fidelity=include_low, reason="top-k sweep",
        )
        imported = int(report["imported"])
    _drive(problem, target)
    constraints = tuple(getattr(problem.objective, "constraints", ()) or ())
    if constraints:
        feasible = [obs for obs in target.dataset.observations if target._is_feasible(obs)]
        best = max((float(obs.y) for obs in feasible), default=None)
    else:
        incumbent = target.best()
        best = None if incumbent is None else float(incumbent.y)
    return {
        "top_k": top_k,
        "seed": seed,
        "imported": imported,
        "best": None if best is None else round(best, 4),
        "n_observations": len(target.dataset.observations),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", default="mf_constrained")
    parser.add_argument("--top-ks", type=int, nargs="+", default=[0, 1, 2, 3, 5, 8])
    parser.add_argument("--budget", type=int, default=30)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--reliability", type=float, default=1.0)
    parser.add_argument("--include-low-fidelity", action="store_true")
    parser.add_argument("--out", default=str(OUT / "results.csv"))
    args = parser.parse_args()

    problem = builtin_problems()[args.problem]
    shutil.rmtree(OUT, ignore_errors=True)
    rows: list[dict] = []
    for seed in args.seeds:
        source = build_source(problem, seed, args.budget)
        for top_k in args.top_ks:
            rows.append(
                run_arm(
                    problem, source, top_k, seed, args.budget, args.reliability,
                    args.include_low_fidelity,
                )
            )
    baseline = statistics.mean(
        row["best"] for row in rows if row["top_k"] == 0 and row["best"] is not None
    )
    for top_k in args.top_ks:
        arm = [row for row in rows if row["top_k"] == top_k and row["best"] is not None]
        if not arm:
            continue
        mean = statistics.mean(row["best"] for row in arm)
        print(json.dumps({
            "top_k": top_k,
            "n": len(arm),
            "best_mean": round(mean, 4),
            "delta_vs_cold": round(mean - baseline, 4),
            "imported_mean": round(statistics.mean(row["imported"] for row in arm), 2),
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