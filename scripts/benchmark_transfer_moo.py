"""Does a Pareto-front warm start help a new multi-objective campaign?

Source campaign runs ``moo_constrained`` first; the target campaign then either
starts from scratch or transfers the source front (``transfer_from``) with a
couple of ``top_k`` / ``prior_reliability`` settings.  Equal budget, same seeds.

Run:
    py -3 scripts/benchmark_transfer_moo.py --source-budget 20 --target-budget 20 --seeds 0 1 2
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

OUT = ROOT / "runs" / "benchmark_transfer_moo"
PROBLEM = "moo_constrained"


def _context(budget: int, batch: int, problem_name: str = PROBLEM) -> OptimizationContext:
    problem = builtin_problems()[problem_name]
    return OptimizationContext.from_space(
        problem.space, budget_total=budget, batch_size=batch,
        n_objectives=problem.objective.n_objectives,
        n_constraints=len(problem.objective.constraints),
        noise="medium",
    )


def _drive(study: Study, problem, budget: int, batch: int) -> None:
    fidelity_objectives = getattr(problem, "objectives_fidelity_fn", None)
    constraints_fn = getattr(problem, "constraints_fn", None)
    for _ in range(max(200, 8 * budget)):
        action = study.next_action()
        kind = action["action"]
        if kind == "stop":
            break
        if kind == "ask":
            study.ask(int(action.get("ask_n") or batch))
        elif kind == "confirm":
            study.confirm(int(action.get("confirm_n") or batch))
        elif kind == "tell":
            results = []
            for item in study.pending:
                fidelity = float(item.get("fidelity", 1.0) or 1.0)
                values = (
                    fidelity_objectives(item["x"], fidelity)
                    if callable(fidelity_objectives)
                    else problem.objectives_fn(item["x"])
                )
                results.append({
                    "id": item["id"],
                    "y": float(values["yield"]),
                    "objectives": {"cost": float(values["cost"])},
                    "fidelity": fidelity,
                    "cost": fidelity,
                    "constraints": (
                        dict(constraints_fn(item["x"])) if callable(constraints_fn) else {}
                    ),
                })
            study.tell(results)


def make_source(seed: int, budget: int, batch: int, problem_name: str = PROBLEM) -> Path:
    problem = builtin_problems()[problem_name]
    run_dir = OUT / f"source_{problem_name}_s{seed}"
    shutil.rmtree(run_dir, ignore_errors=True)
    study = Study.create(
        run_dir, problem.space, problem.objective, algorithm="cqehvi",
        budget=budget, batch_size=batch,
        context=_context(budget, batch, problem_name), seed=seed,
    )
    _drive(study, problem, budget, batch)
    return run_dir


def run_target(
    seed: int,
    budget: int,
    batch: int,
    source: Path | None = None,
    top_k: int = 3,
    reliability: float = 1.0,
    problem_name: str = PROBLEM,
    include_low_fidelity: bool = False,
) -> dict:
    problem = builtin_problems()[problem_name]
    label = (
        "baseline"
        if source is None
        else f"transfer_k{top_k}_r{reliability}{'_lf' if include_low_fidelity else ''}"
    )
    run_dir = OUT / f"target_{problem_name}_{label}_s{seed}"
    shutil.rmtree(run_dir, ignore_errors=True)
    study = Study.create(
        run_dir, problem.space, problem.objective, algorithm="cqehvi",
        budget=budget, batch_size=batch,
        context=_context(budget, batch, problem_name), seed=seed,
    )
    transfer_info = None
    if source is not None:
        # transfer_from loads the source itself (path or Study accepted elsewhere)
        transfer_info = study.transfer_from(
            source,
            top_k=top_k,
            prior_reliability=reliability,
            include_low_fidelity=include_low_fidelity,
        )
    _drive(study, problem, budget, batch)
    summary = study.pareto_summary()
    feasible = study.feasible_pareto_summary()
    return {
        "arm": label,
        "seed": seed,
        "n_observations": len(study.dataset.observations),
        "n_front": summary["n_front"],
        "hypervolume": summary["hypervolume"],
        "hypervolume_feasible": feasible["hypervolume"],
        "n_feasible_front": feasible["n_front"],
        "transferred": 0 if transfer_info is None else transfer_info.get("imported", 0),
        "transfer_selection": None if transfer_info is None else transfer_info.get("selection"),
        "include_low_fidelity": include_low_fidelity,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-budget", type=int, default=20)
    parser.add_argument("--target-budget", type=int, default=20)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--with-low-fidelity", action="store_true",
        help="额外测试把低保真源点一起迁移（多保真源时更省目标预算）",
    )
    parser.add_argument(
        "--problem", default=PROBLEM,
        help="基准问题（默认 moo_constrained；mf_moo_constrained 走多保真）",
    )
    parser.add_argument("--out", default=str(OUT / "results.csv"))
    args = parser.parse_args()

    rows = []
    for seed in args.seeds:
        source = make_source(seed, args.source_budget, args.batch, args.problem)
        arms = [
            (None, 0, 1.0, False),
            (source, 3, 1.0, False),
            (source, 5, 1.0, False),
            (source, 5, 0.5, False),
        ]
        if args.with_low_fidelity:
            # Multi-fidelity sources: transferring cheap points too costs less of
            # the target budget than transferring full-fidelity ones.
            arms += [(source, 5, 1.0, True), (source, 8, 1.0, True)]
        for source_path, top_k, reliability, low_fidelity in arms:
            rows.append(run_target(
                seed, args.target_budget, args.batch,
                source=source_path, top_k=top_k, reliability=reliability,
                problem_name=args.problem, include_low_fidelity=low_fidelity,
            ))
        print(f"seed {seed} done")

    by_arm: dict[str, list[dict]] = {}
    for row in rows:
        by_arm.setdefault(row["arm"], []).append(row)
    for arm, subset in by_arm.items():
        hv = [
            row["hypervolume_feasible"]
            for row in subset
            if row["hypervolume_feasible"] is not None
        ]
        print(json.dumps({
            "arm": arm,
            "n": len(subset),
            "feasible_hv_mean": round(statistics.mean(hv), 2) if hv else None,
            "n_feasible_front_mean": round(
                statistics.mean(row["n_feasible_front"] for row in subset), 2
            ),
            "transferred_mean": round(
                statistics.mean(row["transferred"] for row in subset), 2
            ),
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