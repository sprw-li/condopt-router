"""Does warm-starting from a prior campaign help? (same problem, equal budget)

Arm A (cold): 18 evaluations from scratch with bayes.
Arm B (warm): a 6-evaluation sobol source campaign, then ``transfer_from`` its
best points into a bayes target that spends the remaining 12 evaluations.
Both arms evaluate the same total number of points.

Run:
    py -3 scripts/benchmark_transfer.py --seeds 0 1 2
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
from condopt.study import Study  # noqa: E402

OUT = ROOT / "runs" / "benchmark_transfer"


def _evaluate_pending(study: Study, problem) -> None:
    results = []
    for item in study.pending:
        payload = {"id": item["id"]}
        if problem.objective.is_multi_objective:
            payload["objectives"] = problem.objectives_fn(item["x"])
        else:
            payload["y"] = problem.fn(item["x"])
        if problem.constraints_fn is not None:
            payload["constraints"] = problem.constraints_fn(item["x"])
        results.append(payload)
    study.tell(results)


def _drive(study: Study, problem, batch_size: int) -> None:
    while study.remaining() > 0:
        n = min(batch_size, study.remaining())
        study.ask(n)
        _evaluate_pending(study, problem)


def run_arm(
    arm: str,
    problem_name: str = "rosenbrock10",
    total_budget: int = 18,
    source_budget: int = 6,
    batch_size: int = 3,
    seed: int = 0,
) -> dict:
    problem = builtin_problems()[problem_name]
    run_dir = OUT / f"{arm}_{problem_name}_{seed}"
    shutil.rmtree(run_dir, ignore_errors=True)
    transferred = 0
    if arm == "warm":
        source_dir = OUT / f"source_{problem_name}_{seed}"
        shutil.rmtree(source_dir, ignore_errors=True)
        source = Study.create(
            source_dir, problem.space, problem.objective,
            algorithm="sobol", budget=source_budget, batch_size=batch_size, seed=seed,
        )
        _drive(source, problem, batch_size)
        target = Study.create(
            run_dir, problem.space, problem.objective,
            algorithm="bayes", budget=total_budget, batch_size=batch_size, seed=seed,
        )
        # keep the best 2/3 of the source points as priors
        report = target.transfer_from(source_dir, top_k=max(1, source_budget * 2 // 3))
        transferred = int(report.get("imported", 0))
    else:
        target = Study.create(
            run_dir, problem.space, problem.objective,
            algorithm="bayes", budget=total_budget, batch_size=batch_size, seed=seed,
        )
    _drive(target, problem, batch_size)
    best = target.best()
    return {
        "arm": arm,
        "problem": problem_name,
        "seed": seed,
        "transferred": transferred,
        "n_observations": len(target.dataset),
        "best_score": None if best is None else round(float(best.score(
            problem.objective.name, problem.objective.direction
        )), 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", nargs="+", default=["rosenbrock10", "sphere2d"])
    parser.add_argument("--total-budget", type=int, default=18)
    parser.add_argument("--source-budget", type=int, default=6)
    parser.add_argument("--batch", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--out", default=str(OUT / "results.csv"))
    args = parser.parse_args()

    rows = []
    for problem_name in args.problems:
        for arm in ("cold", "warm"):
            for seed in args.seeds:
                row = run_arm(
                    arm, problem_name, args.total_budget, args.source_budget,
                    args.batch, seed,
                )
                rows.append(row)
                print(json.dumps(row, ensure_ascii=False))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("\nsummary")
    for problem_name in args.problems:
        for arm in ("cold", "warm"):
            subset = [
                row["best_score"] for row in rows
                if row["problem"] == problem_name and row["arm"] == arm
            ]
            print(json.dumps({
                "problem": problem_name, "arm": arm, "n": len(subset),
                "best_score_mean": round(statistics.mean(subset), 4),
            }, ensure_ascii=False))
    print(f"CSV: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())