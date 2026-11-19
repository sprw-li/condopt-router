"""Transfer sensitivity: how many priors, and what happens with a bad source?

Good source: evaluated with the real oracle.  Bad source: same conditions but
labels shuffled (simulates a dissimilar campaign / wrong metadata).  For each
source quality the target transfers top_k priors and then optimizes with the
remaining budget.

Run:
    py -3 scripts/sweep_transfer.py --seeds 0 1 2 --problem rosenbrock10
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from condopt.benchmarks import builtin_problems  # noqa: E402
from condopt.study import Study  # noqa: E402

OUT = ROOT / "runs" / "sweep_transfer"


def _drive(study: Study, problem, batch_size: int) -> None:
    while study.remaining() > 0:
        n = min(batch_size, study.remaining())
        study.ask(n)
        results = []
        for item in study.pending:
            results.append({"id": item["id"], "y": problem.fn(item["x"])})
        study.tell(results)


def run_point(
    quality: str,
    top_k: int,
    problem_name: str = "rosenbrock10",
    total_budget: int = 24,
    source_budget: int = 8,
    batch_size: int = 4,
    seed: int = 0,
    prior_reliability: float = 1.0,
) -> dict:
    problem = builtin_problems()[problem_name]
    source_dir = OUT / f"src_{quality}_{problem_name}_{seed}"
    shutil.rmtree(source_dir, ignore_errors=True)
    source = Study.create(
        source_dir, problem.space, problem.objective,
        algorithm="sobol", budget=source_budget, batch_size=batch_size, seed=seed,
    )
    # evaluate source with real labels, then optionally corrupt them
    source.ask(source_budget)
    labels = [problem.fn(item["x"]) for item in source.pending]
    if quality == "bad":
        rng = random.Random(seed)
        rng.shuffle(labels)
    source.tell([
        {"id": item["id"], "y": labels[index]}
        for index, item in enumerate(source.pending)
    ])

    target_dir = OUT / f"tgt_{quality}_{top_k}_{prior_reliability}_{problem_name}_{seed}"
    shutil.rmtree(target_dir, ignore_errors=True)
    target = Study.create(
        target_dir, problem.space, problem.objective,
        algorithm="bayes", budget=total_budget, batch_size=batch_size, seed=seed,
    )
    report = target.transfer_from(
        source_dir, top_k=top_k, prior_reliability=prior_reliability
    )
    _drive(target, problem, batch_size)
    best = target.best()
    score = None if best is None else float(best.score(
        problem.objective.name, problem.objective.direction
    ))
    return {
        "quality": quality,
        "top_k": top_k,
        "prior_reliability": prior_reliability,
        "seed": seed,
        "imported": report["imported"],
        "n_observations": len(target.dataset),
        "best_score": None if score is None else round(score, 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", default="rosenbrock10")
    parser.add_argument("--total-budget", type=int, default=24)
    parser.add_argument("--source-budget", type=int, default=8)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--reliabilities", type=float, nargs="+", default=[1.0])
    parser.add_argument("--qualities", nargs="+", default=["good", "bad"])
    parser.add_argument("--out", default=str(OUT / "results.csv"))
    args = parser.parse_args()

    rows = []
    for quality in args.qualities:
        for reliability in args.reliabilities:
            for top_k in args.top_k:
                scores = []
                for seed in args.seeds:
                    row = run_point(
                        quality, top_k, args.problem, args.total_budget,
                        args.source_budget, args.batch, seed, reliability,
                    )
                    rows.append(row)
                    scores.append(row["best_score"])
                print(json.dumps({
                    "quality": quality, "reliability": reliability, "top_k": top_k,
                    "n": len(scores),
                    "best_score_mean": round(statistics.mean(scores), 4),
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