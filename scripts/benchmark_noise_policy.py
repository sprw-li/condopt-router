"""Does the "high noise -> UCB + large alpha" advice actually pay off?

``prepare.noise_advice`` tells the agent to switch to UCB with a larger alpha and
add a baseline arm when the measured noise is high.  This measures that advice on a
6-D mixed reaction landscape with injected noise (``run_algorithm(noise_std=...)``).

Arms:
* ``ei_default`` -- bayes with the default acquisition (EI) and alpha;
* ``ucb_alpha10`` -- the advice: UCB + alpha 10;
* ``random`` -- model-free baseline arm (what the advice says to keep for contrast).

Run:
    py -3 scripts/benchmark_noise_policy.py --budget 30 --seeds 0 1 2
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

from condopt.benchmarks import builtin_problems, run_algorithm  # noqa: E402

OUT = ROOT / "runs" / "benchmark_noise_policy"

#: ``alpha`` is capped at 1.0 in the schema (it is an observation-noise variance
#: added to the GP diagonal), so "larger alpha" means the ceiling; UCB also exposes
#: ``kappa`` (0-10) which is where a high-noise robust acquisition really lives.
ARMS = (
    ("ei_default", "bayes", {"acquisition": "ei"}),
    ("ei_alpha1", "bayes", {"acquisition": "ei", "alpha": 1.0}),
    ("ucb_kappa8", "bayes", {"acquisition": "ucb", "kappa": 8.0, "alpha": 1.0}),
    ("random", "random", {}),
)


def run_arm(
    arm: tuple[str, str, dict], problem_name: str, noise_std: float,
    budget: int, batch: int, seed: int,
) -> dict:
    label, algorithm, params = arm
    problem = builtin_problems()[problem_name]
    result = run_algorithm(
        problem, algorithm, budget=budget, batch_size=batch, seed=seed,
        params=params, noise_std=noise_std,
    )
    return {
        "arm": label,
        "algorithm": algorithm,
        "problem": problem_name,
        "noise_std": noise_std,
        "budget": budget,
        "seed": seed,
        "best_score": round(float(result.best_score), 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", default="mixed_reaction")
    parser.add_argument("--noise-std", type=float, nargs="+", default=[0.0, 5.0, 12.0])
    parser.add_argument("--budget", type=int, default=30)
    parser.add_argument("--batch", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--out", default=str(OUT / "results.csv"))
    args = parser.parse_args()

    shutil.rmtree(OUT, ignore_errors=True)
    rows: list[dict] = []
    for noise_std in args.noise_std:
        for arm in ARMS:
            for seed in args.seeds:
                rows.append(
                    run_arm(arm, args.problem, noise_std, args.budget, args.batch, seed)
                )
        print(json.dumps({"noise_std": noise_std, "arms": {
            label: round(
                statistics.mean(
                    row["best_score"] for row in rows
                    if row["noise_std"] == noise_std and row["arm"] == label
                ), 4,
            )
            for label, _, _ in ARMS
        }}, ensure_ascii=False))

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