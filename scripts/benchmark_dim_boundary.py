"""Where exactly does CMA-ES overtake GP-BO? (dimension boundary)

The router switches to ``cmaes`` at 8-14 smooth continuous dimensions, but the
archived evidence only covers **10** dimensions (rosenbrock10 / ackley10).  This
sweeps dimensions 6/8/10/12 with the same landscapes to check the boundary.

Run:
    py -3 scripts/benchmark_dim_boundary.py --dims 6 8 10 12 --budget 60 --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from condopt.base import Objective  # noqa: E402
from condopt.benchmarks import BenchmarkProblem, run_algorithm  # noqa: E402
from condopt.space import SearchSpace  # noqa: E402

OUT = ROOT / "runs" / "benchmark_dim_boundary"


def _space(dim: int) -> SearchSpace:
    return SearchSpace.from_dict([
        {"name": f"x{i}", "kind": "continuous", "low": 0.0, "high": 1.0}
        for i in range(dim)
    ])


def rosenbrock(config: dict, dim: int) -> float:
    values = [4.0 * float(config[f"x{i}"]) - 2.0 for i in range(dim)]
    total = 0.0
    for i in range(dim - 1):
        total += 100.0 * (values[i + 1] - values[i] ** 2) ** 2 + (1.0 - values[i]) ** 2
    return -total


def ackley(config: dict, dim: int) -> float:
    values = [10.0 * float(config[f"x{i}"]) - 5.0 for i in range(dim)]
    n = len(values)
    sum_sq = sum(v * v for v in values) / n
    sum_cos = sum(math.cos(2.0 * math.pi * v) for v in values) / n
    return -(20.0 * math.exp(-0.2 * math.sqrt(sum_sq)) + math.exp(sum_cos) - 20.0 - math.e)


LANDSCAPES = {"rosenbrock": rosenbrock, "ackley": ackley}
ALGORITHMS = ("bayes", "cmaes")


def run_arm(landscape: str, dim: int, algorithm: str, budget: int, batch: int, seed: int) -> dict:
    problem = BenchmarkProblem(
        name=f"{landscape}{dim}",
        space=_space(dim),
        objective=Objective(name="score", direction="max"),
        fn=lambda config: LANDSCAPES[landscape](config, dim),
        description=f"{landscape} on {dim}-D unit cube (optimum 0)",
        optimum=0.0,
    )
    outcome = run_algorithm(
        problem, algorithm, budget=budget, batch_size=batch, seed=seed,
    )
    return {
        "landscape": landscape,
        "dim": dim,
        "algorithm": algorithm,
        "seed": seed,
        "best": round(float(outcome.best_score), 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dims", type=int, nargs="+", default=[6, 8, 10, 12])
    parser.add_argument("--budget", type=int, default=60)
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--out", default=str(OUT / "results.csv"))
    args = parser.parse_args()

    shutil.rmtree(OUT, ignore_errors=True)
    rows: list[dict] = []
    for landscape in LANDSCAPES:
        for dim in args.dims:
            for algorithm in ALGORITHMS:
                for seed in args.seeds:
                    rows.append(run_arm(landscape, dim, algorithm, args.budget, args.batch, seed))
            means = {}
            for algorithm in ALGORITHMS:
                arm = [
                    row for row in rows
                    if row["landscape"] == landscape
                    and row["dim"] == dim
                    and row["algorithm"] == algorithm
                ]
                means[algorithm] = round(statistics.mean(row["best"] for row in arm), 4)
            winner = max(means, key=lambda name: means[name])
            print(json.dumps({
                "landscape": landscape,
                "dim": dim,
                **means,
                "winner": winner,
                "gap": round(means["cmaes"] - means["bayes"], 4),
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