"""Constrained multi-objective benchmark: does ``cqehvi`` pay off?

Compares algorithms on ``moo_constrained`` (2 objectives + hard constraints)
with metrics that matter for a real campaign:

* **feasible rate** -- fraction of executed experiments that satisfy every
  hard constraint (experiments spent on infeasible conditions are wasted);
* **feasible HV ratio** -- hypervolume of the *feasible* non-dominated front
  divided by the feasible reference front's hypervolume;
* **first feasible** -- index of the first feasible experiment.

The generic ``scripts/benchmark.py`` reports the raw best score, which for a
constrained problem may be an infeasible point -- that is exactly the failure
mode ``cqehvi`` is designed to avoid.

    py -3 scripts/benchmark_cqehvi.py --budget 40 --batch 4 --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from condopt.benchmarks import (  # noqa: E402
    builtin_problems,
    moo_constrained_reference_front,
    run_algorithm,
)
from condopt.pareto import hypervolume_2d, non_dominated_mask  # noqa: E402


def is_feasible(values: dict, limits: dict) -> bool:
    for name, spec in limits.items():
        value = values.get(name)
        if value is None:
            return False
        threshold = float(spec.get("value", 0.0))
        if str(spec.get("op", "<=")) == "<=":
            if value > threshold:
                return False
        elif value < threshold:
            return False
    return True


def evaluate_run(problem, result) -> dict:
    limits = problem.objective.constraint_limits
    feasible_flags = [is_feasible(values, limits) for values in result.constraint_vectors]
    feasible_idx = [index for index, flag in enumerate(feasible_flags) if flag]
    reference = moo_constrained_reference_front()
    reference_point = (60.0, -30.0)
    reference_hv = hypervolume_2d(reference, reference_point)
    hv_ratio = 0.0
    if feasible_idx:
        scores = np.asarray(
            [
                [result.objective_vectors[i]["yield"], -result.objective_vectors[i]["cost"]]
                for i in feasible_idx
            ],
            dtype=float,
        )
        hv_ratio = float(hypervolume_2d(scores[non_dominated_mask(scores)], reference_point) / reference_hv)
    return {
        "feasible_count": len(feasible_idx),
        "total": len(feasible_flags),
        "feasible_rate": len(feasible_idx) / len(feasible_flags) if feasible_flags else 0.0,
        "first_feasible": feasible_idx[0] if feasible_idx else None,
        "feasible_hv_ratio": hv_ratio,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="constrained MOO benchmark (moo_constrained)")
    parser.add_argument("--problem", default="moo_constrained")
    parser.add_argument(
        "--algorithms", nargs="*", default=["cqehvi", "qehvi", "parego", "random", "sobol"]
    )
    parser.add_argument("--budget", type=int, default=40)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--mc-draws", type=int, default=64)
    parser.add_argument(
        "--batch-strategy",
        default=None,
        choices=("local_penalty", "kriging_believer", "constant_liar"),
        help="cqehvi batch selection strategy",
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--out", help="CSV output path")
    args = parser.parse_args()

    problems = builtin_problems()
    if args.problem not in problems:
        print(f"unknown problem {args.problem!r}; known: {sorted(problems)}", file=sys.stderr)
        return 2
    problem = problems[args.problem]
    if not problem.objective.constraints:
        print(f"{args.problem} has no constraints; use scripts/benchmark.py", file=sys.stderr)
        return 2

    rows: list[dict] = []
    for algorithm in args.algorithms:
        for seed in args.seeds:
            params = None
            if algorithm in ("qehvi", "cqehvi"):
                params = {"n_initial": 6, "mc_draws": args.mc_draws}
                if algorithm == "cqehvi" and args.batch_strategy:
                    params["batch_strategy"] = args.batch_strategy
            result = run_algorithm(
                problem,
                algorithm,
                budget=args.budget,
                batch_size=args.batch,
                seed=seed,
                params=params,
            )
            row = {"algorithm": algorithm, "seed": seed}
            row.update(evaluate_run(problem, result))
            rows.append(row)

    if args.json:
        print(json.dumps({"problem": args.problem, "rows": rows}, ensure_ascii=False, indent=2))
    else:
        header = f"{'algorithm':<10}{'seed':>5}{'feasible':>10}{'rate':>8}{'first':>8}{'HV ratio':>10}"
        print(header)
        print("-" * len(header))
        for row in rows:
            first = "-" if row["first_feasible"] is None else str(row["first_feasible"])
            count = f"{row['feasible_count']}/{row['total']}"
            print(
                f"{row['algorithm']:<10}{row['seed']:>5}{count:>10}"
                f"{row['feasible_rate']:>8.2f}{first:>8}{row['feasible_hv_ratio']:>10.3f}"
            )
        print()
        print("指标：可行实验数（越多越不浪费预算）/ 可行率 / 首次可行实验序号 / 可行前沿 HV ratio（越大越好）。")
        print("参考前沿：moo_constrained 的可行域稠密网格（T≤78 °C、time≤50 min）。")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            fh.write("algorithm,seed,feasible_count,total,feasible_rate,first_feasible,feasible_hv_ratio\n")
            for row in rows:
                first = "" if row["first_feasible"] is None else row["first_feasible"]
                fh.write(
                    f"{row['algorithm']},{row['seed']},{row['feasible_count']},{row['total']},"
                    f"{row['feasible_rate']:.4f},{first},{row['feasible_hv_ratio']:.6f}\n"
                )
        print(f"CSV: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
