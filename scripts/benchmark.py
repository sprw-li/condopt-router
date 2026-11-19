"""Run the synthetic benchmark suite and print a comparison table.

    py -3 scripts/benchmark.py --budget 20 --batch 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from condopt.benchmarks import builtin_problems, run_suite, summarize  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="condopt benchmark runner")
    parser.add_argument("--problems", nargs="*", default=None)
    parser.add_argument("--algorithms", nargs="*", default=None)
    parser.add_argument("--budget", type=int, default=20)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--out", help="CSV output path")
    args = parser.parse_args()

    results = run_suite(
        problems=args.problems,
        algorithms=args.algorithms,
        budget=args.budget,
        batch_size=args.batch,
        seeds=args.seeds,
    )
    summary = summarize(results)

    if args.json:
        print(json.dumps({"results": [r.to_dict() for r in results], "summary": summary}, ensure_ascii=False, indent=2))
        return 0

    problems = list(dict.fromkeys(item["problem"] for item in summary))
    algorithms = list(dict.fromkeys(item["algorithm"] for item in summary))
    header = f"{'problem':<18}" + "".join(f"{name:>12}" for name in algorithms)
    print(header)
    print("-" * len(header))
    for problem in problems:
        row = f"{problem:<18}"
        values: list[float | None] = []
        for algorithm in algorithms:
            match = next(
                (item for item in summary if item["problem"] == problem and item["algorithm"] == algorithm),
                None,
            )
            value = None if match is None else match["mean_best_score"]
            values.append(value)
            row += f"{'-':>12}" if value is None else f"{value:>12.4g}"
        valid = [(value, i) for i, value in enumerate(values) if value is not None]
        if valid:
            best = max(valid)
            row += f"   best: {algorithms[best[1]]}"
        print(row)
    print()
    print("说明：数值为多次随机种子下的平均最优得分（越大越好）；每条轨迹见 --json。")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            fh.write("problem,algorithm,n_runs,mean_best_score,std_best_score\n")
            for item in summary:
                fh.write(
                    f"{item['problem']},{item['algorithm']},{item['n_runs']},"
                    f"{item['mean_best_score']:.6g},{item['std_best_score']:.6g}\n"
                )
        print(f"CSV: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
