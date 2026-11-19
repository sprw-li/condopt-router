"""CLI: calibrate router weights on a small benchmark matrix.

Usage:
    py -3 scripts/calibrate_router.py --budget 10 --batch 2 --seeds 0 --iterations 80
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from condopt.calibration import (  # noqa: E402
    best_algorithm_per_problem,
    collect_records,
    evaluate_weights,
    lopo_validate,
    random_search_weights,
)

DEFAULT_PROBLEMS = [
    "branin",
    "sphere2d",
    "mixed_reaction",
    "discrete_catalyst",
    "suzuki_like",
    "high_cardinality_solvent",
    "constraint_easy",
    "moo_yield_cost",
    "moo_constrained",
    "mf_suzuki",
    "mf_constrained",
]
DEFAULT_ALGORITHMS = [
    "random", "sobol", "lhs", "grid", "bayes", "tpe",
    "constrained", "parego", "qehvi", "cqehvi", "mfbo",
]


def seed_warning(seeds: list[int]) -> str | None:
    """Single-seed labels are noisy; say so instead of trusting them blindly."""
    if len(seeds) < 2:
        return (
            "只跑了 1 个 seed：单 seed 的“最佳算法”标签噪声很大"
            "（例如 2 维问题上 cmaes 会被偶然判为最佳，而 3 seeds 复核显示 bayes 更好）；"
            "结论请以多 seed 复核为准"
        )
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Condopt router weight calibration (MVP)")
    parser.add_argument("--problems", nargs="*", default=DEFAULT_PROBLEMS)
    parser.add_argument("--algorithms", nargs="*", default=DEFAULT_ALGORITHMS)
    parser.add_argument("--budget", type=int, default=10)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=[0, 1],
        help="每问题重复的随机种子；默认 2 个，单 seed 标签噪声大（脚本会警告）",
    )
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise", default="medium", choices=["none", "low", "medium", "high"])
    parser.add_argument("--smoothness", default="mixed", choices=["smooth", "mixed", "rough"])
    parser.add_argument("--out", default="runs/calibration")
    args = parser.parse_args()

    notice = seed_warning(list(args.seeds))
    if notice:
        print(f"[warn] {notice}")
    records = collect_records(
        args.problems,
        args.algorithms,
        budget=args.budget,
        batch_size=args.batch,
        seeds=args.seeds,
        noise=args.noise,
        smoothness=args.smoothness,
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "records.json").write_text(
        json.dumps([record.__dict__ for record in records], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    result = random_search_weights(records, iterations=args.iterations, seed=args.seed)
    lopo = lopo_validate(records, iterations=max(20, args.iterations // 2), seed=args.seed)
    weights_path = out_dir / "weights.json"
    weights_path.write_text(
        json.dumps(
            {
                "weights": result["weights"],
                "in_sample_metrics": result["metrics"],
                "default_metrics": result["default_metrics"],
                "lopo_metrics": {k: lopo[k] for k in ("top1", "top2", "mean_regret", "n_problems")},
                "labels": best_algorithm_per_problem(records),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    from collections import Counter

    metrics = Counter(record.metric_name for record in records)
    print("metric coverage:", dict(metrics))
    print("labels:", json.dumps(best_algorithm_per_problem(records), ensure_ascii=False))
    print("default:", json.dumps(evaluate_weights(records, None), ensure_ascii=False))
    print("in-sample (optimistic):", json.dumps(result["metrics"], ensure_ascii=False))
    print(
        "LOPO (evidence):",
        json.dumps({k: lopo[k] for k in ("top1", "top2", "mean_regret", "n_problems")}, ensure_ascii=False),
    )
    print("weights:", json.dumps(result["weights"], ensure_ascii=False))
    print("written:", weights_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
