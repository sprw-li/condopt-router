"""How accurate must a descriptor table be? Controlled corruption sweep.

Uses the same hidden-latent setup as ``benchmark_descriptors.py`` (60
categories, 2-D latent, budget 30).  For each corruption level the descriptor
table handed to the optimizer is ``latent + eps * N(0, 1)``; ``eps = 0`` is the
perfect table and ``eps = 99`` is effectively garbage (compare with one-hot).

Run:
    py -3 scripts/benchmark_descriptor_quality.py --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from condopt.base import Objective  # noqa: E402
from condopt.space import SearchSpace  # noqa: E402
from condopt.study import Study  # noqa: E402

OUT = ROOT / "runs" / "benchmark_descriptor_quality"
CATEGORY_COUNT = 60
CATEGORIES = [f"C{index}" for index in range(CATEGORY_COUNT)]
_LATENT = {
    name: (float(0.02 + 0.016 * index), float(0.95 - 0.015 * index))
    for index, name in enumerate(CATEGORIES)
}


def true_yield(category: str) -> float:
    d0, d1 = _LATENT[category]
    return 60.0 + 40.0 * np.exp(-(((d0 - 0.75) ** 2 + (d1 - 0.25) ** 2) / 0.08))


def build_space(epsilon: float, seed: int) -> SearchSpace:
    if epsilon >= 99:  # one-hot control arm
        return SearchSpace.from_dict([
            {"name": "category", "kind": "categorical", "choices": CATEGORIES}
        ])
    rng = np.random.default_rng(seed + 1234)
    descriptors = {}
    for name, (d0, d1) in _LATENT.items():
        noise = rng.normal(0.0, 1.0, size=2)
        descriptors[name] = [d0 + epsilon * float(noise[0]), d1 + epsilon * float(noise[1])]
    return SearchSpace.from_dict([{
        "name": "category", "kind": "categorical", "choices": CATEGORIES,
        "descriptors": descriptors, "descriptor_names": ["latent0", "latent1"],
    }])


def run_arm(epsilon: float, budget: int = 30, batch_size: int = 6, seed: int = 0) -> dict:
    import shutil

    run_dir = OUT / f"eps_{epsilon}_{seed}"
    shutil.rmtree(run_dir, ignore_errors=True)
    study = Study.create(
        run_dir, build_space(epsilon, seed), Objective(name="y", direction="max"),
        algorithm="bayes", budget=budget, batch_size=batch_size, seed=seed,
    )
    done = 0
    while done < budget:
        n = min(batch_size, budget - done)
        pending = study.ask(n)
        study.tell([
            {"id": item["id"], "y": true_yield(item["x"]["category"])} for item in pending
        ])
        done += n
    best = study.best()
    return {
        "epsilon": epsilon,
        "seed": seed,
        "best_yield": None if best is None else round(float(best.y), 3),
        "best_category": None if best is None else best.x["category"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=30)
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epsilons", type=float, nargs="+", default=[0.0, 0.1, 0.25, 0.5, 1.0, 99.0])
    parser.add_argument("--out", default=str(OUT / "results.csv"))
    args = parser.parse_args()

    rows = []
    for epsilon in args.epsilons:
        scores = []
        for seed in args.seeds:
            row = run_arm(epsilon, budget=args.budget, batch_size=args.batch, seed=seed)
            rows.append(row)
            scores.append(row["best_yield"])
        print(json.dumps({
            "epsilon": epsilon,
            "n": len(scores),
            "best_yield_mean": round(statistics.mean(scores), 3),
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