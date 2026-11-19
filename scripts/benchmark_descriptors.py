"""Does replacing one-hot with physical descriptors help in a categorical space?

Synthetic but controlled: 12 categories carry a hidden 2-D descriptor; the true
objective is a smooth function of that descriptor, so categories close in
descriptor space should behave similarly.  Arm A uses one-hot features, arm B
uses the descriptors.  Everything else (algorithm, budget, seeds) is identical.

Run:
    py -3 scripts/benchmark_descriptors.py --budget 24 --seeds 0 1 2
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

OUT = ROOT / "runs" / "benchmark_descriptors"

CATEGORY_COUNT = 60
CATEGORIES = [f"C{index}" for index in range(CATEGORY_COUNT)]
_LATENT = {
    name: (float(0.02 + 0.016 * index), float(0.95 - 0.015 * index))
    for index, name in enumerate(CATEGORIES)
}
_OPTIMUM = max(
    _LATENT.values(),
    key=lambda point: -(point[0] - 0.75) ** 2 - (point[1] - 0.25) ** 2,
)


def true_yield(category: str) -> float:
    d0, d1 = _LATENT[category]
    return 60.0 + 40.0 * np.exp(-(((d0 - 0.75) ** 2 + (d1 - 0.25) ** 2) / 0.08))


def build_space(use_descriptors: bool) -> SearchSpace:
    variable = {
        "name": "category", "kind": "categorical", "choices": CATEGORIES,
    }
    if use_descriptors:
        variable["descriptors"] = {name: list(_LATENT[name]) for name in CATEGORIES}
        variable["descriptor_names"] = ["latent0", "latent1"]
    return SearchSpace.from_dict([variable])


def run_arm(use_descriptors: bool, budget: int = 24, batch_size: int = 4, seed: int = 0) -> dict:
    import shutil

    run_dir = OUT / f"arm_{'desc' if use_descriptors else 'onehot'}_{seed}"
    shutil.rmtree(run_dir, ignore_errors=True)  # reruns must start from a clean study
    study = Study.create(
        run_dir,
        build_space(use_descriptors),
        Objective(name="y", direction="max"),
        algorithm="bayes",
        budget=budget,
        batch_size=batch_size,
        seed=seed,
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
        "arm": "descriptors" if use_descriptors else "one_hot",
        "seed": seed,
        "budget": budget,
        "best_yield": None if best is None else round(float(best.y), 3),
        "best_category": None if best is None else best.x["category"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=30)
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--out", default=str(OUT / "results.csv"))
    args = parser.parse_args()

    rows = []
    for use_descriptors in (True, False):
        for seed in args.seeds:
            row = run_arm(use_descriptors, budget=args.budget, batch_size=args.batch, seed=seed)
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("\nsummary")
    for arm in ("descriptors", "one_hot"):
        subset = [row["best_yield"] for row in rows if row["arm"] == arm]
        print(json.dumps({
            "arm": arm,
            "n": len(subset),
            "best_yield_mean": round(statistics.mean(subset), 3),
        }, ensure_ascii=False))
    print(f"CSV: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())