"""Does ``transfer-audit`` actually separate good sources from bad ones?

Four source kinds share the target's space/objective signature (so
``transfer_from`` would accept them) but differ in how well they predict the
target:

1. ``consistent`` — same problem, different seed (unbiased truth);
2. ``shifted``    — same problem with the optimum moved (related but different);
3. ``shuffled``   — labels randomly permuted (no relation);
4. ``flipped``    — labels negated (anti-correlated).

The target study measures a few points, then audits each source and reports the
suggested reliability.  Good evidence = monotone separation.

Run:
    py -3 scripts/benchmark_transfer_audit.py --seeds 0 1 2
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

from condopt.base import Objective  # noqa: E402
from condopt.space import SearchSpace  # noqa: E402
from condopt.study import Study  # noqa: E402

OUT = ROOT / "runs" / "benchmark_transfer_audit"

# Same space/objective signature for every source: transfer_from would accept them.
SPACE = SearchSpace.from_dict([
    {"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0},
])
OBJECTIVE = Objective(name="y", direction="max")

SOURCE_KINDS = ("consistent", "shifted", "shuffled", "flipped")


def truth(x: float) -> float:
    return -((x - 0.3) ** 2)


def run_source(kind: str, seed: int, n: int = 12) -> Path:
    run_dir = OUT / f"source_{kind}_s{seed}"
    shutil.rmtree(run_dir, ignore_errors=True)
    study = Study.create(
        run_dir, SPACE, OBJECTIVE, algorithm="random",
        budget=n, batch_size=n, seed=seed,
    )
    rng = random.Random(seed)
    rows = []
    for index in range(n):
        x = (index + 0.5) / n
        value = truth(x)
        if kind == "shifted":
            value = -((x - 0.7) ** 2)
        rows.append({"x": {"x": x}, "y": value, "fidelity": 1.0, "cost": 1.0})
    if kind == "shuffled":
        labels = [row["y"] for row in rows]
        rng.shuffle(labels)
        for row, label in zip(rows, labels):
            row["y"] = label
    if kind == "flipped":
        for row in rows:
            row["y"] = -row["y"]
    study.import_observations(rows)
    return run_dir


def run_target(seed: int, n: int = 6) -> Path:
    run_dir = OUT / f"target_s{seed}"
    shutil.rmtree(run_dir, ignore_errors=True)
    study = Study.create(
        run_dir, SPACE, OBJECTIVE, algorithm="random",
        budget=n, batch_size=n, seed=seed,
    )
    rows = []
    for index in range(n):
        x = (index + 0.5) / n
        rows.append({"x": {"x": x}, "y": truth(x), "fidelity": 1.0, "cost": 1.0})
    study.import_observations(rows)
    return run_dir


def audit(seed: int) -> list[dict]:
    target = Study.load(run_target(seed))
    rows = []
    for kind in SOURCE_KINDS:
        source = run_source(kind, seed)
        report = target.assess_transfer(source)
        rows.append({
            "seed": seed,
            "kind": kind,
            "ready": report.get("ready"),
            "spearman": report.get("spearman"),
            "mae": report.get("mae"),
            "suggested_reliability": report.get("suggested_reliability"),
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--out", default=str(OUT / "results.csv"))
    args = parser.parse_args()

    rows = []
    for seed in args.seeds:
        rows.extend(audit(seed))
    by_kind: dict[str, list[dict]] = {}
    for row in rows:
        by_kind.setdefault(row["kind"], []).append(row)
    for kind in SOURCE_KINDS:
        subset = by_kind[kind]
        reliabilities = [
            row["suggested_reliability"]
            for row in subset
            if row["suggested_reliability"] is not None
        ]
        spearman = [row["spearman"] for row in subset if row["spearman"] is not None]
        print(json.dumps({
            "kind": kind,
            "n": len(subset),
            "reliability_mean": round(statistics.mean(reliabilities), 3) if reliabilities else None,
            "reliability_values": [round(value, 3) for value in reliabilities],
            "spearman_mean": round(statistics.mean(spearman), 3) if spearman else None,
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