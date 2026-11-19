"""Can transfer-audit rank *real* source halves by usefulness?

The ART Table 2 set has only six real points.  Enumerate all C(6,3)=20 source/target
splits, audit each one, and ask two questions:

* does the suggested reliability separate splits whose source holds the measured
  best condition (99%) from those that do not?
* does the low-confidence safeguard fire on all of them (as it should for n=3)?

Run:
    py -3 scripts/benchmark_real_audit_splits.py
"""

from __future__ import annotations

import csv
import itertools
import json
import shutil
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from condopt.base import Objective  # noqa: E402
from condopt.space import SearchSpace  # noqa: E402
from condopt.study import Study  # noqa: E402

CSV_PATH = ROOT / "examples" / "paper_campaigns" / "art_table2_real.csv"
OUT = ROOT / "runs" / "benchmark_real_audit_splits"
REAL_BEST_YIELD = 99.0


def load_rows() -> list[dict]:
    return list(csv.DictReader(CSV_PATH.read_text(encoding="utf-8").splitlines()))


def build_space(rows: list[dict]) -> SearchSpace:
    names = [name for name in rows[0] if name not in ("yield", "source")]
    variables = []
    for name in names:
        choices = sorted({row[name] for row in rows})
        variables.append({"name": name, "kind": "categorical", "choices": choices})
    return SearchSpace.from_dict(variables)


def points(rows: list[dict]) -> list[dict]:
    return [
        {
            "x": {name: row[name] for name in rows[0] if name not in ("yield", "source")},
            "y": float(row["yield"]),
        }
        for row in rows
    ]


def audit_split(index: int, source_rows: list[dict], target_rows: list[dict], space: SearchSpace) -> dict:
    source_dir = OUT / f"split{index:02d}_source"
    target_dir = OUT / f"split{index:02d}_target"
    shutil.rmtree(source_dir, ignore_errors=True)
    shutil.rmtree(target_dir, ignore_errors=True)
    objective = Objective(name="yield", direction="max")
    source = Study.create(
        source_dir, space, objective, algorithm="tpe", budget=6, batch_size=3, seed=0
    )
    source.import_observations(points(source_rows))
    target = Study.create(
        target_dir, space, objective, algorithm="tpe", budget=6, batch_size=3, seed=0
    )
    target.import_observations(points(target_rows))
    report = target.assess_transfer(source.dir)
    source_best = max(float(row["yield"]) for row in source_rows)
    target_best = max(float(row["yield"]) for row in target_rows)
    return {
        "split": index,
        "source_best": source_best,
        "target_best": target_best,
        "source_has_best": source_best >= REAL_BEST_YIELD - 1e-9,
        "ready": report.get("ready"),
        "suggested_reliability": report.get("suggested_reliability"),
        "spearman": report.get("spearman"),
        "confidence": report.get("confidence"),
        "n_compared": report.get("n_compared"),
    }


def main() -> int:
    shutil.rmtree(OUT, ignore_errors=True)
    rows = load_rows()
    space = build_space(rows)
    results = []
    for index, combo in enumerate(itertools.combinations(range(len(rows)), 3)):
        source_rows = [rows[i] for i in combo]
        target_rows = [rows[i] for i in range(len(rows)) if i not in combo]
        results.append(audit_split(index, source_rows, target_rows, space))
    with_best = [row for row in results if row["source_has_best"]]
    without_best = [row for row in results if not row["source_has_best"]]

    def _mean(values):
        values = [value for value in values if value is not None]
        return round(statistics.mean(values), 3) if values else None

    summary = {
        "n_splits": len(results),
        "n_source_has_best": len(with_best),
        "reliability_with_best": _mean(row["suggested_reliability"] for row in with_best),
        "reliability_without_best": _mean(
            row["suggested_reliability"] for row in without_best
        ),
        "spearman_with_best": _mean(row["spearman"] for row in with_best),
        "spearman_without_best": _mean(row["spearman"] for row in without_best),
        "low_confidence_rate": round(
            statistics.mean(1.0 if row["confidence"] == "low" else 0.0 for row in results), 3
        ),
    }
    print(json.dumps(summary, ensure_ascii=False))
    out = OUT / "results.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"CSV: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())