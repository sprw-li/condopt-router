"""A/B the default router rules against calibrated multipliers, out-of-sample.

Both sides are evaluated leave-one-problem-out:

* default rules need no training, so each held-out problem is scored directly;
* the calibrated side uses the LOPO numbers already stored in ``weights.json``
  (each fold trains on the other problems only).

Run:
    py -3 scripts/ab_router_weights.py --records runs/calibration_new_benchmarks/records.json \
        --weights runs/calibration_new_benchmarks/weights.json --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from condopt.calibration import (  # noqa: E402
    RunRecord,
    evaluate_weights,
    set_algorithm_weights,
)


def _load_records(path: Path) -> list[RunRecord]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    allowed = {"problem", "algorithm", "score", "context", "metric_name"}
    return [RunRecord(**{k: v for k, v in row.items() if k in allowed}) for row in payload]


def default_lopo(records: list[RunRecord]) -> dict:
    problems = sorted({record.problem for record in records})
    folds = {}
    for held_out in problems:
        held = [record for record in records if record.problem == held_out]
        folds[held_out] = evaluate_weights(held, None)
    n = max(1, len(folds))
    return {
        "top1": sum(item["top1"] for item in folds.values()) / n,
        "top2": sum(item["top2"] for item in folds.values()) / n,
        "mean_regret": sum(item["mean_regret"] for item in folds.values()) / n,
        "n_problems": len(folds),
        "folds": folds,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    records = _load_records(Path(args.records))
    weights_payload = json.loads(Path(args.weights).read_text(encoding="utf-8"))
    calibrated = weights_payload.get("lopo_metrics")
    default = default_lopo(records)
    set_algorithm_weights(None)  # restore default rules for later callers

    report = {
        "n_problems": default["n_problems"],
        "default_lopo": {k: default[k] for k in ("top1", "top2", "mean_regret")},
        "calibrated_lopo": calibrated,
        "winner": None,
    }
    if calibrated:
        default_key = (default["top1"], default["top2"], -default["mean_regret"])
        calibrated_key = (
            calibrated.get("top1", 0.0),
            calibrated.get("top2", 0.0),
            -calibrated.get("mean_regret", 0.0),
        )
        report["winner"] = "calibrated" if calibrated_key > default_key else "default"

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    else:
        print(f"problems: {report['n_problems']}")
        print(f"default (LOPO):    {json.dumps(report['default_lopo'], ensure_ascii=False)}")
        print(f"calibrated (LOPO): {json.dumps(calibrated, ensure_ascii=False)}")
        print(f"winner: {report['winner']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())