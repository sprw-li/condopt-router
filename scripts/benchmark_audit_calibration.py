"""Calibrate ``transfer-audit`` thresholds against measured transfer benefit.

``assess_transfer`` turns a source campaign into a suggested ``prior_reliability``
via Spearman correlation + normalised MAE.  Its thresholds (0.8 / 0.5) are heuristics
from ``research/TRANSFER_AUDIT.md``.  This script builds sources of controlled quality
(identical / mildly noisy / biased / shuffled labels) and checks two things:

1. does the audit score rank sources in the same order as the measured benefit?
2. do the 0.8 / 0.5 thresholds separate positive from negative transfer?

Run:
    py -3 scripts/benchmark_audit_calibration.py --seeds 0 1
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from condopt.base import Objective, Observation  # noqa: E402
from condopt.benchmarks import builtin_problems  # noqa: E402
from condopt.router import OptimizationContext  # noqa: E402
from condopt.study import Study  # noqa: E402

OUT = ROOT / "runs" / "benchmark_audit_calibration"

#: quality label -> (bias, noise_std, shuffle)
#: The coarse four-arm grid is what earlier rounds used; the fine grid adds intermediate
#: biases so a reliability threshold can actually be swept against measured benefit.
QUALITIES = (
    ("identical", 0.0, 0.0, False),
    ("mild_noise", 0.0, 3.0, False),
    ("biased", -12.0, 4.0, False),
    ("shuffled", 0.0, 0.0, True),
)

FINE_QUALITIES = (
    ("bias0", 0.0, 2.0, False),
    ("bias3", -3.0, 2.0, False),
    ("bias6", -6.0, 2.0, False),
    ("bias9", -9.0, 2.0, False),
    ("bias12", -12.0, 2.0, False),
    ("bias18", -18.0, 2.0, False),
)


def _separation(rows: list[dict]) -> dict:
    """Best achievable separation of positive vs negative transfer by an audit cut.

    Returns the accuracy of every candidate cut so the report can say whether *any*
    threshold would have worked (and if not, that the knob is not the problem).
    """
    graded = [
        row for row in rows
        if row.get("audit") is not None and row.get("delta_vs_cold") is not None
    ]
    if not graded:
        return {"n": 0, "best_threshold": None, "best_accuracy": None, "cuts": []}
    positives = [row for row in graded if row["delta_vs_cold"] > 0]
    cuts = []
    for step in range(21):
        threshold = step / 20
        correct = sum(
            1 for row in graded
            if (row["audit"] >= threshold) == (row["delta_vs_cold"] > 0)
        )
        cuts.append({"threshold": round(threshold, 2), "accuracy": round(correct / len(graded), 3)})
    best = max(cuts, key=lambda item: item["accuracy"])
    majority = max(len(positives), len(graded) - len(positives)) / len(graded)
    return {
        "n": len(graded),
        "n_positive": len(positives),
        "n_negative": len(graded) - len(positives),
        "majority_baseline": round(majority, 3),
        "best_threshold": best["threshold"],
        "best_accuracy": best["accuracy"],
        "cuts": cuts,
    }


def _context(problem, budget: int, batch: int) -> OptimizationContext:
    return OptimizationContext.from_space(
        problem.space, budget_total=budget, batch_size=batch, multi_fidelity=False,
        n_objectives=1, n_constraints=0, noise="medium",
    )


def _target(problem, budget: int, seed: int, run_dir: Path) -> Study:
    shutil.rmtree(run_dir, ignore_errors=True)
    return Study.create(
        run_dir, problem.space, problem.objective, algorithm="auto",
        budget=budget, batch_size=3, context=_context(problem, budget, 3), seed=seed,
    )


def _evaluate(problem, item) -> dict:
    return {"id": item["id"], "y": float(problem.fn(item["x"])), "fidelity": 1.0}


def _drive(problem, study: Study) -> None:
    for _ in range(max(120, 8 * int(getattr(study, "budget", 16) or 16))):
        action = study.next_action()
        if action["action"] == "stop":
            break
        if action["action"] == "ask":
            study.ask(int(action.get("ask_n") or 3))
        elif action["action"] == "confirm":
            study.confirm(int(action.get("confirm_n") or 3))
        else:
            study.tell([_evaluate(problem, item) for item in study.pending])
            study.record_progress()


def build_source(
    problem, seed: int, quality: tuple[str, float, float, bool], n_points: int = 24
) -> Study:
    label, bias, noise_std, shuffle = quality
    run_dir = OUT / f"source_{label}_s{seed}"
    shutil.rmtree(run_dir, ignore_errors=True)
    study = Study.create(
        run_dir, problem.space, problem.objective, algorithm="random",
        budget=n_points, batch_size=3, context=_context(problem, n_points, 3), seed=seed,
    )
    rng = np.random.default_rng(seed * 977 + 13)
    configs = problem.space.sample_uniform(n_points, rng)
    values = [
        float(problem.fn(config)) + bias + float(rng.normal(0.0, noise_std))
        for config in configs
    ]
    if shuffle:
        values = list(values)
        random.Random(seed * 31 + 7).shuffle(values)
    observations = [
        Observation(x=config, y=value, fidelity=1.0, meta={"source": label})
        for config, value in zip(configs, values)
    ]
    study.import_observations(observations)
    return study


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", default="mixed_reaction")
    parser.add_argument("--budget", type=int, default=16)
    parser.add_argument("--source-points", type=int, default=24)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--fine", action="store_true", help="use the finer bias grid")
    parser.add_argument("--out", default=str(OUT / "results.csv"))
    args = parser.parse_args()
    qualities = FINE_QUALITIES if args.fine else QUALITIES

    problem = builtin_problems()[args.problem]
    shutil.rmtree(OUT, ignore_errors=True)
    rows: list[dict] = []
    for seed in args.seeds:
        cold_study = _target(problem, args.budget, seed, OUT / f"cold_s{seed}")
        _drive(problem, cold_study)
        cold_incumbent = cold_study.best()
        cold_best = None if cold_incumbent is None else float(cold_incumbent.y)
        for quality in qualities:
            label = quality[0]
            source = build_source(problem, seed, quality, args.source_points)
            target = _target(problem, args.budget, seed, OUT / f"target_{label}_s{seed}")
            for _ in range(3):
                pending = target.ask(2)
                target.tell([_evaluate(problem, item) for item in pending])
            audit = target.assess_transfer(source.dir)
            imported = target.transfer_from(
                source.dir, top_k=args.top_k, prior_reliability=1.0, reason="audit calibration"
            )["imported"]
            _drive(problem, target)
            incumbent = target.best()
            best = None if incumbent is None else float(incumbent.y)
            rows.append({
                "quality": label,
                "seed": seed,
                "audit": None if audit.get("suggested_reliability") is None
                else round(float(audit["suggested_reliability"]), 4),
                "spearman": audit.get("spearman"),
                "imported": imported,
                "best": None if best is None else round(best, 4),
                "cold_best": None if cold_best is None else round(cold_best, 4),
                "delta_vs_cold": None if best is None or cold_best is None
                else round(best - cold_best, 4),
            })
        print(json.dumps({"seed": seed, "cold_best": rows[-1]["cold_best"]}, ensure_ascii=False))

    for quality in qualities:
        label = quality[0]
        subset = [row for row in rows if row["quality"] == label and row["audit"] is not None]
        if not subset:
            continue
        print(json.dumps({
            "quality": label,
            "n": len(subset),
            "audit_mean": round(statistics.mean(row["audit"] for row in subset), 4),
            "delta_vs_cold_mean": round(
                statistics.mean(row["delta_vs_cold"] for row in subset), 4
            ),
        }, ensure_ascii=False))
    print(json.dumps({"separation": _separation(rows)}, ensure_ascii=False))

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