"""Does "start small, then widen after seeing the target" beat a fixed top_k?

``reliability_grade`` now advises: use ``top_k`` 1-3 first, watch two batches of local
measurements, then widen.  This measures that advice against fixed policies, with a
source of *unknown* quality (the realistic case).

Arms:
* ``fixed_k1`` / ``fixed_k3`` / ``fixed_k5`` -- import once, then optimize;
* ``staged_audit`` -- import k=1, run two batches, re-run ``assess_transfer`` with the
  fresh local evidence, and widen to k=5 only when the audit does not veto.

Run:
    py -3 scripts/benchmark_transfer_adaptive.py --seeds 0 1
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

import numpy as np  # noqa: E402

from condopt.base import Observation  # noqa: E402
from condopt.benchmarks import builtin_problems  # noqa: E402
from condopt.router import OptimizationContext  # noqa: E402
from condopt.study import Study  # noqa: E402

OUT = ROOT / "runs" / "benchmark_transfer_adaptive"

#: source quality -> (bias, noise_std, shuffle)
QUALITIES = (
    ("good", 0.0, 2.0, False),
    ("biased", -6.0, 2.0, False),
    ("shuffled", 0.0, 2.0, True),
)

POLICIES = ("fixed_k1", "fixed_k3", "fixed_k5", "staged_audit", "staged_soft", "staged_recheck")


def _context(problem, budget: int, batch: int) -> OptimizationContext:
    return OptimizationContext.from_space(
        problem.space, budget_total=budget, batch_size=batch, multi_fidelity=False,
        n_objectives=1, n_constraints=0, noise="medium",
    )


def _evaluate(problem, item) -> dict:
    return {"id": item["id"], "y": float(problem.fn(item["x"])), "fidelity": 1.0}


def _drive(problem, study: Study, stop_after_batches: int | None = None) -> int:
    batches = 0
    for _ in range(max(120, 8 * int(getattr(study, "budget", 30) or 30))):
        action = study.next_action()
        if action["action"] == "stop":
            break
        if stop_after_batches is not None and batches >= stop_after_batches:
            break
        if action["action"] == "ask":
            study.ask(int(action.get("ask_n") or 3))
        elif action["action"] == "confirm":
            if stop_after_batches is not None:
                break
            study.confirm(int(action.get("confirm_n") or 3))
        else:
            study.tell([_evaluate(problem, item) for item in study.pending])
            study.record_progress()
            batches += 1
    return batches


def build_source(problem, seed: int, quality: tuple[str, float, float, bool], n_points: int) -> Study:
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
    study.import_observations([
        Observation(x=config, y=value, fidelity=1.0, meta={"source": label})
        for config, value in zip(configs, values)
    ])
    return study


def run_arm(
    problem, source: Study, policy: str, seed: int, budget: int, audit_veto: float
) -> dict:
    run_dir = OUT / f"{policy}_s{seed}"
    shutil.rmtree(run_dir, ignore_errors=True)
    target = Study.create(
        run_dir, problem.space, problem.objective, algorithm="auto",
        budget=budget, batch_size=3, context=_context(problem, budget, 3), seed=seed,
    )
    imported = 0
    vetoed = False
    if policy.startswith("fixed_k"):
        k = int(policy.split("_k")[1])
        imported = int(target.transfer_from(
            source.dir, top_k=k, prior_reliability=1.0, reason="fixed policy"
        )["imported"])
        _drive(problem, target)
    elif policy == "staged_audit":
        imported = int(target.transfer_from(
            source.dir, top_k=1, prior_reliability=1.0, reason="staged policy: first look"
        )["imported"])
        _drive(problem, target, stop_after_batches=2)
        audit = target.assess_transfer(source.dir)
        suggestion = audit.get("suggested_reliability")
        if audit.get("ready") and suggestion is not None and suggestion >= audit_veto:
            imported += int(target.transfer_from(
                source.dir, top_k=5, prior_reliability=1.0,
                reason="staged policy: audit did not veto, widen",
            )["imported"])
        else:
            vetoed = True
        _drive(problem, target)
    elif policy == "staged_soft":
        # Widen only after the audit passes, and import with the audit's own suggested
        # reliability instead of full trust -- a cheap guard when the veto mis-fires.
        report = target.transfer_staged(
            source.dir, after_batches=2, min_reliability=audit_veto
        )
        imported = int(report.get("imported") or 0)
        _drive(problem, target, stop_after_batches=2)
        audit = target.assess_transfer(source.dir)
        suggestion = audit.get("suggested_reliability")
        if audit.get("ready") and suggestion is not None and suggestion >= audit_veto:
            imported += int(target.transfer_from(
                source.dir, top_k=5, prior_reliability=float(suggestion),
                reason="staged soft: widen with the audit's own reliability",
            )["imported"])
        else:
            vetoed = True
        _drive(problem, target)
    else:
        # Ask the *library* API instead of hand-rolling it, and keep re-asking after a
        # veto so the question "can a veto be overturned by more local evidence?"
        # gets answered by measurement rather than by assumption.
        first = target.transfer_staged(source.dir, after_batches=2, min_reliability=audit_veto)
        imported = int(first.get("imported") or 0)
        rechecks = 0
        while True:
            _drive(problem, target, stop_after_batches=1)
            if target.remaining() <= 0:
                break
            report = target.transfer_staged(
                source.dir, after_batches=2, min_reliability=audit_veto
            )
            rechecks += 1
            if report["stage"] == "widened":
                imported += int(report.get("imported") or 0)
                break
            if rechecks >= 6:
                vetoed = True
                break
        _drive(problem, target)

    incumbent = target.best()
    best = None if incumbent is None else float(incumbent.y)
    return {
        "quality": next(key for key in ("good", "biased", "shuffled") if key in str(source.dir)),
        "policy": policy,
        "seed": seed,
        "imported": imported,
        "vetoed": vetoed,
        "rechecks": locals().get("rechecks", 0),
        "best": None if best is None else round(best, 4),
        "n_obs": len(target.dataset.observations),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", default="ackley10")
    parser.add_argument("--budget", type=int, default=24)
    parser.add_argument("--source-points", type=int, default=30)
    parser.add_argument("--audit-veto", type=float, default=0.6,
                        help="widen only when the audit suggests at least this reliability")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument(
        "--sweep-veto", type=float, nargs="+", default=None,
        help="run the staged policy at each audit-veto threshold instead of the fixed arms",
    )
    parser.add_argument("--out", default=str(OUT / "results.csv"))
    args = parser.parse_args()

    problem = builtin_problems()[args.problem]
    shutil.rmtree(OUT, ignore_errors=True)
    rows: list[dict] = []
    for seed in args.seeds:
        cold = Study.create(
            OUT / f"cold_s{seed}", problem.space, problem.objective, algorithm="auto",
            budget=args.budget, batch_size=3, context=_context(problem, args.budget, 3),
            seed=seed,
        )
        _drive(problem, cold)
        incumbent = cold.best()
        cold_best = None if incumbent is None else float(incumbent.y)
        print(json.dumps({"seed": seed, "cold_best": round(cold_best, 4)}, ensure_ascii=False))
        for quality in QUALITIES:
            source = build_source(problem, seed, quality, args.source_points)
            if args.sweep_veto:
                for threshold in args.sweep_veto:
                    row = run_arm(
                        problem, source, "staged_audit", seed, args.budget, threshold
                    )
                    row["quality"] = quality[0]
                    row["policy"] = f"staged_t{threshold}"
                    row["veto_threshold"] = threshold
                    row["cold_best"] = round(cold_best, 4)
                    row["delta_vs_cold"] = round(row["best"] - cold_best, 4)
                    rows.append(row)
                continue
            for policy in POLICIES:
                row = run_arm(problem, source, policy, seed, args.budget, args.audit_veto)
                row["quality"] = quality[0]
                row["cold_best"] = round(cold_best, 4)
                row["delta_vs_cold"] = round(row["best"] - cold_best, 4)
                rows.append(row)

    policies = (
        [f"staged_t{threshold}" for threshold in args.sweep_veto]
        if args.sweep_veto else list(POLICIES)
    )
    for quality in QUALITIES:
        for policy in policies:
            subset = [row for row in rows if row["quality"] == quality[0] and row["policy"] == policy]
            if not subset:
                continue
            print(json.dumps({
                "quality": quality[0],
                "policy": policy,
                "n": len(subset),
                "best_mean": round(statistics.mean(row["best"] for row in subset), 4),
                "delta_vs_cold_mean": round(
                    statistics.mean(row["delta_vs_cold"] for row in subset), 4
                ),
                "imported_mean": round(statistics.mean(row["imported"] for row in subset), 2),
                "vetoed": sum(1 for row in subset if row["vetoed"]),
            }, ensure_ascii=False))
    if args.sweep_veto:
        summary = []
        for threshold in args.sweep_veto:
            label = f"staged_t{threshold}"
            per_quality = {}
            for quality in QUALITIES:
                subset = [
                    row for row in rows
                    if row["quality"] == quality[0] and row["policy"] == label
                ]
                if subset:
                    per_quality[quality[0]] = {
                        "delta": round(statistics.mean(row["delta_vs_cold"] for row in subset), 4),
                        "veto_rate": round(
                            sum(1 for row in subset if row["vetoed"]) / len(subset), 2
                        ),
                    }
            worst = min((value["delta"] for value in per_quality.values()), default=None)
            summary.append({"threshold": threshold, "per_quality": per_quality, "worst_delta": worst})
        print(json.dumps({"veto_sweep": summary}, ensure_ascii=False))

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