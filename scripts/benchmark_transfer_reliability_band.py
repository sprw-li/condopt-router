"""Does ``prior_reliability`` actually reach the model?

``transfer_from`` raises the imported observation's ``objective_std`` when the
source is doubted, and that value is supposed to make the GP discount the prior.
This checks the two halves separately:

* the mechanism: ``mfbo`` builds a per-point ``alpha`` array (it used to ignore
  ``objective_std`` entirely, so ``prior_reliability`` was a no-op there);
* the effect: on ``mf_constrained`` (budget 20, 3 seeds) the band arms are compared
  with a cold start.

Run:
    py -3 scripts/benchmark_transfer_reliability_band.py --seeds 0 1 2 --budget 20
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from condopt.benchmarks import builtin_problems  # noqa: E402
from condopt.router import OptimizationContext  # noqa: E402
from condopt.study import Study  # noqa: E402

OUT = ROOT / "runs" / "benchmark_transfer_band"
ARMS: tuple[float | None, ...] = (None, 1.0, 0.8, 0.6, 0.4)


def _evaluate(problem, item) -> dict:
    x = item["x"]
    fidelity = float(item.get("fidelity", 1.0))
    fidelity_fn = getattr(problem, "fidelity_fn", None)
    y = float(fidelity_fn(x, fidelity)) if callable(fidelity_fn) else float(problem.fn(x))
    payload: dict = {"id": item["id"], "y": y, "fidelity": fidelity, "cost": fidelity}
    constraints = getattr(problem, "constraints_fn", None)
    if callable(constraints):
        payload["constraints"] = constraints(x)
    return payload


def _drive(problem, study: Study) -> None:
    # cap scales with the study budget (a fixed cap silently truncates long runs)
    for _ in range(max(120, 8 * int(getattr(study, 'budget', 30) or 30))):
        action = study.next_action()
        kind = action["action"]
        if kind == "stop":
            break
        if kind == "ask":
            study.ask(int(action.get("ask_n") or 3))
        elif kind == "confirm":
            study.confirm(int(action.get("confirm_n") or 3))
        else:
            study.tell([_evaluate(problem, item) for item in study.pending])
            study.record_progress()


def _context(problem, budget: int) -> OptimizationContext:
    return OptimizationContext.from_space(
        problem.space,
        budget_total=budget,
        batch_size=3,
        multi_fidelity=callable(getattr(problem, "fidelity_fn", None)),
        n_objectives=problem.objective.n_objectives,
        n_constraints=len(getattr(problem.objective, "constraints", ()) or ()),
        noise="medium",
    )


def build_source(problem, seed: int, budget: int) -> Study:
    run_dir = OUT / f"source_s{seed}"
    shutil.rmtree(run_dir, ignore_errors=True)
    study = Study.create(
        run_dir, problem.space, problem.objective, algorithm="auto",
        budget=budget, batch_size=3, context=_context(problem, budget), seed=seed + 100,
    )
    _drive(problem, study)
    return study


def run_arm(
    problem,
    source: Study,
    reliability: float | None,
    seed: int,
    budget: int,
    top_k: int = 3,
    include_low_fidelity: bool = False,
) -> dict:
    tag = "cold" if reliability is None else f"r{reliability}"
    run_dir = OUT / f"{tag}_s{seed}"
    shutil.rmtree(run_dir, ignore_errors=True)
    target = Study.create(
        run_dir, problem.space, problem.objective, algorithm="mfbo",
        budget=budget + 3, batch_size=3, context=_context(problem, budget + 3), seed=seed,
    )
    imported = 0
    if reliability is not None:
        report = target.transfer_from(
            source.dir, top_k=top_k, prior_reliability=reliability,
            include_low_fidelity=include_low_fidelity, reason="band test",
        )
        imported = int(report["imported"])
    _drive(problem, target)
    constraints = tuple(getattr(problem.objective, "constraints", ()) or ())
    if constraints:
        feasible = [obs for obs in target.dataset.observations if target._is_feasible(obs)]
        best = max((float(obs.y) for obs in feasible), default=None)
    else:
        incumbent = target.best()
        best = None if incumbent is None else float(incumbent.y)
    return {
        "arm": tag,
        "reliability": reliability,
        "seed": seed,
        "imported": imported,
        "best": None if best is None else round(best, 4),
        "n_observations": len(target.dataset.observations),
    }


def model_uses_reliability(seed: int = 0, budget: int = 20) -> dict:
    """Mechanism check: does the prior's doubt reach the GP's per-point alpha?"""
    import numpy as np

    problem = builtin_problems()["mf_constrained"]
    source = build_source(problem, seed, budget)
    out: dict[str, float] = {}
    for reliability in (1.0, 0.4):
        run_dir = OUT / f"mech_r{reliability}"
        shutil.rmtree(run_dir, ignore_errors=True)
        target = Study.create(
            run_dir, problem.space, problem.objective, algorithm="mfbo",
            budget=budget + 3, batch_size=3,
            context=_context(problem, budget + 3), seed=seed,
        )
        target.transfer_from(source.dir, top_k=1, prior_reliability=reliability)
        imported = list(target.dataset.observations)
        assert imported, "transfer imported nothing"
        # Build the optimizer directly on the imported priors: the study builds its
        # optimizer lazily inside ask(), so this is the deterministic way to inspect
        # the per-point alpha the GP actually receives.
        from condopt import registry

        algorithm = registry.build(
            "mfbo", target.space, target.objective, prior=imported, seed=seed
        )
        features = np.vstack([
            algorithm._features(obs.x, obs.fidelity) for obs in imported
        ])
        values = np.asarray(
            [
                obs.score(target.objective.name, target.objective.direction)
                for obs in imported
            ],
            dtype=float,
        )
        model = algorithm._fit(features, values)
        alpha = np.asarray(model.alpha, dtype=float)
        out[f"r{reliability}"] = float(np.max(alpha))
        out[f"r{reliability}_objective_std"] = float(imported[0].objective_std)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--budget", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--include-low-fidelity", action="store_true")
    parser.add_argument("--out", default=str(OUT / "results.csv"))
    args = parser.parse_args()

    problem = builtin_problems()["mf_constrained"]
    shutil.rmtree(OUT, ignore_errors=True)
    rows: list[dict] = []
    for seed in args.seeds:
        source = build_source(problem, seed, args.budget)
        for reliability in ARMS:
            rows.append(
                run_arm(
                    problem, source, reliability, seed, args.budget,
                    top_k=args.top_k,
                    include_low_fidelity=args.include_low_fidelity,
                )
            )

    baseline = statistics.mean(
        row["best"] for row in rows if row["arm"] == "cold" and row["best"] is not None
    )
    for reliability in ARMS:
        tag = "cold" if reliability is None else f"r{reliability}"
        arm = [row for row in rows if row["arm"] == tag and row["best"] is not None]
        if not arm:
            continue
        mean = statistics.mean(row["best"] for row in arm)
        print(json.dumps({
            "arm": tag,
            "n": len(arm),
            "best_mean": round(mean, 4),
            "delta_vs_cold": round(mean - baseline, 4),
            "imported_mean": round(statistics.mean(row["imported"] for row in arm), 2),
        }, ensure_ascii=False))

    mechanism = model_uses_reliability(seed=args.seeds[0], budget=args.budget)
    print(json.dumps({"mechanism_alpha_max": mechanism}, ensure_ascii=False))

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