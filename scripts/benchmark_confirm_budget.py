"""Does the confirmation batch size matter on a small budget?

Budgets of 10-29 now use 2 confirmation replicates instead of 3.  This measures
the trade-off on a small campaign (branin, budget 12, batch 3, 5 seeds): the
confirmation replicates are evaluations that cannot explore.

Run:
    py -3 scripts/benchmark_confirm_budget.py --seeds 0 1 2 3 4
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
from condopt.study import Study  # noqa: E402

OUT = ROOT / "runs" / "benchmark_confirm_budget"


def run_arm(
    confirm_n: int | None,
    seed: int,
    budget: int,
    batch: int,
    problem_name: str = "branin",
) -> dict:
    problem = builtin_problems()[problem_name]
    multi_fidelity = callable(getattr(problem, "fidelity_fn", None))
    run_dir = OUT / f"confirm_{confirm_n}_s{seed}"
    shutil.rmtree(run_dir, ignore_errors=True)
    meta = {"confirm_n": confirm_n} if confirm_n is not None else None
    study = Study.create(
        run_dir, problem.space, problem.objective, algorithm="bayes",
        budget=budget, batch_size=batch, seed=seed, meta=meta,
    )
    for _ in range(max(120, 8 * budget)):
        action = study.next_action()
        kind = action["action"]
        if kind == "stop":
            break
        if kind == "ask":
            study.ask(int(action.get("ask_n") or batch))
        elif kind == "confirm":
            study.confirm(int(action.get("confirm_n") or batch))
        else:
            study.tell([_measure(problem, item, multi_fidelity) for item in study.pending])
            study.record_progress()
    observations = study.dataset.observations
    confirmations = [obs for obs in observations if obs.meta.get("purpose") == "confirmation"]
    constraints = tuple(getattr(problem.objective, "constraints", ()) or ())
    if constraints:
        feasible = [obs for obs in observations if study._is_feasible(obs)]
        best_value = max((float(obs.y) for obs in feasible), default=None)
    else:
        best = study.best()
        best_value = None if best is None else float(best.y)
    checklist = study.delivery_checklist()
    return {
        "confirm_n": confirm_n if confirm_n is not None else "auto",
        "seed": seed,
        "n_observations": len(observations),
        "n_confirmations": len(confirmations),
        "n_exploration": len(observations) - len(confirmations),
        "best": None if best_value is None else round(best_value, 4),
        "delivery_ok": bool(checklist) and all(item["ok"] for item in checklist),
        "problem": problem_name,
    }


def _measure(problem, item, multi_fidelity: bool) -> dict:
    x = item["x"]
    fidelity = float(item.get("fidelity", 1.0)) if multi_fidelity else 1.0
    fidelity_fn = getattr(problem, "fidelity_fn", None)
    y = (
        float(fidelity_fn(x, fidelity))
        if multi_fidelity and callable(fidelity_fn)
        else float(problem.fn(x))
    )
    payload: dict = {"id": item["id"], "y": y, "fidelity": fidelity, "cost": fidelity}
    constraints = getattr(problem, "constraints_fn", None)
    if callable(constraints):
        payload["constraints"] = constraints(x)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=12)
    parser.add_argument("--batch", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--problem", default="branin")
    parser.add_argument("--out", default=str(OUT / "results.csv"))
    args = parser.parse_args()

    shutil.rmtree(OUT, ignore_errors=True)
    rows = []
    for arm in (1, 2, 3, None):
        for seed in args.seeds:
            rows.append(run_arm(arm, seed, args.budget, args.batch, args.problem))
        subset = [
            row for row in rows
            if row["confirm_n"] == (arm if arm else "auto")
        ]
        bests = [row["best"] for row in subset if row["best"] is not None]
        print(json.dumps({
            "confirm_n": arm if arm else "auto",
            "n": len(subset),
            "best_mean": round(statistics.mean(bests), 4) if bests else None,
            "exploration_mean": round(
                statistics.mean(row["n_exploration"] for row in subset), 2
            ),
            "confirmations_mean": round(
                statistics.mean(row["n_confirmations"] for row in subset), 2
            ),
            "delivery_ok": sum(row["delivery_ok"] for row in subset),
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