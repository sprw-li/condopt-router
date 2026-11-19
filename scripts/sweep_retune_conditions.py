"""Does the stagnation retune still help under noise / hard constraints?

Compares ``continue`` vs ``tune`` (``Study.retune_for_exploration``) while the
executor reports noisy or constrained measurements, scoring the *true*
noise-free objective.

Run:
    py -3 scripts/sweep_retune_conditions.py --budget 60 --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from condopt.benchmarks import builtin_problems  # noqa: E402
from condopt.study import Study  # noqa: E402

OUT = ROOT / "runs" / "sweep_retune_conditions"


def _measure(problem, config, noise_std: float, rng) -> dict:
    payload: dict = {}
    if problem.objective.is_multi_objective:
        payload["objectives"] = problem.objectives_fn(config)
    else:
        payload["y"] = problem.fn(config)
    if problem.constraints_fn is not None:
        payload["constraints"] = problem.constraints_fn(config)
    if noise_std > 0:
        if "y" in payload and payload["y"] is not None:
            payload["y"] = float(payload["y"]) + float(rng.normal(0, noise_std))
            payload["objective_std"] = noise_std
        if "objectives" in payload:
            payload["objectives"] = {
                name: float(value) + float(rng.normal(0, noise_std))
                for name, value in payload["objectives"].items()
            }
            payload["objective_std"] = noise_std
    return payload


def run_arm(
    arm: str,
    problem_name: str,
    noise_std: float = 0.0,
    budget: int = 60,
    batch_size: int = 4,
    seed: int = 0,
    max_steps: int = 200,
) -> dict:
    problem = builtin_problems()[problem_name]
    run_dir = OUT / f"{arm}_{problem_name}_n{noise_std}_{seed}"
    shutil.rmtree(run_dir, ignore_errors=True)
    study = Study.create(
        run_dir, problem.space, problem.objective,
        algorithm="auto", budget=budget, batch_size=batch_size, seed=seed,
    )
    rng = np.random.default_rng(seed + 555)
    retunes = 0
    stopped_at: int | None = None
    for _ in range(max_steps):
        action = study.next_action()
        kind = action["action"]
        if kind == "tell":
            results = []
            for item in study.pending:
                payload = _measure(problem, item["x"], noise_std, rng)
                payload["id"] = item["id"]
                results.append(payload)
            study.tell(results)
        elif kind == "ask":
            n = int(action.get("ask_n") or batch_size)
            study.ask(max(1, min(n, study.remaining())))
        elif kind == "confirm":
            study.confirm(int(action.get("confirm_n") or 3))
        elif kind == "stop":
            category = (action.get("diagnostics") or {}).get("stop_category")
            if category != "stagnation" or study.remaining() <= 0:
                stopped_at = len(study.dataset.observations)
                break
            if arm == "tune":
                retunes += 1
                study.retune_for_exploration(reason=f"noisy retune #{retunes}")
            study.ask(max(1, min(batch_size, study.remaining())))
        if study.remaining() <= 0 and not study.pending:
            break

    # true (noise-free) best over the conditions actually measured
    true_scores = [float(problem.fn(obs.x)) for obs in study.dataset.observations]
    best_true = max(true_scores) if true_scores else None
    feasible = 0
    if problem.constraints_fn is not None:
        for obs in study.dataset.observations:
            values = problem.constraints_fn(obs.x)
            limits = problem.objective.constraint_limits or {}
            ok = True
            for name in problem.objective.constraints:
                limit = limits.get(name) or {}
                threshold = float(limit.get("value", 0.0))
                op = str(limit.get("op", "<="))
                value = float(values[name])
                if op == "<=" and value > threshold + 1e-9:
                    ok = False
                if op == ">=" and value < threshold - 1e-9:
                    ok = False
            feasible += int(ok)
    return {
        "arm": arm,
        "problem": problem_name,
        "noise_std": noise_std,
        "seed": seed,
        "n_observations": len(study.dataset.observations),
        "retunes": retunes,
        "stopped_early_at": stopped_at,
        "best_true": None if best_true is None else round(best_true, 4),
        "n_feasible": feasible,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", nargs="+", default=["rosenbrock10", "constraint_hard"])
    parser.add_argument("--noise", type=float, nargs="+", default=[0.0, 3.0])
    parser.add_argument("--budget", type=int, default=60)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--out", default=str(OUT / "results.csv"))
    args = parser.parse_args()

    rows = []
    for problem_name in args.problems:
        for noise_std in args.noise:
            for arm in ("continue", "tune"):
                for seed in args.seeds:
                    rows.append(run_arm(
                        arm, problem_name, noise_std, args.budget, args.batch, seed
                    ))
                subset = [
                    row for row in rows
                    if row["problem"] == problem_name
                    and row["noise_std"] == noise_std
                    and row["arm"] == arm
                ]
                print(json.dumps({
                    "problem": problem_name,
                    "noise_std": noise_std,
                    "arm": arm,
                    "n": len(subset),
                    "best_true_mean": round(
                        statistics.mean(row["best_true"] for row in subset), 4
                    ),
                    "retunes_mean": round(
                        statistics.mean(row["retunes"] for row in subset), 2
                    ),
                    "n_feasible_mean": round(
                        statistics.mean(row["n_feasible"] for row in subset), 2
                    ),
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