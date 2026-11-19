"""Unattended campaign template: doctor -> step-machine loop -> doctor.

Drives ``Study.next_action`` end-to-end with any oracle function.  The default
oracle is a synthetic benchmark problem, but the same loop is what a robot
scheduler would use: execute the pending batch, feed results back, follow the
``tell/confirm/ask/stop`` decision until the study stops, then self-check.

Run:
    py -3 scripts/unattended_campaign.py --problem sphere2d --budget 16 --batch 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from condopt.benchmarks import builtin_problems  # noqa: E402
from condopt.study import Study  # noqa: E402


def evaluate_pending(study: Study, problem) -> list[dict]:
    """Oracle call for every pending suggestion (replace with real execution)."""
    results = []
    for item in study.pending:
        condition = item["x"]
        payload: dict = {"id": item["id"]}
        if problem.objective.is_multi_objective:
            payload["objectives"] = problem.objectives_fn(condition)
        else:
            payload["y"] = problem.fn(condition)
        if problem.constraints_fn is not None:
            payload["constraints"] = problem.constraints_fn(condition)
        results.append(payload)
    return results


def run_campaign(
    run_dir: Path,
    problem_name: str = "sphere2d",
    budget: int = 16,
    batch_size: int = 4,
    seed: int = 0,
    max_steps: int = 50,
) -> dict:
    problem = builtin_problems()[problem_name]
    if (Path(run_dir) / "config.json").exists():
        study = Study.load(run_dir)
    else:
        study = Study.create(
            run_dir,
            problem.space,
            problem.objective,
            algorithm="auto",
            budget=budget,
            batch_size=batch_size,
            seed=seed,
        )
    before = study.check_integrity()
    trace = []
    for _ in range(max_steps):
        action = study.next_action()
        kind = action["action"]
        trace.append(kind)
        if kind == "tell":
            study.tell(evaluate_pending(study, problem))
        elif kind == "ask":
            n = int(action.get("ask_n") or min(batch_size, max(1, study.remaining())))
            study.ask(max(1, min(n, study.remaining())))
        elif kind == "confirm":
            study.confirm(int(action.get("confirm_n") or 3))
        elif kind == "stop":
            break
        else:  # pragma: no cover - future actions
            raise RuntimeError(f"unknown action {kind!r}")
    after = study.check_integrity()
    best = study.best()
    return {
        "run_dir": str(study.dir),
        "problem": problem_name,
        "algorithm": study.algorithm,
        "steps": len(trace),
        "trace": trace,
        "n_observations": len(study.dataset),
        "remaining": study.remaining(),
        "best": None if best is None else {"x": best.x, "y": best.y},
        "integrity_before": before["ok"],
        "integrity_after": after["ok"],
        "stop": bool(trace and trace[-1] == "stop"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=str(ROOT / "runs" / "unattended_campaign"))
    parser.add_argument("--problem", default="sphere2d")
    parser.add_argument("--budget", type=int, default=16)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = run_campaign(
        Path(args.run), args.problem, args.budget, args.batch, args.seed, args.max_steps
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    else:
        print(
            f"{report['problem']} | {report['algorithm']} | steps={report['steps']} "
            f"obs={report['n_observations']} stop={report['stop']} "
            f"integrity={report['integrity_before']}/{report['integrity_after']}"
        )
        print(f"trace: {' -> '.join(report['trace'])}")
        if report["best"]:
            print(f"best: {report['best']['y']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())