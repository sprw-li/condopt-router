"""Long-run stability drill: bigger budget, timing, file growth, integrity.

Drives a mixed-space campaign to ``--budget`` evaluations through the
``next_action`` state machine, timing every ask/tell, then reports dataset size,
run-dir footprint, decision count, integrity and digest time.

Run:
    py -3 scripts/long_run_drill.py --budget 120 --batch 6
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from condopt.benchmarks import builtin_problems  # noqa: E402
from condopt.study import Study  # noqa: E402

OUT = ROOT / "runs" / "long_run_drill"


def _dir_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def run_drill(
    problem_name: str = "mixed_reaction",
    budget: int = 120,
    batch_size: int = 6,
    seed: int = 0,
    max_steps: int = 200,
    force_budget: bool = False,
) -> dict:
    problem = builtin_problems()[problem_name]
    shutil.rmtree(OUT, ignore_errors=True)
    study = Study.create(
        OUT, problem.space, problem.objective,
        algorithm="auto", budget=budget, batch_size=batch_size, seed=seed,
    )
    ask_times: list[float] = []
    tell_times: list[float] = []
    trace: list[str] = []
    forced = 0
    stop_category = "budget_exhausted"
    stop_reason = "预算已用尽"
    for _ in range(max_steps):
        action = study.next_action()
        kind = action["action"]
        trace.append(kind)
        if kind == "tell":
            started = time.perf_counter()
            study.tell([
                {"id": item["id"], "y": problem.fn(item["x"])} for item in study.pending
            ])
            tell_times.append(time.perf_counter() - started)
        elif kind == "ask":
            n = int(action.get("ask_n") or batch_size)
            started = time.perf_counter()
            study.ask(max(1, min(n, study.remaining())))
            ask_times.append(time.perf_counter() - started)
        elif kind == "confirm":
            study.confirm(int(action.get("confirm_n") or 3))
        elif kind == "stop":
            category = (action.get("diagnostics") or {}).get("stop_category")
            # ``force_budget`` emulates an operator who keeps the campaign going
            # after a stagnation stop instead of accepting the plateau.
            if force_budget and category == "stagnation" and study.remaining() > 0:
                trace[-1] = "stop->override"
                study.ask(max(1, min(batch_size, study.remaining())))
                forced += 1
                continue
            stop_category = category
            stop_reason = action.get("reason")
            break

    started = time.perf_counter()
    integrity = study.check_integrity()
    integrity_time = time.perf_counter() - started

    from condopt.cli import main as cli_main

    started = time.perf_counter()
    cli_main(["digest", "--run", str(study.dir), "--format", "md"])
    digest_time = time.perf_counter() - started

    ids = [
        obs.meta.get("suggestion_id")
        for obs in study.dataset.observations
        if obs.meta.get("suggestion_id")
    ]
    best = study.best()
    report = {
        "problem": problem_name,
        "budget": budget,
        "algorithm": study.algorithm,
        "n_observations": len(study.dataset.observations),
        "n_decisions": len(study.decisions()),
        "n_rounds": len(study.rounds),
        "pending_left": len(study.pending),
        "duplicate_ids": len(ids) - len(set(ids)),
        "irregular_values": sum(
            1 for obs in study.dataset.observations if obs.y is None
        ),
        "ask_calls": len(ask_times),
        "ask_mean_s": round(statistics.mean(ask_times), 3) if ask_times else None,
        "ask_p95_s": (
            round(sorted(ask_times)[max(0, int(0.95 * len(ask_times)) - 1)], 3)
            if ask_times else None
        ),
        "tell_mean_s": round(statistics.mean(tell_times), 3) if tell_times else None,
        "integrity_ok": integrity["ok"],
        "integrity_time_s": round(integrity_time, 3),
        "digest_time_s": round(digest_time, 3),
        "run_dir_kb": round(_dir_bytes(study.dir) / 1024.0, 1),
        "dataset_kb": round((study.dir / "dataset.jsonl").stat().st_size / 1024.0, 1),
        "best_score": None if best is None else round(float(best.y), 6),
        "stop_category": stop_category,
        "stop_reason": stop_reason,
        "forced_continuations": forced,
        "trace_tail": trace[-8:],
    }
    report["ok"] = bool(
        integrity["ok"]
        and len(ids) == len(set(ids))
        and report["n_observations"] <= budget
        and report["n_observations"] > 0
        and report["irregular_values"] == 0
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", default="mixed_reaction")
    parser.add_argument("--budget", type=int, default=120)
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--force-budget",
        action="store_true",
        help="停滞停止后继续取点，直到真正用完预算（模拟操作者选择继续探索）",
    )
    args = parser.parse_args()
    report = run_drill(
        args.problem, args.budget, args.batch, args.seed,
        force_budget=args.force_budget,
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(report, ensure_ascii=False))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())