"""End-to-end drill: unattended loop + transfer + doctor + digest.

Builds a source campaign, warm-starts a target campaign from it (audit ->
impact preview -> transfer), drives the target with the ``next_action`` state
machine until it stops, then self-checks the run and writes a digest.

Run:
    py -3 scripts/e2e_drill.py
    py -3 scripts/e2e_drill.py --json
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from condopt.benchmarks import builtin_problems  # noqa: E402
from condopt.study import Study  # noqa: E402

OUT = ROOT / "runs" / "e2e_drill"


def _evaluate(study: Study, problem) -> list[dict]:
    results = []
    for item in study.pending:
        x = item["x"]
        fidelity = float(item.get("fidelity", 1.0))
        payload: dict = {"id": item["id"], "fidelity": fidelity, "cost": fidelity}
        if problem.objective.is_multi_objective:
            fidelity_fn = getattr(problem, "objectives_fidelity_fn", None)
            payload["objectives"] = (
                fidelity_fn(x, fidelity)
                if callable(fidelity_fn)
                else problem.objectives_fn(x)
            )
            payload["y"] = float(list(payload["objectives"].values())[0])
        else:
            fidelity_fn = getattr(problem, "fidelity_fn", None)
            payload["y"] = (
                float(fidelity_fn(x, fidelity))
                if callable(fidelity_fn)
                else float(problem.fn(x))
            )
        constraints = getattr(problem, "constraints_fn", None)
        if callable(constraints):
            payload["constraints"] = constraints(x)
        results.append(payload)
    return results


def _drive(study: Study, problem, max_steps: int = 40) -> list[str]:
    trace = []
    for _ in range(max_steps):
        action = study.next_action()
        kind = action["action"]
        trace.append(kind)
        if kind == "tell":
            study.tell(_evaluate(study, problem))
        elif kind == "ask":
            n = int(action.get("ask_n") or 3)
            study.ask(max(1, min(n, study.remaining())))
        elif kind == "confirm":
            study.confirm(int(action.get("confirm_n") or 2))
        elif kind == "stop":
            break
    return trace


def run_drill(problem_name: str = "sphere2d", target_budget: int = 14) -> dict:
    problem = builtin_problems()[problem_name]
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True, exist_ok=True)

    from condopt.router import OptimizationContext

    def _context(budget: int) -> OptimizationContext:
        return OptimizationContext.from_space(
            problem.space,
            budget_total=budget,
            batch_size=4,
            multi_fidelity=callable(getattr(problem, "fidelity_fn", None)),
            n_objectives=problem.objective.n_objectives,
            n_constraints=len(getattr(problem.objective, "constraints", ()) or ()),
            noise="medium",
        )

    source = Study.create(
        OUT / "source", problem.space, problem.objective,
        algorithm="sobol", budget=8, batch_size=4, seed=0,
        context=_context(8),
    )
    source_trace = _drive(source, problem)

    target = Study.create(
        OUT / "target", problem.space, problem.objective,
        algorithm="auto", budget=target_budget, batch_size=4, seed=0,
        context=_context(target_budget),
    )
    target_trace = _drive(target, problem, max_steps=2)  # first round only
    audit = target.assess_transfer(source.dir)
    impact = target.simulate_transfer(
        source.dir, top_k=3,
        prior_reliability=audit.get("suggested_reliability") or 0.6,
    )
    transfer = target.transfer_from(
        source.dir, top_k=3,
        prior_reliability=audit.get("suggested_reliability") or 0.6,
        reason="e2e drill",
    )
    target_trace += _drive(target, problem)

    integrity = target.check_integrity()
    from condopt.cli import main as cli_main

    cli_main(["digest", "--run", str(target.dir), "--format", "md"])
    digest_path = target.dir / "digest.md"

    ids = [
        obs.meta.get("suggestion_id")
        for obs in target.dataset.observations
        if obs.meta.get("suggestion_id")
    ]
    best = target.best()
    report = {
        "problem": problem_name,
        "source_trace": source_trace,
        "target_trace": target_trace,
        "audit_ready": audit.get("ready"),
        "suggested_reliability": audit.get("suggested_reliability"),
        "impact_changed": impact.get("changed"),
        "transferred": transfer.get("imported"),
        "n_observations": len(target.dataset.observations),
        "pending_left": len(target.pending),
        "duplicate_ids": len(ids) - len(set(ids)),
        "integrity_ok": integrity["ok"],
        "integrity_issues": integrity["issues"],
        "digest_written": digest_path.exists(),
        "best_score": None if best is None else round(float(best.y), 6),
        "delivery_checklist": target.delivery_checklist(),
        "delivery_ok": all(item["ok"] for item in target.delivery_checklist()),
        "report_has_checklist": "交付自查:" in target.report(),
        "ok": bool(
            integrity["ok"]
            and not target.pending
            and len(ids) == len(set(ids))
            and (transfer.get("imported") or 0) > 0
            and digest_path.exists()
            and all(item["ok"] for item in target.delivery_checklist())
            and "交付自查:" in target.report()
        ),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--problem",
        default="sphere2d",
        choices=sorted(builtin_problems()),
        help="any builtin problem (MF/constrained shapes are supported)",
    )
    args = parser.parse_args()
    report = run_drill(args.problem)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(json.dumps({
            key: report[key]
            for key in (
                "problem", "transferred", "suggested_reliability", "impact_changed",
                "n_observations", "pending_left", "duplicate_ids", "integrity_ok",
                "delivery_ok",
                "digest_written", "best_score", "ok",
            )
        }, ensure_ascii=False))
        print(f"traces: source {'->'.join(report['source_trace'])} | "
              f"target {'->'.join(report['target_trace'])}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())