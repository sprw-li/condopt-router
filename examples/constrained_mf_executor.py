"""Constrained + multi-fidelity executor example.

Combines the two executor responsibilities that were previously shown
separately: read the suggested ``fidelity`` **and** report a constraint value,
under a hard cost budget.

Problem: ``mf_constrained`` (Suzuki-style yield with a fidelity bias, T <= 80 degC).

Run:
    py -3 examples/constrained_mf_executor.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from condopt.benchmarks import builtin_problems  # noqa: E402
from condopt.router import OptimizationContext  # noqa: E402
from condopt.study import Study  # noqa: E402

OUT = ROOT / "runs" / "constrained_mf_executor"


# ``algorithm="mfbo"`` keeps the demo deterministic; ``"auto"`` lets the router
# decide (at a tiny budget it advises space filling -- see docs/ROUTING_POLICY.md).
def run_campaign(cost_budget: float = 10.0, budget: int = 24, seed: int = 0,
    algorithm: str = "mfbo",
) -> dict:
    shutil.rmtree(OUT, ignore_errors=True)
    problem = builtin_problems()["mf_constrained"]
    context = OptimizationContext.from_space(
        problem.space,
        budget_total=budget,
        batch_size=3,
        multi_fidelity=True,
        n_constraints=1,
        noise="medium",
    )
    study = Study.create(
        OUT,
        problem.space,
        problem.objective,
        algorithm=algorithm,
        budget=budget,
        batch_size=3,
        cost_budget=cost_budget,
        context=context,
        seed=seed,
    )
    steps = 0
    while steps < 40:
        steps += 1
        action = study.next_action()
        kind = action["action"]
        if kind == "ask":
            study.ask(int(action.get("ask_n") or 3))
        elif kind == "confirm":
            study.confirm(int(action.get("confirm_n") or 3))
        elif kind == "tell":
            results = []
            for item in study.pending:
                fidelity = float(item.get("fidelity", 1.0) or 1.0)
                results.append({
                    "id": item["id"],
                    "fidelity": fidelity,
                    "y": float(problem.fidelity_fn(item["x"], fidelity)),
                    "constraints": problem.constraints_fn(item["x"]),
                    "cost": fidelity,
                })
            study.tell(results)
        elif kind == "stop":
            break

    observations = study.dataset.observations
    feasible_full = [
        obs for obs in observations
        if obs.fidelity >= 1.0 - 1e-9 and float(obs.constraints.get("T_max", 1e9)) <= 80.0
    ]
    levels: dict[str, int] = {}
    for observation in observations:
        key = f"{float(observation.fidelity):.2f}"
        levels[key] = levels.get(key, 0) + 1
    best = study.best()
    best_feasible = max((float(obs.y) for obs in feasible_full), default=None)
    report = {
        "algorithm": study.algorithm,
        "steps": steps,
        "n_observations": len(observations),
        "cost_spent": round(study.cost_spent(), 3),
        "cost_budget": cost_budget,
        "cost_within_budget": study.cost_spent() <= cost_budget + 1e-9,
        "fidelity_levels": levels,
        "n_feasible_full_fidelity": len(feasible_full),
        "best_full_fidelity": None if best is None else round(float(best.y), 3),
        "best_feasible_full_fidelity": (
            None if best_feasible is None else round(best_feasible, 3)
        ),
        "fidelity_summary": study.fidelity_summary(),
        "integrity_ok": study.check_integrity()["ok"],
    }
    report["ok"] = bool(
        report["cost_within_budget"]
        and report["integrity_ok"]
        and report["n_observations"] > 0
    )
    if report["best_feasible_full_fidelity"] is None:
        report["warning"] = (
            "没有任何可行的全保真观测：交付缺少可行的全保真证据；"
            "应对当前可行的低保真最优点做全保真确认（step 会给出 "
            "collect_feasible_full_fidelity 建议）"
        )
    return report


def main() -> int:
    report = run_campaign()
    (OUT / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        key: report[key]
        for key in (
            "algorithm", "n_observations", "cost_spent", "cost_within_budget",
            "fidelity_levels", "n_feasible_full_fidelity",
            "best_feasible_full_fidelity", "integrity_ok", "ok",
        )
    }, ensure_ascii=False))
    print(f"report: {OUT / 'report.json'}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())