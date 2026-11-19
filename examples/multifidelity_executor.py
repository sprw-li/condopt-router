"""Multi-fidelity + cost write-back executor example.

Shows how an executor reads the suggested ``fidelity`` from pending items,
charges the matching cost, and how the optimizer balances cheap biased runs
against expensive full-fidelity measurements under a cost budget.

Run:
    py -3 examples/multifidelity_executor.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from condopt.benchmarks import mf_suzuki_cost, mf_suzuki_problem  # noqa: E402
from condopt.router import OptimizationContext  # noqa: E402
from condopt.study import Study  # noqa: E402

OUT = ROOT / "runs" / "multifidelity_executor"


# ``algorithm="mfbo"`` keeps the demo deterministic; ``"auto"`` lets the router
# decide (at a tiny budget it advises space filling -- see docs/ROUTING_POLICY.md).
def run_campaign(cost_budget: float = 12.0, budget: int = 24, seed: int = 0,
    algorithm: str = "mfbo",
) -> dict:
    shutil.rmtree(OUT, ignore_errors=True)
    problem = mf_suzuki_problem()
    context = OptimizationContext.from_space(
        problem.space, budget_total=budget, batch_size=3, multi_fidelity=True
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
                    "cost": mf_suzuki_cost(fidelity),
                })
            study.tell(results)
        elif kind == "stop":
            break
    levels: dict[str, int] = {}
    for observation in study.dataset.observations:
        key = f"{float(observation.fidelity):.2f}"
        levels[key] = levels.get(key, 0) + 1
    best = study.best()
    report = {
        "algorithm": study.algorithm,
        "steps": steps,
        "n_observations": len(study.dataset.observations),
        "cost_spent": round(study.cost_spent(), 3),
        "cost_budget": cost_budget,
        "fidelity_levels": levels,
        "fidelity_summary": study.fidelity_summary(),
        "best_full_fidelity": None if best is None else round(float(best.y), 3),
        "recommendation": None if study.recommend() is None else study.recommend()["condition"],
    }
    return report


def main() -> int:
    report = run_campaign()
    (OUT / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "algorithm": report["algorithm"],
        "n_observations": report["n_observations"],
        "cost_spent": report["cost_spent"],
        "fidelity_levels": report["fidelity_levels"],
        "best_full_fidelity": report["best_full_fidelity"],
    }, ensure_ascii=False))
    print(f"report: {OUT / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())