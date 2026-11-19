"""Audit every registered algorithm on a capability-matched benchmark.

Guarantees the router's offer set is actually runnable: each algorithm is built
on a problem whose flags match its declared capabilities, asked for a small
batch, told the results, and asked once more.

Run:
    py -3 scripts/audit_algorithms.py --budget 6 --batch 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from condopt import registry  # noqa: E402
from condopt.benchmarks import builtin_problems, run_algorithm  # noqa: E402
from condopt.dataset import Observation  # noqa: E402


def representative_problem(capabilities: set[str], algorithm: str = ""):
    problems = builtin_problems()
    if algorithm == "qehvi_nd":
        return problems["moo3_constrained"]
    if "multi_objective" in capabilities and "multi_fidelity" in capabilities:
        return problems["mf_moo"]
    if "multi_fidelity" in capabilities and "constraints" in capabilities:
        return problems["mf_constrained"]
    if "multi_fidelity" in capabilities:
        return problems["mf_suzuki"]
    if "multi_objective" in capabilities and "constraints" in capabilities:
        return problems["moo_constrained"]
    if "multi_objective" in capabilities:
        return problems["moo_yield_cost"]
    if "constraints" in capabilities:
        return problems["constraint_easy"]
    if "categorical" in capabilities:
        return problems["mixed_reaction"]
    return problems["sphere2d"]


def run_one(name: str, budget: int = 6, batch_size: int = 2) -> dict:
    spec = registry.spec(name)
    capabilities = set(spec.capabilities)
    problem = representative_problem(capabilities, name)
    optimizer = registry.build(name, problem.space, problem.objective, seed=0)
    observations: list[Observation] = []
    if getattr(optimizer, "supports_fidelity", False):
        done = 0
        while done < budget:
            n = min(batch_size, budget - done)
            pairs = optimizer.ask_with_fidelity(n)
            if not pairs:
                break
            batch = []
            for config, fidelity in pairs:
                level = float(fidelity)
                objectives = (
                    problem.objectives_fidelity_fn(config, level)
                    if problem.objectives_fidelity_fn is not None
                    else problem.objectives_fn(config)
                    if problem.objectives_fn is not None
                    else None
                )
                constraints = problem.constraints_fn(config) if problem.constraints_fn else {}
                value = problem.fidelity_fn(config, level) if problem.fidelity_fn else problem.fn(config)
                batch.append(Observation(
                    x=config, y=value, objectives=objectives or {},
                    constraints=constraints, fidelity=level, cost=level,
                ))
            optimizer.tell(batch)
            observations.extend(batch)
            done += len(batch)
    else:
        result = run_algorithm(
            problem, name, budget=budget, batch_size=batch_size, seed=0,
            noise_std=float(getattr(problem, "noise_std", 0.0) or 0.0),
        )
        observations = list(result.trajectory)
    return {
        "algorithm": name,
        "problem": problem.name,
        "n_observations": len(observations),
        "ok": len(observations) > 0,
    }


def audit(budget: int = 6, batch_size: int = 2) -> list[dict]:
    rows = []
    for name in sorted(registry.names()):
        try:
            rows.append(run_one(name, budget=budget, batch_size=batch_size))
        except Exception as exc:  # noqa: BLE001 - audit reports, it does not hide
            rows.append({
                "algorithm": name,
                "problem": None,
                "n_observations": 0,
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=6)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    rows = audit(budget=args.budget, batch_size=args.batch)
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        for row in rows:
            status = "ok" if row["ok"] else "FAIL"
            detail = row.get("error", f"{row['n_observations']} obs")
            print(f"{row['algorithm']:<12} {str(row['problem']):<18} {status:<5} {detail}")
    failures = [row for row in rows if not row["ok"]]
    print(f"audited={len(rows)} failures={len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())