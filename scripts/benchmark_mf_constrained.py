"""Compare constrained multi-fidelity BO against single-fidelity constrained BO.

Problem: ``mf_constrained`` (multi-fidelity Suzuki + T <= 80 C hard constraint).
Metric: best *feasible full-fidelity* yield under the same resource budget.

Run:
    py -3 scripts/benchmark_mf_constrained.py --budget 20 --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from condopt.benchmarks import builtin_problems, mf_suzuki  # noqa: E402
from condopt.dataset import Observation  # noqa: E402
from condopt.registry import build  # noqa: E402

OUT = ROOT / "runs" / "benchmark_mf_constrained"


def _feasible(y: float, temperature: float) -> bool:
    return temperature <= 80.0 + 1e-9


def run(
    algorithm: str,
    budget: float = 20.0,
    batch_size: int = 2,
    seed: int = 0,
    params: dict | None = None,
) -> dict:
    problem = builtin_problems()["mf_constrained"]
    optimizer = build(
        algorithm, problem.space, problem.objective, params=params or {}, seed=seed
    )
    observations: list[Observation] = []
    spent = 0.0
    while spent < budget - 1e-9:
        if algorithm == "mfbo":
            pairs = optimizer.ask_with_fidelity(batch_size, cost_remaining=budget - spent)
        else:
            pairs = [(config, 1.0) for config in optimizer.ask(batch_size)]
        batch = []
        for config, fidelity in pairs:
            if spent + fidelity > budget + 1e-9 and algorithm == "mfbo":
                continue
            observation = Observation(
                x=config,
                y=mf_suzuki(config, float(fidelity)),
                constraints={"T_max": float(config["T"])},
                fidelity=float(fidelity),
            )
            batch.append(observation)
            spent += float(fidelity)
        if not batch:
            break
        optimizer.tell(batch)
        observations.extend(batch)
    full = [o for o in observations if o.fidelity >= 1.0 - 1e-9]
    feasible_full = [o for o in full if _feasible(o.y, float(o.x["T"]))]
    best = max((o.y for o in feasible_full), default=None)
    return {
        "algorithm": algorithm,
        "seed": seed,
        "budget": budget,
        "n_observations": len(observations),
        "cost_spent": round(spent, 4),
        "n_full_fidelity": len(full),
        "n_feasible_full": len(feasible_full),
        "feasible_full_rate": round(len(feasible_full) / len(full), 4) if full else None,
        "best_feasible_full_yield": None if best is None else round(float(best), 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=float, default=20.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--algorithms", nargs="+", default=["mfbo", "constrained"])
    parser.add_argument("--out", default=str(OUT / "results.csv"))
    args = parser.parse_args()

    rows = []
    for algorithm in args.algorithms:
        for seed in args.seeds:
            row = run(algorithm, budget=args.budget, seed=seed)
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("\nsummary")
    for algorithm in args.algorithms:
        subset = [r for r in rows if r["algorithm"] == algorithm]
        best = [r["best_feasible_full_yield"] for r in subset if r["best_feasible_full_yield"] is not None]
        rates = [r["feasible_full_rate"] for r in subset if r["feasible_full_rate"] is not None]
        print(json.dumps({
            "algorithm": algorithm,
            "n": len(subset),
            "best_feasible_yield_mean": None if not best else round(statistics.mean(best), 3),
            "feasible_full_rate_mean": None if not rates else round(statistics.mean(rates), 3),
            "cost_mean": round(statistics.mean([r["cost_spent"] for r in subset]), 3),
        }, ensure_ascii=False))
    print(f"CSV: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())