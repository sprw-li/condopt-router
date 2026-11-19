"""Route-and-build audit across a capability matrix.

For every (objectives, constraints, multi-fidelity, dims, noise, ...) context:
route it, build the chosen algorithm on a matching space, and ask for a small
batch.  Any routing choice that cannot actually run is a bug.

Run:
    py -3 scripts/audit_routing.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from condopt import registry  # noqa: E402
from condopt.base import Objective  # noqa: E402
from condopt.router import OptimizationContext, _required_capabilities, route  # noqa: E402
from condopt.space import SearchSpace  # noqa: E402


def make_space(n_dims: int, mixed: bool) -> SearchSpace:
    if not mixed:
        return SearchSpace.from_dict(
            [{"name": f"x{i}", "kind": "continuous", "low": 0.0, "high": 1.0}
             for i in range(n_dims)]
        )
    continuous = max(1, n_dims - 2)
    fields = [
        {"name": f"x{i}", "kind": "continuous", "low": 0.0, "high": 1.0}
        for i in range(continuous)
    ]
    fields.append({"name": "cat", "kind": "categorical", "choices": ["a", "b", "c"]})
    fields.append({"name": "n", "kind": "integer", "low": 1, "high": 4})
    return SearchSpace.from_dict(fields)


def make_objective(n_objectives: int, n_constraints: int) -> Objective:
    constraints = tuple(f"c{i}" for i in range(n_constraints))
    limits = {name: {"op": "<=", "value": 0.0} for name in constraints}
    if n_objectives == 1:
        return Objective(
            name="y", direction="max", constraints=constraints, constraint_limits=limits
        )
    names = tuple(f"y{i}" for i in range(n_objectives))
    directions = tuple("max" if index % 2 == 0 else "min" for index in range(n_objectives))
    return Objective(
        name=names[0], direction="max", objectives=names, directions=directions,
        constraints=constraints, constraint_limits=limits,
        ref_point=tuple(0.5 for _ in names),
    )


def audit() -> list[dict]:
    rows: list[dict] = []
    grid = itertools.product(
        (2, 6, 12),            # n_dims
        (1, 2, 3),             # n_objectives
        (0, 1),                # n_constraints
        (False, True),         # multi_fidelity
        (False, True),         # mixed space
        ("low", "high"),       # noise
        (4, 40),               # budget_total (4 exercises the tiny-budget relaxation)
    )
    for (
        n_dims, n_objectives, n_constraints, multi_fidelity, mixed, noise, budget
    ) in grid:
        if mixed and n_dims < 3:
            continue
        if multi_fidelity and mixed:
            continue  # multi-fidelity benchmarks are numeric; covered elsewhere
        space = make_space(n_dims, mixed)
        objective = make_objective(n_objectives, n_constraints)
        context = OptimizationContext(
            n_dims=n_dims,
            n_continuous=n_dims if not mixed else max(1, n_dims - 2),
            budget_total=budget,
            batch_size=4,
            n_objectives=n_objectives,
            n_constraints=n_constraints,
            multi_fidelity=multi_fidelity,
            noise=noise,
            smoothness="smooth",
            n_categorical=2 if mixed else 0,
        )
        row = {
            "n_dims": n_dims, "n_objectives": n_objectives,
            "n_constraints": n_constraints, "multi_fidelity": multi_fidelity,
            "mixed": mixed, "noise": noise, "budget_total": budget,
        }
        try:
            recommendation = route(context)
            row["algorithm"] = recommendation.algorithm
            required = _required_capabilities(context)
            capabilities = set(registry.spec(recommendation.algorithm).capabilities)
            candidates = sorted(
                name
                for name in registry.names()
                if required <= set(registry.spec(name).capabilities)
            )
            row["candidates"] = candidates
            row["n_candidates"] = len(candidates)
            row["warnings"] = list(recommendation.warnings)
            alt_violations = sorted(
                item["algorithm"]
                for item in recommendation.alternatives
                if not required <= set(registry.spec(item["algorithm"]).capabilities)
            )
            row["alternative_violations"] = alt_violations
            underpowered = getattr(recommendation, "underpowered_budget", None) or {}
            relaxed = bool(underpowered.get("capability_relaxed"))
            row["capability_relaxed"] = relaxed
            if relaxed:
                # documented tiny-budget relaxation: a space-filling sampler is
                # recommended although it does not declare the context's
                # capabilities; the study still filters and sorts the results.
                row["relaxed_algorithms"] = sorted(
                    {recommendation.algorithm, *alt_violations}
                )
                registry.build(
                    recommendation.algorithm, space, objective, params={}, seed=0
                ).ask(2)
                row["ok"] = True
                rows.append(row)
                continue
            if not required <= capabilities:
                row.update(ok=False, error=f"missing capabilities {sorted(required - capabilities)}")
                rows.append(row)
                continue
            if alt_violations:
                row.update(ok=False, error=f"alternatives miss capabilities {alt_violations}")
                rows.append(row)
                continue
            optimizer = registry.build(
                recommendation.algorithm, space, objective, params={}, seed=0
            )
            if getattr(optimizer, "supports_fidelity", False):
                optimizer.ask_with_fidelity(2)
            else:
                optimizer.ask(2)
            row["ok"] = True
        except Exception as exc:  # noqa: BLE001 - audit reports failures
            row.setdefault("algorithm", None)
            row.update(ok=False, error=f"{type(exc).__name__}: {exc}")
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    rows = audit()
    failures = [row for row in rows if not row["ok"]]
    single = [row for row in rows if row.get("n_candidates") == 1]
    alternative_violations = [
        row for row in rows
        if row.get("alternative_violations") and not row.get("capability_relaxed")
    ]
    relaxed_rows = [row for row in rows if row.get("capability_relaxed")]
    combos: dict[str, set[str]] = {}
    for row in rows:
        key = (
            f"obj={row['n_objectives']},con={row['n_constraints']},"
            f"mf={int(bool(row['multi_fidelity']))}"
        )
        combos.setdefault(key, set()).update(row.get("candidates", []))
    if not args.json:
        print("capability coverage (required set -> candidates)")
        for key in sorted(combos):
            print(f"  {key}: {', '.join(sorted(combos[key])) or 'NONE'}")
        if alternative_violations:
            print(f"contexts advertising an unusable alternative: {len(alternative_violations)}")
            for row in alternative_violations[:10]:
                print(
                    f"  obj={row['n_objectives']} con={row['n_constraints']} "
                    f"mf={row['multi_fidelity']} -> {row['alternative_violations']}"
                )
        if single:
            print(f"single-candidate contexts: {len(single)}")
            for row in single[:10]:
                print(
                    f"  obj={row['n_objectives']} con={row['n_constraints']} "
                    f"mf={row['multi_fidelity']} -> {row['algorithm']}"
                )
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        for row in failures:
            print(json.dumps(row, ensure_ascii=False))
    print(
        f"contexts={len(rows)} failures={len(failures)} "
        f"single_candidate={len(single)} "
        f"alternative_capability_violations={len(alternative_violations)} "
        f"capability_relaxed={len(relaxed_rows)}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())