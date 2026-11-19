"""Reconcile the documented defaults with the values actually in the code.

``docs/DEFAULTS.md`` is only trustworthy if the numbers it quotes match the
implementation.  This checker pins the most load-bearing ones to their source of
truth (registry parameter defaults, ``Study.create`` signature, router scoring
paths, and the explicit thresholds in ``study.py``).

Run:
    py -3 scripts/check_defaults_sync.py
    py -3 scripts/check_defaults_sync.py --json
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from condopt import registry  # noqa: E402
from condopt.router import OptimizationContext, route  # noqa: E402
from condopt.study import Study  # noqa: E402

SPEC_DEFAULTS = (
    ("mfbo", "acquisition", "ei"),
    ("mfbo", "min_full_fraction", 0.25),
    ("mfbo", "feasibility_aware_full", True),
    ("mfbo", "feasibility_full_threshold", 0.0),
    ("mfbo", "two_phase_feasibility", True),
    ("mfparego", "rho", 0.05),
    ("qehvi_nd", "mc_prefilter", 64),
    ("qehvi_nd", "n_candidates", 256),
    ("cmaes", "sigma0", 0.1),
    ("cmaes", "c_mu", 0.5),
)

STUDY_SIGNATURE = (
    ("patience", 3),
    ("stop_on_stagnation", True),
    ("batch_size", None),
)

STUDY_SOURCE_SNIPPETS = (
    ("patience clamp", "self.patience = max(2, requested_patience)"),
    ("low-feasibility threshold", "feasible_fraction < 0.05"),
    ("low-feasibility min observations", "len(dataset) >= 10"),
    ("transfer warning threshold", "int(top_k) > 4"),
    ("cost hard cap", "cost_cap = self.remaining_cost()"),
    ("confirmation batch rule", "base_confirm = default_confirm_n(self.budget)"),
    ("confirmation batch helper", "    return 3"),
)

ROUTER_EXPECTATIONS = (
    ("mf+2obj", dict(n_dims=4, n_continuous=4, budget_total=40, batch_size=4,
                     n_objectives=2, multi_fidelity=True, smoothness="smooth"), "mfparego"),
    ("mf+3obj", dict(n_dims=4, n_continuous=4, budget_total=40, batch_size=4,
                     n_objectives=3, multi_fidelity=True, smoothness="smooth"), "mfparego"),
    ("constrained", dict(n_dims=3, n_continuous=3, budget_total=30, batch_size=3,
                         n_objectives=1, n_constraints=1, noise="low"), "constrained"),
)


def _spec_default(algorithm: str, parameter: str):
    for item in registry.spec(algorithm).params:
        if item.name == parameter:
            return item.default
    raise KeyError(f"{algorithm}.{parameter}")


def check() -> dict:
    mismatches: list[str] = []

    for algorithm, parameter, expected in SPEC_DEFAULTS:
        try:
            actual = _spec_default(algorithm, parameter)
        except KeyError as exc:
            mismatches.append(str(exc))
            continue
        if actual != expected:
            mismatches.append(f"{algorithm}.{parameter}: code={actual!r} doc={expected!r}")

    signature = inspect.signature(Study.create)
    for name, expected in STUDY_SIGNATURE:
        if expected is None:
            continue
        actual = signature.parameters[name].default
        if actual != expected:
            mismatches.append(f"Study.create({name}): code={actual!r} doc={expected!r}")

    study_source = (ROOT / "src" / "condopt" / "study.py").read_text(encoding="utf-8")
    for label, snippet in STUDY_SOURCE_SNIPPETS:
        if snippet not in study_source:
            mismatches.append(f"{label}: expected source snippet missing ({snippet!r})")

    router_source = (ROOT / "src" / "condopt" / "router.py").read_text(encoding="utf-8")
    if 'mfparego' not in router_source:
        mismatches.append("router no longer mentions mfparego")

    route_results = {}
    for label, kwargs, expected in ROUTER_EXPECTATIONS:
        algorithm = route(OptimizationContext(**kwargs)).algorithm
        route_results[label] = algorithm
        if algorithm != expected:
            mismatches.append(f"route[{label}]: code={algorithm!r} doc={expected!r}")

    return {
        "spec_defaults_checked": len(SPEC_DEFAULTS),
        "snippets_checked": len(STUDY_SOURCE_SNIPPETS),
        "route_results": route_results,
        "mismatches": mismatches,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = check()
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        for item in report["mismatches"]:
            print(f"MISMATCH {item}")
        print(
            f"defaults_sync: {report['spec_defaults_checked']} spec defaults, "
            f"{report['snippets_checked']} source snippets, "
            f"{len(report['route_results'])} route checks, "
            f"{len(report['mismatches'])} mismatches"
        )
    return 1 if report["mismatches"] else 0


if __name__ == "__main__":
    raise SystemExit(main())