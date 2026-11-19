"""Paper-derived thermoset campaign with the full combined decision problem.

Real inputs from Chem. Mater. 2026 (thermoset p-SDL): component library,
<=3 component rule, cure_time <= 10 s, delta_T <= 70 C.  Synthetic oracle and
synthetic plate bias (documented) because no measured campaign record exists.

Combines: 2 objectives (E, elongation) + hard constraints + plate effect + spec.
Drives the automatic loop with next_action and reports the decision chain.

Run:
    py -3 examples/paper_campaigns/run_combined_decision.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from condopt import Goal, Objective, SearchSpace, Spec, Study  # noqa: E402
from condopt.router import OptimizationContext  # noqa: E402
from run_paper_campaigns import MONOMERS, CROSSLINKERS, thermoset_proxy  # noqa: E402

OUT = Path(__file__).resolve().parent / "runs" / "thermoset_combined_decision"
RESULTS = Path(__file__).resolve().parent / "thermoset_combined_decision_results.json"

PLATE_BIAS = {
    "P1": {"E_MPa": 2.5, "elongation_pct": -15.0},
    "P2": {"E_MPa": -2.5, "elongation_pct": 15.0},
}

SEEDS = [
    {"monomer1": "BCOE", "monomer2": "none", "crosslinker": "DEGDMA", "m2_pct": 0.0, "xl_pct": 12.6},
    {"monomer1": "IDA", "monomer2": "none", "crosslinker": "PEGDA-250", "m2_pct": 0.0, "xl_pct": 25.0},
    {"monomer1": "2-HEA", "monomer2": "none", "crosslinker": "PEGDA-700", "m2_pct": 0.0, "xl_pct": 20.0},
]


def build_space() -> SearchSpace:
    return SearchSpace.from_dict([
        {"name": "monomer1", "kind": "categorical", "choices": MONOMERS},
        {"name": "monomer2", "kind": "categorical", "choices": ["none", *MONOMERS]},
        {"name": "crosslinker", "kind": "categorical", "choices": CROSSLINKERS},
        {"name": "m2_pct", "kind": "continuous", "low": 0.0, "high": 60.0},
        {"name": "xl_pct", "kind": "continuous", "low": 0.0, "high": 50.0},
    ])


def measure(config: dict, plate: str | None) -> dict:
    props, _score, cure = thermoset_proxy(config)
    bias = PLATE_BIAS.get(str(plate), {})
    return {
        "objectives": {
            "E_MPa": props["E_MPa"] + bias.get("E_MPa", 0.0),
            "elongation_pct": props["elongation_pct"] + bias.get("elongation_pct", 0.0),
        },
        "constraints": {"cure_time_s": cure, "delta_T_C": props["delta_T_C"]},
    }


def main() -> int:
    space = build_space()
    objective = Objective(
        name="E_MPa",
        direction="max",
        objectives=("E_MPa", "elongation_pct"),
        directions=("max", "max"),
        constraints=("cure_time_s", "delta_T_C"),
        constraint_limits={
            "cure_time_s": {"op": "<=", "value": 10.0},
            "delta_T_C": {"op": "<=", "value": 70.0},
        },
    )
    spec = Spec(goals=(Goal("elongation_pct", mode="target", target=75.0, tolerance=8.0, scale=25.0),))
    if OUT.exists():
        shutil.rmtree(OUT)
    context = OptimizationContext.from_space(
        space, budget_total=20, batch_size=4, n_objectives=2, n_constraints=2,
        domain="materials", smoothness="rough", noise="high",
    )
    study = Study.create(
        OUT, space, objective, spec=spec, context=context, budget=20, batch_size=4, seed=0,
        params={"n_initial": 4, "mc_draws": 32},
        meta={
            "paper": "Scalable Thermoset Informatics Integrated with Autonomous Experimentation",
            "doi": "10.1021/acs.chemmater.6c01600",
            "proxy_note": "property oracle and plate bias are synthetic",
        },
    )
    study.import_observations([
        {"x": config, **measure(config, None), "source": "paper_seed"}
        for config in SEEDS
    ])
    trace = []
    for step in range(20):
        action = study.next_action()
        trace.append({"action": action["action"], "ask_n": action.get("ask_n"),
                      "suggested": action.get("suggested_actions", [])})
        if action["action"] == "stop":
            break
        if action["action"] == "tell":
            study.tell([
                {"id": item["id"], **measure(item["x"], item.get("plate")), "source": "paper_proxy"}
                for item in study.pending
            ])
        elif action["action"] == "ask":
            plate = "P1" if step % 2 == 0 else "P2"
            study.ask(int(action.get("ask_n") or 4), plate=plate)
        elif action["action"] == "confirm":
            study.confirm(int(action.get("confirm_n") or 3))
    decision = study.decision(explain=True)
    result = {
        "algorithm": study.algorithm,
        "actions": trace,
        "decision": decision,
        "pareto": study.pareto_summary(),
        "spec": study.summary()["spec"],
        "run_dir": str(OUT),
    }
    RESULTS.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "algorithm": result["algorithm"],
        "actions": [item["action"] for item in trace],
        "selection_basis": None if decision is None else decision["selection_basis"],
        "plate_effect_detected": None if decision is None else decision["plate_effect_detected"],
        "in_spec": None if decision is None else decision["in_spec"],
        "n_front": result["pareto"]["n_front"],
        "explain": None if decision is None else decision["explain"],
        "written": str(RESULTS),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())