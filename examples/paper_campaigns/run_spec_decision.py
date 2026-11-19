"""Paper-derived thermoset campaign driven by the decision layer.

Stage 1 uses the paper-derived target windows; the synthetic proxy cannot reach
UTS=7.5, so the decision layer stops with in_spec=0 and suggests ``review_spec``.
Stage 2 uses ``suggest_spec`` (target moved to the observed bound) and reruns
the same loop, demonstrating the calibration -> suggestion -> rerun path.

The component library and process constraints come from Chem. Mater. 2026
(thermoset p-SDL); the property oracle is the documented synthetic proxy in
``run_paper_campaigns.py``.

Run:
    py -3 examples/paper_campaigns/run_spec_decision.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from condopt import Goal, SearchSpace, Spec, Study  # noqa: E402
from condopt.router import OptimizationContext  # noqa: E402
from run_paper_campaigns import MONOMERS, CROSSLINKERS, thermoset_proxy  # noqa: E402

OUT = Path(__file__).resolve().parent / "runs" / "thermoset_spec_decision"
OUT_SUGGESTED = OUT.parent / "thermoset_spec_decision_suggested"
RESULTS = Path(__file__).resolve().parent / "thermoset_spec_decision_results.json"

SEEDS = [
    {"monomer1": "BCOE", "monomer2": "none", "crosslinker": "DEGDMA", "m2_pct": 0.0, "xl_pct": 12.6},
    {"monomer1": "IDA", "monomer2": "none", "crosslinker": "PEGDA-250", "m2_pct": 0.0, "xl_pct": 25.0},
    {"monomer1": "2-HEA", "monomer2": "none", "crosslinker": "PEGDA-700", "m2_pct": 0.0, "xl_pct": 20.0},
    {"monomer1": "IBOA", "monomer2": "BCOE", "crosslinker": "PEGDA-250", "m2_pct": 30.0, "xl_pct": 15.0},
]


def build_spec() -> Spec:
    return Spec(
        goals=(
            Goal("E_MPa", mode="target", target=4.5, tolerance=0.5, scale=1.5),
            Goal("UTS_MPa", mode="target", target=7.5, tolerance=0.7, scale=2.0),
            Goal("elongation_pct", mode="target", target=75.0, tolerance=8.0, scale=25.0),
        ),
        constraints={
            "cure_time_s": {"op": "<=", "value": 10.0},
            "delta_T_C": {"op": "<=", "value": 70.0},
        },
    )


def build_space() -> SearchSpace:
    return SearchSpace.from_dict([
        {"name": "monomer1", "kind": "categorical", "choices": MONOMERS},
        {"name": "monomer2", "kind": "categorical", "choices": ["none", *MONOMERS]},
        {"name": "crosslinker", "kind": "categorical", "choices": CROSSLINKERS},
        {"name": "m2_pct", "kind": "continuous", "low": 0.0, "high": 60.0},
        {"name": "xl_pct", "kind": "continuous", "low": 0.0, "high": 50.0},
    ])


def simulate(study: Study) -> None:
    results = []
    for item in study.pending:
        props, _score, cure = thermoset_proxy(item["x"])
        results.append({
            "id": item["id"],
            "objectives": {
                "E_MPa": props["E_MPa"],
                "UTS_MPa": props["UTS_MPa"],
                "elongation_pct": props["elongation_pct"],
            },
            "constraints": {"cure_time_s": cure, "delta_T_C": props["delta_T_C"]},
            "source": "paper_proxy",
        })
    study.tell(results)


def seed(study: Study) -> None:
    payload = []
    for config in SEEDS:
        props, _score, cure = thermoset_proxy(config)
        payload.append({
            "x": config,
            "objectives": {
                "E_MPa": props["E_MPa"],
                "UTS_MPa": props["UTS_MPa"],
                "elongation_pct": props["elongation_pct"],
            },
            "constraints": {"cure_time_s": cure, "delta_T_C": props["delta_T_C"]},
            "source": "paper_seed",
        })
    study.import_observations(payload)


def run_study(spec: Spec, run_dir: Path, seed_value: int = 0) -> tuple[Study, list[dict]]:
    if run_dir.exists():
        shutil.rmtree(run_dir)
    space = build_space()
    context = OptimizationContext.from_space(
        space, budget_total=20, batch_size=4, n_objectives=1, n_constraints=2,
        domain="materials", smoothness="rough", noise="high",
    )
    study = Study.create(
        run_dir, space, spec=spec, context=context, budget=20, batch_size=4,
        seed=seed_value,
        meta={
            "paper": "Scalable Thermoset Informatics Integrated with Autonomous Experimentation",
            "doi": "10.1021/acs.chemmater.6c01600",
            "pmcid": "PMC13718556",
            "proxy_note": "property oracle is synthetic; target windows derived from paper Section 2",
        },
    )
    seed(study)
    trace = []
    for _ in range(20):
        action = study.next_action()
        trace.append({"action": action["action"], "ask_n": action.get("ask_n"),
                      "suggested": action.get("suggested_actions", [])})
        if action["action"] == "stop":
            break
        if action["action"] == "tell":
            simulate(study)
        elif action["action"] == "ask":
            study.ask(int(action.get("ask_n") or 4))
        elif action["action"] == "confirm":
            study.confirm(int(action.get("confirm_n") or 3))
    return study, trace


def summarise(study: Study, spec: Spec, trace: list[dict]) -> dict:
    recommendation = study.recommend()
    calibrated, calibration = spec.calibrate_observations(study.dataset.observations)
    return {
        "algorithm": study.algorithm,
        "actions": trace,
        "summary_spec": study.summary()["spec"],
        "recommendation": recommendation,
        "spec_calibration": calibration,
        "spec_review": spec.achievability(
            [spec.observation_values(observation) for observation in study.dataset.observations]
        ),
        "calibrated_spec": calibrated.to_dict(),
        "run_dir": str(study.dir),
    }


def main() -> int:
    base_spec = build_spec()
    study, trace = run_study(base_spec, OUT)
    base = summarise(study, base_spec, trace)

    suggested_spec, suggestion = base_spec.suggest_spec(
        [base_spec.observation_values(observation) for observation in study.dataset.observations]
    )
    study2, trace2 = run_study(suggested_spec, OUT_SUGGESTED)
    suggested = summarise(study2, suggested_spec, trace2)

    result = {
        "base": base,
        "suggestion": suggestion,
        "suggested_spec": suggested_spec.to_dict(),
        "suggested_campaign": suggested,
    }
    RESULTS.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "base": {
            "actions": [item["action"] for item in trace],
            "in_spec": base["recommendation"]["spec"]["in_spec"] if base["recommendation"]["spec"] else None,
            "spec_review": base["spec_review"]["metrics"],
        },
        "suggested_spec": {
            "targets": [goal.target for goal in suggested_spec.goals],
            "changed": suggestion["changed"],
        },
        "suggested_campaign": {
            "actions": [item["action"] for item in trace2],
            "in_spec": suggested["recommendation"]["spec"]["in_spec"] if suggested["recommendation"]["spec"] else None,
            "objectives": suggested["recommendation"]["objectives"],
            "confirmation": suggested["recommendation"]["confirmation"],
        },
        "written": str(RESULTS),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())