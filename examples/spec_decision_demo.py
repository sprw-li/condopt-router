"""Decision-layer demo: target windows + constraints + stopping + confirmation.

This is the direction we argued for: a campaign is not "maximize yield"; it is
"hit a set of specifications under constraints, stop when the evidence is good
enough, then confirm the recommendation with replicates".

Run:
    py -3 examples/spec_decision_demo.py

The oracle is synthetic (documented below); the point is the decision layer:
Spec -> scalar objective -> router picks `constrained` -> ask/tell -> stop ->
confirmation batch -> recommendation with spec compliance and replicate error.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from condopt import Goal, SearchSpace, Spec, Study  # noqa: E402
from condopt.render import render_study  # noqa: E402

OUT = Path(__file__).resolve().parent / "runs" / "spec_decision_demo"


def oracle(config: dict, rng: np.random.Generator) -> tuple[dict, dict]:
    """Synthetic thermoset-like process. NOT measurements."""
    temp = float(config["T"]) / 100.0
    time_ = float(config["time"]) / 60.0
    ratio = float(config["ratio"])
    catalyst = 1.0 if config["catalyst"] == "A" else 0.6
    e_mod = 2.0 + 2.5 * ratio + 1.0 * catalyst + 0.4 * temp + rng.normal(0, 0.15)
    uts = 4.0 + 2.0 * ratio + 1.5 * catalyst + 0.5 * time_ + rng.normal(0, 0.25)
    elongation = 120.0 - 40.0 * ratio + 10.0 * time_ + rng.normal(0, 4.0)
    cure_time = 4.0 + 3.0 * ratio + 1.0 * time_ + rng.normal(0, 0.25)
    delta_t = 20.0 + 25.0 * ratio + 5.0 * temp + rng.normal(0, 1.5)
    objectives = {"E_MPa": e_mod, "UTS_MPa": uts, "elongation_pct": elongation}
    constraints = {"cure_time_s": max(cure_time, 0.1), "delta_T_C": max(delta_t, 0.1)}
    return objectives, constraints


def main() -> int:
    space = SearchSpace.from_dict(
        [
            {"name": "T", "kind": "continuous", "low": 40.0, "high": 120.0, "unit": "C"},
            {"name": "time", "kind": "continuous", "low": 5.0, "high": 120.0, "unit": "min"},
            {"name": "ratio", "kind": "continuous", "low": 0.1, "high": 0.8},
            {"name": "catalyst", "kind": "categorical", "choices": ["A", "B"]},
        ]
    )
    spec = Spec(
        goals=(
            Goal("E_MPa", mode="target", target=4.5, tolerance=0.4, scale=2.0),
            Goal("UTS_MPa", mode="target", target=6.8, tolerance=0.6, scale=2.5),
            Goal("elongation_pct", mode="target", target=95.0, tolerance=7.0, scale=25.0),
        ),
        constraints={
            "cure_time_s": {"op": "<=", "value": 10.0},
            "delta_T_C": {"op": "<=", "value": 70.0},
        },
    )
    if OUT.exists():
        import shutil

        shutil.rmtree(OUT)
    study = Study.create(
        OUT,
        space,
        spec=spec,
        budget=32,
        batch_size=4,
        seed=0,
        meta={"target_score": -0.05, "max_rounds": 6},
    )
    rng = np.random.default_rng(7)
    stop_reason = ""
    for _ in range(6):
        pending = study.ask(4)
        results = []
        for item in pending:
            objectives, constraints = oracle(item["x"], rng)
            results.append({"id": item["id"], "objectives": objectives, "constraints": constraints})
        study.tell(results)
        stop, stop_reason = study.should_stop()
        if stop:
            break
    # Confirmation replicates of the current best condition.
    confirmation = study.confirm(3)
    confirm_results = []
    for item in confirmation:
        objectives, constraints = oracle(item["x"], rng)
        confirm_results.append({"id": item["id"], "objectives": objectives,
                                "constraints": constraints, "source": "replicate"})
    study.tell(confirm_results)
    recommendation = study.recommend()
    render_study(OUT)
    payload = {
        "algorithm": study.algorithm,
        "spec": spec.to_dict(),
        "summary_spec": study.summary()["spec"],
        "caller_stop_reason": stop_reason,
        "recommendation": recommendation,
        "run_dir": str(OUT),
    }
    output = Path(__file__).resolve().parent / "spec_decision_demo_results.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "algorithm": study.algorithm,
        "score": recommendation["score"],
        "in_spec": recommendation["spec"]["in_spec"] if recommendation["spec"] else None,
        "condition": recommendation["condition"],
        "confirmation": recommendation["confirmation"],
        "should_stop": recommendation["should_stop"],
        "stop_reason": recommendation["stop_reason"],
        "written": str(output),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
