"""End-to-end decision-layer campaign: spec + plates + next_action loop + report.

This is the integration test the individual features were missing:

1. a specification with target windows and hard constraints;
2. a synthetic-but-noisy oracle with *plate-dependent bias* (batch effect);
3. the agent-driven loop driven by ``Study.next_action()``:
   tell -> ask / confirm / stop;
4. confirmation replicates of the recommended (plate-adjusted) condition;
5. final report + recommendation + spec calibration suggestion.

The oracle is synthetic (documented in code); this file validates the decision
chain, not chemistry.

Run:
    py -3 examples/full_campaign_demo.py
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

OUT = Path(__file__).resolve().parent / "runs" / "full_campaign_demo"
RESULTS = Path(__file__).resolve().parent / "full_campaign_demo_results.json"


def oracle(config: dict, plate: str | None, rng: np.random.Generator) -> tuple[dict, dict]:
    """Synthetic formulation with systematic plate bias. NOT measurements."""
    temp = float(config["T"]) / 100.0
    time_ = float(config["time"]) / 60.0
    ratio = float(config["ratio"])
    catalyst = 1.0 if config["catalyst"] == "A" else 0.6
    plate_shift = {
        "P1": (0.9, -0.7, 7.0),
        "P2": (-0.8, 0.7, -6.0),
        "P3": (0.6, -0.5, 5.0),
        "P4": (-0.6, 0.5, -5.0),
    }.get(str(plate), (0.0, 0.0, 0.0))
    e_mod = 2.0 + 2.5 * ratio + 1.0 * catalyst + 0.4 * temp + plate_shift[0] + rng.normal(0, 0.12)
    uts = 4.0 + 2.0 * ratio + 1.5 * catalyst + 0.5 * time_ + plate_shift[1] + rng.normal(0, 0.18)
    elongation = 120.0 - 40.0 * ratio + 10.0 * time_ + plate_shift[2] + rng.normal(0, 3.0)
    cure_time = 4.0 + 3.0 * ratio + 1.0 * time_ + rng.normal(0, 0.20)
    delta_t = 20.0 + 25.0 * ratio + 5.0 * temp + rng.normal(0, 1.2)
    objectives = {
        "E_MPa": float(e_mod),
        "UTS_MPa": float(uts),
        "elongation_pct": float(elongation),
    }
    constraints = {
        "cure_time_s": float(max(cure_time, 0.1)),
        "delta_T_C": float(max(delta_t, 0.1)),
    }
    return objectives, constraints


def build_spec() -> Spec:
    return Spec(
        goals=(
            Goal("E_MPa", mode="target", target=4.5, tolerance=0.5, scale=2.0),
            Goal("UTS_MPa", mode="target", target=6.8, tolerance=0.6, scale=2.5),
            Goal("elongation_pct", mode="target", target=95.0, tolerance=7.0, scale=25.0),
        ),
        constraints={
            "cure_time_s": {"op": "<=", "value": 10.0},
            "delta_T_C": {"op": "<=", "value": 70.0},
        },
    )


def run_campaign(
    run_dir: Path,
    budget: int = 32,
    batch_size: int = 4,
    seed: int = 0,
    render: bool = True,
) -> dict:
    space = SearchSpace.from_dict(
        [
            {"name": "T", "kind": "continuous", "low": 40.0, "high": 120.0},
            {"name": "time", "kind": "continuous", "low": 5.0, "high": 120.0},
            {"name": "ratio", "kind": "continuous", "low": 0.1, "high": 0.8},
            {"name": "catalyst", "kind": "categorical", "choices": ["A", "B"]},
        ]
    )
    spec = build_spec()
    if run_dir.exists():
        import shutil

        shutil.rmtree(run_dir)
    study = Study.create(
        run_dir,
        space,
        spec=spec,
        budget=budget,
        batch_size=batch_size,
        seed=seed,
        meta={"target_score": -0.05, "auto_confirm": True, "confirm_n": 3},
    )
    rng = np.random.default_rng(seed + 11)
    action_trace = []
    for _ in range(40):
        action = study.next_action()
        action_trace.append({
            "action": action["action"],
            "remaining": action["remaining"],
            "suggested": action.get("suggested_actions", []),
        })
        if action["action"] == "stop":
            break
        if action["action"] == "tell":
            pending = study.pending
            results = []
            for item in pending:
                objectives, constraints = oracle(item["x"], item.get("plate"), rng)
                results.append({
                    "id": item["id"],
                    "objectives": objectives,
                    "constraints": constraints,
                    "source": "synthetic",
                })
            study.tell(results)
        elif action["action"] == "ask":
            study.ask(int(action.get("ask_n") or batch_size))
        elif action["action"] == "confirm":
            study.confirm(int(action.get("confirm_n") or 3))
    recommendation = study.recommend()
    calibrated, calibration_report = spec.calibrate_observations(study.dataset.observations)
    if render:
        render_study(run_dir)
    return {
        "algorithm": study.algorithm,
        "actions": action_trace,
        "summary_spec": study.summary()["spec"],
        "plate_summary": study.plate_summary(),
        "recommendation": recommendation,
        "spec_calibration": calibration_report,
        "calibrated_spec": calibrated.to_dict(),
        "report_excerpt": "\n".join(study.report().splitlines()[:22]),
        "run_dir": str(run_dir),
    }


def main() -> int:
    payload = run_campaign(OUT)
    RESULTS.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    recommendation = payload["recommendation"]
    print(json.dumps({
        "algorithm": payload["algorithm"],
        "n_actions": len(payload["actions"]),
        "actions": [item["action"] for item in payload["actions"]],
        "selection_basis": recommendation["selection_basis"],
        "score": recommendation["score"],
        "in_spec": recommendation["spec"]["in_spec"] if recommendation["spec"] else None,
        "confirmation": recommendation["confirmation"],
        "plate_effect": recommendation["plate_effects"]["detected"],
        "should_stop": recommendation["should_stop"],
        "written": str(RESULTS),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
