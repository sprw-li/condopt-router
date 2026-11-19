"""Tests for multi-objective support: ParEGO, router and Study reporting."""

import numpy as np
import pytest

from condopt import Objective, SearchSpace, Study
from condopt.benchmarks import builtin_problems, moo_reference_front
from condopt.pareto import hypervolume_2d, non_dominated_mask
from condopt.registry import build
from condopt.router import OptimizationContext, route


def test_router_picks_parego_for_multi_objective():
    ctx = OptimizationContext(n_dims=2, n_continuous=2, budget_total=32, n_objectives=2)
    recommendation = route(ctx)
    assert recommendation.algorithm == "parego"
    assert recommendation.params["n_initial"] >= 3
    assert recommendation.design["main"] == "parego"
    recommendation.validate()


def test_parego_beats_random_front_on_moo_benchmark():
    problem = builtin_problems()["moo_yield_cost"]
    reference = moo_reference_front()
    reference_point = (60.0, -30.0)
    reference_hv = hypervolume_2d(reference, reference_point)
    assert reference_hv > 0

    optimizer = build("parego", problem.space, problem.objective, params={"n_initial": 6}, seed=0)
    observations = []
    for _ in range(6):
        batch = [problem.make_observation(config) for config in optimizer.ask(4)]
        optimizer.tell(batch)
        observations.extend(batch)
    scores = np.vstack([[obs.objectives["yield"], -obs.objectives["cost"]] for obs in observations])
    front = scores[non_dominated_mask(scores)]
    assert len(front) >= 3
    hv_ratio = hypervolume_2d(front, reference_point) / reference_hv
    assert hv_ratio > 0.7, f"ParEGO HV ratio too low: {hv_ratio}"


def test_parego_handles_single_objective_contract():
    space = SearchSpace.from_dict(
        [{"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0}]
    )
    objective = Objective(name="score", direction="max")
    optimizer = build("parego", space, objective, seed=0)
    from condopt import Observation

    for _ in range(8):
        config = optimizer.ask(1)[0]
        optimizer.tell([Observation(x=config, y=float(1.0 - (config["x"] - 0.4) ** 2))])


def test_parego_freezes_normalization_window():
    problem = builtin_problems()["moo_yield_cost"]
    optimizer = build("parego", problem.space, problem.objective, params={"n_initial": 4}, seed=0)
    base = [problem.make_observation({"T": 80.0 + i, "time": 50.0 + i}) for i in range(4)]
    optimizer.tell(base)
    low_before, high_before = optimizer._window()
    # An extreme new point must not move the frozen window.
    optimizer.tell([problem.make_observation({"T": 60.0, "time": 5.0})])
    low_after, high_after = optimizer._window()
    assert np.allclose(low_before, low_after)
    assert np.allclose(high_before, high_after)


def test_parego_skips_incomplete_objectives():
    from condopt import Observation

    problem = builtin_problems()["moo_yield_cost"]
    optimizer = build("parego", problem.space, problem.objective, params={"n_initial": 4}, seed=0)
    complete = problem.make_observation({"T": 80.0, "time": 60.0})
    incomplete = Observation(x={"T": 90.0, "time": 60.0}, objectives={"yield": 85.0})
    optimizer.tell([complete, incomplete])
    assert optimizer._skipped == 1
    assert len(optimizer._Y) == 1


def test_parego_does_not_freeze_window_on_partial_batch():
    from condopt import Objective

    problem = builtin_problems()["moo_yield_cost"]
    # Deliberately no explicit objective_bounds: exercise the data-frozen path.
    objective = Objective(
        name="yield",
        direction="max",
        objectives=("yield", "cost"),
        directions=("max", "min"),
    )
    optimizer = build("parego", problem.space, objective, params={"n_initial": 4}, seed=0)
    optimizer.tell([problem.make_observation({"T": 80.0, "time": 60.0})])
    low_one, high_one = optimizer._window()
    # A partial batch must not freeze the window; more data must widen it.
    optimizer.tell(
        [
            problem.make_observation({"T": 60.0, "time": 5.0}),
            problem.make_observation({"T": 115.0, "time": 120.0}),
            problem.make_observation({"T": 90.0, "time": 80.0}),
            problem.make_observation({"T": 70.0, "time": 40.0}),
        ]
    )
    low_many, high_many = optimizer._window()
    assert np.any(low_many < low_one)
    assert np.any(high_many > high_one)


def test_study_multi_objective_report(tmp_path):
    problem = builtin_problems()["moo_yield_cost"]
    study = Study.create(
        tmp_path / "moo",
        problem.space,
        problem.objective,
        budget=6,
        batch_size=2,
        seed=0,
    )
    assert study.algorithm == "parego"
    pending = study.ask(2)
    study.tell(
        [
            {"id": item["id"], "objectives": {"yield": 80.0 + index, "cost": 10.0 + index}}
            for index, item in enumerate(pending)
        ]
    )
    summary = study.summary()
    assert summary["pareto"]["n_front"] >= 1
    assert summary["pareto"]["hypervolume"] is not None
    assert summary["pareto"]["hypervolume"] > 0
    text = study.report()
    assert "Pareto" in text

def test_multiobjective_recommendation_uses_pareto_knee(tmp_path):
    from condopt import Study

    problem = builtin_problems()["moo_yield_cost"]
    study = Study.create(
        tmp_path / "moo_recommend",
        problem.space,
        problem.objective,
        algorithm="parego",
        budget=8,
        batch_size=2,
        seed=0,
    )
    seeds = [
        {"T": 80.0, "time": 60.0, "yield": 90.0, "cost": 20.0},
        {"T": 90.0, "time": 50.0, "yield": 88.0, "cost": 12.0},
        {"T": 70.0, "time": 80.0, "yield": 85.0, "cost": 8.0},
        {"T": 100.0, "time": 30.0, "yield": 80.0, "cost": 6.0},
    ]
    study.import_observations([
        {"x": {name: row[name] for name in study.space.names},
         "objectives": {"yield": row["yield"], "cost": row["cost"]}}
        for row in seeds
    ])
    recommendation = study.recommend()
    assert recommendation["selection_basis"] == "pareto_knee"
    assert recommendation["pareto"]["n_front"] >= 2
    assert recommendation["objectives"] in [
        {"yield": row["yield"], "cost": row["cost"]} for row in seeds
    ]
    confirm = study.confirm(2)
    assert all(item["x"] == recommendation["condition"] for item in confirm)

def test_multiobjective_recommendation_corrects_plate_effect(tmp_path):
    from condopt import Study

    problem = builtin_problems()["moo_yield_cost"]
    study = Study.create(
        tmp_path / "moo_plate",
        problem.space,
        problem.objective,
        algorithm="parego",
        budget=10,
        batch_size=2,
        seed=0,
    )
    rows = [
        {"T": 80.0, "time": 60.0, "yield": 95.0, "cost": 20.0, "plate": "P1"},
        {"T": 90.0, "time": 50.0, "yield": 90.0, "cost": 10.0, "plate": "P1"},
        {"T": 70.0, "time": 80.0, "yield": 70.0, "cost": 5.0, "plate": "P2"},
        {"T": 100.0, "time": 30.0, "yield": 65.0, "cost": 15.0, "plate": "P2"},
    ]
    study.import_observations([
        {
            "x": {name: row[name] for name in study.space.names},
            "objectives": {"yield": row["yield"], "cost": row["cost"]},
            "meta": {"plate": row["plate"]},
        }
        for row in rows
    ])
    assert study.plate_effects("yield")["detected"] is True
    recommendation = study.recommend()
    assert recommendation["selection_basis"] == "pareto_knee_plate_adjusted"
    assert recommendation["plate_effects"]["detected"] is True
    assert recommendation["objectives_adjusted"] is not None
    confirm = study.confirm(2)
    assert all(item["x"] == recommendation["condition"] for item in confirm)

def test_multiobjective_recommendation_prefers_in_spec_front_member(tmp_path):
    from condopt.spec import Goal, Spec

    problem = builtin_problems()["moo_yield_cost"]
    spec = Spec(goals=(Goal("yield", mode="target", target=90.0, tolerance=1.0, scale=5.0),))
    study = Study.create(
        tmp_path / "moo_spec",
        problem.space,
        problem.objective,
        spec=spec,
        algorithm="parego",
        budget=10,
        batch_size=2,
        seed=0,
    )
    rows = [
        {"T": 80.0, "time": 60.0, "yield": 95.0, "cost": 20.0},
        {"T": 90.0, "time": 50.0, "yield": 90.0, "cost": 15.0},
        {"T": 70.0, "time": 80.0, "yield": 80.0, "cost": 10.0},
        {"T": 100.0, "time": 30.0, "yield": 70.0, "cost": 5.0},
    ]
    study.import_observations([
        {"x": {name: row[name] for name in study.space.names},
         "objectives": {"yield": row["yield"], "cost": row["cost"]}}
        for row in rows
    ])
    recommendation = study.recommend()
    assert recommendation["selection_basis"] == "pareto_spec"
    assert recommendation["objectives"]["yield"] == 90.0
    assert recommendation["spec"]["in_spec"] is True
    confirm = study.confirm(2)
    assert all(item["x"] == recommendation["condition"] for item in confirm)

def test_knee_weights_change_multiobjective_recommendation(tmp_path):
    from condopt.cli import main

    problem = builtin_problems()["moo_yield_cost"]
    study = Study.create(
        tmp_path / "knee_weights",
        problem.space,
        problem.objective,
        algorithm="parego",
        budget=10,
        batch_size=2,
        seed=0,
    )
    rows = [
        {"T": 80.0, "time": 60.0, "yield": 95.0, "cost": 20.0},
        {"T": 90.0, "time": 50.0, "yield": 90.0, "cost": 15.0},
        {"T": 70.0, "time": 80.0, "yield": 80.0, "cost": 10.0},
        {"T": 100.0, "time": 30.0, "yield": 70.0, "cost": 5.0},
    ]
    study.import_observations([
        {"x": {name: row[name] for name in study.space.names},
         "objectives": {"yield": row["yield"], "cost": row["cost"]}}
        for row in rows
    ])
    default = study.recommend()
    assert default["objectives"]["yield"] == 80.0
    assert main([
        "knee-weights", "--run", str(study.dir), "--weights", "yield=100,cost=1", "--json",
    ]) == 0
    study = Study.load(study.dir)
    weighted = study.recommend()
    assert weighted["objectives"]["yield"] == 95.0
    assert weighted["selection_basis"] == "pareto_knee"
    action = study.next_action()
    assert action["diagnostics"]["knee_weights"] == {"yield": 100.0, "cost": 1.0}
    with pytest.raises(ValueError, match="unknown objective"):
        study.set_knee_weights({"nope": 1.0})

def test_report_shows_knee_weights(tmp_path):
    problem = builtin_problems()["moo_yield_cost"]
    study = Study.create(
        tmp_path / "report_knee_weights",
        problem.space,
        problem.objective,
        algorithm="parego",
        budget=6,
        batch_size=2,
        seed=0,
    )
    study.import_observations([
        {"x": {"T": 80.0, "time": 60.0}, "objectives": {"yield": 90.0, "cost": 20.0}},
        {"x": {"T": 90.0, "time": 50.0}, "objectives": {"yield": 80.0, "cost": 10.0}},
    ])
    study.set_knee_weights({"yield": 3.0, "cost": 1.0})
    assert "knee 权重" in study.report()

def test_combined_moo_constraints_plate_and_spec(tmp_path):
    from condopt.spec import Goal, Spec

    problem = builtin_problems()["moo_constrained"]
    spec = Spec(goals=(Goal("yield", mode="target", target=88.0, tolerance=1.5, scale=5.0),))
    study = Study.create(
        tmp_path / "combined",
        problem.space,
        problem.objective,
        spec=spec,
        algorithm="cqehvi",
        budget=10,
        batch_size=2,
        seed=0,
        params={"n_initial": 4, "mc_draws": 32},
    )
    rows = [
        {"T": 70.0, "time": 30.0, "yield": 90.0, "cost": 20.0, "T_max": 70.0, "plate": "P1"},
        {"T": 72.0, "time": 35.0, "yield": 88.0, "cost": 15.0, "T_max": 72.0, "plate": "P1"},
        {"T": 65.0, "time": 60.0, "yield": 80.0, "cost": 10.0, "T_max": 60.0, "plate": "P2"},
        {"T": 68.0, "time": 50.0, "yield": 78.0, "cost": 8.0, "T_max": 65.0, "plate": "P2"},
    ]
    study.import_observations([
        {
            "x": {name: row[name] for name in study.space.names},
            "objectives": {"yield": row["yield"], "cost": row["cost"]},
            "constraints": {"T_max": row["T_max"]},
            "meta": {"plate": row["plate"]},
        }
        for row in rows
    ])
    recommendation = study.recommend()
    assert recommendation["selection_basis"] == "pareto_spec_plate_adjusted"
    assert recommendation["plate_effects"]["detected"] is True
    assert recommendation["spec"] is not None
    assert recommendation["spec_raw"] is not None
    assert recommendation["pareto"]["n_front"] >= 1
    action = study.next_action()
    assert action["diagnostics"]["plate_effect_detected"] is True
    assert "use_plate_adjusted_recommendation" in action["suggested_actions"]
    confirm = study.confirm(2)
    assert all(item["x"] == recommendation["condition"] for item in confirm)

def test_summary_exposes_decision_for_multiobjective(tmp_path):
    problem = builtin_problems()["moo_yield_cost"]
    study = Study.create(
        tmp_path / "summary_decision",
        problem.space,
        problem.objective,
        algorithm="parego",
        budget=6,
        batch_size=2,
        seed=0,
    )
    study.import_observations([
        {"x": {"T": 80.0, "time": 60.0}, "objectives": {"yield": 90.0, "cost": 20.0}},
        {"x": {"T": 90.0, "time": 50.0}, "objectives": {"yield": 80.0, "cost": 10.0}},
    ])
    payload = study.summary()
    assert payload["decision"]["selection_basis"] == "pareto_knee"
    assert payload["spec"] is None

def test_decision_cli_outputs_json(tmp_path, capsys):
    import json

    from condopt.cli import main

    problem = builtin_problems()["moo_yield_cost"]
    study = Study.create(
        tmp_path / "decision_cli",
        problem.space,
        problem.objective,
        algorithm="parego",
        budget=6,
        batch_size=2,
        seed=0,
    )
    study.import_observations([
        {"x": {"T": 80.0, "time": 60.0}, "objectives": {"yield": 90.0, "cost": 20.0}},
        {"x": {"T": 90.0, "time": 50.0}, "objectives": {"yield": 80.0, "cost": 10.0}},
    ])
    assert main(["decision", "--run", str(study.dir), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["decision"]["selection_basis"] == "pareto_knee"

def test_decision_explain_mentions_reasons(tmp_path):
    problem = builtin_problems()["moo_yield_cost"]
    study = Study.create(
        tmp_path / "decision_explain",
        problem.space,
        problem.objective,
        algorithm="parego",
        budget=6,
        batch_size=2,
        seed=0,
    )
    study.import_observations([
        {"x": {"T": 80.0, "time": 60.0}, "objectives": {"yield": 90.0, "cost": 20.0}},
        {"x": {"T": 90.0, "time": 50.0}, "objectives": {"yield": 80.0, "cost": 10.0}},
    ])
    decision = study.decision(explain=True)
    assert decision["explain"]
    assert any("推荐依据" in line for line in decision["explain"])

def test_multiobjective_spec_review_includes_suggested_spec(tmp_path):
    from condopt.spec import Goal, Spec

    problem = builtin_problems()["moo_yield_cost"]
    spec = Spec(goals=(Goal("yield", mode="target", target=200.0, tolerance=2.0, scale=10.0),))
    study = Study.create(
        tmp_path / "moo_spec_review",
        problem.space,
        problem.objective,
        spec=spec,
        algorithm="parego",
        budget=10,
        batch_size=2,
        seed=0,
    )
    study.import_observations([
        {"x": {"T": 80.0 + index, "time": 60.0 - index},
         "objectives": {"yield": 50.0 + index, "cost": 20.0 - index}}
        for index in range(5)
    ])
    action = study.next_action()
    assert action["diagnostics"]["spec_in_spec_count"] == 0
    assert "review_spec" in action["suggested_actions"]
    review = action["diagnostics"]["spec_review"]
    assert review["metrics"]["yield"]["target_within_observed"] is False
    assert review["suggested_spec"]["goals"][0]["target"] == 54.0

def test_parego_supports_constraints():
    import numpy as np

    from condopt.registry import build

    problem = builtin_problems()["moo_constrained"]
    optimizer = build(
        "parego",
        problem.space,
        problem.objective,
        params={"n_initial": 2, "n_candidates": 64},
        seed=0,
    )
    observations = [
        problem.make_observation({"T": 70.0, "time": 30.0}),
        problem.make_observation({"T": 110.0, "time": 100.0}),
    ]
    optimizer.tell(observations)
    features = [{"T": 70.0, "time": 30.0}, {"T": 110.0, "time": 100.0}]
    probability = optimizer._feasibility_probability(features)
    assert probability[0] > probability[1]
    config = optimizer.ask(1)[0]
    assert set(config) == {"T", "time"}

def test_three_objective_constrained_problem_routes_to_parego_and_runs():
    from condopt.benchmarks import run_algorithm

    problem = builtin_problems()["moo3_constrained"]
    ctx = OptimizationContext(
        n_dims=2, n_continuous=2, budget_total=20,
        n_objectives=3, n_constraints=1,
    )
    recommendation = route(ctx)
    assert recommendation.algorithm == "parego"

    result = run_algorithm(problem, "parego", budget=8, batch_size=2, seed=0)
    assert result.budget == 8
    assert len(result.objective_vectors) == 8
    assert len(result.objective_vectors[0]) == 3
    assert len(result.constraint_vectors) == 8

def test_cli_pareto_exports_front_csv(tmp_path):
    import json

    from condopt.cli import main

    problem = builtin_problems()["moo3_constrained"]
    study = Study.create(
        tmp_path / "pareto_export",
        problem.space,
        problem.objective,
        algorithm="parego",
        budget=8,
        batch_size=2,
        seed=0,
    )
    study.import_observations([
        {"x": {"T": 70.0, "time": 30.0},
         "objectives": {"yield": 90.0, "cost": 10.0, "purity": 97.0},
         "constraints": {"T_max": 70.0}},
        {"x": {"T": 65.0, "time": 60.0},
         "objectives": {"yield": 80.0, "cost": 8.0, "purity": 95.0},
         "constraints": {"T_max": 65.0}},
        {"x": {"T": 75.0, "time": 45.0},
         "objectives": {"yield": 85.0, "cost": 12.0, "purity": 96.0},
         "constraints": {"T_max": 75.0}},
    ])
    assert main(["pareto", "--run", str(study.dir), "--json"]) == 0
    export = study.dir / "pareto_front.csv"
    assert export.exists()
    lines = export.read_text(encoding="utf-8-sig").splitlines()
    assert "obj.yield" in lines[0] and "obj.purity" in lines[0]
    assert len(lines) - 1 == study.pareto_summary()["n_front"]

def test_benchmark_noise_script_runs():
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_noise.py"
    spec = importlib.util.spec_from_file_location("benchmark_noise", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    row = module.run(True, sigma=3.0, budget=6, batch_size=2, seed=0)
    assert row["budget"] == 6
    assert row["hv_true"] >= 0.0

def test_hypervolume_nd_matches_2d_and_analytic_3d():
    import numpy as np

    from condopt.pareto import hypervolume_2d, hypervolume_nd

    rng = np.random.default_rng(0)
    reference = (0.0, 0.0)
    for _ in range(10):
        points = rng.random((6, 2)) * 3.0
        assert hypervolume_nd(points, reference) == pytest.approx(
            hypervolume_2d(points, reference), rel=1e-9, abs=1e-12
        )
    # single point in 3D: box volume
    assert hypervolume_nd([[2.0, 3.0, 4.0]], [0.0, 0.0, 0.0]) == pytest.approx(24.0)
    # two non-dominated points: union = box(p) + box(q) - overlap
    points = np.array([[2.0, 1.0], [1.0, 2.0]])
    hv = hypervolume_nd(points, [0.0, 0.0])
    assert hv == pytest.approx(3.0)
    # adding a point never decreases HV
    base = np.array([[2.0, 2.0, 2.0], [1.0, 3.0, 2.0]])
    extra = np.vstack([base, [[1.5, 1.5, 2.5]]])
    assert hypervolume_nd(extra, [0, 0, 0]) >= hypervolume_nd(base, [0, 0, 0])

def test_three_objective_study_reports_hypervolume(tmp_path):
    import numpy as np

    from condopt.pareto import hypervolume_nd

    problem = builtin_problems()["moo3_constrained"]
    study = Study.create(
        tmp_path / "moo3_hv",
        problem.space,
        problem.objective,
        algorithm="parego",
        budget=8,
        batch_size=2,
        seed=0,
    )
    study.import_observations([
        {"x": {"T": 70.0, "time": 30.0},
         "objectives": {"yield": 90.0, "cost": 10.0, "purity": 97.0},
         "constraints": {"T_max": 70.0}},
        {"x": {"T": 65.0, "time": 60.0},
         "objectives": {"yield": 80.0, "cost": 8.0, "purity": 95.0},
         "constraints": {"T_max": 65.0}},
    ])
    summary = study.pareto_summary()
    assert summary["hypervolume"] is not None
    assert summary["reference_point"] is not None
    signs = np.asarray([1.0, -1.0, 1.0])
    front = np.asarray(
        [[90.0, 10.0, 97.0], [80.0, 8.0, 95.0]], dtype=float
    ) * signs
    expected = hypervolume_nd(front, np.asarray(summary["reference_point"]))
    assert summary["hypervolume"] == pytest.approx(expected, rel=1e-9)
    report = study.report()
    assert "HV" in report or "hypervolume" in report.lower()

def test_hypervolume_contribution_matches_full_difference():
    import numpy as np

    from condopt.pareto import hypervolume_contribution, hypervolume_nd

    rng = np.random.default_rng(3)
    reference = np.zeros(3)
    for _ in range(20):
        front = rng.random((8, 3)) * 2.0 + 0.1
        point = rng.random(3) * 2.0 + 0.1
        base = hypervolume_nd(front, reference)
        full = hypervolume_nd(np.vstack([front, point]), reference) - base
        contribution = hypervolume_contribution(point, front, reference)
        assert contribution == pytest.approx(max(0.0, full), rel=1e-9, abs=1e-9)
    # dominated point contributes nothing
    front = np.array([[1.0, 1.0, 1.0]])
    assert hypervolume_contribution([0.5, 0.5, 0.5], front, reference) == 0.0

def test_cqnehvi_runs_on_noisy_constrained_problem():
    from condopt.benchmarks import run_algorithm

    problem = builtin_problems()["moo_constrained"]
    result = run_algorithm(
        problem, "cqnehvi", budget=8, batch_size=2, seed=0, noise_std=2.0
    )
    assert result.budget == 8
    assert len(result.objective_vectors) == 8
    assert len(result.constraint_vectors) == 8


def test_router_keeps_cqehvi_and_lists_cqnehvi_as_optional():
    from condopt.router import OptimizationContext, route

    noisy = OptimizationContext(
        n_dims=3, n_continuous=3, budget_total=48, batch_size=4,
        n_objectives=2, n_constraints=1, noise="high", smoothness="smooth",
    )
    recommendation = route(noisy)
    # benchmark-validated default: cqehvi even under noise; cqnehvi is opt-in
    assert recommendation.algorithm == "cqehvi"
    assert "cqnehvi" in recommendation.candidate_scores

    quiet = OptimizationContext(
        n_dims=3, n_continuous=3, budget_total=48, batch_size=4,
        n_objectives=2, n_constraints=1, noise="low", smoothness="smooth",
    )
    assert route(quiet).algorithm == "cqehvi"

def test_feasible_pareto_summary_ignores_biased_low_fidelity(tmp_path):
    """Mixed fidelities must not inflate the deliverable hypervolume."""
    from condopt.base import Objective
    from condopt.space import SearchSpace
    from condopt.study import Study

    space = SearchSpace.from_dict([
        {"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0}
    ])
    objective = Objective(
        name="yield", direction="max",
        objectives=("yield", "cost"), directions=("max", "min"),
        constraints=("c",), constraint_limits={"c": {"op": "<=", "value": 0.5}},
    )
    study = Study.create(
        tmp_path / "mixed_fidelity", space, objective, algorithm="mfparego",
        budget=8, batch_size=4, seed=0,
    )
    study.import_observations([
        # full fidelity: modest, feasible values
        {"x": {"x": 0.1}, "y": 10.0, "objectives": {"cost": 10.0},
         "constraints": {"c": 0.1}, "fidelity": 1.0},
        {"x": {"x": 0.2}, "y": 11.0, "objectives": {"cost": 9.0},
         "constraints": {"c": 0.2}, "fidelity": 1.0},
        # low fidelity: wildly inflated (as a biased surrogate would be)
        {"x": {"x": 0.3}, "y": 900.0, "objectives": {"cost": 1.0},
         "constraints": {"c": 0.3}, "fidelity": 0.25},
        {"x": {"x": 0.4}, "y": 950.0, "objectives": {"cost": 0.5},
         "constraints": {"c": 0.4}, "fidelity": 0.25},
    ])
    raw = study.pareto_summary()
    feasible = study.feasible_pareto_summary()
    assert raw["fidelity_filter"] == "full"
    assert feasible["fidelity_filter"] == "full"  # same policy as the raw front
    assert feasible["n_feasible"] == 2
    assert feasible["hypervolume"] is not None
    assert feasible["hypervolume"] <= raw["hypervolume"] + 1e-9
    assert feasible["hypervolume"] < 100.0  # the inflated points must not enter

