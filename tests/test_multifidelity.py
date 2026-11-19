"""Tests for the multi-fidelity interface (mfbo + Study fidelity plumbing)."""

import math

import pytest

from condopt import Objective, Observation, SearchSpace, Study
from condopt.registry import build
from condopt.router import OptimizationContext, route


def simple_space() -> SearchSpace:
    return SearchSpace.from_dict(
        [{"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0}]
    )


def test_router_picks_mfbo_with_fidelity_channel():
    ctx = OptimizationContext(
        n_dims=2, n_continuous=2, budget_total=30, multi_fidelity=True
    )
    recommendation = route(ctx)
    assert recommendation.algorithm == "mfbo"
    assert "1.0" in recommendation.params["fidelity_levels"]
    recommendation.validate()


def test_fidelity_cost_table_interpolates_and_rejects_missing_levels():
    space = simple_space()
    optimizer = build(
        "mfbo",
        space,
        Objective(name="y", direction="max"),
        params={"fidelity_costs": "0.25:0.5,1.0:2.0"},
        seed=0,
    )
    # 0.5 is interpolated between 0.25 (0.5) and 1.0 (2.0) -> 1.0
    assert optimizer._cost(0.5) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="no cost for level"):
        optimizer._cost(2.0)


def test_mfbo_budget_uses_fidelity_cost_table():
    space = simple_space()
    optimizer = build(
        "mfbo",
        space,
        Objective(name="y", direction="max"),
        params={"fidelity_costs": "0.25:0.5,0.5:0.9,1.0:2.0"},
        seed=0,
    )
    pairs = optimizer.ask_with_fidelity(2, cost_remaining=1.0)
    spent = sum(optimizer._cost(fidelity) for _, fidelity in pairs)
    assert pairs
    assert spent <= 1.0 + 1e-9
    # 0.25 costs 0.5, so two cheapest points (1.0) fit; 0.5 costs 0.9.
    assert all(optimizer._cost(fidelity) <= 0.9 + 1e-9 for _, fidelity in pairs)


def test_min_full_fraction_is_soft_under_tight_budget():
    space = simple_space()
    optimizer = build(
        "mfbo",
        space,
        Objective(name="y", direction="max"),
        params={"n_initial": 1, "min_full_fraction": 1.0},
        seed=0,
    )
    # min_full_fraction=1.0 is a soft preference: with only 0.3 budget the
    # cheapest full-fidelity run is unaffordable, so it must degrade, not crash
    # or overspend.
    pairs = optimizer.ask_with_fidelity(2, cost_remaining=0.3)
    assert pairs
    assert all(fidelity < 1.0 for _, fidelity in pairs)
    assert sum(fidelity for _, fidelity in pairs) <= 0.3 + 1e-9
    assert optimizer.last_cost_trimmed >= 0
    assert optimizer.last_full_requested >= 1
    assert optimizer.last_full_returned == 0


def test_mfbo_call_level_cost_override():
    space = simple_space()
    optimizer = build("mfbo", space, Objective(name="y", direction="max"), seed=0)
    pairs = optimizer.ask_with_fidelity(3, cost_remaining=0.3)
    assert sum(optimizer._cost(fidelity) for _, fidelity in pairs) <= 0.3 + 1e-9
    assert optimizer.cost_remaining == pytest.approx(0.3)


def test_mfbo_budget_uses_cost_exponent(tmp_path):
    space = simple_space()
    context = OptimizationContext.from_space(
        space, budget_total=8, batch_size=4, multi_fidelity=True
    )
    study = Study.create(
        tmp_path / "mf_exp",
        space,
        Objective(name="y", direction="max"),
        context=context,
        params={"cost_exponent": 2.0},
        budget=8,
        batch_size=4,
        cost_budget=0.3,
        seed=0,
    )
    pending = study.ask(4)
    spent = sum(item["fidelity"] ** 2.0 for item in pending)
    assert spent <= 0.3 + 1e-9
    assert all(item["fidelity"] <= 0.5 + 1e-9 for item in pending)


def test_mfbo_raises_when_budget_below_cheapest_level(tmp_path):
    space = simple_space()
    context = OptimizationContext.from_space(
        space, budget_total=8, batch_size=2, multi_fidelity=True
    )
    study = Study.create(
        tmp_path / "mf_too_poor",
        space,
        Objective(name="y", direction="max"),
        context=context,
        budget=8,
        batch_size=2,
        cost_budget=0.1,
        seed=0,
    )
    # exhausted budget returns no suggestion (graceful stop for agent loops)
    assert study.ask(2) == []


def test_mfbo_respects_remaining_cost_budget(tmp_path):
    space = simple_space()
    context = OptimizationContext.from_space(
        space, budget_total=8, batch_size=4, multi_fidelity=True
    )
    study = Study.create(
        tmp_path / "mf_budget",
        space,
        Objective(name="y", direction="max"),
        context=context,
        budget=8,
        batch_size=4,
        meta={"cost_budget": 0.6},
        seed=0,
    )
    pending = study.ask(4)
    total_cost = sum(item["fidelity"] for item in pending)
    assert 0 < len(pending) <= 4
    assert total_cost <= 0.6 + 1e-9
    assert all(item["fidelity"] <= 0.5 for item in pending)
    if len(pending) < 4:
        trims = [d for d in study.decisions() if d["action"] == "cost_trim"]
        assert trims
        payload = trims[-1]["payload"]
        assert {"requested", "returned", "trimmed", "full_requested", "full_returned"} <= set(payload)


def test_mfbo_incumbent_uses_full_fidelity_only():
    space = simple_space()
    optimizer = build("mfbo", space, Objective(name="y", direction="max"), seed=0)
    optimizer.tell(
        [
            Observation(x={"x": 0.1}, y=99.0, fidelity=0.25),
            Observation(x={"x": 0.2}, y=80.0, fidelity=1.0),
            Observation(x={"x": 0.3}, y=90.0, fidelity=0.5),
            Observation(x={"x": 0.4}, y=70.0, fidelity=1.0),
        ]
    )
    # A biased cheap run (99) must not become the EI incumbent.
    assert optimizer._best_scalar() == pytest.approx(80.0)


def test_mfbo_initial_design_uses_lowest_fidelity():
    space = simple_space()
    optimizer = build("mfbo", space, Objective(name="y", direction="max"), seed=0)
    pairs = optimizer.ask_with_fidelity(4)
    levels = optimizer._levels()
    # n_initial auto for a 1-D space is 4, so all four design points are cheap.
    assert [fidelity for _, fidelity in pairs] == [levels[0]] * 4


def test_mfbo_ask_with_fidelity_contract():
    space = simple_space()
    optimizer = build("mfbo", space, Objective(name="y", direction="max"), seed=0)
    assert optimizer.supports_fidelity is True
    pairs = optimizer.ask_with_fidelity(6)
    assert len(pairs) == 6
    levels = [fidelity for _, fidelity in pairs]
    assert all(0.0 < fidelity <= 1.0 for fidelity in levels)
    assert any(fidelity < 1.0 for fidelity in levels)
    # tell, then the model-based phase must still return valid pairs
    optimizer.tell(
        [
            Observation(x=config, y=float(1.0 - (config["x"] - 0.3) ** 2), fidelity=fidelity)
            for config, fidelity in pairs
        ]
    )
    more = optimizer.ask_with_fidelity(4)
    assert len(more) == 4
    full = [fidelity for _, fidelity in more if fidelity >= 1.0 - 1e-9]
    assert len(full) >= math.ceil(0.25 * 4)


def test_study_multi_fidelity_flow(tmp_path):
    space = simple_space()
    context = OptimizationContext.from_space(
        space, budget_total=10, batch_size=2, multi_fidelity=True
    )
    study = Study.create(
        tmp_path / "mf",
        space,
        Objective(name="y", direction="max"),
        context=context,
        budget=10,
        batch_size=2,
        seed=0,
    )
    assert study.algorithm == "mfbo"
    pending = study.ask(2)
    assert all(0.0 < item["fidelity"] <= 1.0 for item in pending)
    csv_text = (tmp_path / "mf" / "suggestions.csv").read_text(encoding="utf-8")
    assert csv_text.splitlines()[0].startswith("id,batch,algorithm,fidelity")
    study.tell(
        [
            {
                "id": item["id"],
                "y": 0.5,
                "fidelity": item["fidelity"],
            }
            for item in pending
        ]
    )
    observations = study.dataset.observations
    assert [obs.fidelity for obs in observations] == [item["fidelity"] for item in pending]
    # No explicit cost in the results: cheap runs default to cost=fidelity.
    expected_cost = sum(
        item["fidelity"] for item in pending if item["fidelity"] < 1.0 - 1e-9
    )
    assert study.cost_spent() == pytest.approx(expected_cost)

def test_step_reports_cost_budget_and_suggests_review(tmp_path):
    study = Study.create(
        tmp_path / "mf_cost",
        simple_space(),
        Objective(name="y", direction="max"),
        algorithm="mfbo",
        budget=10,
        batch_size=2,
        seed=0,
        cost_budget=1.0,
        params={"fidelity_costs": "0.25:0.5,1.0:2.0"},
    )
    pending = study.ask(2)
    study.tell([{"id": item["id"], "y": float(index)} for index, item in enumerate(pending)])
    action = study.next_action()
    assert action["diagnostics"]["multi_fidelity"] is True
    assert action["diagnostics"]["cost_spent"] is not None
    assert action["diagnostics"]["remaining_cost"] is not None
    if action["diagnostics"]["remaining_cost"] < 1.0:
        assert "review_cost_budget" in action["suggested_actions"]
        assert "review_cost_budget" in action["suggested_action_reasons"]

def test_router_picks_mfbo_when_fidelity_and_constraints_requested():
    from condopt import registry

    ctx = OptimizationContext(
        n_dims=2, n_continuous=2, budget_total=30,
        multi_fidelity=True, n_constraints=2,
    )
    recommendation = route(ctx)
    assert recommendation.algorithm == "mfbo"
    assert "constraints" in registry.spec("mfbo").capabilities

def test_step_warns_when_pinned_algorithm_ignores_constraints(tmp_path):
    objective = Objective(
        name="y",
        direction="max",
        constraints=("c",),
        constraint_limits={"c": {"op": "<=", "value": 1.0}},
    )
    study = Study.create(
        tmp_path / "pinned_mf",
        simple_space(),
        objective,
        algorithm="bayes",
        budget=8,
        batch_size=2,
        seed=0,
    )
    pending = study.ask(2)
    study.tell([{"id": item["id"], "y": 1.0, "constraints": {"c": 0.5}} for item in pending])
    action = study.next_action()
    assert action["diagnostics"]["constraints_supported"] is False
    assert "switch_to_constraint_aware" in action["suggested_actions"]
    assert "override" in action["suggested_action_reasons"]["switch_to_constraint_aware"]

def test_mfbo_constraint_feasibility_probability_prefers_feasible():
    import numpy as np

    objective = Objective(
        name="y",
        direction="max",
        constraints=("c",),
        constraint_limits={"c": {"op": "<=", "value": 1.0}},
    )
    optimizer = build(
        "mfbo", simple_space(), objective, params={"n_initial": 0}, seed=0
    )
    optimizer.tell([
        Observation(x={"x": 0.1}, y=1.0, constraints={"c": 0.2}, fidelity=1.0),
        Observation(x={"x": 0.9}, y=1.0, constraints={"c": 3.0}, fidelity=1.0),
    ])
    features = np.vstack([
        optimizer._features({"x": 0.1}, 1.0),
        optimizer._features({"x": 0.9}, 1.0),
    ])
    probability = optimizer._feasibility_probability(features)
    assert probability[0] > probability[1]
    pairs = optimizer.ask_with_fidelity(1)
    assert len(pairs) == 1

def test_mf_constrained_problem_runs_with_mfbo():
    from condopt.benchmarks import builtin_problems, mf_suzuki

    problem = builtin_problems()["mf_constrained"]
    assert problem.objective.constraints == ("T_max",)
    optimizer = build(
        "mfbo",
        problem.space,
        problem.objective,
        params={"n_initial": 4, "n_candidates": 128},
        seed=0,
    )
    observations = []
    for _ in range(3):
        pairs = optimizer.ask_with_fidelity(2)
        for config, fidelity in pairs:
            observations.append(Observation(
                x=config,
                y=mf_suzuki(config, fidelity),
                constraints={"T_max": float(config["T"])},
                fidelity=fidelity,
            ))
        optimizer.tell(observations[-len(pairs):])
    assert optimizer._C is not None and optimizer._C.shape[1] == 1
    assert len(observations) >= 6

def test_benchmark_mf_constrained_script_runs():
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_mf_constrained.py"
    spec = importlib.util.spec_from_file_location("benchmark_mf_constrained", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    row = module.run("mfbo", budget=6.0, batch_size=2, seed=0)
    assert row["cost_spent"] <= 6.0 + 1e-6
    assert row["n_observations"] > 0
    assert row["algorithm"] == "mfbo"

def test_router_prefers_multi_objective_over_fidelity_when_both_requested():
    ctx = OptimizationContext(
        n_dims=2, n_continuous=2, budget_total=30,
        multi_fidelity=True, n_objectives=2,
    )
    recommendation = route(ctx)
    # four independent settings favour mfparego; momfbo stays as candidate
    assert recommendation.algorithm == "mfparego"
    assert "momfbo" in recommendation.candidate_scores

def test_step_warns_when_context_wants_fidelity_but_algorithm_lacks_it(tmp_path):
    space = simple_space()
    context = OptimizationContext.from_space(
        space, budget_total=8, batch_size=2, multi_fidelity=True
    )
    study = Study.create(
        tmp_path / "pinned_no_mf",
        space,
        Objective(name="y", direction="max"),
        algorithm="parego",
        context=context,
        budget=8,
        batch_size=2,
        seed=0,
    )
    pending = study.ask(2)
    study.tell([{"id": item["id"], "y": 1.0, "objectives": {"y": 1.0}} for item in pending])
    action = study.next_action()
    assert action["diagnostics"]["multi_fidelity_context"] is True
    assert action["diagnostics"]["multi_fidelity_supported"] is False
    assert "switch_to_multi_fidelity" in action["suggested_actions"]

def test_router_picks_mfparego_for_multi_objective_multi_fidelity():
    ctx = OptimizationContext(
        n_dims=2, n_continuous=2, budget_total=30,
        multi_fidelity=True, n_objectives=2,
    )
    recommendation = route(ctx)
    assert recommendation.algorithm == "mfparego"
    rec_ctx = OptimizationContext(
        n_dims=2, n_continuous=2, budget_total=30,
        multi_fidelity=True, n_objectives=2, n_constraints=1,
    )
    assert route(rec_ctx).algorithm == "mfparego"


def test_momfbo_runs_on_multi_objective_problem(tmp_path):
    from condopt.benchmarks import builtin_problems
    from condopt.registry import build

    problem = builtin_problems()["moo_yield_cost"]
    optimizer = build(
        "momfbo",
        problem.space,
        problem.objective,
        params={"n_initial": 2, "n_candidates": 64},
        seed=0,
    )
    observations = [
        problem.make_observation({"T": 80.0, "time": 40.0}),
        problem.make_observation({"T": 90.0, "time": 60.0}),
        problem.make_observation({"T": 70.0, "time": 90.0}),
    ]
    for observation in observations:
        observation.fidelity = 1.0
    optimizer.tell(observations)
    pairs = optimizer.ask_with_fidelity(2)
    assert len(pairs) == 2
    assert all(0.0 < fidelity <= 1.0 for _, fidelity in pairs)

def test_mf_moo_problem_runs_with_momfbo():
    from condopt.benchmarks import builtin_problems
    from condopt.dataset import Observation
    from condopt.registry import build

    problem = builtin_problems()["mf_moo"]
    optimizer = build(
        "momfbo",
        problem.space,
        problem.objective,
        params={"n_initial": 2, "n_candidates": 64},
        seed=0,
    )
    observations = []
    for _ in range(2):
        pairs = optimizer.ask_with_fidelity(2)
        batch = []
        for config, fidelity in pairs:
            objectives = problem.objectives_fidelity_fn(config, fidelity)
            batch.append(Observation(
                x=config, objectives=objectives, fidelity=fidelity, cost=fidelity
            ))
        optimizer.tell(batch)
        observations.extend(batch)
    assert len(observations) >= 4
    assert all(set(obs.objectives) == {"yield", "cost"} for obs in observations)

def test_benchmark_momfbo_script_runs():
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_momfbo.py"
    spec = importlib.util.spec_from_file_location("benchmark_momfbo", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    row = module.run("momfbo", budget=6.0, batch_size=2, seed=0)
    assert row["cost_spent"] <= 6.0 + 1e-6
    assert row["n_observations"] > 0
    assert row["hv_full_fidelity"] >= 0.0

def test_next_action_stops_when_cost_budget_exhausted(tmp_path):
    space = simple_space()
    context = OptimizationContext.from_space(
        space, budget_total=8, batch_size=2, multi_fidelity=True
    )
    study = Study.create(
        tmp_path / "cost_stop",
        space,
        Objective(name="y", direction="max"),
        context=context,
        algorithm="mfbo",
        budget=8,
        batch_size=2,
        cost_budget=0.5,
        seed=0,
    )
    study.import_observations([{"x": {"x": 0.5}, "y": 1.0, "cost": 0.5}])
    action = study.next_action()
    assert action["action"] == "stop"
    assert "成本预算" in action["reason"]

def test_step_suggests_review_fidelity_channel_on_large_bias(tmp_path):
    space = simple_space()
    context = OptimizationContext.from_space(
        space, budget_total=8, batch_size=2, multi_fidelity=True
    )
    study = Study.create(
        tmp_path / "fidelity_bias",
        space,
        Objective(name="y", direction="max"),
        context=context,
        algorithm="mfbo",
        budget=8,
        batch_size=2,
        seed=0,
    )
    study.import_observations([
        {"x": {"x": 0.2}, "y": 50.0, "fidelity": 0.5, "cost": 0.5},
        {"x": {"x": 0.2}, "y": 80.0, "fidelity": 1.0, "cost": 1.0},
        {"x": {"x": 0.8}, "y": 45.0, "fidelity": 0.5, "cost": 0.5},
        {"x": {"x": 0.8}, "y": 75.0, "fidelity": 1.0, "cost": 1.0},
    ])
    summary = study.fidelity_summary()
    assert summary["paired"]["n"] == 2
    assert summary["paired"]["mean_bias"] == pytest.approx(30.0)
    assert summary["normalized_bias"] == pytest.approx(30.0 / 35.0)
    action = study.next_action()
    assert "review_fidelity_channel" in action["suggested_actions"]
    assert action["diagnostics"]["fidelity_paired_n"] == 2

def test_momfbo_propagates_objective_noise_to_scalar():
    import numpy as np

    from condopt.benchmarks import builtin_problems
    from condopt.registry import build

    problem = builtin_problems()["moo_yield_cost"]
    optimizer = build(
        "momfbo",
        problem.space,
        problem.objective,
        params={"n_initial": 2, "n_candidates": 64},
        seed=0,
    )
    observations = []
    for index, config in enumerate(
        [{"T": 80.0, "time": 40.0}, {"T": 90.0, "time": 60.0}, {"T": 70.0, "time": 90.0}]
    ):
        observation = problem.make_observation(config)
        observation.fidelity = 1.0
        observation.objective_std = 2.0 + index
        observations.append(observation)
    optimizer.tell(observations)
    assert optimizer._Ystd.shape == optimizer._Y.shape
    weights = np.array([0.5, 0.5])
    scalar_std = optimizer._scalar_std(weights)
    assert np.isfinite(scalar_std).all()
    assert (scalar_std > 0).all()

def test_step_suggests_collect_full_fidelity_when_none_recorded(tmp_path):
    space = simple_space()
    context = OptimizationContext.from_space(
        space, budget_total=8, batch_size=2, multi_fidelity=True
    )
    study = Study.create(
        tmp_path / "no_full_fidelity",
        space,
        Objective(name="y", direction="max"),
        context=context,
        algorithm="mfbo",
        budget=8,
        batch_size=2,
        seed=0,
    )
    study.import_observations([
        {"x": {"x": 0.2}, "y": 50.0, "fidelity": 0.5, "cost": 0.5},
        {"x": {"x": 0.4}, "y": 55.0, "fidelity": 0.5, "cost": 0.5},
        {"x": {"x": 0.6}, "y": 52.0, "fidelity": 0.5, "cost": 0.5},
        {"x": {"x": 0.8}, "y": 48.0, "fidelity": 0.5, "cost": 0.5},
    ])
    action = study.next_action()
    assert action["diagnostics"]["n_full_fidelity"] == 0
    assert "collect_full_fidelity" in action["suggested_actions"]
    assert "collect_full_fidelity" in action["suggested_action_reasons"]

def test_template_includes_fidelity_for_multi_fidelity_study(tmp_path, capsys):
    import csv as csv_module

    from condopt.cli import main

    space = simple_space()
    context = OptimizationContext.from_space(
        space, budget_total=8, batch_size=2, multi_fidelity=True
    )
    study = Study.create(
        tmp_path / "mf_template",
        space,
        Objective(name="y", direction="max"),
        context=context,
        algorithm="mfbo",
        budget=8,
        batch_size=2,
        seed=0,
    )
    study.ask(2)
    assert main(["template", "--run", str(study.dir), "--json"]) == 0
    capsys.readouterr()
    rows = list(csv_module.DictReader(
        (study.dir / "results_template.csv").read_text(encoding="utf-8-sig").splitlines()
    ))
    assert "fidelity" in rows[0]
    assert all(row["fidelity"] for row in rows)
    # filling the same file back keeps the requested fidelity
    for index, row in enumerate(rows):
        row["y"] = str(10.0 + index)
    filled = tmp_path / "filled_mf.csv"
    with filled.open("w", encoding="utf-8", newline="") as handle:
        writer = csv_module.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assert main(["tell", "--run", str(study.dir), "--results", str(filled), "--json"]) == 0
    capsys.readouterr()
    requested = {row["id"]: float(row["fidelity"]) for row in rows}
    for observation in study.dataset.observations:
        assert float(observation.fidelity) == requested[observation.meta["suggestion_id"]]

def test_moo_recommendation_ignores_low_fidelity_points(tmp_path):
    from condopt.benchmarks import builtin_problems

    problem = builtin_problems()["moo_yield_cost"]
    study = Study.create(
        tmp_path / "moo_mf_recommend",
        problem.space,
        problem.objective,
        context=OptimizationContext.from_space(
            problem.space, budget_total=8, batch_size=2, multi_fidelity=True
        ),
        algorithm="momfbo",
        budget=8,
        batch_size=2,
        seed=0,
    )
    study.import_observations([
        # attractive but biased low-fidelity point
        {"x": {"T": 82.0, "time": 55.0},
         "objectives": {"yield": 99.0, "cost": 7.0}, "fidelity": 0.5, "cost": 0.5},
        # true full-fidelity points
        {"x": {"T": 80.0, "time": 50.0},
         "objectives": {"yield": 80.0, "cost": 10.0}, "fidelity": 1.0, "cost": 1.0},
        {"x": {"T": 70.0, "time": 40.0},
         "objectives": {"yield": 75.0, "cost": 12.0}, "fidelity": 1.0, "cost": 1.0},
    ])
    recommendation = study.recommend()
    assert recommendation["fidelity_filter"] == "full"
    assert recommendation["condition"]["T"] in (80.0, 70.0)
    assert "全保真" in recommendation["selection_reason"]

def test_pareto_summary_excludes_low_fidelity_points(tmp_path):
    from condopt.benchmarks import builtin_problems

    problem = builtin_problems()["moo_yield_cost"]
    study = Study.create(
        tmp_path / "moo_mf_pareto",
        problem.space,
        problem.objective,
        context=OptimizationContext.from_space(
            problem.space, budget_total=8, batch_size=2, multi_fidelity=True
        ),
        algorithm="momfbo",
        budget=8,
        batch_size=2,
        seed=0,
    )
    study.import_observations([
        {"x": {"T": 82.0, "time": 55.0},
         "objectives": {"yield": 99.0, "cost": 7.0}, "fidelity": 0.5, "cost": 0.5},
        {"x": {"T": 80.0, "time": 50.0},
         "objectives": {"yield": 80.0, "cost": 10.0}, "fidelity": 1.0, "cost": 1.0},
    ])
    summary = study.pareto_summary()
    assert summary["fidelity_filter"] == "full"
    assert summary["n_low_fidelity_excluded"] == 1
    assert all(item["x"]["T"] != 82.0 for item in summary["front"])

def test_cost_budget_is_a_hard_cap_on_suggestions(tmp_path):
    from condopt.benchmarks import mf_suzuki_problem
    from condopt.router import OptimizationContext
    from condopt.study import Study

    problem = mf_suzuki_problem()
    context = OptimizationContext.from_space(
        problem.space, budget_total=20, batch_size=5, multi_fidelity=True
    )
    study = Study.create(
        tmp_path / "hard_cap",
        problem.space,
        problem.objective,
        algorithm="mfbo",
        budget=20,
        batch_size=5,
        cost_budget=1.0,
        context=context,
        seed=0,
    )
    pending = study.ask(5)
    planned = sum(float(item.get("fidelity", 1.0) or 1.0) for item in pending)
    assert planned <= 1.0 + 1e-9
    assert len(pending) >= 1
    assert study.remaining_cost() == 1.0
    decisions = (study.dir / "decisions.jsonl").read_text(encoding="utf-8")
    # either the optimizer trimmed on its own cost model or the study's hard cap did
    assert "cost_trim" in decisions or "cost_budget_trim" in decisions

def test_confirm_respects_cost_budget(tmp_path):
    from condopt.benchmarks import mf_suzuki_problem
    from condopt.router import OptimizationContext
    from condopt.study import Study

    problem = mf_suzuki_problem()
    context = OptimizationContext.from_space(
        problem.space, budget_total=10, batch_size=3, multi_fidelity=True
    )
    study = Study.create(
        tmp_path / "confirm_cap",
        problem.space,
        problem.objective,
        algorithm="mfbo",
        budget=10,
        batch_size=3,
        cost_budget=1.5,
        context=context,
        seed=0,
    )
    study.import_observations([
        {"x": problem.best_config, "y": 110.0, "fidelity": 1.0, "cost": 1.0},
        {"x": problem.best_config, "y": 60.0, "fidelity": 0.25, "cost": 0.25},
        {"x": problem.best_config, "y": 62.0, "fidelity": 0.25, "cost": 0.25},
    ])
    remaining = study.remaining_cost()
    assert remaining is not None and remaining < 1.0  # cannot afford a full run
    assert study.confirm(3) == []
    decisions = (study.dir / "decisions.jsonl").read_text(encoding="utf-8")
    assert "cost_budget_trim" in decisions

def test_multifidelity_executor_example_respects_cost_budget():
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "examples" / "multifidelity_executor.py"
    spec = importlib.util.spec_from_file_location("mf_executor_example", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    report = module.run_campaign(cost_budget=2.0, budget=8, seed=0)
    assert report["cost_spent"] <= 2.0 + 1e-9
    assert report["n_observations"] >= 1
    assert report["algorithm"] in ("mfbo", "momfbo")

def test_mfparego_runs_on_multifidelity_multiobjective():
    from condopt.benchmarks import builtin_problems, run_algorithm

    problem = builtin_problems()["mf_moo"]
    result = run_algorithm(problem, "mfparego", budget=12, batch_size=4, seed=0)
    assert result.budget == 12
    assert len(result.objective_vectors) == 12
    assert all(value == value for value in result.trajectory)  # finite trajectory


def test_router_lists_mfparego_as_multi_fidelity_multi_objective_candidate():
    from condopt.router import OptimizationContext, route

    context = OptimizationContext(
        n_dims=4, n_continuous=4, budget_total=40, batch_size=4,
        n_objectives=2, multi_fidelity=True, smoothness="smooth",
    )
    recommendation = route(context)
    assert recommendation.algorithm == "mfparego"  # benchmark-validated default
    assert recommendation.params
    recommendation.validate()

    single = OptimizationContext(
        n_dims=4, n_continuous=4, budget_total=40, batch_size=4,
        n_objectives=1, multi_fidelity=True, smoothness="smooth",
    )
    assert route(single).algorithm == "mfbo"

def test_router_multi_fidelity_multiobjective_params_valid():
    from condopt.router import OptimizationContext, route

    context = OptimizationContext(
        n_dims=4, n_continuous=4, budget_total=40, batch_size=4,
        n_objectives=2, multi_fidelity=True, smoothness="smooth",
    )
    recommendation = route(context)
    assert recommendation.algorithm == "mfparego"
    assert "min_full_fraction" in recommendation.params
    assert "momfbo" in recommendation.candidate_scores
    recommendation.validate()

def test_sweep_momfbo_full_script_runs():
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "sweep_momfbo_full.py"
    spec = importlib.util.spec_from_file_location("sweep_momfbo_full", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    row = module.evaluate(
        "momfbo", {"min_full_fraction": 0.4}, [0, 1], 8.0, 2, "mf_moo_constrained"
    )
    assert row["n"] == 2
    assert row["hv_feasible_mean"] >= 0.0

def test_mfparego_survives_incomplete_objective_vectors(tmp_path):
    from condopt.base import Objective
    from condopt.space import SearchSpace
    from condopt.study import Study

    space = SearchSpace.from_dict([
        {"name": "T", "kind": "continuous", "low": 60.0, "high": 115.0},
        {"name": "time", "kind": "continuous", "low": 5.0, "high": 120.0},
    ])
    objective = Objective(
        name="yield",
        direction="max",
        objectives=("yield", "cost"),
        directions=("max", "min"),
        ref_point=(60.0, 30.0),
    )
    study = Study.create(
        tmp_path / "mfparego_incomplete",
        space,
        objective,
        algorithm="mfparego",
        budget=12,
        batch_size=4,
        seed=0,
    )
    study.import_observations([
        {"x": {"T": 80.0, "time": 50.0}, "objectives": {"yield": 90.0, "cost": 10.0}},
        {"x": {"T": 82.0, "time": 55.0}, "objectives": {"yield": 92.0, "cost": 11.0}},
        {"x": {"T": 70.0, "time": 40.0}, "objectives": {"yield": 85.0, "cost": 8.0}},
        # incomplete: missing the second objective on purpose
        {"x": {"T": 75.0, "time": 60.0}, "objectives": {"yield": 88.0}},
    ])
    pending = study.ask(4)
    assert len(pending) == 4
    study.tell([
        {
            "id": item["id"],
            "fidelity": item.get("fidelity", 1.0),
            "objectives": {"yield": 80.0, "cost": 9.0},
        }
        for item in pending
    ])
    assert len(study.dataset.observations) == 8
    more = study.ask(2)
    assert len(more) == 2

def test_mqehvi_nd_runs_with_fidelity_and_stays_optional():
    from condopt.benchmarks import builtin_problems, run_algorithm
    from condopt.router import OptimizationContext, route

    problem = builtin_problems()["mf_moo"]
    result = run_algorithm(problem, "mqehvi_nd", budget=12, batch_size=4, seed=0)
    assert result.budget == 12
    assert len(result.objective_vectors) == 12

    for n_objectives, expected in ((2, "mfparego"), (3, "mfparego")):
        context = OptimizationContext(
            n_dims=4, n_continuous=4, budget_total=40, batch_size=4,
            n_objectives=n_objectives, multi_fidelity=True, smoothness="smooth",
        )
        recommendation = route(context)
        assert recommendation.algorithm == expected
        assert "mqehvi_nd" in recommendation.candidate_scores

def test_mf_moo4_problem_and_benchmark_path():
    from condopt.benchmarks import builtin_problems

    problem = builtin_problems()["mf_moo4"]
    assert problem.objective.n_objectives == 4
    assert problem.objectives_fidelity_fn is not None
    values = problem.objectives_fidelity_fn(problem.best_config, 1.0)
    assert set(values) == {"yield", "cost", "selectivity", "purity"}

    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_momfbo.py"
    spec = importlib.util.spec_from_file_location("benchmark_momfbo_4", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    row = module.run("mfparego", budget=8.0, batch_size=2, seed=0, problem_name="mf_moo4")
    assert row["hv_feasible_full"] >= 0.0
    assert row["problem"] == "mf_moo4"

def test_mf_solvent_descriptor_variant_shares_landscape():
    from condopt.benchmarks import builtin_problems

    base = builtin_problems()["mf_solvent"]
    described = builtin_problems()["mf_solvent_desc"]
    assert base.objective.n_objectives == described.objective.n_objectives == 2
    assert described.space.feature_dim < base.space.feature_dim
    assert base.objectives_fidelity_fn(base.best_config, 1.0) == described.objectives_fidelity_fn(
        described.best_config, 1.0
    )
    low = base.objectives_fidelity_fn(base.best_config, 0.25)["yield"]
    full = base.objectives_fidelity_fn(base.best_config, 1.0)["yield"]
    assert low < full  # cheap runs are biased low on yield

def test_infeasible_full_fidelity_runs_are_flagged(tmp_path):
    from condopt.base import Objective
    from condopt.space import SearchSpace
    from condopt.study import Study

    space = SearchSpace.from_dict(
        [{"name": "T", "kind": "continuous", "low": 40.0, "high": 120.0}]
    )
    objective = Objective(
        name="yield",
        direction="max",
        constraints=("T_max",),
        constraint_limits={"T_max": {"op": "<=", "value": 80.0}},
    )
    study = Study.create(
        tmp_path / "infeasible_full",
        space,
        objective,
        algorithm="mfbo",
        budget=8,
        batch_size=4,
        seed=0,
    )
    study.import_observations([
        {"x": {"T": 110.0}, "y": 90.0, "fidelity": 1.0, "cost": 1.0,
         "constraints": {"T_max": 110.0}},
        {"x": {"T": 112.0}, "y": 88.0, "fidelity": 1.0, "cost": 1.0,
         "constraints": {"T_max": 112.0}},
        {"x": {"T": 70.0}, "y": 60.0, "fidelity": 0.25, "cost": 0.25,
         "constraints": {"T_max": 70.0}},
    ])
    action = study.next_action()
    diagnostics = action["diagnostics"]
    assert diagnostics["n_full_fidelity"] == 2
    assert diagnostics["n_feasible_full_fidelity"] == 0
    assert "collect_feasible_full_fidelity" in action["suggested_actions"]
    assert action["suggested_action_reasons"]["collect_feasible_full_fidelity"]


def test_constrained_mf_executor_example_runs():
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "examples" / "constrained_mf_executor.py"
    spec = importlib.util.spec_from_file_location("constrained_mf_executor", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    report = module.run_campaign(cost_budget=4.0, budget=10, seed=0)
    assert report["ok"] is True
    assert report["cost_within_budget"] is True
    assert report["integrity_ok"] is True
    assert report["fidelity_levels"]

def test_feasibility_aware_full_benchmark_script_runs():
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_feasible_full.py"
    spec = importlib.util.spec_from_file_location("benchmark_feasible_full", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    row = module.run_one(True, 0.5, budget=10, batch_size=3, cost_budget=4.0, seed=0)
    assert row["n_observations"] > 0
    assert row["n_full"] >= 0
    assert row["cost_spent"] <= 4.0 + 1e-9

def test_hard_feasibility_benchmark_script_runs():
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_hard_feasibility.py"
    spec = importlib.util.spec_from_file_location("benchmark_hard_feasibility", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    tight = module.run_one(True, budget=12, batch_size=3, seed=0)
    loose = module.run_one(True, budget=12, batch_size=3, seed=0, bounds_encoded=True)
    # one evaluation may be held back to confirm the final winner
    assert 11 <= tight["n_observations"] <= 12
    assert 11 <= loose["n_observations"] <= 12
    # one evaluation may be reserved to confirm the final winner
    assert 11 <= loose["n_feasible"] <= 12  # everything inside the box is feasible
    assert module.is_feasible({"T": 60.0, "time": 55.0, "solvent": "DMF", "base_eq": 2})
    assert not module.is_feasible({"T": 90.0, "time": 55.0, "solvent": "DMF", "base_eq": 2})

def test_cost_budget_shortfall_stops_instead_of_spinning(tmp_path):
    """Regression: an unaffordable confirmation used to loop forever."""
    from condopt.base import Objective
    from condopt.space import SearchSpace
    from condopt.study import Study

    space = SearchSpace.from_dict([
        {"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0}
    ])
    objective = Objective(name="y", direction="max")
    study = Study.create(
        tmp_path / "cost_spin",
        space,
        objective,
        algorithm="mfbo",
        budget=12,
        batch_size=3,
        cost_budget=2.5,
        seed=0,
    )
    study.import_observations([
        {"x": {"x": 0.5}, "y": 1.0, "fidelity": 1.0, "cost": 1.0},
    ])
    for _ in range(12):
        action = study.next_action()
        if action["action"] == "stop":
            break
        if action["action"] == "ask":
            study.ask(int(action.get("ask_n") or 3))
        elif action["action"] == "confirm":
            study.confirm(int(action.get("confirm_n") or 3))
        else:
            study.tell([
                {"id": item["id"], "y": 0.5,
                 "fidelity": float(item.get("fidelity", 1.0) or 1.0),
                 "cost": float(item.get("fidelity", 1.0) or 1.0)}
                for item in study.pending
            ])
    else:
        raise AssertionError("study never stopped")
    assert action["action"] == "stop"
    assert action["diagnostics"]["stop_category"] in ("cost_exhausted", "budget_exhausted")
    assert "成本" in action["diagnostics"]["stop_reason"]

def test_stale_persisted_constraints_do_not_break_a_warm_start(tmp_path):
    """Regression: a state saved after the first batch must not shadow rebuilt data."""
    import numpy as np

    from condopt.algorithms.mfbo import MultiFidelityBO
    from condopt.base import Observation, Objective
    from condopt.space import SearchSpace

    space = SearchSpace.from_dict([
        {"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0}
    ])
    objective = Objective(
        name="y", direction="max", constraints=("c",),
        constraint_limits={"c": {"op": "<=", "value": 1.0}},
    )
    prior = [
        Observation(
            x={"x": i / 20.0}, y=float(i), fidelity=1.0,
            constraints={"c": float(i - 10)}, cost=1.0,
        )
        for i in range(17)
    ]
    algo = MultiFidelityBO(space, objective, seed=0, prior=prior)
    assert algo._C is not None and algo._C.shape[0] == 17
    # a stale state from an earlier batch (1 row) must be ignored
    algo.load_state_dict({"kind": "mfbo", "C": [[0.0]], "init_queue": None})
    assert algo._C.shape[0] == 17
    assert algo._feasible_mask(np.ones(len(algo._y), dtype=bool)) is not None
    proposals = algo.ask_with_fidelity(2, cost_remaining=5.0)
    assert len(proposals) == 2

