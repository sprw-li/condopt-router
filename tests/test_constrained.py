"""Tests for constrained Bayesian optimization."""

import numpy as np
import pytest

from condopt import Objective, Observation, SearchSpace
from condopt.registry import build
from condopt.router import OptimizationContext, route


def constrained_space() -> SearchSpace:
    return SearchSpace.from_dict(
        [{"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0}]
    )


def constrained_objective() -> Objective:
    return Objective(
        name="f",
        direction="max",
        constraints=("x_max",),
        constraint_limits={"x_max": {"op": "<=", "value": 0.5}},
    )


def test_router_picks_constrained_algorithm():
    ctx = OptimizationContext(
        n_dims=2, n_continuous=2, budget_total=24, n_constraints=1
    )
    recommendation = route(ctx)
    assert recommendation.algorithm == "constrained"
    assert recommendation.params["mc_draws"] >= 16
    recommendation.validate()


def test_constrained_optimizer_finds_feasible_optimum():
    space = constrained_space()
    objective = constrained_objective()

    def f(config: dict) -> float:
        return -((config["x"] - 0.8) ** 2)  # unconstrained optimum at 0.8

    def constraint(config: dict) -> float:
        return config["x"]  # feasible iff x <= 0.5

    optimizer = build(
        "constrained", space, objective, params={"n_initial": 4, "mc_draws": 64}, seed=0
    )
    observations = []
    for _ in range(8):
        for config in optimizer.ask(2):
            observation = Observation(x=config, y=f(config), constraints={"x_max": constraint(config)})
            observations.append(observation)
            optimizer.tell([observation])
    feasible = [obs for obs in observations if obs.constraints["x_max"] <= 0.5]
    assert feasible, "constrained optimizer never found a feasible point"
    best_feasible = max(obs.y for obs in feasible)
    assert best_feasible > -0.2, f"feasible optimum poor: {best_feasible}"


def test_constrained_requires_limits():
    space = constrained_space()
    objective = Objective(name="f", direction="max", constraints=("x_max",))
    with pytest.raises(ValueError, match="constraint_limits"):
        build("constrained", space, objective, seed=0)


def test_constraint_definition_validation():
    with pytest.raises(ValueError):
        Objective(name="f", constraints=("c",), constraint_limits={"c": {"op": "~", "value": 1}})


def test_constrained_algorithm_on_constraint_benchmarks():
    from condopt.benchmarks import builtin_problems

    expectations = {"constraint_easy": 94.0, "constraint_hard": 92.0}
    for name, threshold in expectations.items():
        problem = builtin_problems()[name]
        optimizer = build(
            "constrained",
            problem.space,
            problem.objective,
            params={"n_initial": 6, "mc_draws": 64},
            seed=0,
        )
        observations = []
        for _ in range(5):
            batch = [problem.make_observation(config) for config in optimizer.ask(4)]
            optimizer.tell(batch)
            observations.extend(batch)
        feasible = [obs for obs in observations if obs.constraints["g"] <= 0]
        assert feasible, f"{name}: no feasible point found"
        best_feasible = max(obs.y for obs in feasible)
        assert best_feasible > threshold, f"{name}: best feasible {best_feasible}"


def test_study_with_constraints(tmp_path):
    from condopt import Study

    space = constrained_space()
    objective = constrained_objective()
    study = Study.create(
        tmp_path / "constrained",
        space,
        objective,
        budget=6,
        batch_size=2,
        seed=0,
    )
    assert study.algorithm == "constrained"
    pending = study.ask(2)
    results = [
        {
            "id": item["id"],
            "y": -((item["x"]["x"] - 0.8) ** 2),
            "constraints": {"x_max": item["x"]["x"]},
        }
        for item in pending
    ]
    study.tell(results)
    assert len(study.dataset) == 2
    assert "constrained" in study.report()

def test_confirm_prefers_a_feasible_condition_when_full_runs_were_infeasible(tmp_path):
    from condopt.base import Objective
    from condopt.space import SearchSpace
    from condopt.study import Study

    space = SearchSpace.from_dict([
        {"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0}
    ])
    objective = Objective(
        name="y",
        direction="max",
        constraints=("c",),
        constraint_limits={"c": {"op": "<=", "value": 0.5}},
    )
    study = Study.create(
        tmp_path / "confirm_feasible",
        space,
        objective,
        algorithm="random",
        budget=8,
        batch_size=2,
        seed=0,
    )
    study.import_observations([
        # expensive but infeasible
        {"x": {"x": 0.9}, "y": 90.0, "fidelity": 1.0, "cost": 1.0, "constraints": {"c": 0.9}},
        # cheap but feasible and decent
        {"x": {"x": 0.2}, "y": 60.0, "fidelity": 0.25, "cost": 0.25, "constraints": {"c": 0.2}},
        {"x": {"x": 0.3}, "y": 55.0, "fidelity": 0.25, "cost": 0.25, "constraints": {"c": 0.3}},
    ])
    pending = study.confirm(1)
    assert len(pending) == 1
    assert pending[0]["x"]["x"] == 0.2  # the feasible point, not the high-yield one
    decisions = (study.dir / "decisions.jsonl").read_text(encoding="utf-8")
    assert "best_feasible_observed" in decisions

def test_two_phase_feasibility_branch_runs_without_feasible_history(tmp_path):
    from condopt.base import Objective
    from condopt.space import SearchSpace
    from condopt.study import Study

    space = SearchSpace.from_dict([
        {"name": "T", "kind": "continuous", "low": 40.0, "high": 120.0},
        {"name": "t", "kind": "continuous", "low": 5.0, "high": 60.0},
    ])
    objective = Objective(
        name="yield",
        direction="max",
        constraints=("T_max",),
        constraint_limits={"T_max": {"op": "<=", "value": 65.0}},
    )
    study = Study.create(
        tmp_path / "two_phase",
        space,
        objective,
        algorithm="mfbo",
        params={"two_phase_feasibility": True},
        budget=10,
        batch_size=3,
        seed=0,
    )
    study.import_observations([
        {"x": {"T": 110.0, "t": 40.0}, "y": 90.0, "fidelity": 0.25, "cost": 0.25,
         "constraints": {"T_max": 110.0}},
        {"x": {"T": 105.0, "t": 30.0}, "y": 88.0, "fidelity": 0.25, "cost": 0.25,
         "constraints": {"T_max": 105.0}},
        {"x": {"T": 100.0, "t": 20.0}, "y": 86.0, "fidelity": 0.25, "cost": 0.25,
         "constraints": {"T_max": 100.0}},
    ])
    pairs = study.ask(3)
    assert len(pairs) == 3
    for item in pairs:
        space.validate_config(item["x"])

def test_low_observed_feasibility_suggests_encoding_limits(tmp_path):
    from condopt.base import Objective
    from condopt.space import SearchSpace
    from condopt.study import Study

    space = SearchSpace.from_dict([
        {"name": "T", "kind": "continuous", "low": 40.0, "high": 120.0}
    ])
    objective = Objective(
        name="yield",
        direction="max",
        constraints=("T_max",),
        constraint_limits={"T_max": {"op": "<=", "value": 0.0}},
    )
    study = Study.create(
        tmp_path / "low_feas",
        space,
        objective,
        algorithm="bayes",
        budget=20,
        batch_size=4,
        seed=0,
    )
    # T_max is a violation magnitude: only T <= 60 is feasible -> tiny fraction
    study.import_observations([
        {"x": {"T": value}, "y": 1.0, "constraints": {"T_max": value - 60.0}}
        for value in (100.0, 105.0, 110.0, 115.0, 95.0, 90.0, 85.0, 80.0, 75.0, 70.0, 65.0, 118.0)
    ])
    action = study.next_action()
    diagnostics = action["diagnostics"]
    assert diagnostics["observed_feasible_fraction"] == 0.0
    assert "encode_limits_as_bounds" in action["suggested_actions"]
    assert "可行域过小" in action["suggested_action_reasons"]["encode_limits_as_bounds"]

    # a healthy feasible fraction must not trigger it
    healthy = Study.create(
        tmp_path / "ok_feas",
        space,
        objective,
        algorithm="bayes",
        budget=20,
        batch_size=4,
        seed=0,
    )
    # ``T_max`` is a violation magnitude (feasible iff ``T <= 60``), and the declared
    # space is ``T in [40, 120]``: the values must stay inside it.  The previous list
    # used T = 10..59, which was out of bounds and only "worked" because continuous
    # bounds were not validated (fixed in A10).
    healthy.import_observations([
        {"x": {"T": value}, "y": 1.0, "constraints": {"T_max": value - 60.0}}
        for value in (40.0, 42.0, 45.0, 48.0, 50.0, 52.0, 55.0, 58.0, 59.0, 57.0, 56.0, 54.0)
    ])
    assert "encode_limits_as_bounds" not in healthy.next_action()["suggested_actions"]

