"""Tests for the feasibility-aware qEHVI (``cqehvi``): 2 objectives + hard constraints."""

import numpy as np
import pytest

from condopt import Objective, Observation, SearchSpace
from condopt.benchmarks import (
    builtin_problems,
    moo_constrained_reference_front,
    run_algorithm,
)
from condopt.pareto import hypervolume_2d, non_dominated_mask
from condopt import registry
from condopt.registry import build
from condopt.router import OptimizationContext, route


def constrained_moo_objective() -> Objective:
    return Objective(
        name="yield",
        direction="max",
        objectives=("yield", "cost"),
        directions=("max", "min"),
        constraints=("T_max",),
        constraint_limits={"T_max": {"op": "<=", "value": 78.0}},
        ref_point=(60.0, 30.0),
    )


def two_dim_space() -> SearchSpace:
    return SearchSpace.from_dict(
        [
            {"name": "T", "kind": "continuous", "low": 60.0, "high": 115.0},
            {"name": "time", "kind": "continuous", "low": 5.0, "high": 120.0},
        ]
    )


def test_router_picks_cqehvi_for_constrained_multi_objective():
    ctx = OptimizationContext(
        n_dims=2,
        n_continuous=2,
        budget_total=40,
        batch_size=4,
        n_objectives=2,
        n_constraints=2,
        noise="low",
    )
    rec = route(ctx)
    assert rec.algorithm == "cqehvi", rec.candidate_scores
    assert rec.params["mc_draws"] >= 16
    assert rec.params["feasibility_threshold"] >= 0.0
    assert rec.design["main"] == "cqehvi"
    assert rec.confidence == "high"
    rec.validate()


def test_cqehvi_requires_two_objectives():
    space = two_dim_space()
    with pytest.raises(ValueError, match="exactly 2 objectives"):
        build(
            "cqehvi",
            space,
            Objective(
                name="y",
                direction="max",
                constraints=("c",),
                constraint_limits={"c": {"op": "<=", "value": 1.0}},
            ),
            seed=0,
        )


def test_cqehvi_requires_a_constraint():
    space = two_dim_space()
    objective = Objective(
        name="yield",
        direction="max",
        objectives=("yield", "cost"),
        directions=("max", "min"),
    )
    with pytest.raises(ValueError, match="at least one constraint"):
        build("cqehvi", space, objective, seed=0)


def test_cqehvi_requires_constraint_limits():
    space = two_dim_space()
    objective = Objective(name="yield", direction="max", objectives=("yield", "cost"))
    objective.constraints = ("T_max",)
    objective.constraint_limits = {}
    with pytest.raises(ValueError, match="constraint_limits"):
        build("cqehvi", space, objective, seed=0)


def test_cqehvi_feasibility_respects_margin():
    optimizer = build(
        "cqehvi",
        two_dim_space(),
        constrained_moo_objective(),
        params={"constraint_margin": 1.0},
        seed=0,
    )
    rows = [
        ({"T": 70.0, "time": 20.0}, {"yield": 80.0, "cost": 8.0}, {"T_max": 70.0}),
        ({"T": 75.0, "time": 20.0}, {"yield": 82.0, "cost": 8.5}, {"T_max": 77.0}),
        ({"T": 80.0, "time": 20.0}, {"yield": 84.0, "cost": 9.0}, {"T_max": 80.0}),
    ]
    optimizer.tell(
        [
            Observation(x=x, objectives=objectives, constraints=constraints)
            for x, objectives, constraints in rows
        ]
    )
    # limit 78 with margin 1 -> feasible iff observed T_max <= 77
    assert optimizer._feasible.tolist() == [True, True, False]


def test_cqehvi_ignores_missing_constraint_values():
    optimizer = build(
        "cqehvi", two_dim_space(), constrained_moo_objective(), params={"n_initial": 2}, seed=0
    )
    optimizer.tell(
        [
            Observation(
                x={"T": 70.0, "time": 20.0},
                objectives={"yield": 80.0, "cost": 8.0},
                constraints={"T_max": 70.0},
            ),
            Observation(
                x={"T": 72.0, "time": 20.0},
                objectives={"yield": 81.0, "cost": 8.1},
            ),
        ]
    )
    assert optimizer._feasible.tolist() == [True, False]
    config = optimizer.ask(1)[0]
    two_dim_space().validate_config(config)


def test_cqehvi_run_reports_feasible_best():
    """The benchmark metric must be the *feasible* best, not the raw maximum."""
    problem = builtin_problems()["moo_constrained"]
    result = run_algorithm(
        problem,
        "cqehvi",
        budget=24,
        batch_size=4,
        seed=0,
        params={"n_initial": 6, "mc_draws": 64},
    )
    assert result.budget == 24
    assert result.best_feasible_score is not None
    # feasible yield optimum is ~88.1 C at T=78, time=50; unconstrained ~90
    assert result.best_feasible_score > 80.0


def test_cqehvi_beats_unconstrained_on_feasibility_and_hv():
    problem = builtin_problems()["moo_constrained"]
    reference = moo_constrained_reference_front()
    reference_point = (60.0, -30.0)
    reference_hv = hypervolume_2d(reference, reference_point)
    assert reference_hv > 0

    def feasible(observation) -> bool:
        limits = problem.objective.constraint_limits
        for name, spec in limits.items():
            value = observation.constraints.get(name)
            if value is None:
                return False
            threshold = float(spec["value"])
            if str(spec.get("op", "<=")) == "<=":
                if value > threshold:
                    return False
            elif value < threshold:
                return False
        return True

    def campaign(algorithm: str, params: dict) -> tuple[int, float]:
        optimizer = build(algorithm, problem.space, problem.objective, params=params, seed=0)
        observations = []
        for _ in range(6):
            batch = [problem.make_observation(config) for config in optimizer.ask(4)]
            optimizer.tell(batch)
            observations.extend(batch)
        feasible_observations = [obs for obs in observations if feasible(obs)]
        assert feasible_observations, f"{algorithm}: no feasible point found"
        scores = np.vstack(
            [
                [obs.objectives["yield"], -obs.objectives["cost"]]
                for obs in feasible_observations
            ]
        )
        front = scores[non_dominated_mask(scores)]
        return len(feasible_observations), hypervolume_2d(front, reference_point) / reference_hv

    constrained_n, constrained_hv = campaign(
        "cqehvi", {"n_initial": 6, "mc_draws": 64}
    )
    plain_n, plain_hv = campaign("qehvi", {"n_initial": 6, "mc_draws": 64})
    assert constrained_n >= 2 * plain_n, (constrained_n, plain_n)
    assert constrained_hv > plain_hv, (constrained_hv, plain_hv)
    assert constrained_hv > 0.9


def test_cqehvi_parameters_are_registered_and_valid():
    spec = registry.spec("cqehvi")
    assert "constraints" in spec.capabilities
    assert "multi_objective" in spec.capabilities
    names = [param.name for param in spec.params]
    for required in ("mc_draws", "constraint_margin", "feasibility_threshold", "batch_strategy"):
        assert required in names
    assert registry.validate_params("cqehvi", {}) == spec.defaults()


def test_cqehvi_constant_liar_batch_runs():
    problem = builtin_problems()["moo_constrained"]
    result = run_algorithm(
        problem,
        "cqehvi",
        budget=8,
        batch_size=4,
        seed=0,
        params={"n_initial": 4, "mc_draws": 32, "batch_strategy": "constant_liar"},
    )
    assert result.budget == 8
    assert len(result.objective_vectors) == 8
    assert len(result.constraint_vectors) == 8


def test_study_with_constrained_multi_objective(tmp_path):
    from condopt import Study

    problem = builtin_problems()["moo_constrained"]
    study = Study.create(
        tmp_path / "cqehvi",
        problem.space,
        problem.objective,
        budget=6,
        batch_size=2,
        seed=0,
    )
    # 2 objectives + constraints: cqehvi (low noise path) or cqnehvi (noisy path)
    assert study.algorithm in ("cqehvi", "cqnehvi")
    pending = study.ask(2)
    assert len(pending) == 2
    study.tell(
        [
            {
                "id": item["id"],
                "objectives": {"yield": 80.0 + index, "cost": 8.0 + index},
                "constraints": {"T_max": 70.0, "time_max": 20.0},
            }
            for index, item in enumerate(pending)
        ]
    )
    assert len(study.dataset) == 2
    summary = study.summary()
    assert summary["pareto"]["n_front"] >= 1
