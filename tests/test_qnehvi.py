"""Tests for noisy multi-objective qNEHVI and qEHVI robustness fixes."""

import numpy as np
import pytest

from condopt import Objective, Observation
from condopt.algorithms.qehvi import QEHVI
from condopt.benchmarks import builtin_problems, moo_reference_front
from condopt.pareto import hypervolume_2d, non_dominated_mask
from condopt.registry import build
from condopt.router import OptimizationContext, route


def test_router_switches_between_qehvi_and_qnehvi():
    low = OptimizationContext(
        n_dims=2, n_continuous=2, budget_total=60, n_objectives=2, noise="low"
    )
    assert route(low).algorithm == "qehvi"
    medium = OptimizationContext(
        n_dims=2, n_continuous=2, budget_total=60, n_objectives=2, noise="medium"
    )
    recommendation = route(medium)
    assert recommendation.algorithm == "qnehvi"
    recommendation.validate()


def test_qnehvi_runs_on_noisy_benchmark():
    problem = builtin_problems()["moo_yield_cost"]
    reference = moo_reference_front()
    ref_point = (60.0, -30.0)
    reference_hv = hypervolume_2d(reference, ref_point)

    optimizer = build(
        "qnehvi",
        problem.space,
        problem.objective,
        params={
            "n_initial": 6,
            "n_candidates": 256,
            "n_base": 8,
            "n_y": 32,
            "noise_std": 0.5,
        },
        seed=0,
    )
    rng = np.random.default_rng(0)
    observations = []
    for _ in range(10):
        for config in optimizer.ask(4):
            objectives = problem.objectives_fn(config)
            noisy = {
                "yield": objectives["yield"] + rng.normal(0.0, 0.5),
                "cost": max(0.0, objectives["cost"] + rng.normal(0.0, 0.3)),
            }
            observation = Observation(x=config, objectives=noisy, objective_std=0.5)
            observations.append(observation)
            optimizer.tell([observation])
    # Evaluate the TRUE objective (noise only affected the observations we told).
    scores = np.vstack(
        [
            [problem.objectives_fn(obs.x)["yield"], -problem.objectives_fn(obs.x)["cost"]]
            for obs in observations
        ]
    )
    front = scores[non_dominated_mask(scores)]
    hv_ratio = hypervolume_2d(front, ref_point) / reference_hv
    assert hv_ratio > 0.7, f"qNEHVI HV ratio too low: {hv_ratio}"


def test_qnehvi_warns_only_once_when_no_noise_information():
    import warnings as warnings_module

    problem = builtin_problems()["moo_yield_cost"]
    optimizer = build(
        "qnehvi",
        problem.space,
        problem.objective,
        params={"n_initial": 2, "n_candidates": 128, "n_base": 4, "n_y": 16, "noise_std": 0.0},
        seed=0,
    )
    optimizer.tell(
        [
            problem.make_observation({"T": 80.0, "time": 60.0}),
            problem.make_observation({"T": 90.0, "time": 70.0}),
            problem.make_observation({"T": 100.0, "time": 80.0}),
        ]
    )
    with warnings_module.catch_warnings(record=True) as caught:
        warnings_module.simplefilter("always")
        optimizer.ask(1)
        optimizer.tell([problem.make_observation({"T": 85.0, "time": 75.0})])
        optimizer.ask(1)
    noise_warnings = [w for w in caught if "no observation-noise" in str(w.message)]
    assert len(noise_warnings) == 1


def test_qehvi_counts_all_observations_and_raises_on_invalid_ref():
    problem = builtin_problems()["moo_yield_cost"]
    optimizer = build("qehvi", problem.space, problem.objective, seed=0)
    complete = problem.make_observation({"T": 80.0, "time": 60.0})
    incomplete = Observation(x={"T": 90.0, "time": 60.0}, objectives={"yield": 85.0})
    optimizer.tell([complete, incomplete])
    assert optimizer.t == 2
    assert optimizer._skipped == 1

    bad_objective = Objective(
        name="yield",
        direction="max",
        objectives=("yield", "cost"),
        directions=("max", "min"),
        ref_point=(1000.0, 1000.0),
    )
    bad = build("qehvi", problem.space, bad_objective, params={"n_initial": 2}, seed=0)
    bad.tell(
        [
            problem.make_observation({"T": 80.0, "time": 60.0}),
            problem.make_observation({"T": 90.0, "time": 70.0}),
            problem.make_observation({"T": 70.0, "time": 50.0}),
            problem.make_observation({"T": 85.0, "time": 80.0}),
        ]
    )
    with pytest.raises(ValueError, match="dominates"):
        bad.ask(1)


def test_qnehvi_ref_is_persisted_in_state():
    problem = builtin_problems()["moo_yield_cost"]
    optimizer = build(
        "qnehvi",
        problem.space,
        problem.objective,
        params={"n_initial": 2, "n_candidates": 128, "n_base": 4, "n_y": 16, "noise_std": 0.2},
        seed=0,
    )
    optimizer.tell(
        [
            problem.make_observation({"T": 80.0, "time": 60.0}),
            problem.make_observation({"T": 90.0, "time": 70.0}),
            problem.make_observation({"T": 100.0, "time": 80.0}),
        ]
    )
    optimizer.ask(1)
    state = optimizer.state_dict()
    clone = build(
        "qnehvi",
        problem.space,
        problem.objective,
        params={"n_initial": 2, "n_candidates": 128, "n_base": 4, "n_y": 16, "noise_std": 0.2},
        seed=0,
    )
    clone.load_state_dict(state)
    assert np.allclose(clone._signed_references(), optimizer._signed_references())
    assert len(clone.ask(1)) == 1


def test_parego_counts_all_observations():
    problem = builtin_problems()["moo_yield_cost"]
    optimizer = build("parego", problem.space, problem.objective, seed=0)
    complete = problem.make_observation({"T": 80.0, "time": 60.0})
    incomplete = Observation(x={"T": 90.0, "time": 60.0}, objectives={"yield": 85.0})
    optimizer.tell([complete, incomplete])
    assert optimizer.t == 2
    assert len(optimizer._Y) == 1


def test_qehvi_derived_ref_ignores_incomplete_rows():
    problem = builtin_problems()["moo_yield_cost"]
    objective_without_ref = Objective(
        name="yield",
        direction="max",
        objectives=("yield", "cost"),
        directions=("max", "min"),
    )
    optimizer = build(
        "qehvi", problem.space, objective_without_ref, params={"n_initial": 2, "n_candidates": 128}, seed=0
    )
    optimizer.tell(
        [
            problem.make_observation({"T": 80.0, "time": 60.0}),
            problem.make_observation({"T": 90.0, "time": 70.0}),
            problem.make_observation({"T": 100.0, "time": 80.0}),
            Observation(x={"T": 70.0, "time": 50.0}, objectives={"yield": 80.0}),
        ]
    )
    configs = optimizer.ask(1)
    assert len(configs) == 1
    assert np.all(np.isfinite(optimizer._signed_references()))


def test_qnehvi_warns_when_no_noise_information():
    problem = builtin_problems()["moo_yield_cost"]
    optimizer = build(
        "qnehvi",
        problem.space,
        problem.objective,
        params={"n_initial": 2, "n_candidates": 128, "n_base": 4, "n_y": 16, "noise_std": 0.0},
        seed=0,
    )
    optimizer.tell(
        [
            problem.make_observation({"T": 80.0, "time": 60.0}),
            problem.make_observation({"T": 90.0, "time": 70.0}),
            problem.make_observation({"T": 100.0, "time": 80.0}),
        ]
    )
    with pytest.warns(RuntimeWarning, match="no observation-noise"):
        optimizer.ask(1)
