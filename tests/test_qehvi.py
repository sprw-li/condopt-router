"""Tests for 2D EHVI: partitioning, DHV and the qEHVI optimizer."""

import numpy as np
import pytest

from condopt import Objective, SearchSpace
from condopt.benchmarks import builtin_problems, moo_reference_front
from condopt.pareto import (
    dhv_2d,
    hypervolume_2d,
    hypervolume_from_boxes,
    non_dominated_mask,
    partition_2d,
)
from condopt.registry import build
from condopt.router import OptimizationContext, route


def test_partition_and_dhv_hand_computed():
    ref = np.zeros(2)
    front = np.asarray([[3.0, 1.0], [2.0, 2.0]])
    boxes = partition_2d(front, ref)
    assert hypervolume_from_boxes(boxes) == pytest.approx(5.0)
    assert hypervolume_2d(front, ref) == pytest.approx(5.0)
    assert dhv_2d(np.asarray([3.0, 2.0]), boxes, ref) == pytest.approx(1.0)
    assert dhv_2d(np.asarray([4.0, 2.0]), boxes, ref) == pytest.approx(3.0)
    assert dhv_2d(np.asarray([1.0, 1.0]), boxes, ref) == pytest.approx(0.0)
    assert dhv_2d(np.asarray([2.5, 1.5]), boxes, ref) == pytest.approx(0.25)


def test_partition_three_point_front_matches_hypervolume():
    ref = np.zeros(2)
    front = np.asarray([[3.0, 1.0], [2.0, 2.0], [1.0, 3.0]])
    boxes = partition_2d(front, ref)
    assert hypervolume_from_boxes(boxes) == pytest.approx(hypervolume_2d(front, ref))
    assert hypervolume_from_boxes(boxes) == pytest.approx(6.0)


def test_router_switches_to_qehvi_for_two_objectives():
    ctx = OptimizationContext(
        n_dims=2,
        n_continuous=2,
        budget_total=60,
        n_objectives=2,
        noise="low",
    )
    recommendation = route(ctx)
    assert recommendation.algorithm == "qehvi"
    recommendation.validate()

    small = OptimizationContext(
        n_dims=2, n_continuous=2, budget_total=20, n_objectives=2, noise="low"
    )
    assert route(small).algorithm == "parego"


def test_qehvi_requires_two_objectives():
    space = SearchSpace.from_dict(
        [{"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0}]
    )
    with pytest.raises(ValueError, match="exactly 2 objectives"):
        build("qehvi", space, Objective(name="f", direction="max"), seed=0)


def test_qehvi_improves_hypervolume_on_moo_benchmark():
    problem = builtin_problems()["moo_yield_cost"]
    reference = moo_reference_front()
    reference_point = (60.0, -30.0)
    reference_hv = hypervolume_2d(reference, reference_point)

    optimizer = build(
        "qehvi",
        problem.space,
        problem.objective,
        params={"n_initial": 6, "mc_draws": 64},
        seed=0,
    )
    observations = []
    for _ in range(10):
        batch = [problem.make_observation(config) for config in optimizer.ask(4)]
        optimizer.tell(batch)
        observations.extend(batch)
    scores = np.vstack([[obs.objectives["yield"], -obs.objectives["cost"]] for obs in observations])
    front = scores[non_dominated_mask(scores)]
    hv_ratio = hypervolume_2d(front, reference_point) / reference_hv
    assert hv_ratio > 0.9, f"qEHVI HV ratio too low: {hv_ratio}"
