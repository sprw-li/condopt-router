"""Tests for Pareto utilities."""

import numpy as np
import pytest

from condopt.base import Objective
from condopt.dataset import Observation
from condopt.pareto import (
    hypervolume_2d,
    hypervolume_ratio,
    non_dominated_mask,
    observation_values,
    signed_scores,
    summarise,
)


def test_non_dominated_mask_basic():
    scores = np.asarray([[3.0, 1.0], [2.0, 2.0], [1.0, 3.0], [1.0, 1.0]])
    mask = non_dominated_mask(scores)
    assert mask.tolist() == [True, True, True, False]


def test_fast_2d_matches_bruteforce():
    rng = np.random.default_rng(0)
    scores = rng.random((60, 2))
    fast = non_dominated_mask(scores)
    brute = np.ones(60, dtype=bool)
    for i in range(60):
        for j in range(60):
            if i != j and np.all(scores[j] >= scores[i]) and np.any(scores[j] > scores[i]):
                brute[i] = False
                break
    assert fast.tolist() == brute.tolist()


def test_hypervolume_2d_known_values():
    points = np.asarray([[2.0, 2.0], [3.0, 1.0]])
    assert hypervolume_2d(points, (0.0, 0.0)) == pytest.approx(5.0)
    assert hypervolume_2d(points, (10.0, 10.0)) == pytest.approx(0.0)  # below reference
    shifted = np.asarray([[12.0, 12.0], [13.0, 11.0]])
    assert hypervolume_2d(shifted, (10.0, 10.0)) == pytest.approx(5.0)
    assert hypervolume_2d(np.empty((0, 2)), (0.0, 0.0)) == 0.0


def test_hypervolume_ratio():
    reference = np.asarray([[4.0, 4.0]])
    candidate = np.asarray([[2.0, 2.0]])
    assert hypervolume_ratio(candidate, reference, (0.0, 0.0)) == pytest.approx(0.25)


def test_observation_values_and_directions():
    objective = Objective(
        name="yield",
        direction="max",
        objectives=("yield", "cost"),
        directions=("max", "min"),
    )
    observation = Observation(x={"T": 80.0}, objectives={"yield": 90.0, "cost": 12.0})
    values = observation_values(observation, objective)
    assert values.tolist() == [90.0, 12.0]
    scores = signed_scores(values, objective)
    assert scores.tolist() == [90.0, -12.0]


def test_summarise_uses_objective_ref_point():
    objective = Objective(
        name="yield",
        direction="max",
        objectives=("yield", "cost"),
        directions=("max", "min"),
        ref_point=(60.0, 30.0),
    )
    observations = [
        Observation(x={"T": 80.0}, objectives={"yield": 90.0, "cost": 12.0}),
        Observation(x={"T": 70.0}, objectives={"yield": 80.0, "cost": 10.0}),
    ]
    result = summarise(observations, objective)
    assert result.reference_point is not None
    assert result.reference_point.tolist() == [60.0, -30.0]
    assert result.hypervolume is not None and result.hypervolume > 0


def test_summarise_multi_objective():
    objective = Objective(
        name="yield", direction="max", objectives=("yield", "cost"), directions=("max", "min")
    )
    observations = [
        Observation(x={"T": 80.0}, objectives={"yield": 90.0, "cost": 12.0}),
        Observation(x={"T": 70.0}, objectives={"yield": 80.0, "cost": 10.0}),
        Observation(x={"T": 60.0}, objectives={"yield": 70.0, "cost": 12.0}),  # dominated
    ]
    result = summarise(observations, objective, reference=(60.0, -30.0))
    assert result.n_front == 2
    assert result.hypervolume is not None and result.hypervolume > 0
