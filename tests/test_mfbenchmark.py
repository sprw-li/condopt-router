"""Tests for the multi-fidelity benchmark (`mf_suzuki`) and mfbo on it."""

import pytest

from condopt import Observation
from condopt.benchmarks import cost_aware_best, mf_suzuki, mf_suzuki_problem
from condopt.registry import build


def test_mf_suzuki_optima():
    problem = mf_suzuki_problem()
    best = problem.best_config
    assert mf_suzuki(best, 1.0) == pytest.approx(110.0)
    assert mf_suzuki(best, 0.5) == pytest.approx(110.0 - 15.0 * (0.5**1.5), abs=1e-9)
    assert mf_suzuki(best, 0.25) < mf_suzuki(best, 0.5) < mf_suzuki(best, 1.0)


def test_single_fidelity_optimizers_report_full_fidelity():
    problem = mf_suzuki_problem()
    optimizer = build("bayes", problem.space, problem.objective, seed=0)
    pairs = optimizer.ask_with_fidelity(2)
    assert optimizer.supports_fidelity is False
    assert [fidelity for _, fidelity in pairs] == [1.0, 1.0]


def test_mfbo_uses_cheap_fidelities_and_finds_good_full_fidelity():
    problem = mf_suzuki_problem()
    optimizer = build(
        "mfbo",
        problem.space,
        problem.objective,
        params={"n_initial": 4, "n_candidates": 256, "fidelity_levels": "0.25,0.5,1.0"},
        seed=0,
    )
    observations = []
    for _ in range(6):
        for config, fidelity in optimizer.ask_with_fidelity(4):
            value = mf_suzuki(config, fidelity)
            observation = Observation(x=config, y=value, fidelity=fidelity, cost=fidelity)
            observations.append(observation)
            optimizer.tell([observation])
    stats = cost_aware_best(observations)
    assert stats["best_full_fidelity"] is not None
    assert stats["best_full_fidelity"] > 90.0
    # mfbo should spend less than 24 full-fidelity-equivalents
    assert stats["total_cost"] < 24.0
    assert any(obs.fidelity < 1.0 for obs in observations)
