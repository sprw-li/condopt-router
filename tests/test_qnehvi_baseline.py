"""A5: qNEHVI's baseline must come from the model posterior, with per-objective noise.

External review: "baseline posterior = False" -- the nested MC perturbed the observed
labels with i.i.d. noise instead of sampling the posterior at ``X_baseline``, and a
single ``objective_std`` was copied onto both objectives.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from condopt import registry  # noqa: E402
from condopt.base import Objective, Observation  # noqa: E402
from condopt.space import SearchSpace  # noqa: E402


def _optimizer(seed: int = 0):
    space = SearchSpace.from_dict(
        [{"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0},
         {"name": "z", "kind": "continuous", "low": 0.0, "high": 1.0}]
    )
    objective = Objective(
        name="yield", direction="max",
        objectives=("yield", "cost"), directions=("max", "min"),
        ref_point=(0.0, 2.0),
        objective_bounds={"yield": {"low": 0.0, "high": 1.0},
                          "cost": {"low": 0.0, "high": 1.0}},
    )
    optimizer = registry.build("qnehvi", space, objective, params={}, seed=seed)
    rng = np.random.default_rng(seed)
    observations = []
    for _ in range(8):
        x = float(rng.uniform(0.0, 1.0))
        z = float(rng.uniform(0.0, 1.0))
        yield_value = 0.2 + 0.6 * x
        cost = 0.4 + 1.2 * z
        observations.append(
            Observation(
                x={"x": x, "z": z},
                objectives={"yield": yield_value, "cost": cost},
                objectives_std={"yield": 0.08, "cost": 0.01},
            )
        )
    optimizer.tell(observations)
    return optimizer


def test_baseline_is_sampled_from_the_model_at_the_observed_points():
    optimizer = _optimizer()
    X = np.asarray(optimizer._X, dtype=float).copy()
    calls: list[np.ndarray] = []
    original = optimizer._sample_posterior_multi

    def spy(model, features, n_draws):
        calls.append(np.asarray(features, dtype=float))
        return original(model, features, n_draws)

    optimizer._sample_posterior_multi = spy  # type: ignore[method-assign]
    optimizer.ask(1)

    assert calls, "no posterior sampling happened at all"
    baseline_calls = [
        features for features in calls
        if features.shape == X.shape and np.allclose(features, X)
    ]
    assert baseline_calls, (
        "the baseline was not sampled from the model at X_baseline -- the nested MC "
        "must not fall back to perturbing the observed labels"
    )
    assert len(baseline_calls) >= 2, "both objectives need a baseline posterior sample"


def test_per_objective_noise_is_used():
    optimizer = _optimizer()
    # yield std 0.08, cost std 0.01 -- a scalar-only implementation would give [s, s]
    assert optimizer._noise_std.shape[1] == 2
    assert not np.allclose(optimizer._noise_std[:, 0], optimizer._noise_std[:, 1])
    assert np.allclose(optimizer._noise_std[:, 0], 0.08)
    assert np.allclose(optimizer._noise_std[:, 1], 0.01)


def test_scalar_objective_std_still_fills_both_objectives():
    optimizer = _optimizer()
    optimizer._noise_std = np.zeros((0, 2))
    optimizer._Y = np.zeros((0, 2))
    optimizer.tell([
        Observation(x={"x": 0.3, "z": 0.3},
                    objectives={"yield": 0.4, "cost": 1.0}, objective_std=0.05)
    ])
    assert np.allclose(optimizer._noise_std[0], [0.05, 0.05])
