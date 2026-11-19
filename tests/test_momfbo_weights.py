"""A2 (backlog): momfbo must score the incumbent with the weights it fitted on.

Review finding: ``_propose`` drew Dirichlet weights, fitted the GP on that
scalarisation, then asked ``_best_scalar()`` for an incumbent computed with a *uniform*
weight vector (measured 0.9421 vs 0.525), so ``imp = mu - best`` was systematically
inflated.  ``parego``/``mfparego`` keep the two consistent; this locks momfbo too.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from condopt import registry  # noqa: E402
from condopt.base import Objective, Observation  # noqa: E402
from condopt.space import SearchSpace  # noqa: E402

pytest.importorskip("sklearn")


def _moo_prior(rng: np.random.Generator, n: int = 10) -> list[Observation]:
    space = SearchSpace.from_dict(
        [{"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0}]
    )
    return [
        Observation(
            x={"x": float(value)},
            objectives={"yield": float(value), "cost": float(1.0 - value)},
        )
        for value in rng.uniform(0.0, 1.0, size=n)
    ]


def test_momfbo_incumbent_uses_the_fitted_weights():
    rng = np.random.default_rng(0)
    space = SearchSpace.from_dict(
        [{"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0}]
    )
    objective = Objective(
        name="yield", direction="max",
        objectives=("yield", "cost"), directions=("max", "min"),
        ref_point=(0.0, 2.0),
        objective_bounds={"yield": {"low": 0.0, "high": 1.0},
                          "cost": {"low": 0.0, "high": 1.0}},
    )
    optimizer = registry.build(
        "momfbo", space, objective, params={}, seed=3, prior=_moo_prior(rng)
    )

    seen: list[np.ndarray | None] = []
    original = optimizer._best_scalar

    def spy(weights=None):
        seen.append(None if weights is None else np.asarray(weights, dtype=float))
        return original(weights)

    optimizer._best_scalar = spy  # type: ignore[method-assign]
    optimizer.ask(1)

    assert seen, "the incumbent was never scored"
    passed = seen[-1]
    assert passed is not None, (
        "momfbo scored the incumbent with the default uniform weights while the model "
        "was fitted on Dirichlet weights -- imp = mu - best is then inconsistent"
    )
    uniform = np.full(objective.n_objectives, 1.0 / objective.n_objectives)
    assert not np.allclose(passed, uniform), "expected the round's Dirichlet weights"


def test_best_scalar_still_defaults_to_uniform_for_other_callers():
    rng = np.random.default_rng(1)
    space = SearchSpace.from_dict(
        [{"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0}]
    )
    objective = Objective(
        name="yield", direction="max",
        objectives=("yield", "cost"), directions=("max", "min"),
        ref_point=(0.0, 2.0),
        objective_bounds={"yield": {"low": 0.0, "high": 1.0},
                          "cost": {"low": 0.0, "high": 1.0}},
    )
    optimizer = registry.build(
        "momfbo", space, objective, params={}, seed=3, prior=_moo_prior(rng)
    )
    assert np.isfinite(optimizer._best_scalar())
    assert np.isfinite(optimizer._best_scalar(np.array([0.8, 0.2])))
