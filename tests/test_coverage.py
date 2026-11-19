"""Routing coverage regression guard.

Every capability combination must have a runnable recommendation, and the
multi-fidelity multi-objective regime must keep at least two candidates so a
single algorithm cannot become a single point of failure.
"""

from __future__ import annotations

import itertools

import pytest

from condopt import registry
from condopt.router import OptimizationContext, _required_capabilities, route


def _contexts():
    for n_dims, n_objectives, n_constraints, multi_fidelity, noise in itertools.product(
        (2, 6, 12), (1, 2, 3), (0, 1), (False, True), ("low", "high")
    ):
        yield OptimizationContext(
            n_dims=n_dims,
            n_continuous=n_dims,
            budget_total=40,
            batch_size=4,
            n_objectives=n_objectives,
            n_constraints=n_constraints,
            multi_fidelity=multi_fidelity,
            noise=noise,
            smoothness="smooth",
        )


def test_every_context_has_a_capable_recommendation():
    for context in _contexts():
        recommendation = route(context)
        required = _required_capabilities(context)
        capabilities = set(registry.spec(recommendation.algorithm).capabilities)
        assert required <= capabilities, (
            context, recommendation.algorithm, sorted(required - capabilities)
        )
        recommendation.validate()


def test_multi_fidelity_multi_objective_has_at_least_two_candidates():
    for context in _contexts():
        if not context.multi_fidelity or context.n_objectives < 2:
            continue
        required = _required_capabilities(context)
        candidates = [
            name
            for name in registry.names()
            if required <= set(registry.spec(name).capabilities)
        ]
        assert len(candidates) >= 2, (context, candidates)
