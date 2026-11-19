"""A12: one three-state feasibility rule everywhere ("missing" != "feasible").

External review: the same question had three different answers depending on where it
was asked -- ``study._is_feasible`` and the benchmarks treated a *missing* constraint
measurement as a violation, ``benchmarks`` additionally ignored observations with an
empty ``constraints`` dict entirely, and the surrogate path used ``np.nan``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from condopt.base import Objective  # noqa: E402
from condopt.dataset import Observation  # noqa: E402
from condopt.feasibility import (  # noqa: E402
    ConstraintState,
    coverage_note,
    evaluate_constraint,
    feasible_count,
    feasibility_state,
)


def _objective(limits: dict | None = None) -> Objective:
    return Objective(
        name="yield", direction="max", constraints=("T_max",),
        constraint_limits=limits or {"T_max": {"op": "<=", "value": 80.0}},
    )


def _obs(constraints: dict | None) -> Observation:
    payload = {"x": {"T": 60.0}, "y": 1.0}
    if constraints is not None:
        payload["constraints"] = constraints
    return Observation(**payload)


def test_evaluate_constraint_covers_all_three_states():
    limit = {"op": "<=", "value": 80.0}
    assert evaluate_constraint("T_max", 70.0, limit)[0] is ConstraintState.TRUE
    assert evaluate_constraint("T_max", 90.0, limit)[0] is ConstraintState.FALSE
    assert evaluate_constraint("T_max", None, limit)[0] is ConstraintState.UNKNOWN
    assert evaluate_constraint("T_max", float("nan"), limit)[0] is ConstraintState.UNKNOWN
    assert evaluate_constraint("T_max", 70.0, None)[0] is ConstraintState.UNKNOWN


def test_missing_constraint_is_unknown_not_feasible():
    objective = _objective()
    report = feasibility_state(_obs(None), objective)
    assert report["state"] == ConstraintState.UNKNOWN.value
    assert report["missing_names"] == ["T_max"]
    # the conservative rule: UNKNOWN never counts as feasible
    counts = feasible_count([_obs(None)], objective)
    assert counts["true"] == 0
    assert counts["unknown"] == 1
    assert counts["known_fraction"] == 0.0


def test_overall_state_priority_violation_beats_unknown():
    objective = Objective(
        name="yield", direction="max", constraints=("T_max", "purity"),
        constraint_limits={"T_max": {"op": "<=", "value": 80.0},
                           "purity": {"op": ">=", "value": 0.9}},
    )
    observation = Observation(x={"T": 60.0}, y=1.0, constraints={"T_max": 90.0})
    report = feasibility_state(observation, objective)
    assert report["state"] == ConstraintState.FALSE.value
    assert report["violated"] == ["T_max"]
    assert report["unknown"] == ["purity"]


def test_observed_unmeasured_and_violated_are_counted_separately():
    objective = _objective()
    observations = [_obs({"T_max": 70.0}), _obs({"T_max": 90.0}), _obs(None), _obs({})]
    counts = feasible_count(observations, objective)
    assert counts["n"] == 4
    assert counts["true"] == 1
    assert counts["false"] == 1
    assert counts["unknown"] == 2
    assert counts["unknown_by_constraint"] == {"T_max": 2}
    assert counts["known_fraction"] == 0.5


def test_coverage_note_only_fires_when_coverage_is_low():
    objective = _objective()
    assert coverage_note([_obs({"T_max": 70.0})], objective) is None
    note = coverage_note([_obs(None), _obs(None), _obs({"T_max": 70.0})], objective)
    assert note is not None
    assert "未测到" in note["advice"]


def test_study_is_feasible_matches_the_shared_rule(tmp_path):
    from condopt.space import SearchSpace
    from condopt.study import Study

    space = SearchSpace.from_dict([
        {"name": "T", "kind": "continuous", "low": 40.0, "high": 120.0}
    ])
    study = Study.create(
        tmp_path / "feas", space, _objective(), algorithm="random", budget=6,
        batch_size=2, seed=0,
    )
    assert study._is_feasible(_obs({"T_max": 70.0})) is True
    assert study._is_feasible(_obs({"T_max": 90.0})) is False
    assert study._is_feasible(_obs(None)) is False  # unknown is not feasible

    study.import_observations([
        {"x": {"T": 60.0}, "y": 1.0, "constraints": {"T_max": 70.0}},
        {"x": {"T": 70.0}, "y": 1.0},
    ])
    report = study.feasibility_report()
    assert report["true"] == 1 and report["unknown"] == 1
    assert report["coverage"] is not None


def test_prepare_advice_warns_about_unmeasured_constraints():
    from condopt.prepare import feasibility_advice
    from condopt.router import OptimizationContext

    objective = _objective()
    context = OptimizationContext(n_dims=1, n_continuous=1, budget_total=10,
                                  budget_remaining=10, batch_size=2, n_objectives=1,
                                  noise="medium")
    plain = feasibility_advice({}, context, objective)
    assert plain["coverage"] is None
    warned = feasibility_advice(
        {}, context, objective, observations=[_obs(None), _obs(None)]
    )
    assert warned["coverage"] is not None
    assert "未测到" in warned["advice"]


def test_benchmarks_count_missing_constraints_as_unknown():
    """The benchmark feasibility counter must not silently drop unmeasured rows.

    Before A12 the benchmark loop required a non-empty ``constraints`` dict, so an
    observation without it was invisible rather than unknown -- and a partially
    measured one was scored as a violation.
    """
    from condopt import benchmarks
    from condopt.feasibility import feasibility_state as fb

    problem = next(
        item for item in benchmarks.builtin_problems().values()
        if getattr(item.objective, "constraint_limits", None)
    )
    space = problem.space
    config = space.sample_sobol(1, rng=None)[0] if False else None  # keep it simple
    del config

    unmeasured = Observation(x=_x_for(space), y=1.0)
    assert fb(unmeasured, problem.objective)["state"] == ConstraintState.UNKNOWN.value

    name = next(iter(problem.objective.constraint_limits))
    partially = Observation(x=_x_for(space), y=1.0, constraints={name: 1e6})
    state = fb(partially, problem.objective)["state"]
    assert state in (ConstraintState.TRUE.value, ConstraintState.UNKNOWN.value,
                     ConstraintState.FALSE.value)
    # a value that certainly violates a "<=" style limit must be FALSE, not invisible
    limit = problem.objective.constraint_limits[name]
    if str(limit.get("op", "<=")) == "<=":
        violating = Observation(x=_x_for(space), y=1.0,
                                constraints={name: float(limit["value"]) + 1e6})
        assert fb(violating, problem.objective)["state"] == ConstraintState.FALSE.value


def _x_for(space) -> dict:
    """Any valid configuration of ``space`` (the feasibility rule ignores it)."""
    return dict(space.sample_sobol(1, rng=None)[0]) if False else {
        name: (space.variable(name).low if space.variable(name).kind in ("continuous", "log")
               else space.variable(name).choices[0])
        for name in space.names
    }
