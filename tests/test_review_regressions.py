"""Regressions for defects found by the independent reviews (2026-11-11).

Each test names the review finding it locks down, so the defect cannot come back
silently.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from condopt.base import Objective  # noqa: E402
from condopt.router import OptimizationContext, route  # noqa: E402
from condopt.space import SearchSpace  # noqa: E402
from condopt.study import Study  # noqa: E402


def test_initial_design_leaves_room_for_the_reported_algorithm():
    """Review finding: "router says tpe, but 20 Sobol points run and TPE never does"."""
    context = OptimizationContext(
        n_dims=3, n_continuous=1, n_categorical=2, feature_dim=20,
        budget_total=20, budget_remaining=20, batch_size=2,
        n_objectives=1, noise="medium",
    )
    recommendation = route(context)
    assert recommendation.algorithm == "tpe"
    design = recommendation.plan[0]
    main = recommendation.plan[-1]
    assert design["n"] < context.remaining, "the initial design consumed the whole budget"
    assert main["n"] >= 1, "the reported algorithm got zero evaluations"
    assert recommendation.params["n_initial"] <= context.remaining - 1


def test_target_score_stop_respects_a_minimised_objective():
    """Review finding: direction=min with target=0.1 and best=5.0 stopped as "reached"."""
    with tempfile.TemporaryDirectory() as tmpdir:
        space = SearchSpace.from_dict(
            [{"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0}]
        )
        study = Study.create(
            Path(tmpdir) / "min", space, Objective(name="loss", direction="min"),
            algorithm="random", budget=10, batch_size=2, seed=0,
            meta={"target_score": 0.1},
        )
        pending = study.ask(2)
        study.tell([{"id": item["id"], "y": 5.0} for item in pending])
        stop, reason = study.should_stop()
        assert stop is False, f"5.0 has not reached a 0.1 target for a minimised loss: {reason}"

        # and it *does* stop once the target is genuinely reached
        pending = study.ask(2)
        study.tell([{"id": item["id"], "y": 0.05} for item in pending])
        stop, reason = study.should_stop()
        assert stop is True, reason
        assert "0.1" in reason


def test_warm_start_grace_marker_survives_a_reload():
    """Review finding: the marker lived in memory, so the CLI lost it between commands."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run = Path(tmpdir) / "grace"
        space = SearchSpace.from_dict(
            [{"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0}]
        )
        study = Study.create(
            run, space, Objective(name="y", direction="max"),
            algorithm="random", budget=40, batch_size=2, seed=0,
        )
        for _ in range(3):
            pending = study.ask(2)
            study.tell([{"id": item["id"], "y": 1.0} for item in pending])
        study.import_observations([{"x": {"x": 0.9}, "y": 99.0}])
        reloaded = Study.load(run)
        assert reloaded._stagnation_guard_rounds() == 3
        assert (run / "warm_start.json").exists()

def test_cmaes_actually_adapts_inside_a_study():
    """Review finding: under ``Study`` cmaes sampled its frozen initial N(0.5, sigma0^2).

    The Study rebuilds the optimizer each round with ``prior=<observations>`` and then
    restores ``opt_state.json`` on top; because it never calls ``tell``, the snapshot
    kept the initial mean forever, so the router recommended cmaes for >=8-D smooth
    problems while the search was effectively random.  This asserts that the proposals
    move toward the optimum instead.
    """
    import numpy as np

    with tempfile.TemporaryDirectory() as tmpdir:
        space = SearchSpace.from_dict(
            [{"name": f"x{i}", "kind": "continuous", "low": 0.0, "high": 1.0}
             for i in range(10)]
        )
        study = Study.create(
            Path(tmpdir) / "cmaes", space, Objective(name="y", direction="min"),
            algorithm="cmaes", budget=40, batch_size=4, seed=0,
        )
        batch_means: list[float] = []
        for _ in range(8):
            pending = study.ask(4)
            batch_means.append(
                float(np.mean([value for item in pending for value in item["x"].values()]))
            )
            study.tell([
                {"id": item["id"],
                 "y": sum((value - 0.1) ** 2 for value in item["x"].values())}
                for item in pending
            ])
        first, last = batch_means[1], batch_means[-1]
        assert last < first - 0.02, (
            "cmaes proposals did not move toward the optimum inside a Study: "
            f"{[round(value, 3) for value in batch_means]}"
        )

