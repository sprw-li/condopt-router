"""Tests for calibrated router weights."""

import json

import pytest

from condopt import SearchSpace
from condopt.cli import main
from condopt.router import (
    OptimizationContext,
    algorithm_weights,
    route,
    set_algorithm_weights,
)


def smooth_context() -> OptimizationContext:
    return OptimizationContext(
        n_dims=5, n_continuous=5, budget_total=40, smoothness="smooth"
    )


def test_weights_override_default_choice():
    ctx = smooth_context()
    assert route(ctx).algorithm == "bayes"
    try:
        set_algorithm_weights({"random": 10.0})
        assert route(ctx).algorithm == "random"
        assert route(ctx).candidate_scores["random"] > route(ctx).candidate_scores["bayes"]
    finally:
        set_algorithm_weights(None)
    assert route(ctx).algorithm == "bayes"
    assert algorithm_weights() == {}


def test_weight_validation():
    with pytest.raises(ValueError, match="unknown algorithm"):
        set_algorithm_weights({"no-such-algorithm": 2.0})
    with pytest.raises(ValueError, match=">= 0"):
        set_algorithm_weights({"bayes": -1.0})
    set_algorithm_weights(None)


def test_cli_route_with_weights_file(tmp_path, capsys):
    space = SearchSpace.from_dict(
        [{"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0}]
    )
    space_path = tmp_path / "space.json"
    space.to_json(str(space_path))
    weights_path = tmp_path / "weights.json"
    weights_path.write_text(
        json.dumps({"weights": {"random": 100.0}}), encoding="utf-8"
    )
    try:
        code = main(
            [
                "route",
                "--space",
                str(space_path),
                "--budget",
                "40",
                "--weights",
                str(weights_path),
                "--json",
            ]
        )
        assert code == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["algorithm"] == "random"
    finally:
        set_algorithm_weights(None)

def test_calibration_warns_on_single_seed_labels():
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "calibrate_router.py"
    spec = importlib.util.spec_from_file_location("calibrate_router", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert module.seed_warning([0]) is not None
    assert "单 seed" in module.seed_warning([0])
    assert module.seed_warning([0, 1]) is None

def test_calibration_defaults_to_two_seeds():
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "calibrate_router.py"
    source = script.read_text(encoding="utf-8")
    assert "default=[0, 1]" in source
    assert "单 seed 标签噪声大" in source

