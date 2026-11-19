"""A10: out-of-range measurements are rejected (or clipped with an audit record).

External review: continuous/log bounds were never validated, so a measured 150 could
be told into a ``[60, 120]`` space; every downstream ``decode`` then silently clipped
it, and the recorded condition was not the one the optimizer saw.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from condopt.base import Objective  # noqa: E402
from condopt.space import SearchSpace  # noqa: E402
from condopt.study import Study  # noqa: E402

SPACE = {
    "variables": [
        {"name": "T", "kind": "continuous", "low": 40.0, "high": 120.0, "unit": "degC"},
        {"name": "time", "kind": "log", "low": 5.0, "high": 240.0},
    ]
}


def _study(tmp_path: Path, name: str, **kwargs) -> Study:
    space = SearchSpace.from_dict(SPACE)
    return Study.create(
        tmp_path / name, space, Objective(name="y", direction="max"),
        algorithm="random", budget=10, batch_size=2, seed=0, **kwargs,
    )


def test_out_of_range_measurement_is_rejected(tmp_path):
    study = _study(tmp_path, "strict")
    pending = study.ask(2)
    bogus = {**pending[0]["x"], "T": 150.0}
    with pytest.raises(ValueError) as error:
        study.tell([{"id": pending[0]["id"], "x": bogus, "y": 1.0}])
    message = str(error.value)
    assert "outside [40.0, 120.0]" in message
    assert "clip_out_of_bounds" in message, "the error must say how to opt into clipping"


def test_clipping_is_opt_in_and_audited(tmp_path):
    study = _study(tmp_path, "clip", meta={"clip_out_of_bounds": True})
    pending = study.ask(2)
    bogus = {**pending[0]["x"], "T": 150.0, "time": 1.0}
    recorded = study.tell([{"id": pending[0]["id"], "x": bogus, "y": 1.0}])
    observation = recorded[0]
    assert observation.x["T"] == 120.0
    assert observation.x["time"] == 5.0
    assert observation.meta["bounds_clipped"]["T"] == {"original": 150.0, "clipped": 120.0}
    decisions = (study.dir / "decisions.jsonl").read_text(encoding="utf-8")
    assert "clip_out_of_bounds" in decisions


def test_one_ulp_round_trip_is_not_treated_as_out_of_range():
    space = SearchSpace.from_dict(SPACE)
    variable = space.variable("time")
    # decode() legitimately returns this for u=0
    variable.validate_value(4.999999999999999)
    variable.validate_value(240.0000000000001)
    with pytest.raises(ValueError):
        variable.validate_value(241.0)
    # encode must therefore still work at the boundary
    assert space.encode({"T": 40.0, "time": 5.0})[1] == pytest.approx(0.0)


def test_coerce_bounds_reports_the_adjustments():
    space = SearchSpace.from_dict(SPACE)
    config, adjustments = space.coerce_bounds({"T": 200.0, "time": 5.0})
    assert config["T"] == 120.0
    assert adjustments == {"T": (200.0, 120.0)}


def test_cli_exposes_the_flag():
    cli = (ROOT / "src" / "condopt" / "cli.py").read_text(encoding="utf-8")
    assert "--clip-out-of-bounds" in cli
