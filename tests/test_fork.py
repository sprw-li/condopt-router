"""Tests for fork: create a calibrated/suggested spec campaign from history."""

from __future__ import annotations

import json

import pytest

from condopt import SearchSpace, Study
from condopt.cli import main
from condopt.spec import Goal, Spec


def make_old(tmp_path):
    space = SearchSpace.from_dict(
        [{"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0}]
    )
    spec = Spec(goals=(Goal("y", mode="target", target=80.0, tolerance=1.0, scale=5.0),))
    study = Study.create(
        tmp_path / "old", space, spec=spec, budget=10, batch_size=2, seed=0
    )
    study.import_observations([
        {"x": {"x": 0.1}, "objectives": {"y": 50.0}},
        {"x": {"x": 0.5}, "objectives": {"y": 60.0}},
        {"x": {"x": 0.9}, "objectives": {"y": 70.0}},
    ])
    return study


def test_fork_suggested_spec_and_warm_start(tmp_path, capsys):
    old = make_old(tmp_path)
    new_dir = tmp_path / "new"
    assert main([
        "fork", "--run", str(old.dir), "--new-run", str(new_dir),
        "--spec-mode", "suggested", "--top-k", "2", "--json",
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["imported"] == 2
    assert new_dir.exists()
    assert (new_dir / "fork_report.json").exists()
    study = Study.load(new_dir)
    assert study.spec.goals[0].target == 70.0
    assert study.meta["forked_from"] == str(old.dir)
    assert len(study.dataset.observations) == 2
    report = study.report()
    assert "fork from" in report
    assert "规格改动" in report
    action = study.next_action()
    assert action["diagnostics"]["forked_from"] == str(old.dir)
    assert action["diagnostics"]["fork_spec_mode"] == "suggested"


def test_fork_rejects_existing_target(tmp_path):
    old = make_old(tmp_path)
    with pytest.raises(SystemExit, match="already exists"):
        main([
            "fork", "--run", str(old.dir), "--new-run", str(old.dir), "--json",
        ])

def test_fork_overrides_budget_and_batch(tmp_path):
    old = make_old(tmp_path)
    new_dir = tmp_path / "new_overrides"
    assert main([
        "fork", "--run", str(old.dir), "--new-run", str(new_dir),
        "--spec-mode", "calibrated", "--top-k", "0", "--budget", "12",
        "--batch-size", "3", "--json",
    ]) == 0
    study = Study.load(new_dir)
    assert study.budget == 12
    assert study.batch_size == 3
    assert len(study.dataset.observations) == 0

def test_fork_uses_external_spec_file(tmp_path):
    old = make_old(tmp_path)
    spec_file = tmp_path / "external_spec.json"
    spec_file.write_text(json.dumps({
        "goals": [{"metric": "y", "mode": "target", "target": 60.0,
                   "tolerance": 1.0, "scale": 5.0}],
        "constraints": {},
    }), encoding="utf-8")
    new_dir = tmp_path / "new_file_spec"
    assert main([
        "fork", "--run", str(old.dir), "--new-run", str(new_dir),
        "--spec-file", str(spec_file), "--top-k", "0", "--json",
    ]) == 0
    study = Study.load(new_dir)
    assert study.spec.goals[0].target == 60.0
    assert study.meta["fork_spec_mode"] == "file"

