"""Tests for the decision layer: specs, stopping, recommendation, confirmation."""

from __future__ import annotations

import pytest

from condopt import Objective, SearchSpace, Study
from condopt.spec import Goal, Spec


def test_goal_modes_and_spec_score():
    assert Goal("y", mode="max").score(10) == pytest.approx(10.0)
    assert Goal("y", mode="min").score(10) == pytest.approx(-10.0)
    target = Goal("y", mode="target", target=80, tolerance=1.0, scale=5)
    assert target.score(80) == pytest.approx(0.0)
    assert target.score(70) == pytest.approx(-2.0)
    assert target.in_spec(80.5) is True and target.in_spec(82) is False
    # a target without tolerance is a soft preference, not a hard spec
    assert Goal("y", mode="target", target=80, scale=5).in_spec(100) is True
    range_goal = Goal("purity", mode="range", low=98, high=100)
    assert range_goal.score(99) == pytest.approx(0.0)
    assert range_goal.in_spec(99) is True and range_goal.in_spec(97) is False
    spec = Spec(goals=(target, range_goal))
    score = spec.score({"y": 80, "purity": 99})
    assert score == pytest.approx(0.0)
    assert spec.compliance({"y": 80, "purity": 99})["in_spec"] is True
    assert spec.compliance({"y": 80, "purity": 97, "cure": 20})["in_spec"] is False


def test_spec_scalar_objective_has_constraints():
    spec = Spec(
        goals=(Goal("yield", mode="target", target=80, scale=5),),
        constraints={"temp": {"op": "<=", "value": 10}},
    )
    objective = spec.scalar_objective()
    assert isinstance(objective, Objective)
    assert objective.name == "spec_score"
    assert objective.direction == "max"
    assert objective.constraints == ("temp",)
    assert objective.constraint_limits["temp"]["value"] == 10


def test_spec_calibrate_uses_robust_scale():
    spec = Spec(goals=(Goal("y", mode="target", target=80.0, scale=5.0),))
    records = [{"y": value} for value in [79.0, 80.0, 81.0, 80.0, 79.5]]
    calibrated, report = spec.calibrate(records, min_samples=3, tolerance_sigmas=2.0)
    goal = calibrated.goals[0]
    entry = report["metrics"]["y"]
    assert entry["n"] == 5
    assert entry["confidence"] == "medium"
    assert goal.scale == pytest.approx(entry["suggested_scale"])
    assert goal.tolerance == pytest.approx(2.0 * goal.scale)
    assert calibrated.constraints == spec.constraints


def make_study(tmp_path):
    space = SearchSpace.from_dict(
        [{"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0}]
    )
    spec = Spec(
        goals=(
            Goal("yield", mode="target", target=80.0, scale=5.0),
            Goal("purity", mode="range", low=98.0, high=100.0),
        ),
        constraints={"temp": {"op": "<=", "value": 10.0}},
    )
    return Study.create(
        tmp_path / "spec_run",
        space,
        spec=spec,
        budget=10,
        batch_size=2,
        seed=0,
        context=None,
        meta={"target_score": 0.0},
    )


def test_study_computes_spec_score_and_summary(tmp_path):
    study = make_study(tmp_path)
    pending = study.ask(2)
    assert len(pending) == 2
    study.tell(
        [
            {"id": pending[0]["id"], "objectives": {"yield": 80.0, "purity": 99.0},
             "constraints": {"temp": 8.0}},
            {"id": pending[1]["id"], "objectives": {"yield": 70.0, "purity": 97.0},
             "constraints": {"temp": 12.0}},
        ]
    )
    observations = study.dataset.observations
    # target at 80 -> 0; purity 99 in range -> 0  => spec score 0
    assert observations[0].y == pytest.approx(0.0)
    assert observations[0].meta["spec"]["in_spec"] is True
    summary = study.summary()
    assert summary["spec"]["in_spec"] == 1
    assert summary["spec"]["n"] == 2
    assert summary["spec"]["best"]["in_spec"] is True
    recommend = study.recommend()
    assert recommend["spec"]["in_spec"] is True
    assert recommend["condition"] == observations[0].x


def test_should_stop_on_target_and_confirm(tmp_path):
    study = make_study(tmp_path)
    pending = study.ask(1)
    study.tell(
        [{"id": pending[0]["id"], "objectives": {"yield": 80.0, "purity": 99.0},
          "constraints": {"temp": 8.0}}]
    )
    stop, reason = study.should_stop()
    assert stop is True
    assert "目标" in reason
    confirm = study.confirm(2)
    assert len(confirm) == 2
    assert all(item["purpose"] == "confirmation" for item in confirm)
    study.tell(
        [
            {"id": item["id"], "objectives": {"yield": 79.5, "purity": 99.0},
             "constraints": {"temp": 8.2}, "source": "replicate"}
            for item in confirm
        ]
    )
    recommend = study.recommend()
    assert recommend["confirmation"] is not None
    assert recommend["confirmation"]["n"] == 2
    assert recommend["confirmation"]["std"] >= 0

def test_prepare_and_cli_spec_roundtrip(tmp_path):
    import json

    from condopt.cli import main

    space = {
        "variables": [
            {"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0},
        ]
    }
    spec_payload = {
        "goals": [{"metric": "yield", "mode": "target", "target": 80, "tolerance": 1.0, "scale": 5.0}],
        "constraints": {"temp": {"op": "<=", "value": 10.0}},
    }
    answers = tmp_path / "answers.json"
    answers.write_text(
        json.dumps({"variables": space["variables"], "spec": spec_payload, "budget": 8}),
        encoding="utf-8",
    )
    prepared = tmp_path / "prepared"
    assert main(["prepare", "--answers", str(answers), "--out", str(prepared), "--json"]) == 0
    assert (prepared / "spec.json").exists()
    objective = json.loads((prepared / "objective.json").read_text(encoding="utf-8"))
    assert objective["name"] == "spec_score"
    assert objective["constraints"] == ["temp"]

    spec_file = prepared / "spec.json"
    run_dir = tmp_path / "run"
    assert main([
        "init", "--run", str(run_dir), "--space", str(prepared / "space.json"),
        "--spec", str(spec_file), "--budget", "6", "--batch", "2", "--json",
    ]) == 0
    study = Study.load(run_dir)
    pending = study.ask(2)
    study.tell(
        [
            {"id": pending[0]["id"], "objectives": {"yield": 80.0}, "constraints": {"temp": 8.0}},
            {"id": pending[1]["id"], "objectives": {"yield": 70.0}, "constraints": {"temp": 12.0}},
        ]
    )
    assert main(["recommend", "--run", str(run_dir), "--json"]) == 0
    assert main(["confirm", "--run", str(run_dir), "--n", "2", "--json"]) == 0
    assert len(Study.load(run_dir)._pending) == 2
    calibrated_path = tmp_path / "spec_calibrated.json"
    assert main(["calibrate-spec", "--run", str(run_dir), "--out", str(calibrated_path), "--json"]) == 0
    assert calibrated_path.exists()
    suggested_path = tmp_path / "spec_suggested.json"
    assert main(["suggest-spec", "--run", str(run_dir), "--out", str(suggested_path), "--json"]) == 0
    assert suggested_path.exists()


def test_next_action_reports_spec_calibration_when_ready(tmp_path):
    study = make_study(tmp_path)
    pending = study.ask(3)
    study.tell(
        [
            {"id": pending[0]["id"], "objectives": {"yield": 79.0, "purity": 99.0},
             "constraints": {"temp": 8.0}},
            {"id": pending[1]["id"], "objectives": {"yield": 80.0, "purity": 99.5},
             "constraints": {"temp": 8.5}},
            {"id": pending[2]["id"], "objectives": {"yield": 81.0, "purity": 99.2},
             "constraints": {"temp": 9.0}},
        ]
    )
    action = study.next_action()
    assert action["diagnostics"]["spec_calibration_ready"] is True
    calibration = action["diagnostics"]["spec_calibration"]
    assert calibration["report"]["metrics"]["yield"]["n"] == 3
    assert "calibrate_spec" in action["suggested_actions"]

def test_prepare_calibrates_spec_from_prior_run(tmp_path):
    import json

    from condopt.cli import main

    source = make_study(tmp_path)
    pending = source.ask(3)
    source.tell([
        {"id": item["id"], "objectives": {"yield": value, "purity": 99.0},
         "constraints": {"temp": 8.0}}
        for item, value in zip(pending, [79.0, 80.0, 81.0])
    ])
    answers = {
        "variables": [{"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0}],
        "spec": {
            "goals": [
                {"metric": "yield", "mode": "target", "target": 80.0,
                 "tolerance": 1.0, "scale": 5.0},
                {"metric": "purity", "mode": "range", "low": 98.0, "high": 100.0},
            ],
            "constraints": {"temp": {"op": "<=", "value": 10.0}},
        },
        "spec_calibrate_from": str(source.dir),
        "budget": 8,
    }
    answers_path = tmp_path / "answers_calibrated.json"
    answers_path.write_text(json.dumps(answers), encoding="utf-8")
    out = tmp_path / "prepared_calibrated"
    assert main(["prepare", "--answers", str(answers_path), "--out", str(out), "--json"]) == 0
    calibrated = json.loads((out / "spec.json").read_text(encoding="utf-8"))
    report = json.loads((out / "spec_calibration.json").read_text(encoding="utf-8"))
    yield_goal = calibrated["goals"][0]
    assert report["metrics"]["yield"]["n"] == 3
    assert yield_goal["tolerance"] != 1.0
    assert yield_goal["tolerance"] == pytest.approx(2.0 * yield_goal["scale"])
    assert report["metrics"]["yield"]["suggested_tolerance"] == pytest.approx(yield_goal["tolerance"])

def test_next_action_flags_unreachable_spec(tmp_path):
    study = make_study(tmp_path)
    pending = study.ask(5)
    study.tell([
        {"id": item["id"], "objectives": {"yield": 50.0 + index, "purity": 90.0},
         "constraints": {"temp": 8.0}}
        for index, item in enumerate(pending)
    ])
    action = study.next_action()
    assert action["diagnostics"]["spec_in_spec_count"] == 0
    assert "review_spec" in action["suggested_actions"]
    review = action["diagnostics"]["spec_review"]
    assert review["metrics"]["yield"]["target_within_observed"] is False
    assert "review_spec" in action["suggested_action_reasons"]
    assert "in-spec" in action["suggested_action_reasons"]["review_spec"]
    suggested_spec = action["diagnostics"]["spec_review"]["suggested_spec"]
    assert suggested_spec["goals"][0]["target"] == 54.0

def test_spec_achievability_flags_unreachable_target():
    spec = Spec(goals=(Goal("y", mode="target", target=80.0, tolerance=1.0, scale=5.0),))
    report = spec.achievability([{"y": 50.0}, {"y": 60.0}])
    entry = report["metrics"]["y"]
    assert entry["observed"] == [50.0, 60.0]
    assert entry["target_within_observed"] is False
    assert entry["suggested_target"] == 60.0
    assert entry["in_spec_count"] == 0

def test_spec_suggest_moves_unreachable_target_into_observed_range():
    spec = Spec(goals=(Goal("y", mode="target", target=80.0, tolerance=1.0, scale=5.0),))
    suggested, report = spec.suggest_spec([{"y": 50.0}, {"y": 60.0}])
    assert spec.goals[0].target == 80.0
    assert suggested.goals[0].target == 60.0
    assert report["changed"][0]["kind"] == "target"
    assert report["achievability"]["metrics"]["y"]["target_within_observed"] is False

def test_spec_worst_case_aggregation_dominated_by_weakest_goal():
    good = Goal("a", mode="target", target=10.0, tolerance=1.0, scale=1.0)
    bad = Goal("b", mode="target", target=10.0, tolerance=1.0, scale=1.0)
    weighted = Spec(goals=(good, bad), aggregation="weighted_sum")
    robust = Spec(goals=(good, bad), aggregation="worst_case")
    values = {"a": 10.0, "b": 8.0}
    assert weighted.score(values) == pytest.approx(-1.0)
    assert robust.score(values) == pytest.approx(-2.0)
    assert robust.scalar_objective().name == "spec_score"

def test_study_recommend_respects_worst_case_aggregation(tmp_path):
    space = SearchSpace.from_dict(
        [{"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0}]
    )
    spec = Spec(
        goals=(
            Goal("a", mode="target", target=10.0, tolerance=1.0, scale=1.0),
            Goal("b", mode="target", target=10.0, tolerance=1.0, scale=1.0),
        ),
        aggregation="worst_case",
    )
    study = Study.create(
        tmp_path / "worst_case_run",
        space,
        spec=spec,
        algorithm="lhs",
        budget=8,
        batch_size=2,
        seed=0,
    )
    study.import_observations([
        {"x": {"x": 0.2}, "objectives": {"a": 9.5, "b": 5.0}},
        {"x": {"x": 0.8}, "objectives": {"a": 7.0, "b": 7.0}},
    ])
    observations = study.dataset.observations
    assert observations[0].y == pytest.approx(-5.0)
    assert observations[1].y == pytest.approx(-3.0)
    recommendation = study.recommend()
    assert recommendation["condition"] == {"x": 0.8}
    assert recommendation["score"] == pytest.approx(-3.0)
    assert study.spec.aggregation == "worst_case"

def test_worst_case_spec_review_and_suggestion(tmp_path):
    space = SearchSpace.from_dict(
        [{"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0}]
    )
    spec = Spec(
        goals=(
            Goal("a", mode="target", target=10.0, tolerance=1.0, scale=1.0),
            Goal("b", mode="target", target=100.0, tolerance=2.0, scale=10.0),
        ),
        aggregation="worst_case",
    )
    study = Study.create(
        tmp_path / "worst_case_review",
        space,
        spec=spec,
        algorithm="lhs",
        budget=10,
        batch_size=2,
        seed=0,
    )
    study.import_observations([
        {"x": {"x": index / 10.0}, "objectives": {"a": 10.0, "b": 50.0 + index}}
        for index in range(5)
    ])
    action = study.next_action()
    assert action["diagnostics"]["spec_in_spec_count"] == 0
    assert "review_spec" in action["suggested_actions"]
    review = action["diagnostics"]["spec_review"]["metrics"]
    assert review["a"]["target_within_observed"] is True
    assert review["b"]["target_within_observed"] is False

    suggested, report = study.spec.suggest_spec([
        study.spec.observation_values(obs) for obs in study.dataset.observations
    ])
    assert suggested.aggregation == "worst_case"
    assert suggested.goals[1].target == 54.0
    assert any(change["metric"] == "b" for change in report["changed"])

def test_calibrate_by_group_ignores_between_plate_shift():
    spec = Spec(goals=(Goal("y", mode="target", target=80.0, tolerance=1.0, scale=5.0),))
    records = [
        {"y": 80.0}, {"y": 80.2}, {"y": 79.8},
        {"y": 90.0}, {"y": 90.2}, {"y": 89.8},
    ]
    groups = ["P1", "P1", "P1", "P2", "P2", "P2"]
    global_calibrated, _ = spec.calibrate(records, min_samples=3, tolerance_sigmas=2.0)
    grouped, report = spec.calibrate_by_group(records, groups, min_samples=3, tolerance_sigmas=2.0)
    assert grouped.goals[0].tolerance < global_calibrated.goals[0].tolerance
    assert set(report["metrics"]["y"]["groups"]) == {"P1", "P2"}
    assert report["metrics"]["y"]["groups"]["P1"]["n"] == 3
    assert report["metrics"]["y"]["global_scale"] > report["metrics"]["y"]["suggested_scale"]

def test_suggest_spec_by_group_keeps_tolerance_small():
    spec = Spec(goals=(Goal("y", mode="target", target=80.0, tolerance=1.0, scale=5.0),))
    records = [
        {"y": 80.0}, {"y": 80.2}, {"y": 79.8},
        {"y": 90.0}, {"y": 90.2}, {"y": 89.8},
    ]
    groups = ["P1", "P1", "P1", "P2", "P2", "P2"]
    global_suggested, _ = spec.suggest_spec(records)
    grouped_suggested, report = spec.suggest_spec(records, groups=groups)
    assert grouped_suggested.goals[0].tolerance < global_suggested.goals[0].tolerance
    assert "groups" in report["calibration"]["metrics"]["y"]

