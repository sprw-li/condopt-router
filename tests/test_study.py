"""Tests for the persistent study workflow and the CLI."""

import json

import pytest

from condopt import Objective, SearchSpace, Study
from condopt.cli import main
from condopt.study import PendingError


def simple_space() -> SearchSpace:
    return SearchSpace.from_dict(
        [
            {"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0},
            {"name": "y", "kind": "continuous", "low": 0.0, "high": 1.0},
        ]
    )


def make_study(tmp_path, budget=10, batch_size=2) -> Study:
    return Study.create(
        tmp_path / "run",
        simple_space(),
        Objective(name="yield", direction="max"),
        budget=budget,
        batch_size=batch_size,
        seed=0,
    )


def test_study_lifecycle(tmp_path):
    study = make_study(tmp_path)
    run_dir = tmp_path / "run"
    assert (run_dir / "problem.json").exists()
    assert (run_dir / "config.json").exists()
    assert study.algorithm == "bayes"

    pending = study.ask()
    assert len(pending) == 2
    assert pending[0]["algorithm"] == "bayes"
    assert (run_dir / "suggestions.csv").exists()

    with pytest.raises(PendingError):
        study.ask()

    observations = study.tell([{"id": item["id"], "y": 0.1 * i} for i, item in enumerate(pending)])
    assert len(observations) == 2
    assert len(study.dataset) == 2
    assert study.best() is not None
    assert study.best().y == pytest.approx(0.1)
    assert study.rounds[-1]["n"] == 2

    second = study.ask(2)
    assert len(second) == 2
    assert study.remaining() == 6


def test_study_reload_preserves_state(tmp_path):
    study = make_study(tmp_path)
    pending = study.ask()
    study.tell([{"id": item["id"], "y": float(i)} for i, item in enumerate(pending)])
    loaded = Study.load(tmp_path / "run")
    assert loaded.algorithm == study.algorithm
    assert loaded.params == study.params
    assert len(loaded.dataset) == 2
    assert loaded.rounds == study.rounds


def test_study_override_is_logged_and_sticks(tmp_path):
    study = make_study(tmp_path)
    with pytest.raises(ValueError):
        study.override("random", None, reason="   ")
    study.override("random", {"dedupe": True}, reason="测试人工覆写")
    assert study.algorithm == "random"
    assert study.manual_override is True
    decisions = study.decisions()
    assert decisions[-1]["action"] == "override"
    assert "测试人工覆写" in decisions[-1]["reason"]
    # the override survives the next auto-route pass
    pending = study.ask(2)
    assert all(item["algorithm"] == "random" for item in pending)
    study.tell([{"id": item["id"], "y": float(i)} for i, item in enumerate(pending)])

    reloaded = Study.load(tmp_path / "run")
    assert reloaded.algorithm == "random"
    assert reloaded.manual_override is True
    # explicit resume re-enables the router
    result = reloaded.resume_auto_route()
    assert result["manual_override"] is False
    assert Study.load(tmp_path / "run").manual_override is False


def test_partial_tell_keeps_unmatched_pending(tmp_path):
    study = make_study(tmp_path, budget=10, batch_size=4)
    pending = study.ask(4)
    first_two = pending[:2]
    observations = study.tell([{"id": item["id"], "y": float(i)} for i, item in enumerate(first_two)])
    assert len(observations) == 2
    assert len(study.pending) == 2
    assert [item["id"] for item in study.pending] == [item["id"] for item in pending[2:]]
    assert len(study.dataset) == 2
    with pytest.raises(PendingError):
        study.ask()
    # finish the batch
    study.tell([{"id": item["id"], "y": 1.0} for item in study.pending])
    assert study.pending == []
    assert len(study.dataset) == 4


def test_repeated_tell_is_idempotent(tmp_path):
    study = make_study(tmp_path)
    pending = study.ask(2)
    results = [{"id": item["id"], "y": float(i)} for i, item in enumerate(pending)]
    study.tell(results)
    assert len(study.dataset) == 2
    assert len(study.rounds) == 1
    # replaying the same results must not duplicate observations or rounds
    replayed = study.tell(results)
    assert replayed == []
    assert len(study.dataset) == 2
    assert len(study.rounds) == 1


def test_study_respects_budget(tmp_path):
    study = make_study(tmp_path, budget=4, batch_size=2)
    pending = study.ask(2)
    study.tell([{"id": item["id"], "y": float(i)} for i, item in enumerate(pending)])
    with pytest.raises(ValueError):
        study.ask(4)


def test_import_prior_observations(tmp_path):
    study = make_study(tmp_path, budget=10, batch_size=2)
    imported = study.import_observations(
        [
            {"x": {"x": 0.1, "y": 0.2}, "y": 1.0, "source": "manual"},
            {"x": {"x": 0.4, "y": 0.8}, "y": 2.0, "source": "manual"},
        ]
    )
    assert len(imported) == 2
    assert len(study.dataset) == 2
    assert study.remaining() == 8
    assert study.best().y == pytest.approx(2.0)
    # warm-started BO should ask without needing an initial design again
    pending = study.ask(2)
    assert len(pending) == 2


def test_best_prefers_full_fidelity_and_remaining_cost(tmp_path):
    from condopt import Study as StudyClass

    space = simple_space()
    study = StudyClass.create(
        tmp_path / "mf_best",
        space,
        Objective(name="y", direction="max"),
        algorithm="mfbo",
        budget=6,
        batch_size=2,
        meta={"cost_budget": 2.0},
        seed=0,
    )
    pending = study.ask(2)
    study.tell(
        [
            {"id": pending[0]["id"], "y": 99.0, "fidelity": 0.25, "cost": 0.25},
            {"id": pending[1]["id"], "y": 80.0, "fidelity": 1.0, "cost": 1.0},
        ]
    )
    assert study.best().y == pytest.approx(80.0)
    assert study.cost_spent() == pytest.approx(1.25)
    assert study.remaining_cost() == pytest.approx(0.75)


def test_cost_spent_tracks_observation_costs(tmp_path):
    study = make_study(tmp_path, budget=6, batch_size=2)
    pending = study.ask(2)
    study.tell([{"id": item["id"], "y": 1.0, "cost": 2.5} for item in pending])
    assert study.cost_spent() == pytest.approx(5.0)
    assert study.summary()["cost_spent"] == pytest.approx(5.0)
    assert study.summary()["remaining_cost"] is None


def test_create_cost_budget_parameter(tmp_path):
    study = Study.create(
        tmp_path / "budget_param",
        simple_space(),
        Objective(name="y", direction="max"),
        budget=6,
        batch_size=2,
        cost_budget=2.5,
    )
    assert study.remaining_cost() == pytest.approx(2.5)


def test_cli_init_cost_budget(tmp_path):
    space_path = tmp_path / "space.json"
    simple_space().to_json(str(space_path))
    run_dir = tmp_path / "cost_budget_run"
    code = main(
        [
            "init",
            "--run",
            str(run_dir),
            "--space",
            str(space_path),
            "--budget",
            "6",
            "--batch",
            "2",
            "--cost-budget",
            "2.5",
            "--json",
        ]
    )
    assert code == 0
    loaded = Study.load(run_dir)
    assert loaded.remaining_cost() == pytest.approx(2.5)


def test_explicit_algorithm_is_pinned(tmp_path):
    study = Study.create(
        tmp_path / "pinned",
        simple_space(),
        Objective(name="y", direction="max"),
        algorithm="mfbo",
        budget=8,
        batch_size=2,
        seed=0,
    )
    assert study.algorithm == "mfbo"
    assert study.manual_override is True
    pending = study.ask(2)
    assert all(item["algorithm"] == "mfbo" for item in pending)
    reloaded = Study.load(tmp_path / "pinned")
    assert reloaded.algorithm == "mfbo"
    assert reloaded.manual_override is True
    assert any(decision["action"] == "manual_selection" for decision in reloaded.decisions())


def test_parse_objective_shorthand():
    from condopt.cli import _parse_objective

    objective = _parse_objective("yield:max")
    assert objective.name == "yield"
    assert objective.direction == "max"
    objective = _parse_objective("loss:min")
    assert objective.direction == "min"


def test_read_results_accepts_bom(tmp_path):
    from condopt.cli import _read_results

    path = tmp_path / "results.json"
    path.write_text("\ufeff" + json.dumps([{"id": "r1-1", "y": 1.0}]), encoding="utf-8")
    assert _read_results(str(path)) == [{"id": "r1-1", "y": 1.0}]


def test_report_contains_key_sections(tmp_path):
    study = make_study(tmp_path)
    study.ask()
    text = study.report()
    assert "Study:" in text
    assert "待测建议" in text
    assert "下一轮建议" in text


def test_cli_end_to_end(tmp_path, capsys):
    space_path = tmp_path / "space.json"
    simple_space().to_json(str(space_path))
    run_dir = tmp_path / "cli-run"

    assert main(["init", "--run", str(run_dir), "--space", str(space_path), "--budget", "20", "--batch", "2", "--json"]) == 0
    init_payload = json.loads(capsys.readouterr().out)
    assert init_payload["algorithm"] == "bayes"

    assert main(["ask", "--run", str(run_dir), "--n", "2", "--json"]) == 0
    pending = json.loads(capsys.readouterr().out)
    assert len(pending) == 2

    results_path = tmp_path / "results.json"
    results_path.write_text(
        json.dumps([{"id": item["id"], "y": 0.5 * i} for i, item in enumerate(pending)]),
        encoding="utf-8",
    )
    assert main(["tell", "--run", str(run_dir), "--results", str(results_path), "--json"]) == 0
    tell_payload = json.loads(capsys.readouterr().out)
    assert tell_payload["n"] == 2
    assert tell_payload["remaining"] == 18

    assert main(["report", "--run", str(run_dir), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["used"] == 2
    assert report["router"]["next"]["algorithm"]


def test_cli_route_and_spec(tmp_path, capsys):
    space_path = tmp_path / "space.json"
    simple_space().to_json(str(space_path))
    assert main(["route", "--space", str(space_path), "--budget", "30", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["algorithm"] in ("bayes", "sobol", "lhs", "random", "grid")
    assert payload["params"]

    assert main(["spec", "--algorithm", "bayes", "--json"]) == 0
    spec_payload = json.loads(capsys.readouterr().out)
    assert spec_payload["name"] == "bayes"
    assert any(param["name"] == "acquisition" for param in spec_payload["params"])

    assert main(["spec", "--json"]) == 0
    all_specs = json.loads(capsys.readouterr().out)
    assert len(all_specs) >= 5

def test_exploration_reroute_returns_to_main_algorithm(tmp_path):
    space = simple_space()
    study = Study.create(
        tmp_path / "reroute",
        space,
        Objective(name="y", direction="max"),
        budget=24,
        batch_size=2,
        seed=0,
        patience=2,
    )
    seen = []
    for _ in range(10):
        pending = study.ask(2)
        seen.append(pending[0]["algorithm"])
        study.tell([{"id": item["id"], "y": 1.0} for item in pending])
    actions = [decision["action"] for decision in study.decisions()]
    assert "auto_reroute" in actions
    assert "resume_route" in actions
    assert "sobol" in seen
    assert seen[-1] != "sobol"

def test_cli_tell_accepts_measurement_csv(tmp_path):
    from condopt.cli import main

    study = Study.create(
        tmp_path / "csv_run",
        SearchSpace.from_dict([{"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0}]),
        Objective(name="y", direction="max"),
        algorithm="lhs",
        budget=8,
        batch_size=2,
        seed=0,
    )
    pending = study.ask(2)
    csv_path = tmp_path / "measurements.csv"
    csv_path.write_text(
        "id,x,y,plate,well\n"
        f"{pending[0]['id']},0.1,10.5,P1,A1\n"
        f"{pending[1]['id']},0.9,12.5,P1,A2\n",
        encoding="utf-8",
    )
    assert main(["tell", "--run", str(study.dir), "--results", str(csv_path), "--json"]) == 0
    observations = study.dataset.observations
    assert len(observations) == 2
    assert observations[0].meta["plate"] == "P1"
    assert observations[1].y == 12.5

def test_cli_tell_accepts_prefixed_csv(tmp_path):
    from condopt.cli import main
    from condopt.spec import Goal, Spec

    spec = Spec(
        goals=(Goal("yield", mode="target", target=80.0, tolerance=1.0, scale=5.0),),
        constraints={"temp": {"op": "<=", "value": 10.0}},
    )
    study = Study.create(
        tmp_path / "csv_spec_run",
        SearchSpace.from_dict([{"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0}]),
        spec=spec,
        algorithm="lhs",
        budget=8,
        batch_size=2,
        seed=0,
    )
    pending = study.ask(2)
    csv_path = tmp_path / "measurements_spec.csv"
    csv_path.write_text(
        "id,x.x,obj.yield,con.temp\n"
        f"{pending[0]['id']},0.2,79.5,8.0\n"
        f"{pending[1]['id']},0.8,70.0,12.0\n",
        encoding="utf-8",
    )
    assert main(["tell", "--run", str(study.dir), "--results", str(csv_path), "--json"]) == 0
    observations = study.dataset.observations
    assert observations[0].objectives["yield"] == 79.5
    assert observations[0].constraints["temp"] == 8.0
    assert observations[0].meta["spec"]["in_spec"] is True

def test_csv_import_sniffs_tab_delimiter(tmp_path):
    from condopt.cli import main

    study = Study.create(
        tmp_path / "tsv_run",
        SearchSpace.from_dict([{"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0}]),
        Objective(name="y", direction="max"),
        algorithm="lhs",
        budget=8,
        batch_size=2,
        seed=0,
    )
    tsv_path = tmp_path / "measurements.tsv"
    tsv_path.write_text("x\ty\n0.3\t11.0\n0.7\t13.0\n", encoding="utf-8")
    assert main(["import", "--run", str(study.dir), "--results", str(tsv_path), "--json"]) == 0
    observations = study.dataset.observations
    assert len(observations) == 2
    assert observations[0].x == {"x": 0.3}
    assert observations[1].y == 13.0

def test_template_roundtrip_through_tell(tmp_path):
    from condopt.cli import main

    study = Study.create(
        tmp_path / "template_run",
        SearchSpace.from_dict([{"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0}]),
        Objective(name="y", direction="max"),
        algorithm="lhs",
        budget=8,
        batch_size=2,
        seed=0,
    )
    pending = study.ask(2)
    assert main(["template", "--run", str(study.dir), "--json"]) == 0
    template = study.dir / "results_template.csv"
    assert template.exists()
    lines = template.read_text(encoding="utf-8-sig").splitlines()
    assert lines[0].startswith("id,plate,well,x.x,y")
    assert len(lines) == 3
    # fill measurements and feed the same file back
    filled = [lines[0]] + [
        line.rsplit(",", 1)[0] + f",{10.0 + index}" for index, line in enumerate(lines[1:])
    ]
    filled_path = tmp_path / "filled.csv"
    filled_path.write_text("\n".join(filled) + "\n", encoding="utf-8")
    assert main(["tell", "--run", str(study.dir), "--results", str(filled_path), "--json"]) == 0
    assert len(study.dataset.observations) == 2

def test_export_dataset_csv(tmp_path):
    from condopt.cli import main

    study = Study.create(
        tmp_path / "export_run",
        SearchSpace.from_dict([{"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0}]),
        Objective(
            name="yield",
            direction="max",
            constraints=("temp",),
            constraint_limits={"temp": {"op": "<=", "value": 10.0}},
        ),
        algorithm="lhs",
        budget=8,
        batch_size=2,
        seed=0,
    )
    study.import_observations([
        {"x": {"x": 0.2}, "y": 10.0, "constraints": {"temp": 8.0}},
        {"x": {"x": 0.8}, "y": 12.0, "constraints": {"temp": 9.0}},
    ])
    assert main(["export", "--run", str(study.dir), "--json"]) == 0
    export = study.dir / "dataset_export.csv"
    assert export.exists()
    lines = export.read_text(encoding="utf-8-sig").splitlines()
    assert "x.x" in lines[0] and "con.temp" in lines[0]
    assert len(lines) == 3

def test_lab_csv_handoff_closed_loop(tmp_path, capsys):
    import csv as csv_module

    from condopt.cli import main
    from condopt.spec import Goal, Spec

    spec = Spec(
        goals=(Goal("yield", mode="target", target=80.0, tolerance=1.0, scale=5.0),),
        constraints={"temp": {"op": "<=", "value": 10.0}},
    )
    study = Study.create(
        tmp_path / "lab_loop",
        SearchSpace.from_dict([{"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0}]),
        spec=spec,
        algorithm="lhs",
        budget=8,
        batch_size=3,
        seed=0,
    )
    study.ask(3, plate="P1")
    assert main(["template", "--run", str(study.dir), "--json"]) == 0
    capsys.readouterr()
    template = study.dir / "results_template.csv"
    rows = list(csv_module.DictReader(template.read_text(encoding="utf-8-sig").splitlines()))
    assert len(rows) == 3
    for index, row in enumerate(rows):
        row["obj.yield"] = str(80.0 + index)
        row["con.temp"] = "8.0"
    filled = tmp_path / "filled_lab.csv"
    with filled.open("w", encoding="utf-8", newline="") as handle:
        writer = csv_module.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assert main(["tell", "--run", str(study.dir), "--results", str(filled), "--json"]) == 0
    capsys.readouterr()
    assert len(study.dataset.observations) == 3
    assert all(obs.meta["plate"] == "P1" for obs in study.dataset.observations)
    assert all(obs.constraints["temp"] == 8.0 for obs in study.dataset.observations)
    assert any(obs.meta["spec"]["in_spec"] for obs in study.dataset.observations)

    assert main(["export", "--run", str(study.dir), "--json"]) == 0
    capsys.readouterr()
    assert (study.dir / "dataset_export.csv").exists()
    recommendation = study.recommend()
    assert recommendation["spec"]["in_spec"] is True

def test_cli_decisions_shows_audit_trail(tmp_path, capsys):
    import json

    from condopt.cli import main

    study = Study.create(
        tmp_path / "decisions_run",
        SearchSpace.from_dict([{"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0}]),
        Objective(name="y", direction="max"),
        algorithm="lhs",
        budget=8,
        batch_size=2,
        seed=0,
    )
    study.override("bayes", {"n_initial": 4}, reason="人工复核：改贝叶斯")
    assert main(["decisions", "--run", str(study.dir), "--action", "override", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["n"] >= 1
    assert payload["decisions"][0]["reason"] == "人工复核：改贝叶斯"

def test_predict_returns_surrogate_mean_and_std(tmp_path):
    study = Study.create(
        tmp_path / "predict_run",
        SearchSpace.from_dict([{"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0}]),
        Objective(name="y", direction="max"),
        algorithm="lhs",
        budget=16,
        batch_size=4,
        seed=0,
    )
    grid = [index / 9 for index in range(10)]
    study.import_observations([
        {"x": {"x": value}, "y": -((value - 0.3) ** 2)} for value in grid
    ])
    prediction = study.predict({"x": 0.3})
    assert prediction["predictions"]["y"]["mean"] == pytest.approx(0.0, abs=0.01)
    assert prediction["predictions"]["y"]["std"] > 0.0
    assert prediction["condition"] == {"x": 0.3}
    # away from the optimum the prediction is clearly worse
    far = study.predict({"x": 0.9})
    assert far["predictions"]["y"]["mean"] < -0.1


def test_cli_predict_outputs_json(tmp_path, capsys):
    import json as json_module

    from condopt.cli import main

    study = Study.create(
        tmp_path / "predict_cli",
        SearchSpace.from_dict([{"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0}]),
        Objective(
            name="y",
            direction="max",
            constraints=("temp",),
            constraint_limits={"temp": {"op": "<=", "value": 10.0}},
        ),
        algorithm="lhs",
        budget=16,
        batch_size=4,
        seed=0,
    )
    study.import_observations([
        {"x": {"x": index / 9}, "y": -((index / 9 - 0.3) ** 2), "constraints": {"temp": 8.0}}
        for index in range(10)
    ])
    assert main(["predict", "--run", str(study.dir), "--config", '{"x": 0.3}', "--json"]) == 0
    payload = json_module.loads(capsys.readouterr().out)
    assert "y" in payload["predictions"]
    assert payload["p_feasible"] is not None
    assert payload["constraints"]["temp"]["threshold"] == 10.0

def test_check_integrity_ok_and_detects_tampering(tmp_path):
    import json as json_module

    study = Study.create(
        tmp_path / "integrity_run",
        SearchSpace.from_dict([{"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0}]),
        Objective(name="y", direction="max"),
        algorithm="bayes",
        budget=8,
        batch_size=2,
        seed=0,
    )
    study.ask(2)
    healthy = study.check_integrity()
    assert healthy["ok"] is True
    assert healthy["counts"]["pending"] == 2

    # duplicate pending id -> issue
    pending_path = study.dir / "pending.json"
    items = json_module.loads(pending_path.read_text(encoding="utf-8"))
    items.append(dict(items[0]))
    pending_path.write_text(
        json_module.dumps(items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    broken = Study.load(study.dir).check_integrity()
    assert broken["ok"] is False
    assert any("duplicate ids" in issue for issue in broken["issues"])

    # optimizer state algorithm mismatch -> issue
    pending_path.write_text(
        json_module.dumps(items[:2], ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    state_path = study.dir / "opt_state.json"
    state = json_module.loads(state_path.read_text(encoding="utf-8"))
    state["algorithm"] = "tpe"
    state_path.write_text(
        json_module.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    mismatched = Study.load(study.dir).check_integrity()
    assert mismatched["ok"] is False
    assert any("opt_state algorithm" in issue for issue in mismatched["issues"])


def test_cli_doctor_reports_and_exits_nonzero_on_issues(tmp_path, capsys):
    import json as json_module

    from condopt.cli import main

    study = Study.create(
        tmp_path / "doctor_run",
        SearchSpace.from_dict([{"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0}]),
        Objective(name="y", direction="max"),
        algorithm="lhs",
        budget=8,
        batch_size=2,
        seed=0,
    )
    assert main(["doctor", "--run", str(study.dir), "--json"]) == 0
    payload = json_module.loads(capsys.readouterr().out)
    assert payload["ok"] is True

    rounds_path = study.dir / "rounds.json"
    rounds_path.write_text("{not json", encoding="utf-8")
    assert main(["doctor", "--run", str(study.dir), "--json"]) == 1
    payload = json_module.loads(capsys.readouterr().out)
    assert payload["ok"] is False

def test_import_observations_requires_condition(tmp_path):
    import pytest as pytest_module

    study = Study.create(
        tmp_path / "import_guard",
        SearchSpace.from_dict([{"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0}]),
        Objective(name="y", direction="max"),
        algorithm="random",
        budget=4,
        batch_size=2,
        seed=0,
    )
    with pytest_module.raises(ValueError, match="needs an 'x' condition"):
        study.import_observations([{"y": 1.0}])

def test_init_cli_budget_first_flag(tmp_path):
    """`condopt init --budget-first` must persist the budget-first stopping policy."""
    import contextlib
    import io
    import json

    from condopt.cli import main as cli_main

    run_dir = tmp_path / "cli_budget_first"
    space_path = tmp_path / "space.json"
    space_path.write_text(json.dumps([
        {"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0}
    ]), encoding="utf-8")
    argv = [
        "init", "--run", str(run_dir), "--objective", "y:max",
        "--space", str(space_path), "--budget", "12", "--batch", "3",
        "--budget-first", "--patience", "2",
    ]
    with contextlib.redirect_stdout(io.StringIO()):
        code = cli_main(argv)
    assert code == 0
    from condopt.study import Study

    study = Study.load(run_dir)
    assert study.meta["stop_on_stagnation"] is False
    assert study.meta["stopping_policy"] == "budget_first"
    assert study.patience == 2

def test_missing_delivery_evidence_is_surfaced_as_a_suggestion(tmp_path):
    from condopt.base import Objective
    from condopt.space import SearchSpace
    from condopt.study import Study

    space = SearchSpace.from_dict([
        {"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0}
    ])
    study = Study.create(
        tmp_path / "delivery", space, Objective(name="y", direction="max"),
        algorithm="random", budget=12, batch_size=3, seed=0,
    )
    study.import_observations([{"x": {"x": 0.4}, "y": 1.0}])  # single replicate
    action = study.next_action()
    assert action["diagnostics"]["delivery_missing"], action["diagnostics"]
    assert "complete_delivery_evidence" in action["suggested_actions"]
    assert "交付证据不完整" in action["suggested_action_reasons"]["complete_delivery_evidence"]

    # after a confirmation replicate the gap closes (confirmation observations are
    # tagged with purpose=confirmation, which is what the checklist looks for)
    from condopt.base import Observation

    study.tell([
        Observation(
            x={"x": 0.4}, y=1.1, fidelity=1.0, cost=1.0,
            meta={"purpose": "confirmation"},
        )
    ])
    checklist = study.delivery_checklist()
    assert all(item["ok"] for item in checklist), checklist
    assert study.next_action()["diagnostics"]["delivery_missing"] == []

def test_default_confirm_n_is_shared_with_prepare():
    from condopt.prepare import deliverable_advice
    from condopt.router import OptimizationContext
    from condopt.base import Objective
    from condopt.study import default_confirm_n

    assert default_confirm_n(12) == 3
    assert default_confirm_n(90) == 3
    advice = deliverable_advice(
        {}, OptimizationContext(n_dims=2, n_continuous=2, budget_total=12),
        Objective(name="y", direction="max"),
    )
    assert advice["confirmation_plan"]["n"] == default_confirm_n(12)


def test_confirmation_batch_scales_with_the_budget(tmp_path):
    from condopt.base import Objective
    from condopt.space import SearchSpace
    from condopt.study import Study

    space = SearchSpace.from_dict([
        {"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0}
    ])

    def run(budget: int) -> list:
        study = Study.create(
            tmp_path / f"confirm_{budget}", space, Objective(name="y", direction="max"),
            algorithm="random", budget=budget, batch_size=3, seed=0,
        )
        for _ in range(60):
            action = study.next_action()
            kind = action["action"]
            if kind == "stop":
                break
            if kind == "ask":
                study.ask(int(action.get("ask_n") or 3))
            elif kind == "confirm":
                study.confirm(int(action.get("confirm_n") or 3))
            else:
                study.tell([
                    {"id": item["id"], "y": -((float(item["x"]["x"]) - 0.3) ** 2),
                     "fidelity": 1.0}
                    for item in study.pending
                ])
        return [obs for obs in study.dataset.observations
                if obs.meta.get("purpose") == "confirmation"]

    small = run(12)
    large = run(90)
    assert 3 <= len(small) <= 4, len(small)      # default batch of 3
    assert 3 <= len(large) <= 4, len(large)      # same default for large budgets

def test_confirm_budget_benchmark_script_runs(tmp_path):
    import importlib.util
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "benchmark_confirm_budget.py"
    spec = importlib.util.spec_from_file_location("benchmark_confirm_budget", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.OUT = tmp_path / "confirm_budget"
    auto = module.run_arm(None, seed=0, budget=9, batch=3)
    small = module.run_arm(1, seed=0, budget=9, batch=3)
    assert auto["n_observations"] == 9
    assert small["n_observations"] == 9
    assert auto["n_confirmations"] >= small["n_confirmations"]

def test_budget_first_campaigns_still_deliver_confirmed_evidence(tmp_path):
    """A budget-first run must not end with an unconfirmed incumbent."""
    from condopt.base import Objective
    from condopt.space import SearchSpace
    from condopt.study import Study

    space = SearchSpace.from_dict([
        {"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0}
    ])
    study = Study.create(
        tmp_path / "bf", space, Objective(name="y", direction="max"),
        algorithm="bayes", budget=40, batch_size=4, seed=0,
        stop_on_stagnation=False,
    )
    for _ in range(120):
        action = study.next_action()
        kind = action["action"]
        if kind == "stop":
            break
        if kind == "ask":
            study.ask(int(action.get("ask_n") or 4))
        elif kind == "confirm":
            study.confirm(int(action.get("confirm_n") or 3))
        else:
            study.tell([
                {"id": item["id"], "y": -((float(item["x"]["x"]) - 0.3) ** 2)}
                for item in study.pending
            ])
    checklist = study.delivery_checklist()
    assert checklist, "no delivery checks produced"
    missing = [item for item in checklist if not item["ok"]]
    assert not missing, missing

def test_warm_start_gets_a_stagnation_grace_period(tmp_path):
    """Imported priors must not trigger a plateau stop immediately."""
    from condopt.base import Objective, Observation
    from condopt.space import SearchSpace
    from condopt.study import Study

    space = SearchSpace.from_dict([
        {"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0}
    ])
    study = Study.create(
        tmp_path / "warm", space, Objective(name="y", direction="max"),
        algorithm="random", budget=60, batch_size=2, seed=0, patience=3,
    )
    # three rounds of mediocre local data, then one strong prior
    for value in (1.0, 1.0, 1.0):
        pending = study.ask(2)
        study.tell([{"id": item["id"], "y": value} for item in pending])
    study.import_observations([Observation(x={"x": 0.9}, y=99.0, fidelity=1.0)])
    stop, reason = study.should_stop()
    assert stop is False, reason
    # local progress must still be visible: improvements after the prior keep the
    # campaign alive even though the imported prior stays "best"
    for value in (2.0, 3.0, 4.0):
        pending = study.ask(2)
        study.tell([{"id": item["id"], "y": value} for item in pending])
    stop, _ = study.should_stop()
    assert stop is False
    # once local progress flattens, the plateau rule fires again
    for _ in range(4):
        pending = study.ask(2)
        study.tell([{"id": item["id"], "y": 4.0} for item in pending])
    stop, _ = study.should_stop()
    assert stop is True

