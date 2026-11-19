"""Tests for the machine-readable automatic loop (priority c).

`Study.next_action()` / `condopt step` combine pending state, should_stop,
recommend and confirm into one auditable action the skill can follow.
"""

import json

import pytest

from condopt import Objective, SearchSpace, Study
from condopt.cli import main


def simple_space() -> SearchSpace:
    return SearchSpace.from_dict(
        [
            {"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0},
            {"name": "y", "kind": "continuous", "low": 0.0, "high": 1.0},
        ]
    )


def make_study(tmp_path, budget=20, batch_size=2, **kwargs) -> Study:
    return Study.create(
        tmp_path / "run",
        simple_space(),
        Objective(name="yield", direction="max"),
        budget=budget,
        batch_size=batch_size,
        seed=0,
        **kwargs,
    )


def tell_all(study: Study, value: float) -> None:
    study.tell([{"id": item["id"], "y": value} for item in study.pending])


def test_step_without_data_asks(tmp_path):
    study = make_study(tmp_path)
    action = study.next_action()
    assert action["action"] == "ask"
    assert action["recommendation"] is None
    assert action["should_stop"] is False
    assert "ask" in action["next_command"]


def test_step_after_ask_requests_tell(tmp_path):
    study = make_study(tmp_path)
    pending = study.ask(2)
    action = study.next_action()
    assert action["action"] == "tell"
    assert action["pending"] == [item["id"] for item in pending]
    assert "tell" in action["next_command"]


def test_step_continues_after_tell(tmp_path):
    study = make_study(tmp_path)
    pending = study.ask(2)
    study.tell([{"id": pending[0]["id"], "y": 1.0}, {"id": pending[1]["id"], "y": 2.0}])
    action = study.next_action()
    assert action["action"] == "ask"
    assert action["used"] == 2
    assert action["remaining"] == 18
    assert action["recommendation"]["condition"] is not None
    assert action["recommendation"]["should_stop"] is False
    assert {"batch_effect", "plate_effect", "spec_calibration_ready"} <= set(action["diagnostics"])
    assert "ask_next_batch" in action["suggested_actions"]


def test_step_stops_when_budget_is_exhausted(tmp_path):
    study = make_study(tmp_path, budget=2)
    pending = study.ask(2)
    study.tell([{"id": item["id"], "y": float(index)} for index, item in enumerate(pending)])
    action = study.next_action()
    assert action["action"] == "stop"
    assert action["should_stop"] is True
    assert "预算" in action["stop_reason"]
    assert action["next_command"].startswith("condopt recommend")


def test_step_confirms_before_claiming_target(tmp_path):
    study = make_study(tmp_path, budget=10, meta={"target_score": 8.0})
    pending = study.ask(2)
    study.tell([{"id": item["id"], "y": 9.0} for item in pending])
    action = study.next_action()
    assert action["action"] == "confirm"
    assert action["confirm_n"] == 3
    assert action["should_stop"] is True
    assert "目标" in action["stop_reason"]
    assert "confirm" in action["next_command"]

    # after the confirmation replicates come back, the loop stops for real
    study.confirm(action["confirm_n"])
    tell_all(study, 9.0)
    final = study.next_action()
    assert final["action"] == "stop"
    assert final["recommendation"]["confirmation"]["n"] == 3
    assert final["recommendation"]["score"] == pytest.approx(9.0)


def test_step_confirms_when_budget_is_almost_spent(tmp_path):
    study = make_study(tmp_path, budget=4)
    pending = study.ask(2)
    study.tell([{"id": pending[0]["id"], "y": 1.0}, {"id": pending[1]["id"], "y": 2.0}])
    action = study.next_action()
    assert action["action"] == "confirm"
    assert action["confirm_n"] == 2
    assert "预算仅剩" in action["reason"]
    study.confirm(2)
    tell_all(study, 2.0)
    assert study.next_action()["action"] == "stop"


def test_step_auto_confirm_can_be_disabled(tmp_path):
    study = make_study(tmp_path, budget=10, meta={"target_score": 8.0, "auto_confirm": False})
    pending = study.ask(2)
    study.tell([{"id": item["id"], "y": 9.0} for item in pending])
    action = study.next_action()
    assert action["action"] == "stop"
    assert action["confirm_n"] is None


def test_step_reports_batch_effect(tmp_path):
    study = make_study(tmp_path)
    study.import_observations(
        [
            {"x": {"x": 0.1, "y": 0.1}, "y": 40.0, "plate": "P1"},
            {"x": {"x": 0.2, "y": 0.1}, "y": 41.0, "plate": "P1"},
            {"x": {"x": 0.3, "y": 0.1}, "y": 60.0, "plate": "P2"},
            {"x": {"x": 0.4, "y": 0.1}, "y": 61.0, "plate": "P2"},
        ]
    )
    action = study.next_action()
    assert action["batch_effect"]["detected"] is True
    assert "检测到批次/孔板效应" in action["batch_effect"]["message"]


def test_step_recommendation_matches_recommend(tmp_path):
    study = make_study(tmp_path)
    pending = study.ask(2)
    study.tell([{"id": item["id"], "y": float(index)} for index, item in enumerate(pending)])
    recommendation = study.recommend()
    action = study.next_action()
    assert set(action["recommendation"]) == set(recommendation)
    assert action["recommendation"]["condition"] == recommendation["condition"]


def test_step_cli_outputs_json(tmp_path, capsys):
    study = make_study(tmp_path)
    study.ask(2)
    assert main(["step", "--run", str(study.dir), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["action"] == "tell"
    assert payload["pending"]
    assert "next_command" in payload


def test_report_mentions_next_action(tmp_path):
    study = make_study(tmp_path)
    pending = study.ask(2)
    study.tell([{"id": item["id"], "y": 1.0} for item in pending])
    report = study.report()
    assert "下一步行动：ask" in report
    assert "condopt ask" in report
    assert "推荐条件" in report


def test_target_stop_uses_confirmation_mean_not_lucky_best(tmp_path):
    study = make_study(tmp_path, budget=10, meta={"target_score": 5.0})
    pending = study.ask(2)
    study.tell([{"id": pending[0]["id"], "y": 10.0}, {"id": pending[1]["id"], "y": 1.0}])
    stop, _ = study.should_stop()
    assert stop is True  # target hit on the raw best; next_action will confirm
    confirm = study.confirm(2)
    study.tell([{"id": item["id"], "y": value} for item, value in zip(confirm, [3.0, 4.0])])
    stop_after, _ = study.should_stop()
    assert stop_after is False  # confirmation mean 3.5 < target 5
    recommend = study.recommend()
    assert recommend["score"] == pytest.approx(3.5)
    assert recommend["should_stop"] is False

def test_unattended_campaign_script_reaches_stop(tmp_path):
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "unattended_campaign.py"
    spec = importlib.util.spec_from_file_location("unattended_campaign", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    report = module.run_campaign(
        tmp_path / "unattended", "sphere2d", budget=12, batch_size=4, seed=0
    )
    assert report["stop"] is True
    assert report["integrity_before"] is True
    assert report["integrity_after"] is True
    assert 0 < report["n_observations"] <= 12
    assert report["best"] is not None
    assert "confirm" in report["trace"] or report["remaining"] == 0

def test_unattended_campaign_resumes_after_interruption(tmp_path):
    import importlib.util
    from pathlib import Path

    from condopt import Study


    script = Path(__file__).resolve().parents[1] / "scripts" / "unattended_campaign.py"
    spec = importlib.util.spec_from_file_location("unattended_campaign", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    run_dir = tmp_path / "resume"
    first = module.run_campaign(run_dir, "sphere2d", budget=12, batch_size=4, seed=0, max_steps=3)
    assert first["stop"] is False  # interrupted mid-campaign

    interrupted = Study.load(run_dir)
    assert interrupted.check_integrity()["ok"] is True

    second = module.run_campaign(run_dir, "sphere2d", budget=12, batch_size=4, seed=0, max_steps=50)
    assert second["stop"] is True
    assert second["integrity_after"] is True
    assert second["n_observations"] <= 12

    ids = [
        obs.meta.get("suggestion_id")
        for obs in Study.load(run_dir).dataset.observations
        if obs.meta.get("suggestion_id")
    ]
    assert len(ids) == len(set(ids)), "resume must not duplicate observations"

def test_pending_and_partial_tell_cli(tmp_path, capsys):
    import json as json_module

    from condopt.base import Objective
    from condopt.cli import main
    from condopt.space import SearchSpace
    from condopt.study import Study

    study = Study.create(
        tmp_path / "executor_run",
        SearchSpace.from_dict([{"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0}]),
        Objective(name="y", direction="max"),
        algorithm="random",
        budget=6,
        batch_size=3,
        seed=0,
    )
    pending = study.ask(3)

    assert main(["pending", "--run", str(study.dir), "--json"]) == 0
    payload = json_module.loads(capsys.readouterr().out)
    assert payload["n_pending"] == 3
    assert payload["pending"][0]["id"] == pending[0]["id"]

    # executor measured only the first well
    results_path = tmp_path / "partial.json"
    results_path.write_text(
        json_module.dumps([{"id": pending[0]["id"], "y": 0.5}]), encoding="utf-8"
    )
    assert main(["tell", "--run", str(study.dir), "--results", str(results_path), "--json"]) == 0
    tell_payload = json_module.loads(capsys.readouterr().out)
    assert tell_payload["n"] == 1
    assert tell_payload["still_pending"] == [pending[1]["id"], pending[2]["id"]]

    decisions = (study.dir / "decisions.jsonl").read_text(encoding="utf-8")
    assert "execution_partial" in decisions

    reloaded = Study.load(study.dir)
    assert len(reloaded.pending) == 2
    assert reloaded.check_integrity()["ok"] is True

def test_stop_categories_and_exploration_restart_suggestion(tmp_path):
    from condopt.base import Objective
    from condopt.space import SearchSpace
    from condopt.study import Study

    space = SearchSpace.from_dict([
        {"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0}
    ])
    study = Study.create(
        tmp_path / "stop_cat",
        space,
        Objective(name="y", direction="max"),
        algorithm="random",
        budget=10,
        batch_size=2,
        seed=0,
        patience=2,
    )
    # flat measurements -> stagnation while budget remains; the incumbent is
    # confirmed first, then the study stops with the stagnation category.
    action = None
    for _ in range(8):
        action = study.next_action()
        if action["action"] == "stop":
            break
        if action["action"] == "confirm":
            n = int(action.get("confirm_n") or 1)
            pending = study.confirm(max(1, min(n, study.remaining())))
        elif action["action"] == "ask":
            n = int(action.get("ask_n") or 2)
            pending = study.ask(max(1, min(n, study.remaining())))
        else:  # tell
            pending = study.pending
        study.tell([{"id": item["id"], "y": 0.5} for item in pending])
    assert action is not None and action["action"] == "stop"
    assert action["diagnostics"]["stop_category"] == "stagnation"
    assert action["diagnostics"]["remaining_evals"] > 0
    assert "consider_exploration_restart" in action["suggested_actions"]
    assert action["suggested_action_reasons"]["consider_exploration_restart"]

    # budget exhausted -> distinct category
    exhausted = Study.create(
        tmp_path / "stop_budget",
        space,
        Objective(name="y", direction="max"),
        algorithm="random",
        budget=2,
        batch_size=2,
        seed=0,
    )
    pending = exhausted.ask(2)
    exhausted.tell([{"id": item["id"], "y": 0.1} for item in pending])
    final = exhausted.next_action()
    assert final["action"] == "stop"
    assert final["diagnostics"]["stop_category"] == "budget_exhausted"


def test_long_run_drill_reaches_budget_when_forced(tmp_path):
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "long_run_drill.py"
    spec = importlib.util.spec_from_file_location("long_run_drill", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    report = module.run_drill(
        "mixed_reaction", budget=24, batch_size=4, seed=0, force_budget=True
    )
    assert report["ok"] is True, report
    # one evaluation may be held back to confirm the final winner
    assert 23 <= report["n_observations"] <= 24
    assert report["duplicate_ids"] == 0
    assert report["integrity_ok"] is True
    assert report["digest_time_s"] >= 0.0

def test_stagnation_benchmark_script_runs():
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_stagnation.py"
    spec = importlib.util.spec_from_file_location("benchmark_stagnation", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    row = module.run_arm("continue", "mixed_reaction", budget=10, batch_size=4, seed=0)
    # one evaluation may be held back to confirm the final winner
    assert 9 <= row["n_observations"] <= 10
    assert row["best_score"] is not None

def test_retune_for_exploration_switches_acquisition_and_logs(tmp_path, capsys):
    import json as json_module

    from condopt.cli import main
    from condopt.base import Objective
    from condopt.space import SearchSpace
    from condopt.study import Study

    study = Study.create(
        tmp_path / "retune",
        SearchSpace.from_dict([{"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0}]),
        Objective(name="y", direction="max"),
        algorithm="bayes",
        budget=10,
        batch_size=2,
        seed=0,
    )
    before = dict(study.params)
    report = study.retune_for_exploration()
    assert report["changed"] == ["acquisition"]
    assert study.params["acquisition"] != before["acquisition"]
    decisions = (study.dir / "decisions.jsonl").read_text(encoding="utf-8")
    assert "retune" in decisions or "重调参" in decisions

    assert main(["retune", "--run", str(study.dir), "--json"]) == 0
    payload = json_module.loads(capsys.readouterr().out)
    assert payload["algorithm"] == "bayes"
    assert payload["changed"]

def test_patience_one_is_clamped_to_two(tmp_path):
    from condopt.base import Objective
    from condopt.space import SearchSpace
    from condopt.study import Study

    study = Study.create(
        tmp_path / "patience_clamp",
        SearchSpace.from_dict([{"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0}]),
        Objective(name="y", direction="max"),
        algorithm="random",
        budget=10,
        batch_size=2,
        seed=0,
        patience=1,
    )
    assert study.patience == 2
    assert study.meta["patience_clamped_from"] == 1

def test_retune_conditions_sweep_script_runs():
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "sweep_retune_conditions.py"
    spec = importlib.util.spec_from_file_location("sweep_retune_conditions", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    row = module.run_arm("tune", "constraint_hard", noise_std=0.0, budget=16, batch_size=4, seed=0)
    assert row["n_observations"] == 16
    assert row["best_true"] is not None

def test_budget_first_stopping_ignores_the_plateau(tmp_path):
    from condopt.base import Objective
    from condopt.space import SearchSpace
    from condopt.study import Study

    space = SearchSpace.from_dict([
        {"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0}
    ])
    default_study = Study.create(
        tmp_path / "stagnation_first", space, Objective(name="y", direction="max"),
        algorithm="random", budget=30, batch_size=3, patience=2, seed=0,
    )
    budget_study = Study.create(
        tmp_path / "budget_first", space, Objective(name="y", direction="max"),
        algorithm="random", budget=30, batch_size=3, patience=2, seed=0,
        stop_on_stagnation=False,
    )
    assert default_study.meta["stop_on_stagnation"] is True
    assert budget_study.meta["stopping_policy"] == "budget_first"

    for study in (default_study, budget_study):
        steps = 0
        while steps < 80 and len(study.dataset.observations) < 30:
            steps += 1
            action = study.next_action()
            if action["action"] == "stop":
                break
            if action["action"] == "ask":
                study.ask(int(action.get("ask_n") or 3))
            elif action["action"] == "confirm":
                study.confirm(int(action.get("confirm_n") or 3))
            else:
                # deliberately flat results: every round reports the same score
                study.tell([
                    {"id": item["id"], "y": 1.0,
                     "fidelity": float(item.get("fidelity", 1.0) or 1.0)}
                    for item in study.pending
                ])

    assert len(default_study.dataset.observations) < 30  # plateau stopped it early
    # budget-first uses the budget, minus the reserved delivery slot(s)
    assert 28 <= len(budget_study.dataset.observations) <= 30

