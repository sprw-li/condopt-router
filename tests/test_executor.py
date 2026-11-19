"""Executor adapter: campaign driving, failure handling, and the demo script."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from condopt.base import Objective
from condopt.executor import ExecutionRequest, Executor, Measurement, run_campaign
from condopt.space import SearchSpace
from condopt.study import Study

ROOT = Path(__file__).resolve().parents[1]


class Quadratic(Executor):
    def measure(self, request: ExecutionRequest) -> Measurement:
        value = -((float(request.x["x"]) - 0.3) ** 2)
        return Measurement(id=request.id, y=value, fidelity=request.fidelity, cost=request.fidelity)


class Broken(Executor):
    def measure(self, request: ExecutionRequest) -> Measurement:
        raise RuntimeError("instrument offline")


def _study(tmp_path, **kwargs) -> Study:
    space = SearchSpace.from_dict([
        {"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0}
    ])
    return Study.create(
        tmp_path / "exec",
        space,
        Objective(name="y", direction="max"),
        algorithm="bayes",
        budget=kwargs.pop("budget", 8),
        batch_size=kwargs.pop("batch_size", 2),
        seed=0,
        **kwargs,
    )


def test_campaign_drives_study_to_completion(tmp_path):
    study = _study(tmp_path, budget=8, batch_size=2)
    report = run_campaign(study, Quadratic(), log_path=tmp_path / "exec" / "log.jsonl")
    assert report["ok"] is True
    assert report["integrity_ok"] is True
    assert 7 <= report["n_observations"] <= 8
    assert report["still_pending"] == 0
    assert report["cost_spent"] == 8.0
    assert report["actions"]["stop"] == 1
    rows = [json.loads(line) for line in
            (tmp_path / "exec" / "log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 8
    assert all(row["failed"] is False for row in rows)


def test_failed_measurements_stay_pending_and_are_logged(tmp_path):
    study = _study(tmp_path, budget=6, batch_size=2)
    report = run_campaign(study, Broken(), log_path=tmp_path / "exec" / "log.jsonl")
    assert report["attempts"] >= 6
    assert report["failures"] >= 6
    assert report["n_observations"] == 0
    rows = [json.loads(line) for line in
            (tmp_path / "exec" / "log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows and all(row["failed"] is True for row in rows)
    assert "instrument offline" in rows[0]["note"]


def test_executor_adapter_example_runs():
    spec = importlib.util.spec_from_file_location(
        "executor_adapter", ROOT / "examples" / "executor_adapter.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    report = module.run(seed=0, budget=10, cost_budget=4.0)
    assert report["ok"] is True
    assert report["failures"] >= 1  # the simulated instrument fails every 7th call
    assert report["still_pending"] == 0
    assert Path(report["execution_log"]).exists()

def test_broken_instrument_stops_after_empty_batches(tmp_path):
    study = _study(tmp_path, budget=20, batch_size=2)
    report = run_campaign(
        study,
        Broken(),
        log_path=tmp_path / "exec" / "log.jsonl",
        max_empty_batches=3,
        max_retries=-1,  # this test exercises the empty-batch guard, not the retry cap
    )
    assert "没有可用的实测结果" in report["stopped_reason"]
    assert report["steps"] < 12  # no spinning up to max_steps
    assert report["still_pending"] >= 1  # the batch is kept for a retry


def test_rerunning_the_campaign_resumes_pending_conditions(tmp_path):
    study = _study(tmp_path, budget=6, batch_size=2)
    first = run_campaign(
        study, Broken(), log_path=tmp_path / "exec" / "log.jsonl", max_empty_batches=2
    )
    assert first["n_observations"] == 0
    resumed_pending = first["still_pending"]
    assert resumed_pending >= 1

    second = run_campaign(study, Quadratic(), log_path=tmp_path / "exec" / "log.jsonl")
    assert second["resumed_pending"] == resumed_pending
    # one evaluation may be reserved to confirm the final winner
    assert 5 <= second["n_observations"] <= 6
    assert second["still_pending"] == 0

class Slow(Executor):
    def measure(self, request: ExecutionRequest) -> Measurement:
        import time as _time

        _time.sleep(0.35)
        return Measurement(id=request.id, y=0.1, fidelity=request.fidelity, cost=request.fidelity)


class Hanging(Executor):
    def measure(self, request: ExecutionRequest) -> Measurement:
        import time as _time

        _time.sleep(5)  # long enough to trip the timeout, short enough for tests
        return Measurement(id=request.id, y=0.1)


def test_parallel_batches_finish_and_match_working_executor(tmp_path):
    study = _study(tmp_path, budget=8, batch_size=4)
    report = run_campaign(
        study, Quadratic(), max_workers=4, log_path=tmp_path / "exec" / "log.jsonl"
    )
    assert 7 <= report["n_observations"] <= 8
    assert report["still_pending"] == 0
    assert report["ok"] is True


def test_timeout_marks_the_measurement_failed_instead_of_blocking(tmp_path):
    study = _study(tmp_path, budget=4, batch_size=2)
    report = run_campaign(
        study,
        Hanging(),
        timeout=0.2,
        max_empty_batches=2,
        log_path=tmp_path / "exec" / "log.jsonl",
    )
    assert report["failures"] >= 2
    assert report["n_observations"] == 0
    assert "timeout" in report["stopped_reason"] or "没有可用的实测结果" in report["stopped_reason"]
    rows = [json.loads(line) for line in
            (tmp_path / "exec" / "log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any("timeout after" in row["note"] for row in rows)

def test_adapter_long_run_script_covers_a_small_budget(tmp_path, monkeypatch):
    import importlib.util

    script = ROOT / "scripts" / "adapter_long_run.py"
    spec = importlib.util.spec_from_file_location("adapter_long_run", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.OUT = tmp_path / "adapter_long_run"
    report = module.run(budget=20, batch_size=4, workers=2, seed=0, patience=100)
    # one evaluation may be reserved to confirm the final winner
    assert 19 <= report["n_observations"] <= 20
    assert report["duplicate_suggestion_ids"] == 0
    assert report["integrity_ok"] is True
    assert report["dataset_kb"] > 0
    assert report["instrument_p95_seconds"] is not None

class TwoObjectives(Executor):
    def measure(self, request: ExecutionRequest) -> Measurement:
        value = float(request.x["x"])
        return Measurement(
            id=request.id,
            y=1.0 - value,
            objectives={"cost": value},
            fidelity=request.fidelity,
            constraints={"c": value},
            cost=request.fidelity,
        )


def test_multi_objective_measurements_reach_the_dataset(tmp_path):
    from condopt.base import Objective

    space = SearchSpace.from_dict([
        {"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0}
    ])
    objective = Objective(
        name="yield", direction="max",
        objectives=("yield", "cost"), directions=("max", "min"),
        constraints=("c",), constraint_limits={"c": {"op": "<=", "value": 0.5}},
    )
    study = Study.create(
        tmp_path / "moo_exec", space, objective, algorithm="cqehvi",
        budget=6, batch_size=3, seed=0,
    )
    report = run_campaign(study, TwoObjectives(), log_path=tmp_path / "moo_exec" / "log.jsonl")
    assert report["n_observations"] == 6
    observations = study.dataset.observations
    assert all("cost" in obs.objectives for obs in observations)
    assert all("c" in obs.constraints for obs in observations)
    assert study.pareto_summary()["n_front"] >= 1


def test_campaign_without_any_usable_measurement_is_not_ok(tmp_path):
    study = _study(tmp_path, budget=4, batch_size=2)
    report = run_campaign(
        study, Broken(), max_empty_batches=2, log_path=tmp_path / "exec" / "log.jsonl"
    )
    assert report["n_observations"] == 0
    assert report["ok"] is False


def test_multiobjective_executor_example_runs():
    spec = importlib.util.spec_from_file_location(
        "multiobjective_executor", ROOT / "examples" / "multiobjective_executor.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.OUT = ROOT / "runs" / "_test_multiobjective_executor"
    report = module.run(seed=0, budget=9, batch_size=3, workers=2)
    assert report["n_observations"] == 9
    assert report["n_objective_vectors"] == 9
    assert report["integrity_ok"] is True
    assert report["ok"] is True

def test_multi_objective_campaign_tracks_hypervolume(tmp_path):
    from condopt.base import Objective

    space = SearchSpace.from_dict([
        {"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0}
    ])
    objective = Objective(
        name="yield", direction="max",
        objectives=("yield", "cost"), directions=("max", "min"),
    )
    study = Study.create(
        tmp_path / "hv_exec", space, objective, algorithm="qehvi",
        budget=9, batch_size=3, seed=0,
    )
    report = run_campaign(study, TwoObjectives(), log_path=tmp_path / "hv_exec" / "log.jsonl")
    progress_file = tmp_path / "hv_exec" / "progress.jsonl"
    assert progress_file.exists()
    rows = [json.loads(line) for line in
            progress_file.read_text(encoding="utf-8").splitlines()]
    assert report["actions"]["tell"] >= 3
    assert len(rows) == report["actions"]["tell"]  # one row per batch that told
    assert report["actions"]["tell_actions"] >= report["actions"]["tell"]
    assert report["progress_points"] == len(rows)
    assert all(row["round"] > 0 for row in rows)
    volumes = [row["hypervolume"] for row in rows if row["hypervolume"] is not None]
    assert volumes == sorted(volumes)  # hypervolume must not decrease
    assert report["final_hypervolume"] == study.pareto_summary()["hypervolume"]
    assert report["final_n_front"] >= 1


def test_constrained_multiobjective_progress_tracks_feasible_hv(tmp_path):
    from condopt.base import Objective

    space = SearchSpace.from_dict([
        {"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0}
    ])
    objective = Objective(
        name="yield", direction="max",
        objectives=("yield", "cost"), directions=("max", "min"),
        constraints=("c",), constraint_limits={"c": {"op": "<=", "value": 0.5}},
    )
    study = Study.create(
        tmp_path / "hv_feas", space, objective, algorithm="cqehvi",
        budget=9, batch_size=3, seed=0,
    )
    report = run_campaign(study, TwoObjectives(), log_path=tmp_path / "hv_feas" / "log.jsonl")
    rows = [
        json.loads(line)
        for line in (tmp_path / "hv_feas" / "progress.jsonl")
        .read_text(encoding="utf-8").splitlines()
    ]
    assert all("feasible_fraction" in row for row in rows)
    assert all(0.0 <= row["feasible_fraction"] <= 1.0 for row in rows)
    assert any(row["hypervolume_feasible"] is not None for row in rows)
    volumes = [
        row["hypervolume_feasible"]
        for row in rows
        if row["hypervolume_feasible"] is not None
    ]
    assert volumes == sorted(volumes)
    for row in rows:
        if row["hypervolume"] is not None and row["hypervolume_feasible"] is not None:
            # The raw front can include infeasible points, so raw HV is an upper
            # bound on what is actually deliverable.
            assert row["hypervolume_feasible"] <= row["hypervolume"] + 1e-9


def test_single_objective_campaign_has_no_progress_file(tmp_path):
    study = _study(tmp_path, budget=6, batch_size=2)
    report = run_campaign(study, Quadratic(), log_path=tmp_path / "exec" / "log.jsonl")
    assert not (tmp_path / "exec" / "progress.jsonl").exists()
    assert "final_hypervolume" not in report

def test_adapter_long_run_supports_multi_objective_problems(tmp_path):
    import importlib.util

    script = ROOT / "scripts" / "adapter_long_run.py"
    spec = importlib.util.spec_from_file_location("adapter_long_run_moo", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.OUT = tmp_path / "adapter_long_run_moo"
    report = module.run(
        budget=8, batch_size=4, workers=2, seed=0, patience=100,
        problem_name="moo_constrained",
    )
    assert 7 <= report["n_observations"] <= 8
    assert report["duplicate_suggestion_ids"] == 0
    assert report["progress_points"] >= 2
    assert report["feasible_hypervolume_curve"] is not None
    assert report["failures"] == 0  # 8 calls < fail_every=11
    assert report["integrity_ok"] is True

def test_mf_moo_progress_keeps_feasible_hv_below_raw(tmp_path):
    import importlib.util

    from condopt.benchmarks import builtin_problems

    script = ROOT / "scripts" / "adapter_long_run.py"
    spec = importlib.util.spec_from_file_location("adapter_long_run_mfmoo", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.OUT = tmp_path / "mfmoo_drill"
    report = module.run(
        budget=8, batch_size=4, workers=1, seed=0, patience=100,
        problem_name="mf_moo_constrained",
    )
    assert 7 <= report["n_observations"] <= 8
    assert report["multi_fidelity"] is True
    assert report["integrity_ok"] is True
    rows = [
        json.loads(line)
        for line in (tmp_path / "mfmoo_drill" / "progress.jsonl")
        .read_text(encoding="utf-8").splitlines()
    ]
    for row in rows:
        if row.get("hypervolume") is not None and row.get("hypervolume_feasible") is not None:
            assert row["hypervolume_feasible"] <= row["hypervolume"] + 1e-9

def test_adapter_long_run_supports_four_objective_problems(tmp_path):
    import importlib.util

    script = ROOT / "scripts" / "adapter_long_run.py"
    spec = importlib.util.spec_from_file_location("adapter_long_run_4obj", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.OUT = tmp_path / "adapter_long_run_4obj"
    report = module.run(
        budget=8, batch_size=4, workers=2, seed=0, patience=100,
        problem_name="mf_moo4",
    )
    assert 7 <= report["n_observations"] <= 8
    assert report["duplicate_suggestion_ids"] == 0
    assert report["final_hypervolume"] is not None
    assert report["integrity_ok"] is True
    # mf_moo4 declares no constraints, so the feasible-only curve stays empty
    assert report["feasible_hypervolume_curve"] is None
    # only full-fidelity runs count as deliverable evidence
    assert 0 < report["n_feasible_full_fidelity"] <= report["n_observations"]

def test_failed_conditions_are_abandoned_after_max_retries(tmp_path):
    study = _study(tmp_path, budget=6, batch_size=2)
    report = run_campaign(
        study,
        Broken(),
        max_empty_batches=99,   # rely on the retry limit instead of the guard
        max_retries=1,          # first failure -> abandoned on the second attempt
        log_path=tmp_path / "exec" / "log.jsonl",
    )
    assert report["cancelled_after_retries"] >= 1
    assert report["still_pending"] == 0  # nothing left to retry forever
    assert report["n_observations"] == 0
    decisions = (tmp_path / "exec" / "decisions.jsonl").read_text(encoding="utf-8")
    assert "cancel_pending" in decisions
    rows = [
        json.loads(line)
        for line in (tmp_path / "exec" / "log.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(row.get("abandoned") for row in rows)


def test_cancel_pending_keeps_the_queue_consistent(tmp_path):
    from condopt.base import Objective
    from condopt.space import SearchSpace
    from condopt.study import Study

    space = SearchSpace.from_dict([
        {"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0}
    ])
    study = Study.create(
        tmp_path / "cancel", space, Objective(name="y", direction="max"),
        algorithm="random", budget=4, batch_size=2, seed=0,
    )
    pending = study.ask(2)
    ids = [item["id"] for item in pending]
    assert len(study.pending) == 2
    cancelled = study.cancel_pending([ids[0]], reason="instrument offline")
    assert cancelled == [ids[0]]
    assert [item["id"] for item in study.pending] == [ids[1]]
    assert study.cancel_pending(["missing-id"]) == []

def test_failures_are_classified_by_cause(tmp_path):
    from condopt.executor import classify_failure, failure_summary

    assert classify_failure("timeout after 30s") == "timeout"
    assert classify_failure("RuntimeError: instrument offline") == "exception:RuntimeError"
    assert classify_failure("", abandoned=True) == "abandoned"
    assert classify_failure("") == "instrument_error"
    assert classify_failure("instrument error (simulated)") == "instrument_error"

    study = _study(tmp_path, budget=4, batch_size=2)
    report = run_campaign(
        study, Broken(), max_empty_batches=2, max_retries=1,
        log_path=tmp_path / "exec" / "log.jsonl",
    )
    kinds = report["failure_kinds"]
    assert sum(kinds.values()) == report["failures"]
    assert any(kind.startswith("exception:RuntimeError") for kind in kinds)
    assert "abandoned" in kinds

    summary = failure_summary(tmp_path / "exec" / "log.jsonl")
    assert summary["failures"] == report["failures"]
    assert summary["total"] == report["attempts"]


def test_timeout_failures_have_their_own_kind(tmp_path):
    study = _study(tmp_path, budget=2, batch_size=2)
    report = run_campaign(
        study, Hanging(), timeout=0.2, max_empty_batches=1, max_retries=-1,
        log_path=tmp_path / "exec" / "log.jsonl",
    )
    # one empty batch is enough: every failure is a timeout, and -- crucially --
    # the campaign returns promptly instead of waiting for the 5 s instrument
    assert report["failure_kinds"].get("timeout") == report["failures"] == 2

def test_report_summarises_the_execution_log(tmp_path):
    study = _study(tmp_path, budget=4, batch_size=2)
    run_campaign(
        study, Broken(), max_empty_batches=2, max_retries=1,
        log_path=study.dir / "execution_log.jsonl",
    )
    report = study.report()
    assert "执行日志：" in report
    assert "exception:RuntimeError" in report
    assert "次失败" in report


def test_report_has_no_log_line_without_an_execution_log(tmp_path):
    study = _study(tmp_path, budget=4, batch_size=2)
    run_campaign(study, Quadratic(), log_path=tmp_path / "exec" / "log.jsonl")
    assert "执行日志：" not in study.report()

def _digest(study_dir, out_path) -> str:
    """Run the digest command and return the generated markdown."""
    import contextlib
    import io

    from condopt.cli import main as cli_main

    with contextlib.redirect_stdout(io.StringIO()):
        code = cli_main(["digest", "--run", str(study_dir), "--out", str(out_path)])
    assert code == 0
    return Path(out_path).read_text(encoding="utf-8")


def test_digest_merges_the_execution_log_for_single_objective_runs(tmp_path):
    import importlib.util
    import sys
    from pathlib import Path

    script = ROOT / "scripts" / "adapter_long_run.py"
    spec = importlib.util.spec_from_file_location("adapter_long_run_digest", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.OUT = tmp_path / "digest_mf"
    report = module.run(
        budget=8, batch_size=4, workers=2, seed=0, patience=100,
        problem_name="mf_constrained",
    )
    digest = _digest(tmp_path / "digest_mf", tmp_path / "digest_mf.md")
    assert "执行日志：" in digest
    assert "预算:" in digest
    assert "下一步行动" in digest
    assert report["failure_kinds"] is not None


def test_digest_merges_the_progress_curve_for_multi_objective_runs(tmp_path):
    import importlib.util
    import sys
    from pathlib import Path

    script = ROOT / "scripts" / "adapter_long_run.py"
    spec = importlib.util.spec_from_file_location("adapter_long_run_digest_moo", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.OUT = tmp_path / "digest_moo"
    module.run(
        budget=8, batch_size=4, workers=2, seed=0, patience=100,
        problem_name="moo_constrained",
    )
    digest = _digest(tmp_path / "digest_moo", tmp_path / "digest_moo.md")
    assert "可交付前沿" in digest
    assert "进展:" in digest
    assert "执行日志：" in digest

def test_delivery_checklist_flags_missing_evidence(tmp_path):
    import importlib.util
    import sys
    from pathlib import Path

    script = ROOT / "scripts" / "adapter_long_run.py"
    spec = importlib.util.spec_from_file_location("adapter_long_run_checklist", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    module.OUT = tmp_path / "checklist_moo"
    module.run(budget=8, batch_size=4, workers=2, seed=0, patience=100,
               problem_name="moo_constrained")
    from condopt.study import Study

    study = Study.load(tmp_path / "checklist_moo")
    items = {item["item"]: item for item in study.delivery_checklist()}
    assert "非支配前沿存在" in items
    assert items["非支配前沿存在"]["ok"] is True
    assert "逐批进展记录" in items
    report = study.report()
    assert "交付自查:" in report
    assert "[OK ]" in report

def test_hand_written_loops_can_record_progress(tmp_path):
    from condopt.base import Objective
    from condopt.space import SearchSpace
    from condopt.study import Study

    space = SearchSpace.from_dict([
        {"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0}
    ])
    objective = Objective(
        name="yield", direction="max",
        objectives=("yield", "cost"), directions=("max", "min"),
    )
    study = Study.create(
        tmp_path / "manual_progress", space, objective, algorithm="qehvi",
        budget=4, batch_size=2, seed=0,
    )
    study.import_observations([
        {"x": {"x": 0.1}, "y": 0.9, "objectives": {"cost": 0.1}},
        {"x": {"x": 0.2}, "y": 0.8, "objectives": {"cost": 0.2}},
    ])
    row = study.record_progress()
    assert row is not None and row["round"] == 2
    assert (tmp_path / "manual_progress" / "progress.jsonl").exists()
    assert study.delivery_checklist()
    progress_item = next(
        item for item in study.delivery_checklist() if item["item"] == "逐批进展记录"
    )
    assert progress_item["ok"] is True
    # single-objective studies have no trajectory
    single = Study.create(
        tmp_path / "manual_single", space, Objective(name="y", direction="max"),
        algorithm="random", budget=4, batch_size=2, seed=0,
    )
    assert single.record_progress() is None

def test_delivery_evidence_benchmark_script_runs(tmp_path):
    import importlib.util
    import sys
    from pathlib import Path

    script = ROOT / "scripts" / "benchmark_delivery_evidence.py"
    spec = importlib.util.spec_from_file_location("benchmark_delivery_evidence", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.OUT = tmp_path / "delivery_evidence"
    row = module.run_seed("moo_yield_cost", seed=0, budget=12)
    assert row["delivery_ok"] is True, row["missing"]
    assert row["missing"] == ""
    assert row["items"] >= 2

def test_e2e_drill_passes_on_a_multifidelity_constrained_problem(tmp_path):
    import importlib.util
    import sys
    from pathlib import Path

    script = ROOT / "scripts" / "e2e_drill.py"
    spec = importlib.util.spec_from_file_location("e2e_drill_mf", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.OUT = tmp_path / "e2e_mf"

    report = module.run_drill("mf_constrained", target_budget=16)
    assert report["ok"] is True, report
    assert report["delivery_ok"] is True
    assert report["report_has_checklist"] is True
    assert report["integrity_ok"] is True

def test_run_campaign_wires_the_staged_prior_recipe(tmp_path):
    """prior_source must trigger the staged recipe inside the campaign loop."""
    from condopt.benchmarks import builtin_problems
    from condopt.executor import Executor, run_campaign
    from condopt.router import OptimizationContext
    from condopt.study import Study

    problem = builtin_problems()["mixed_reaction"]

    class Bench(Executor):
        def measure(self, request):
            return Measurement(id=request.id, y=float(problem.fn(request.x)), fidelity=1.0)

    context = OptimizationContext.from_space(
        problem.space, budget_total=21, batch_size=3, multi_fidelity=False,
        n_objectives=1, n_constraints=0, noise="medium",
    )
    source = Study.create(
        tmp_path / "source", problem.space, problem.objective, algorithm="random",
        budget=18, batch_size=3, context=context, seed=0,
    )
    pending = source.ask(18)
    source.tell([
        {"id": item["id"], "y": float(problem.fn(item["x"])), "fidelity": 1.0}
        for item in pending
    ])
    target = Study.create(
        tmp_path / "target", problem.space, problem.objective, algorithm="random",
        budget=21, batch_size=3, context=context, seed=1,
    )
    report = run_campaign(target, Bench(), prior_source=source.dir)
    assert report["ok"] is True
    events = report["transfer_staged"]
    assert events, "the staged recipe was never consulted"
    assert events[0]["stage"] == "first_look"
    assert events[0]["imported"] == 1
    # it settles exactly once: no repeated audits after widen/veto
    assert len(events) <= 3
    assert any(event["stage"] in {"widened", "vetoed", "waiting"} for event in events)


def test_run_campaign_can_skip_the_staged_recipe(tmp_path):
    from condopt.benchmarks import builtin_problems
    from condopt.executor import Executor, run_campaign
    from condopt.router import OptimizationContext
    from condopt.study import Study

    problem = builtin_problems()["mixed_reaction"]

    class Bench(Executor):
        def measure(self, request):
            return Measurement(id=request.id, y=float(problem.fn(request.x)), fidelity=1.0)

    context = OptimizationContext.from_space(
        problem.space, budget_total=9, batch_size=3, multi_fidelity=False,
        n_objectives=1, n_constraints=0, noise="medium",
    )
    target = Study.create(
        tmp_path / "target", problem.space, problem.objective, algorithm="random",
        budget=9, batch_size=3, context=context, seed=1,
    )
    report = run_campaign(target, Bench(), prior_source=target.dir, staged_transfer=False)
    assert report["transfer_staged"] == []

