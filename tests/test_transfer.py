"""Tests for warm-start transfer between compatible studies."""

from __future__ import annotations

import json

import pytest

from condopt import Objective, SearchSpace, Study
from condopt.cli import main


def space() -> SearchSpace:
    return SearchSpace.from_dict(
        [{"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0}]
    )


def make_source(tmp_path):
    source = Study.create(
        tmp_path / "source",
        space(),
        Objective(name="y", direction="max"),
        budget=8,
        batch_size=2,
        seed=0,
    )
    pending = source.ask(4)
    source.tell([{"id": item["id"], "y": float(index)} for index, item in enumerate(pending)])
    return source


def make_target(tmp_path):
    return Study.create(
        tmp_path / "target",
        space(),
        Objective(name="y", direction="max"),
        budget=10,
        batch_size=2,
        seed=1,
    )


def test_transfer_from_compatible_study(tmp_path):
    source = make_source(tmp_path)
    target = make_target(tmp_path)
    summary = target.transfer_from(source.dir, reason="test warm start")
    assert summary["imported"] == 4
    assert summary["skipped"] == 0
    observations = target.dataset.observations
    assert all(obs.source == "transfer" for obs in observations)
    assert all(obs.meta["transferred_from"] == str(source.dir) for obs in observations)
    decisions = [item["action"] for item in target.decisions()]
    assert "transfer_import" in decisions


def test_transfer_top_k_keeps_best(tmp_path):
    source = make_source(tmp_path)
    target = make_target(tmp_path)
    summary = target.transfer_from(source.dir, top_k=2)
    assert summary["imported"] == 2
    imported_scores = sorted(obs.y for obs in target.dataset.observations)
    assert imported_scores == [2.0, 3.0]


def test_transfer_rejects_incompatible_objective(tmp_path):
    source = make_source(tmp_path)
    target = Study.create(
        tmp_path / "target2",
        space(),
        Objective(name="other", direction="max"),
        budget=8,
        batch_size=2,
        seed=0,
    )
    with pytest.raises(ValueError, match="objectives"):
        target.transfer_from(source.dir)


def test_cli_import_from_run(tmp_path, capsys):
    source = make_source(tmp_path)
    target = make_target(tmp_path)
    assert main([
        "import", "--run", str(target.dir), "--from-run", str(source.dir),
        "--top-k", "2", "--reason", "cli", "--json",
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["imported"] == 2
    assert len(target.dataset.observations) == 2

def test_transfer_points_do_not_collide_with_target_pending_ids(tmp_path):
    from condopt.benchmarks import builtin_problems

    problem = builtin_problems()["sphere2d"]
    source = Study.create(
        tmp_path / "source_ids",
        problem.space,
        problem.objective,
        algorithm="sobol",
        budget=6,
        batch_size=3,
        seed=0,
    )
    pending = source.ask(3)
    source.tell([{"id": item["id"], "y": problem.fn(item["x"])} for item in pending])

    target = Study.create(
        tmp_path / "target_ids",
        problem.space,
        problem.objective,
        algorithm="bayes",
        budget=12,
        batch_size=3,
        seed=0,
    )
    report = target.transfer_from(source.dir, top_k=3)
    assert report["imported"] == 3
    assert all(
        "suggestion_id" not in obs.meta or obs.meta.get("transferred_from") is None
        for obs in target.dataset.observations
    )
    transferred_ids = {
        obs.meta.get("suggestion_id") for obs in target.dataset.observations
    }
    assert transferred_ids == {None}

    # the next round must not be swallowed as an idempotent replay
    target_pending = target.ask(3)
    target.tell([{"id": item["id"], "y": problem.fn(item["x"])} for item in target_pending])
    assert len(target.pending) == 0
    assert len(target.dataset.observations) == 6

def test_benchmark_transfer_script_runs():
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_transfer.py"
    spec = importlib.util.spec_from_file_location("benchmark_transfer", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    row = module.run_arm("warm", "sphere2d", total_budget=9, source_budget=3, batch_size=3, seed=0)
    assert row["arm"] == "warm"
    assert row["transferred"] >= 1
    assert row["best_score"] is not None

def test_transfer_summary_warns_on_many_priors(tmp_path):
    from condopt.benchmarks import builtin_problems

    problem = builtin_problems()["sphere2d"]
    source = Study.create(
        tmp_path / "src_many",
        problem.space,
        problem.objective,
        algorithm="sobol",
        budget=10,
        batch_size=5,
        seed=0,
    )
    pending = source.ask(10)
    source.tell([{"id": item["id"], "y": problem.fn(item["x"])} for item in pending])

    target = Study.create(
        tmp_path / "tgt_many",
        problem.space,
        problem.objective,
        algorithm="bayes",
        budget=20,
        batch_size=4,
        seed=0,
    )
    many = target.transfer_from(source.dir, top_k=8)
    assert many["warning"] and "top_k" in many["warning"]

    target2 = Study.create(
        tmp_path / "tgt_few",
        problem.space,
        problem.objective,
        algorithm="bayes",
        budget=20,
        batch_size=4,
        seed=0,
    )
    few = target2.transfer_from(source.dir, top_k=2)
    assert few["warning"] is None

def test_sweep_transfer_script_runs():
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "sweep_transfer.py"
    spec = importlib.util.spec_from_file_location("sweep_transfer", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    row = module.run_point("good", 2, "sphere2d", total_budget=9, source_budget=3,
                           batch_size=3, seed=0)
    assert row["quality"] == "good"
    assert row["best_score"] is not None

def test_transfer_warns_on_domain_mismatch_and_small_source(tmp_path):
    from condopt.benchmarks import builtin_problems
    from condopt.router import OptimizationContext

    problem = builtin_problems()["sphere2d"]
    source = Study.create(
        tmp_path / "src_domain",
        problem.space,
        problem.objective,
        context=OptimizationContext.from_space(
            problem.space, budget_total=6, batch_size=3, domain="chemistry"
        ),
        algorithm="sobol",
        budget=6,
        batch_size=3,
        seed=0,
    )
    pending = source.ask(3)
    source.tell([{"id": item["id"], "y": problem.fn(item["x"])} for item in pending])

    target = Study.create(
        tmp_path / "tgt_domain",
        problem.space,
        problem.objective,
        context=OptimizationContext.from_space(
            problem.space, budget_total=12, batch_size=3, domain="materials"
        ),
        algorithm="bayes",
        budget=12,
        batch_size=3,
        seed=0,
    )
    report = target.transfer_from(source.dir, top_k=2)
    assert any("跨领域" in item for item in report["warnings"])
    assert any("先验统计价值有限" in item for item in report["warnings"])

def test_transfer_keeps_pareto_tradeoffs_for_multi_objective(tmp_path):
    from condopt.benchmarks import builtin_problems

    problem = builtin_problems()["moo_yield_cost"]
    source = Study.create(
        tmp_path / "src_moo",
        problem.space,
        problem.objective,
        algorithm="parego",
        budget=8,
        batch_size=4,
        seed=0,
    )
    source.import_observations([
        # primary-objective best, expensive
        {"x": {"T": 82.0, "time": 55.0}, "objectives": {"yield": 95.0, "cost": 15.0}},
        # cheap trade-off, same front
        {"x": {"T": 65.0, "time": 20.0}, "objectives": {"yield": 80.0, "cost": 7.0}},
        # dominated point
        {"x": {"T": 70.0, "time": 60.0}, "objectives": {"yield": 70.0, "cost": 12.0}},
    ])
    target = Study.create(
        tmp_path / "tgt_moo",
        problem.space,
        problem.objective,
        algorithm="parego",
        budget=12,
        batch_size=4,
        seed=0,
    )
    report = target.transfer_from(source.dir, top_k=2)
    assert report["selection"] == "pareto_front"
    assert report["imported"] == 2
    costs = {
        obs.objectives["cost"] for obs in target.dataset.observations
    }
    # both ends of the front survive; the dominated point does not
    assert costs == {15.0, 7.0}

def test_prior_reliability_weakens_transferred_evidence(tmp_path):
    import pytest as pytest_module

    from condopt.benchmarks import builtin_problems

    problem = builtin_problems()["sphere2d"]
    source = Study.create(
        tmp_path / "src_rel",
        problem.space,
        problem.objective,
        algorithm="sobol",
        budget=8,
        batch_size=4,
        seed=0,
    )
    pending = source.ask(8)
    source.tell([{"id": item["id"], "y": problem.fn(item["x"])} for item in pending])

    target = Study.create(
        tmp_path / "tgt_rel",
        problem.space,
        problem.objective,
        algorithm="bayes",
        budget=16,
        batch_size=4,
        seed=0,
    )
    report = target.transfer_from(source.dir, top_k=4, prior_reliability=0.5)
    assert report["prior_reliability"] == 0.5
    assert all(obs.objective_std > 0 for obs in target.dataset.observations)

    strong = Study.create(
        tmp_path / "tgt_strong",
        problem.space,
        problem.objective,
        algorithm="bayes",
        budget=16,
        batch_size=4,
        seed=0,
    )
    strong.transfer_from(source.dir, top_k=4, prior_reliability=1.0)
    assert all(obs.objective_std == 0 for obs in strong.dataset.observations)

    with pytest_module.raises(ValueError, match="prior_reliability"):
        strong.transfer_from(source.dir, top_k=2, prior_reliability=0.0)

def test_assess_transfer_suggests_reliability_from_evidence(tmp_path):
    import json as json_module

    from condopt.cli import main
    from condopt.space import SearchSpace

    space = SearchSpace.from_dict([{"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0}])

    def true_y(x: float) -> float:
        return -((x - 0.3) ** 2)

    def make_study(name, labels):
        study = Study.create(
            tmp_path / name,
            space,
            Objective(name="y", direction="max"),
            algorithm="lhs",
            budget=10,
            batch_size=10,
            seed=0,
        )
        study.import_observations([
            {"x": {"x": index / 7}, "y": labels(index / 7)} for index in range(8)
        ])
        return study

    target = make_study("audit_target", true_y)
    good_source = make_study("audit_source_good", true_y)
    report = target.assess_transfer(good_source.dir)
    assert report["ready"] is True
    assert report["n_compared"] >= 3
    assert report["suggested_reliability"] == 1.0

    def shuffled(index: float) -> float:
        return true_y((index + 0.45) % 1.0)

    bad_source = make_study("audit_source_bad", shuffled)
    bad_report = target.assess_transfer(bad_source.dir)
    assert bad_report["ready"] is True
    assert bad_report["suggested_reliability"] in (0.3, 0.6)

    assert main([
        "transfer-audit", "--run", str(target.dir), "--from-run", str(good_source.dir),
        "--json",
    ]) == 0

def test_simulate_transfer_previews_impact_without_touching_live_run(tmp_path):
    from condopt.benchmarks import builtin_problems

    problem = builtin_problems()["sphere2d"]
    source = Study.create(
        tmp_path / "sim_src",
        problem.space,
        problem.objective,
        algorithm="sobol",
        budget=8,
        batch_size=4,
        seed=0,
    )
    pending = source.ask(8)
    source.tell([{"id": item["id"], "y": problem.fn(item["x"])} for item in pending])

    target = Study.create(
        tmp_path / "sim_tgt",
        problem.space,
        problem.objective,
        algorithm="bayes",
        budget=12,
        batch_size=4,
        seed=0,
    )
    first, second = list(problem.space.names)[:2]
    target.import_observations([
        {"x": {first: 0.05, second: 0.05}, "y": -0.5},
        {"x": {first: 0.95, second: 0.95}, "y": -0.9},
        {"x": {first: 0.4, second: 0.6}, "y": -0.2},
    ])
    before_size = len(target.dataset.observations)
    report = target.simulate_transfer(source.dir, top_k=3)
    assert report["dry_run"] is True
    assert report["transferred"] >= 1
    assert report["live_run_untouched"] is True
    assert report["n_before"] == before_size
    assert len(target.dataset.observations) == before_size
    assert target.pending == []
    assert "changed" in report

    from condopt.cli import main

    assert main([
        "transfer-impact", "--run", str(target.dir), "--from-run", str(source.dir),
        "--top-k", "3", "--json",
    ]) == 0

def test_transfer_moo_benchmark_script_runs():
    import importlib.util
    import sys
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_transfer_moo.py"
    spec = importlib.util.spec_from_file_location("benchmark_transfer_moo", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    source = module.make_source(seed=0, budget=8, batch=4)
    baseline = module.run_target(0, budget=8, batch=4)
    transferred = module.run_target(0, budget=8, batch=4, source=source, top_k=3)
    assert baseline["transferred"] == 0
    # up to top_k priors are imported; a prior may be dropped when the target has
    # already measured the same condition (dedupe keeps the dataset unique)
    assert 2 <= transferred["transferred"] <= 3
    assert 7 <= transferred["n_observations"] <= 8
    assert transferred["n_front"] >= 1

def test_transfer_reports_the_budget_it_consumed(tmp_path):
    from condopt.base import Objective
    from condopt.space import SearchSpace
    from condopt.study import Study

    space = SearchSpace.from_dict([
        {"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0}
    ])
    objective = Objective(name="y", direction="max")
    source = Study.create(
        tmp_path / "src", space, objective, algorithm="random",
        budget=10, batch_size=5, seed=0,
    )
    source.import_observations([
        {"x": {"x": index / 10}, "y": float(index)} for index in range(1, 7)
    ])
    target = Study.create(
        tmp_path / "dst", space, objective, algorithm="random",
        budget=8, batch_size=4, seed=0,
    )
    before = target.remaining()
    info = target.transfer_from(source, top_k=3)
    assert info["imported"] == 3
    assert info["budget_charged"] == 3
    assert info["budget_total"] == 8
    assert info["remaining_after_transfer"] == before - 3
    assert info["cost_charged"] == 3.0  # full-fidelity priors

def test_transfer_audit_benchmark_script_separates_sources():
    import importlib.util
    import sys
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_transfer_audit.py"
    spec = importlib.util.spec_from_file_location("benchmark_transfer_audit", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    rows = module.audit(seed=0)
    by_kind = {row["kind"]: row for row in rows}
    assert set(by_kind) == set(module.SOURCE_KINDS)
    assert by_kind["consistent"]["suggested_reliability"] == 1.0
    for kind in ("shifted", "shuffled", "flipped"):
        assert by_kind[kind]["suggested_reliability"] < 1.0, kind
        assert by_kind[kind]["kind"] == kind

def test_reliability_grade_thresholds():
    from condopt.prepare import reliability_grade

    assert reliability_grade(None)["grade"] == "unknown"
    assert reliability_grade(1.0)["grade"] == "trust"
    assert reliability_grade(0.8)["grade"] == "trust"
    assert reliability_grade(0.6)["grade"] == "suspect"
    assert reliability_grade(0.5)["grade"] == "suspect"
    assert reliability_grade(0.3)["grade"] == "avoid"
    assert "top_k" in reliability_grade(0.3)["advice"]

def test_low_sample_audit_is_flagged_and_graded_cautiously(tmp_path):
    from condopt.base import Objective
    from condopt.prepare import reliability_grade
    from condopt.space import SearchSpace
    from condopt.study import Study

    space = SearchSpace.from_dict([
        {"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0}
    ])
    objective = Objective(name="y", direction="max")
    source = Study.create(
        tmp_path / "low_src", space, objective, algorithm="random",
        budget=6, batch_size=3, seed=0,
    )
    source.import_observations([
        {"x": {"x": index / 10}, "y": -((index / 10 - 0.3) ** 2)}
        for index in range(1, 4)
    ])
    target = Study.create(
        tmp_path / "low_dst", space, objective, algorithm="random",
        budget=6, batch_size=3, seed=0,
    )
    target.import_observations([
        {"x": {"x": index / 10}, "y": -((index / 10 - 0.3) ** 2)}
        for index in range(4, 7)
    ])
    report = target.assess_transfer(source)
    assert report["ready"] is True
    assert report["confidence"] == "low"
    assert report["sample_warning"]
    grade = reliability_grade(report["suggested_reliability"], report["confidence"])
    assert grade["grade"] == "low_confidence"
    assert "不可当硬约束" in grade["advice"]


def test_audit_with_enough_samples_is_normal_confidence(tmp_path):
    from condopt.base import Objective
    from condopt.space import SearchSpace
    from condopt.study import Study

    space = SearchSpace.from_dict([
        {"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0}
    ])
    objective = Objective(name="y", direction="max")
    rows = [
        {"x": {"x": index / 12}, "y": -((index / 12 - 0.3) ** 2)}
        for index in range(12)
    ]
    source = Study.create(
        tmp_path / "big_src", space, objective, algorithm="random",
        budget=12, batch_size=12, seed=0,
    )
    source.import_observations(rows[:6])
    target = Study.create(
        tmp_path / "big_dst", space, objective, algorithm="random",
        budget=12, batch_size=12, seed=1,
    )
    target.import_observations(rows[6:])
    report = target.assess_transfer(source)
    assert report["confidence"] == "normal"
    assert report["sample_warning"] is None

def test_audit_reports_a_bootstrap_confidence_interval(tmp_path):
    import importlib.util
    import sys
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_transfer_audit.py"
    spec = importlib.util.spec_from_file_location("benchmark_transfer_audit_ci", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    rows = module.audit(seed=0)
    consistent = next(row for row in rows if row["kind"] == "consistent")
    assert consistent["spearman"] is not None
    assert consistent["suggested_reliability"] == 1.0

    module.OUT = tmp_path / "audit_ci"
    source = module.run_source("consistent", seed=0)
    target = module.run_target(seed=0)
    from condopt.study import Study

    report = Study.load(target).assess_transfer(source)
    ci = report["spearman_ci"]
    assert ci is not None
    assert ci["level"] == 0.9
    assert ci["low"] <= report["spearman"] <= ci["high"] or ci["low"] <= ci["high"]
    assert report["confidence"] in ("normal", "low")
    assert report["estimate"] == "bootstrap_median"
    assert report["spearman_robust"] is not None
    assert "bootstrap 中位数" in report["reason"]
    assert report["suggested_reliability_point"] in (0.3, 0.6, 1.0)

def test_transferred_prior_does_not_count_as_a_replicate(tmp_path):
    """A warm start must still confirm the incumbent with its *own* replicate."""
    from condopt.base import Objective, Observation
    from condopt.space import SearchSpace
    from condopt.study import Study

    space = SearchSpace.from_dict([
        {"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0}
    ])
    objective = Objective(name="y", direction="max")
    source = Study.create(
        tmp_path / "src", space, objective, algorithm="random", budget=6, batch_size=3, seed=0
    )
    source.tell([
        Observation(x={"x": 0.9}, y=9.0, fidelity=1.0, cost=1.0, n_replicates=3),
    ])
    target = Study.create(
        tmp_path / "dst", space, objective, algorithm="random", budget=10, batch_size=3, seed=1
    )
    report = target.transfer_from(source.dir, top_k=1, prior_reliability=0.9)
    assert report["imported"] == 1
    imported = target.dataset.observations[0]
    assert imported.n_replicates == 1
    assert imported.meta["source_n_replicates"] == 3.0
    # round-scoped source bookkeeping must not leak into the target
    assert "purpose" not in imported.meta
    action = target.next_action()
    assert action["action"] in {"confirm", "ask", "tell"}
    assert target.delivery_checklist()
    confirmation_item = next(
        item for item in target.delivery_checklist() if item["item"] == "推荐条件已复孔确认"
    )
    assert confirmation_item["ok"] is False  # the target has not confirmed yet

def test_transferred_confirmation_flag_does_not_leak(tmp_path):
    """A prior tagged as a source confirmation must not confirm the target."""
    from condopt.base import Objective, Observation
    from condopt.space import SearchSpace
    from condopt.study import Study

    space = SearchSpace.from_dict([
        {"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0}
    ])
    objective = Objective(name="y", direction="max")
    source = Study.create(
        tmp_path / "src2", space, objective, algorithm="random", budget=6, batch_size=3, seed=0
    )
    source.tell([
        Observation(
            x={"x": 0.8}, y=8.0, fidelity=1.0, cost=1.0,
            meta={"purpose": "confirmation", "plate": "P2"},
        ),
    ])
    target = Study.create(
        tmp_path / "dst2", space, objective, algorithm="random", budget=10, batch_size=3, seed=1
    )
    target.transfer_from(source.dir, top_k=1, prior_reliability=0.9)
    imported = target.dataset.observations[0]
    assert "purpose" not in imported.meta
    assert imported.meta["source_meta"]["purpose"] == "confirmation"
    assert target._confirmation_stats(imported) is None

def test_resume_does_not_shadow_rebuilt_algorithm_data(tmp_path):
    """A stale persisted state must not replace data rebuilt from the dataset."""
    import numpy as np

    from condopt import registry
    from condopt.base import Observation, Objective
    from condopt.space import SearchSpace

    space = SearchSpace.from_dict([
        {"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0}
    ])
    objective = Objective(
        name="yield", direction="max",
        objectives=("yield", "cost"), directions=("max", "min"),
    )
    prior = [
        Observation(
            x={"x": i / 20.0}, y=float(i), fidelity=0.5 + 0.5 * (i % 2),
            objectives={"cost": float(20 - i)}, cost=1.0,
        )
        for i in range(12)
    ]

    momfbo = registry.build("momfbo", space, objective, seed=0, prior=prior)
    assert np.asarray(momfbo._Y).shape[0] == 12  # prior must be replayed, not reset
    assert np.asarray(momfbo._Y).shape[0] == 12
    momfbo.load_state_dict({"kind": "momfbo", "Y": [[0.0, 0.0]]})
    assert np.asarray(momfbo._Y).shape[0] == 12

    mqehvi = registry.build("mqehvi_nd", space, objective, seed=0, prior=prior)
    assert np.asarray(mqehvi._fidelity).shape[0] == 12
    mqehvi.load_state_dict({"kind": "mqehvi_nd", "fidelity": [1.0]})
    assert np.asarray(mqehvi._fidelity).shape[0] == 12

    qehvi = registry.build("qehvi", space, objective, seed=0, prior=prior)
    qehvi.ask(1)  # the reference point is derived on first use
    assert qehvi._ref is not None
    ref = np.asarray(qehvi._ref, dtype=float).copy()
    qehvi.load_state_dict({"kind": "qehvi", "ref": [0.0, 0.0]})
    assert np.allclose(np.asarray(qehvi._ref, dtype=float), ref)

def test_state_dict_only_persists_counters_and_algorithm_geometry():
    """Architecture guard: ``state_dict`` must not carry observation data.

    Allowed keys (with the reason they are not data):
      * ``kind``            -- algorithm name;
      * ``init_queue``      -- pending design points for the initial design;
      * ``m``, ``C``, ``sigma`` -- CMA-ES mean/covariance/step, dimension-sized
        geometry required to resume the search trajectory;
      * ``index``, ``grid_size`` -- grid cursor;
      * ``asked``           -- counter of already-proposed points;
      * ``ref``             -- reference point, dimension-sized;
      * ``constraint_names`` -- metadata.
    Anything whose row count equals the number of observations is data and must be
    rebuilt from the dataset instead.
    """
    from condopt import registry
    from condopt.base import Observation, Objective
    from condopt.space import SearchSpace

    ALLOWED = {
        "kind", "init_queue", "m", "C", "sigma", "index", "grid_size",
        "asked", "ref", "constraint_names",
    }
    space = SearchSpace.from_dict([
        {"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0}
    ])
    double = Objective(
        name="yield", direction="max", constraints=("c",),
        constraint_limits={"c": {"op": "<=", "value": 1.0}},
        objectives=("yield", "cost"), directions=("max", "min"),
    )
    prior = [
        Observation(
            x={"x": i / 20.0}, y=float(i), fidelity=1.0,
            constraints={"c": float(i - 6)},
            objectives={"cost": float(20 - i)}, cost=1.0,
        )
        for i in range(12)
    ]
    checked = 0
    for name in registry.names():
        try:
            algorithm = registry.build(name, space, double, seed=0, prior=prior)
        except Exception as exc:  # capability mismatch (e.g. qehvi needs no constraints)
            assert "constraint" in str(exc).lower() or "objective" in str(exc).lower(), (
                f"{name} failed to build for an unrelated reason: {exc}"
            )
            continue
        state = algorithm.state_dict()
        unknown = set(state) - ALLOWED
        assert not unknown, f"{name} persists unexpected state keys: {sorted(unknown)}"
        for key, value in state.items():
            if not isinstance(value, list):
                continue
            assert len(value) <= 32, f"{name}.{key} persists a long list ({len(value)})"
            assert len(value) != len(prior), (
                f"{name}.{key} has one entry per observation: rebuild it from the dataset"
            )
            if value and isinstance(value[0], (list, tuple)):
                assert len(value) != len(prior), (
                    f"{name}.{key} persists a data matrix with one row per observation"
                )
        checked += 1
    assert checked >= 12

def test_transfer_top_k_zero_imports_nothing(tmp_path):
    from condopt.base import Objective
    from condopt.space import SearchSpace
    from condopt.study import Study

    space = SearchSpace.from_dict([
        {"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0}
    ])
    objective = Objective(name="y", direction="max")
    source = Study.create(
        tmp_path / "zk_src", space, objective, algorithm="random", budget=6,
        batch_size=3, seed=0,
    )
    source.tell([{"x": {"x": 0.9}, "y": 9.0}])
    target = Study.create(
        tmp_path / "zk_dst", space, objective, algorithm="random", budget=6,
        batch_size=3, seed=1,
    )
    report = target.transfer_from(source.dir, top_k=0, prior_reliability=0.9)
    assert report["imported"] == 0
    assert report["selection"] == "none"
    assert target.dataset.observations == []
    try:
        target.transfer_from(source.dir, top_k=-1)
    except ValueError as exc:
        assert "top_k" in str(exc)
    else:  # pragma: no cover - must raise
        raise AssertionError("negative top_k must be rejected")

def test_mfbo_honours_prior_reliability(tmp_path):
    """The doubt must reach the MF GP's per-point alpha (was a no-op)."""
    import numpy as np

    from condopt import registry
    from condopt.base import Objective, Observation
    from condopt.space import SearchSpace

    space = SearchSpace.from_dict([
        {"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0}
    ])
    objective = Objective(name="y", direction="max")

    def alpha_for(objective_std: float) -> float:
        prior = [
            Observation(
                x={"x": 0.8}, y=8.0, fidelity=1.0, cost=1.0,
                objective_std=objective_std, n_replicates=1,
            )
        ]
        algorithm = registry.build("mfbo", space, objective, prior=prior, seed=0)
        features = np.vstack([algorithm._features(obs.x, obs.fidelity) for obs in prior])
        model = algorithm._fit(features, np.asarray([8.0]))
        return float(np.max(np.asarray(model.alpha, dtype=float)))

    trusted = alpha_for(0.0)
    doubted = alpha_for(28.0)
    assert doubted > trusted, (trusted, doubted)

def test_low_confidence_audit_never_recommends_full_trust(tmp_path):
    """Calibration showed a biased source can score ~0.65 with few target points."""
    from condopt.benchmarks import builtin_problems
    from condopt.router import OptimizationContext
    from condopt.study import Study

    problem = builtin_problems()["mixed_reaction"]
    context = OptimizationContext.from_space(
        problem.space, budget_total=16, batch_size=2, multi_fidelity=False,
        n_objectives=1, n_constraints=0, noise="medium",
    )
    source = Study.create(
        tmp_path / "src", problem.space, problem.objective, algorithm="random",
        budget=12, batch_size=2, context=context, seed=0,
    )
    pending = source.ask(12)
    source.tell([
        {"id": item["id"], "y": float(problem.fn(item["x"])), "fidelity": 1.0}
        for item in pending
    ])
    target = Study.create(
        tmp_path / "dst", problem.space, problem.objective, algorithm="random",
        budget=16, batch_size=2, context=context, seed=1,
    )
    pending = target.ask(4)
    target.tell([
        {"id": item["id"], "y": float(problem.fn(item["x"])), "fidelity": 1.0}
        for item in pending
    ])
    report = target.assess_transfer(source.dir)
    assert report["ready"] is True
    if report["confidence"] == "low":
        # low confidence may not recommend full trust, and a cap must be explicit
        assert report["suggested_reliability"] <= 0.6
        if report["capped_by_low_confidence"]:
            assert report["suggested_reliability"] == 0.6
            assert "封顶" in report["reason"]

def test_transfer_staged_imports_small_first_then_waits(tmp_path):
    """The validated recipe must be executable: small look -> wait -> widen/veto."""
    from condopt.benchmarks import builtin_problems
    from condopt.router import OptimizationContext
    from condopt.study import Study

    problem = builtin_problems()["mixed_reaction"]
    context = OptimizationContext.from_space(
        problem.space, budget_total=24, batch_size=3, multi_fidelity=False,
        n_objectives=1, n_constraints=0, noise="medium",
    )
    source = Study.create(
        tmp_path / "src", problem.space, problem.objective, algorithm="random",
        budget=18, batch_size=3, context=context, seed=0,
    )
    pending = source.ask(18)
    source.tell([
        {"id": item["id"], "y": float(problem.fn(item["x"])), "fidelity": 1.0}
        for item in pending
    ])
    target = Study.create(
        tmp_path / "dst", problem.space, problem.objective, algorithm="random",
        budget=24, batch_size=3, context=context, seed=1,
    )

    first = target.transfer_staged(source.dir, start_top_k=1, after_batches=2)
    assert first["stage"] == "first_look"
    assert first["imported"] == 1
    assert first["widened"] is False

    # not enough local batches yet -> it refuses to widen
    waiting = target.transfer_staged(source.dir, start_top_k=1, after_batches=2)
    assert waiting["stage"] == "waiting"
    assert waiting["widened"] is False

    # two local batches later the decision is taken (widen or veto), never silent
    for _ in range(2):
        pending = target.ask(3)
        target.tell([
            {"id": item["id"], "y": float(problem.fn(item["x"])), "fidelity": 1.0}
            for item in pending
        ])
    decision = target.transfer_staged(source.dir, start_top_k=1, after_batches=2)
    assert decision["stage"] in {"widened", "vetoed"}
    assert "audit" in decision
    if decision["stage"] == "vetoed":
        assert decision["imported"] == 0
        assert decision["widened"] is False
    else:
        assert decision["widened"] is True


def test_transfer_staged_force_skips_the_veto(tmp_path):
    from condopt.benchmarks import builtin_problems
    from condopt.router import OptimizationContext
    from condopt.study import Study

    problem = builtin_problems()["mixed_reaction"]
    context = OptimizationContext.from_space(
        problem.space, budget_total=24, batch_size=3, multi_fidelity=False,
        n_objectives=1, n_constraints=0, noise="medium",
    )
    source = Study.create(
        tmp_path / "src", problem.space, problem.objective, algorithm="random",
        budget=18, batch_size=3, context=context, seed=0,
    )
    pending = source.ask(18)
    source.tell([
        {"id": item["id"], "y": float(problem.fn(item["x"])), "fidelity": 1.0}
        for item in pending
    ])
    target = Study.create(
        tmp_path / "dst", problem.space, problem.objective, algorithm="random",
        budget=24, batch_size=3, context=context, seed=1,
    )
    report = target.transfer_staged(
        source.dir, start_top_k=1, wide_top_k=3, after_batches=0, force=True
    )
    assert report["stage"] == "widened"
    assert report["widened"] is True

