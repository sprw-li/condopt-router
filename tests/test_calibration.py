"""Tests for router weight calibration helpers and the MVP script."""

import pytest

from condopt import SearchSpace
from condopt.calibration import (
    RunRecord,
    best_algorithm_per_problem,
    collect_records,
    evaluate_weights,
    lopo_validate,
    random_search_weights,
)
from condopt.router import OptimizationContext


def smooth_context() -> dict:
    space = SearchSpace.from_dict(
        [
            {"name": "T", "kind": "continuous", "low": 40.0, "high": 120.0},
            {"name": "time", "kind": "continuous", "low": 5.0, "high": 120.0},
        ]
    )
    return OptimizationContext.from_space(
        space, budget_total=40, batch_size=2, smoothness="smooth"
    ).to_dict()


def synthetic_records() -> list[RunRecord]:
    context = smooth_context()
    records = []
    for problem in ("p1", "p2", "p3"):
        # bayes is the hand-tuned default, but the empirical label is sobol
        records.append(RunRecord(problem, "bayes", 80.0, context))
        records.append(RunRecord(problem, "sobol", 90.0, context))
        records.append(RunRecord(problem, "random", 20.0, context))
    return records


def test_best_algorithm_labels():
    labels = best_algorithm_per_problem(synthetic_records())
    assert labels == {"p1": "sobol", "p2": "sobol", "p3": "sobol"}


def test_evaluate_weights_can_override_choice():
    records = synthetic_records()
    default = evaluate_weights(records, None)
    assert default["n_problems"] == 3
    tuned = evaluate_weights(records, {"bayes": 0.4, "sobol": 1.6, "random": 1.0})
    assert tuned["top1"] >= default["top1"]


def test_random_search_returns_weights():
    records = synthetic_records()
    result = random_search_weights(records, iterations=40, seed=0)
    assert set(result) >= {"weights", "metrics", "default_metrics"}
    assert set(result["weights"]) == {"bayes", "sobol", "random"}
    assert result["n_iterations"] == 40


def test_ties_count_as_correct():
    context = smooth_context()
    records = [
        RunRecord("p", "bayes", 80.0, context),
        RunRecord("p", "sobol", 80.0, context),
        RunRecord("p", "random", 20.0, context),
    ]
    metrics = evaluate_weights(records, None)
    assert metrics["top1"] == 1.0
    assert metrics["top2"] == 1.0
    assert metrics["mean_regret"] == 0.0


def test_lopo_validation_is_out_of_sample():
    records = synthetic_records()
    result = lopo_validate(records, iterations=20, seed=0)
    assert result["n_problems"] == 3
    assert 0.0 <= result["top1"] <= 1.0
    assert len(result["folds"]) == 3
    assert all("held_out" in fold for fold in result["folds"])


def test_collect_records_filters_by_capability():
    # qehvi is 2-objective only: it must be skipped on a single-objective problem
    single = collect_records(
        ["sphere2d"], ["random", "qehvi"], budget=4, batch_size=2, seeds=[0]
    )
    assert {record.algorithm for record in single} == {"random"}
    # and a single-objective algorithm must be skipped on a MOO problem
    multi = collect_records(
        ["moo_yield_cost"], ["random", "qehvi"], budget=4, batch_size=2, seeds=[0]
    )
    assert {record.algorithm for record in multi} == {"random", "qehvi"} or {
        record.algorithm for record in multi
    } == {"qehvi"}


def test_benchmark_problem_has_capability_metadata():
    from condopt.benchmarks import builtin_problems

    for problem in builtin_problems().values():
        assert isinstance(problem.tags, tuple)
        assert float(problem.noise_std) >= 0.0


def test_collect_records_context_includes_noise():
    records = collect_records(
        ["sphere2d"], ["random"], budget=4, batch_size=2, seeds=[0], noise="high", smoothness="rough"
    )
    assert records[0].context["noise"] == "high"
    assert records[0].context["smoothness"] == "rough"


def test_collect_records_small_matrix():
    records = collect_records(
        ["sphere2d"], ["random", "sobol"], budget=4, batch_size=2, seeds=[0]
    )
    assert len(records) == 2
    assert {record.algorithm for record in records} == {"random", "sobol"}
    assert all(record.metric_name == "true_best" for record in records)
    assert all(record.context for record in records)

def test_collect_records_uses_true_hv_for_moo():
    records = collect_records(
        ["moo_yield_cost"], ["qehvi"], budget=6, batch_size=2, seeds=[0]
    )
    assert len(records) == 1
    assert records[0].metric_name == "true_hv"
    assert 0.0 <= records[0].score <= 1.05

def test_collect_records_uses_feasible_best_for_constrained():
    records = collect_records(
        ["constraint_easy"], ["constrained"], budget=8, batch_size=2, seeds=[0]
    )
    assert len(records) == 1
    assert records[0].metric_name == "feasible_best"
    assert records[0].score > -1e300

def test_collect_records_uses_cost_aware_metric_for_multifidelity():
    records = collect_records(
        ["mf_suzuki"], ["mfbo"], budget=8, batch_size=2, seeds=[0]
    )
    assert len(records) == 1
    assert records[0].metric_name == "cost_aware_best"
    assert records[0].score > 0

def test_collect_records_uses_cost_aware_feasible_metric_for_mf_constrained():
    records = collect_records(
        ["mf_constrained"], ["mfbo"], budget=8, batch_size=2, seeds=[0]
    )
    assert len(records) == 1
    assert records[0].metric_name == "cost_aware_feasible_best"
    assert records[0].score > -1e300

def test_collect_records_sets_multi_fidelity_context_from_problem_tags():
    records = collect_records(
        ["mf_suzuki"], ["mfbo"], budget=6, batch_size=2, seeds=[0]
    )
    assert len(records) == 1
    assert records[0].context["multi_fidelity"] is True

def test_calibrate_router_defaults_cover_multi_fidelity_and_constraints():
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "calibrate_router.py"
    spec = importlib.util.spec_from_file_location("calibrate_router", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert "mf_constrained" in module.DEFAULT_PROBLEMS
    assert "mfbo" in module.DEFAULT_ALGORITHMS
    assert "constrained" in module.DEFAULT_ALGORITHMS

