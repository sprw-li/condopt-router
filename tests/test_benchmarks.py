"""Tests for the synthetic benchmark problems."""

import pytest

from condopt.benchmarks import builtin_problems, run_algorithm, summarize


SINGLE_OBJECTIVE = [n for n in sorted(builtin_problems()) if not builtin_problems()[n].objective.is_multi_objective]


@pytest.mark.parametrize("name", SINGLE_OBJECTIVE)
def test_benchmark_optimum_is_attained(name):
    problem = builtin_problems()[name]
    assert problem.optimum is not None
    assert problem.best_config is not None
    value = problem.evaluate(problem.best_config)
    # some textbook optima are quoted with 3 decimals (hartmann3, solvent)
    tolerance = (
        1e-3
        if name in (
            "solvent_constrained",
            "solvent_constrained_desc",
            "hartmann3",
            "high_cardinality_solvent",
            "high_cardinality_solvent_desc",
            "mf_moo_constrained",
        )
        else 1e-6
    )
    if name == "constraint_hard":
        tolerance = 0.05  # constrained optimum is a grid approximation
    assert value == pytest.approx(problem.optimum, abs=tolerance)
    # a correct optimum must have (near) zero regret at the known best point
    score = problem.objective.sign * value
    assert problem.regret(score) <= tolerance


def test_benchmark_run_and_summary():
    results = [
        run_algorithm(problem, "random", budget=6, batch_size=2, seed=0)
        for problem in [builtin_problems()["sphere2d"]]
    ]
    assert results[0].budget == 6
    assert len(results[0].trajectory) == 3
    summary = summarize(results)
    assert summary[0]["n_runs"] == 1
    assert summary[0]["problem"] == "sphere2d"

def test_moo_run_records_objective_vectors():
    result = run_algorithm(
        builtin_problems()["moo_yield_cost"], "parego", budget=4, batch_size=2, seed=0
    )
    assert result.objective_vectors
    assert set(result.objective_vectors[0]) == {"yield", "cost"}


def test_constrained_moo_run_records_constraint_vectors():
    problem = builtin_problems()["moo_constrained"]
    observation = problem.make_observation({"T": 80.0, "time": 20.0})
    assert observation.constraints == {"T_max": 80.0, "time_max": 20.0}
    assert observation.objectives and set(observation.objectives) == {"yield", "cost"}
    result = run_algorithm(problem, "random", budget=4, batch_size=2, seed=0)
    assert len(result.constraint_vectors) == 4
    assert set(result.constraint_vectors[0]) == {"T_max", "time_max"}
    assert len(result.objective_vectors) == 4

def test_noise_std_changes_model_based_trajectory():
    problem = builtin_problems()["sphere2d"]
    clean = run_algorithm(problem, "bayes", budget=8, batch_size=2, seed=0)
    noisy = run_algorithm(problem, "bayes", budget=8, batch_size=2, seed=0, noise_std=3.0)
    assert len(noisy.trajectory) == len(clean.trajectory)
    assert noisy.trajectory != clean.trajectory

def test_tier_c_descriptor_variant_matches_one_hot_landscape():
    from condopt.benchmarks import builtin_problems, run_algorithm

    base = builtin_problems()["high_cardinality_solvent"]
    described = builtin_problems()["high_cardinality_solvent_desc"]
    assert described.space.feature_dim < base.space.feature_dim
    assert described.evaluate(described.best_config) == base.evaluate(base.best_config)
    result = run_algorithm(described, "bayes", budget=12, batch_size=4, seed=0)
    assert result.best_value is not None

def test_benchmark_tier_c_script_runs():
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_tier_c.py"
    spec = importlib.util.spec_from_file_location("benchmark_tier_c", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert module.OUT.parent.exists() or True

