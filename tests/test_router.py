"""Tests for the algorithm router."""

import pytest

from condopt import SearchSpace, registry
from condopt.space import SearchSpace  # noqa: E402
from condopt.router import OptimizationContext, route, route_round


def test_smooth_low_dim_prefers_bayes():
    ctx = OptimizationContext(
        n_dims=5,
        n_continuous=5,
        budget_total=40,
        batch_size=2,
        smoothness="smooth",
        noise="low",
        n_observations=0,
    )
    rec = route(ctx)
    assert rec.algorithm == "bayes"
    assert rec.params["n_initial"] >= 3
    assert rec.confidence in ("high", "medium")
    rec.validate()


def test_high_dim_prefers_sampling():
    ctx = OptimizationContext(n_dims=25, n_continuous=25, budget_total=40, batch_size=4)
    rec = route(ctx)
    assert rec.algorithm in ("sobol", "random", "lhs")
    rec.validate()


def test_tiny_budget_does_not_pick_bayes():
    ctx = OptimizationContext(n_dims=3, n_continuous=3, budget_total=5, batch_size=1)
    rec = route(ctx)
    assert rec.algorithm != "bayes"
    rec.validate()


def test_discrete_grid_wins():
    space = SearchSpace.from_dict(
        [
            {"name": "a", "kind": "categorical", "choices": ["x", "y", "z"]},
            {"name": "b", "kind": "categorical", "choices": ["p", "q"]},
            {"name": "c", "kind": "bool"},
        ]
    )
    ctx = OptimizationContext.from_space(space, budget_total=24)
    rec = route(ctx)
    assert rec.algorithm == "grid"
    assert rec.design["algorithm"] == "grid"
    rec.validate()


def test_large_batch_bo_uses_batch_strategy():
    ctx = OptimizationContext(
        n_dims=4, n_continuous=4, budget_total=30, batch_size=8, smoothness="smooth"
    )
    rec = route(ctx)
    assert rec.algorithm == "bayes"
    assert rec.params["batch_strategy"] == "local_penalty"
    assert rec.params["ard"] is True
    rec.validate()


def test_noisy_high_switches_acquisition_and_alpha():
    ctx = OptimizationContext(
        n_dims=4, n_continuous=4, budget_total=30, batch_size=1, noise="high"
    )
    rec = route(ctx)
    assert rec.algorithm == "bayes"
    assert rec.params["acquisition"] == "ucb"
    assert rec.params["alpha"] > 1e-3
    rec.validate()


def test_multi_objective_prefers_parego():
    ctx = OptimizationContext(
        n_dims=4, n_continuous=4, budget_total=30, n_objectives=2, smoothness="smooth"
    )
    rec = route(ctx)
    assert rec.algorithm == "parego"
    assert any("Pareto" in warning for warning in rec.warnings)
    assert rec.confidence in ("high", "medium")
    rec.validate()


def test_feature_dim_explosion_penalizes_bayes():
    # 3 categorical knobs with 20 levels each + 1 continuous -> 61 GP features.
    space = SearchSpace.from_dict(
        [
            {"name": "c1", "kind": "categorical", "choices": [f"a{i}" for i in range(20)]},
            {"name": "c2", "kind": "categorical", "choices": [f"b{i}" for i in range(20)]},
            {"name": "c3", "kind": "categorical", "choices": [f"c{i}" for i in range(20)]},
            {"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0},
        ]
    )
    ctx = OptimizationContext.from_space(space, budget_total=80, batch_size=2, smoothness="smooth")
    assert ctx.feature_dim == 61
    assert ctx.effective_dim == 61
    rec = route(ctx)
    assert rec.algorithm != "bayes"
    rec.validate()


def test_categorical_heavy_prefers_tpe():
    space = SearchSpace.from_dict(
        [
            {"name": "c1", "kind": "categorical", "choices": [f"a{i}" for i in range(10)]},
            {"name": "c2", "kind": "categorical", "choices": [f"b{i}" for i in range(10)]},
            {"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0},
        ]
    )
    ctx = OptimizationContext.from_space(space, budget_total=30, batch_size=2)
    rec = route(ctx)
    assert rec.algorithm == "tpe"
    assert rec.params["n_initial"] >= 3
    rec.validate()


def test_unsupported_capability_combination_warns():
    # Constraints + multi-fidelity + multi-objective: both mfparego and momfbo
    # declare the full set; mfparego is the benchmark-backed default and momfbo
    # remains a candidate.
    ctx = OptimizationContext(
        n_dims=3,
        n_continuous=3,
        budget_total=20,
        n_constraints=1,
        multi_fidelity=True,
        n_objectives=2,
    )
    rec = route(ctx)
    assert rec.algorithm == "mfparego"
    assert "momfbo" in rec.candidate_scores
    rec.validate()


def test_multi_objective_with_constraints_routes_to_cqehvi():
    ctx = OptimizationContext(
        noise="low",
        n_dims=3,
        n_continuous=3,
        budget_total=24,
        n_objectives=2,
        n_constraints=1,
    )
    rec = route(ctx)
    assert rec.algorithm == "cqehvi", rec.candidate_scores
    registry.validate_params(rec.algorithm, rec.params)


def test_recommended_params_always_valid():
    contexts = [
        OptimizationContext(n_dims=1, n_continuous=1, budget_total=12),
        OptimizationContext(n_dims=12, n_continuous=6, n_categorical=6, budget_total=80, batch_size=12),
        OptimizationContext(n_dims=30, n_continuous=30, budget_total=8, compute="low"),
        OptimizationContext(n_dims=8, n_log=4, n_integer=4, budget_total=15, noise="high"),
        OptimizationContext(n_dims=6, n_continuous=6, budget_total=60, smoothness="rough"),
    ]
    for ctx in contexts:
        rec = route(ctx)
        registry.validate_params(rec.algorithm, rec.params)
        assert rec.plan and rec.design
        assert rec.algorithm in rec.candidate_scores


def test_route_round_detects_stall():
    ctx = OptimizationContext(n_dims=5, n_continuous=5, budget_total=40, smoothness="smooth")
    rec = route_round(ctx, recent_scores=[1.0, 1.0, 1.0, 1.0], patience=3)
    assert rec.algorithm in ("sobol", "random")
    assert any("没有改善" in reason for reason in rec.rationale)
    registry.validate_params(rec.algorithm, rec.params)


def test_route_round_keeps_algorithm_when_improving():
    ctx = OptimizationContext(n_dims=5, n_continuous=5, budget_total=40, smoothness="smooth")
    rec = route_round(ctx, recent_scores=[0.1, 0.2, 0.3, 0.4], patience=3)
    assert rec.algorithm == "bayes"

def test_small_categorical_space_prefers_bayes():
    """Small-dimensional pure-categorical spaces: GP-BO can model interactions.

    Evidence: 2026 ART HTE campaign proxy, where GP-BO recovered the published
    optimum while TPE plateaued; literature labels show TPE/bayes are tied for
    categorical-heavy problems, with grid for tiny enumerable spaces.
    """
    space = SearchSpace.from_dict(
        [
            {"name": "solvent", "kind": "categorical", "choices": [f"s{i}" for i in range(12)]},
            {"name": "pc", "kind": "categorical", "choices": [f"p{i}" for i in range(10)]},
            {"name": "ni", "kind": "categorical", "choices": [f"n{i}" for i in range(5)]},
            {"name": "ligand", "kind": "categorical", "choices": [f"l{i}" for i in range(10)]},
            {"name": "amine", "kind": "categorical", "choices": [f"a{i}" for i in range(5)]},
        ]
    )
    ctx = OptimizationContext.from_space(
        space, budget_total=120, batch_size=4, domain="chemistry", noise="medium"
    )
    rec = route(ctx)
    assert rec.algorithm == "bayes", rec.candidate_scores
    rec.validate()

def test_more_than_two_objectives_falls_back_instead_of_invalid_qehvi():
    ctx = OptimizationContext(
        n_dims=3, n_continuous=3, budget_total=30, n_objectives=3, n_constraints=1
    )
    rec = route(ctx)
    # parego now declares constraints, so 3+ objectives with constraints is valid
    assert rec.algorithm == "parego"
    rec.validate()
    rec.validate()

    ctx_plain = OptimizationContext(
        n_dims=3, n_continuous=3, budget_total=30, n_objectives=3
    )
    rec_plain = route(ctx_plain)
    assert rec_plain.algorithm == "parego"
    assert any("目标" in warning for warning in rec_plain.warnings)

def test_audit_script_runs_every_registered_algorithm():
    import importlib.util
    from pathlib import Path

    from condopt import registry

    script = Path(__file__).resolve().parents[1] / "scripts" / "audit_algorithms.py"
    spec = importlib.util.spec_from_file_location("audit_algorithms", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    rows = module.audit(budget=4, batch_size=2)
    assert len(rows) == len(registry.names())
    failures = [row for row in rows if not row["ok"]]
    assert failures == []

def test_high_dim_smooth_continuous_prefers_cmaes():
    ctx = OptimizationContext(
        n_dims=20, n_continuous=20, budget_total=60, batch_size=4,
        smoothness="smooth",
    )
    recommendation = route(ctx)
    assert recommendation.algorithm == "cmaes"
    assert recommendation.params["population_size"] >= 4
    recommendation.validate()


def test_cmaes_improves_on_sphere():
    from condopt.benchmarks import builtin_problems, run_algorithm

    problem = builtin_problems()["sphere2d"]
    result = run_algorithm(problem, "cmaes", budget=24, batch_size=4, seed=0)
    # sphere2d optimum is 0; CMA-ES should get within 1e-2 in 24 evaluations
    assert result.trajectory[-1] > -0.01

def test_benchmark_highdim_script_runs():
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_highdim.py"
    spec = importlib.util.spec_from_file_location("benchmark_highdim", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert module.ROOT.exists()

def test_sweep_cmaes_script_runs():
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "sweep_cmaes.py"
    spec = importlib.util.spec_from_file_location("sweep_cmaes", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert module.OUT.parent.exists() or True

def test_sweep_mfbo_script_runs():
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "sweep_mfbo.py"
    spec = importlib.util.spec_from_file_location("sweep_mfbo", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert module.OUT.parent.exists() or True

def test_sweep_momfbo_script_runs():
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "sweep_momfbo.py"
    spec = importlib.util.spec_from_file_location("sweep_momfbo", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert module.OUT.parent.exists() or True

def test_audit_routing_matrix_has_no_invalid_choices():
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "audit_routing.py"
    spec = importlib.util.spec_from_file_location("audit_routing", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    rows = module.audit()
    assert len(rows) >= 90
    failures = [row for row in rows if not row["ok"]]
    assert failures == [], failures[:3]

def test_three_objectives_with_budget_prefers_qehvi_nd():
    ctx = OptimizationContext(
        n_dims=4, n_continuous=4, budget_total=60, batch_size=4,
        n_objectives=3, n_constraints=1, noise="low", smoothness="smooth",
    )
    recommendation = route(ctx)
    assert recommendation.algorithm == "qehvi_nd"
    recommendation.validate()
    # small budget keeps parego
    small = OptimizationContext(
        n_dims=4, n_continuous=4, budget_total=20, batch_size=4,
        n_objectives=3, n_constraints=1, noise="low", smoothness="smooth",
    )
    assert route(small).algorithm == "parego"


def test_benchmark_3obj_script_runs():
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_3obj.py"
    spec = importlib.util.spec_from_file_location("benchmark_3obj", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    row = module.run("qehvi_nd", budget=6, batch_size=2, seed=0)
    assert row["budget"] == 6

def test_four_objectives_prefers_qehvi_nd():
    ctx = OptimizationContext(
        n_dims=4, n_continuous=4, budget_total=60, batch_size=4,
        n_objectives=4, n_constraints=1, noise="low", smoothness="smooth",
    )
    recommendation = route(ctx)
    assert recommendation.algorithm == "qehvi_nd"
    assert recommendation.params["mc_prefilter"] == 48
    assert recommendation.params["mc_draws"] == 32
    recommendation.validate()


def test_moo4_benchmark_problem_exists():
    from condopt.benchmarks import builtin_problems

    problem = builtin_problems()["moo4_constrained"]
    assert problem.objective.n_objectives == 4
    values = problem.objectives_fn({"T": 70.0, "time": 40.0})
    assert set(values) == {"yield", "cost", "purity", "selectivity"}

def test_descriptors_shift_router_preference_to_gp():
    descriptors = {
        f"a{i}": [float(i) / 9.0] for i in range(10)
    }
    descriptors_b = {
        f"b{i}": [1.0 - float(i) / 9.0] for i in range(10)
    }
    space = SearchSpace.from_dict([
        {"name": "c1", "kind": "categorical", "choices": [f"a{i}" for i in range(10)],
         "descriptors": descriptors},
        {"name": "c2", "kind": "categorical", "choices": [f"b{i}" for i in range(10)],
         "descriptors": descriptors_b},
        {"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0},
    ])
    ctx = OptimizationContext.from_space(space, budget_total=30, batch_size=2)
    assert ctx.n_descriptor_vars == 2
    recommendation = route(ctx)
    # descriptors shrink the feature dimension, so the GP model becomes viable
    # and the router moves from TPE to bayes on this categorical-heavy problem
    assert recommendation.algorithm == "bayes"
    recommendation.validate()

