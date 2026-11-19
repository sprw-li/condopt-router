"""Tests for the skill-questionnaire -> JSON preparation helper."""

import json

import pytest

from condopt.prepare import parse_answers, template, write_problem


def test_template_is_valid_answers():
    space, objective, context = parse_answers(template())
    assert len(space) == 6
    assert objective.name == "yield"
    assert objective.n_objectives == 1
    assert context.batch_size == 4
    assert context.budget_total == 24


def test_aliases_and_derived_fields():
    answers = {
        "variables": [{"name": "x", "kind": "continuous", "low": 0, "high": 1}],
        "objective": {
            "name": "yield",
            "direction": "max",
            "objectives": ["yield", "cost"],
            "directions": ["max", "min"],
        },
        "budget": 20,
        "batch": 2,
        "prior": 4,
        "constraints": 1,
        "domain": "chemistry",
    }
    space, objective, context = parse_answers(answers)
    assert objective.n_objectives == 2
    assert context.n_objectives == 2
    assert context.n_constraints == 1
    assert context.n_observations == 4
    assert context.budget_remaining == 16


def test_missing_variables_raises():
    with pytest.raises(ValueError, match="variables"):
        parse_answers({"budget": 10})


def test_inconsistent_objective_count_raises():
    with pytest.raises(ValueError, match="不一致"):
        parse_answers(
            {
                "variables": [{"name": "x", "kind": "continuous", "low": 0, "high": 1}],
                "objective": {"name": "y", "objectives": ["y", "c"], "directions": ["max", "min"]},
                "n_objectives": 1,
            }
        )


def test_write_problem_creates_files_and_recommendation(tmp_path):
    result = write_problem(template(), tmp_path / "workspace")
    for key in ("space", "context", "objective"):
        assert (tmp_path / "workspace" / f"{key}.json").exists()
        json.loads((tmp_path / "workspace" / f"{key}.json").read_text(encoding="utf-8"))
    assert result["recommendation"]["algorithm"] == "bayes"
    assert result["recommendation"]["params"]


def test_constraint_objective_routes_to_constrained():
    answers = {
        "variables": [{"name": "T", "kind": "continuous", "low": 40, "high": 120}],
        "objective": {
            "name": "yield",
            "direction": "max",
            "constraints": ["T_max"],
            "constraint_limits": {"T_max": {"op": "<=", "value": 110.0}},
        },
        "budget": 24,
        "batch": 2,
    }
    space, objective, context = parse_answers(answers)
    assert objective.constraints == ("T_max",)
    assert context.n_constraints == 1
    from condopt.router import route

    assert route(context).algorithm == "constrained"


def test_constraint_without_limits_raises():
    answers = {
        "variables": [{"name": "T", "kind": "continuous", "low": 40, "high": 120}],
        "objective": {"name": "yield", "direction": "max", "constraints": ["T_max"]},
    }
    with pytest.raises(ValueError, match="constraint_limits"):
        parse_answers(answers)


def test_cli_prepare(tmp_path, capsys):
    from condopt.cli import main

    answers = tmp_path / "answers.json"
    answers.write_text(json.dumps(template(), ensure_ascii=False), encoding="utf-8")
    code = main(["prepare", "--answers", str(answers), "--out", str(tmp_path / "ws"), "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["recommendation"]["algorithm"] == "bayes"
    assert (tmp_path / "ws" / "space.json").exists()

def test_prepare_recommends_a_stopping_policy(tmp_path):
    from condopt.prepare import stopping_advice, template, write_problem
    from condopt.router import OptimizationContext

    small = stopping_advice(OptimizationContext(n_dims=3, n_continuous=3, budget_total=24))
    large = stopping_advice(OptimizationContext(n_dims=3, n_continuous=3, budget_total=120))
    assert small["recommended"] == "stagnation_first"
    assert large["recommended"] == "budget_first"
    assert "69/120" in large["why"]  # cites the measurement
    assert "stop_on_stagnation" in large["how_to_enable"]

    result = write_problem(template(), tmp_path / "prepared")
    assert result["stopping"]["recommended"] in ("budget_first", "stagnation_first")
    assert (tmp_path / "prepared" / "space.json").exists()

def test_prepare_warns_about_a_tiny_feasible_region(tmp_path):
    from condopt.prepare import feasibility_advice, template, write_problem
    from condopt.base import Objective
    from condopt.router import OptimizationContext

    objective = Objective(
        name="y", direction="max", constraints=("a", "b", "c"),
        constraint_limits={n: {"op": "<=", "value": 0.0} for n in ("a", "b", "c")},
    )
    context = OptimizationContext(n_dims=4, n_continuous=4, budget_total=40, n_constraints=3)
    inferred = feasibility_advice({}, context, objective)
    assert inferred["risk"] == "high"
    assert inferred["estimated_feasible_fraction"] == 0.05
    assert "写进变量范围" in inferred["advice"]

    explicit = feasibility_advice({"feasible_fraction_estimate": 0.02}, context, objective)
    assert explicit["risk"] == "high"
    assert explicit["estimated_feasible_fraction"] == 0.02

    relaxed = feasibility_advice(
        {"feasible_fraction_estimate": 0.6},
        OptimizationContext(n_dims=4, n_continuous=4, budget_total=40, n_constraints=1),
        objective,
    )
    assert relaxed["risk"] == "low"

    no_constraints = feasibility_advice(
        {}, OptimizationContext(n_dims=2, n_continuous=2, budget_total=20), Objective()
    )
    assert no_constraints["risk"] == "none"

    result = write_problem(template(), tmp_path / "prepared_feas")
    assert "feasibility" in result

def test_prepare_advises_on_prior_data(tmp_path):
    from condopt.prepare import prior_advice, template, write_problem
    from condopt.router import OptimizationContext

    none = prior_advice({}, OptimizationContext(n_dims=2, n_continuous=2, budget_total=20))
    assert none["has_prior"] is False

    some = prior_advice(
        {"prior": 6},
        OptimizationContext(n_dims=2, n_continuous=2, budget_total=20, n_observations=6),
    )
    assert some["has_prior"] is True
    assert some["n_prior"] == 6
    assert "transfer-audit" in some["advice"]
    assert "top_k" in some["advice"]
    assert "−1190" in some["evidence"] or "-1190" in some["evidence"]

    result = write_problem(template(), tmp_path / "prepared_prior")
    assert result["prior"]["has_prior"] is False

def test_prepare_prior_advice_switches_to_pareto_mode_for_multi_objective():
    from condopt.base import Objective
    from condopt.prepare import prior_advice
    from condopt.router import OptimizationContext

    objective = Objective(
        name="yield", direction="max",
        objectives=("yield", "cost"), directions=("max", "min"),
    )
    context = OptimizationContext(
        n_dims=2, n_continuous=2, budget_total=40, n_objectives=2, n_observations=6
    )
    advice = prior_advice({"prior": 6}, context, objective)
    assert advice["mode"] == "pareto_front"
    assert "非支配前沿" in advice["advice"]
    assert "top_k" in advice["advice"]
    assert "pareto_front" in advice["how_to_enable"]
    assert "11.6" in advice["evidence"]  # cites the measured front-size gain

    single = prior_advice({"prior": 6}, context, Objective(name="y", direction="max"))
    assert single.get("mode") != "pareto_front"

    mf_context = OptimizationContext(
        n_dims=2, n_continuous=2, budget_total=40, n_objectives=2,
        multi_fidelity=True, n_observations=6,
    )
    mf_advice = prior_advice({"prior": 6}, mf_context, objective)
    assert mf_advice["mode"] == "pareto_front_mf"
    assert "预算成本" in mf_advice["advice"]
    assert "629.09" in mf_advice["evidence"]
    assert "620.94" in mf_advice["evidence"]

def test_prepare_advises_on_the_deliverable_metric():
    from condopt.base import Objective
    from condopt.prepare import deliverable_advice
    from condopt.router import OptimizationContext

    single = deliverable_advice(
        {}, OptimizationContext(n_dims=2, n_continuous=2, budget_total=20),
        Objective(name="y", direction="max"),
    )
    assert single["metric"] == "best_value"
    assert any("复孔" in item for item in single["must_include"])

    constrained_moo = deliverable_advice(
        {},
        OptimizationContext(
            n_dims=3, n_continuous=3, budget_total=40, n_objectives=2, n_constraints=1
        ),
        Objective(
            name="yield", direction="max",
            objectives=("yield", "cost"), directions=("max", "min"),
            constraints=("c",), constraint_limits={"c": {"op": "<=", "value": 1.0}},
        ),
    )
    assert constrained_moo["metric"] == "feasible_hypervolume"
    assert any("可行" in item for item in constrained_moo["must_include"])

    mf = deliverable_advice(
        {},
        OptimizationContext(
            n_dims=2, n_continuous=2, budget_total=30, multi_fidelity=True
        ),
        Objective(name="y", direction="max"),
    )
    assert any("全保真" in item for item in mf["must_include"])
    assert "digest" in mf["how_to_check"]

    small = deliverable_advice(
        {}, OptimizationContext(n_dims=2, n_continuous=2, budget_total=12),
        Objective(name="y", direction="max"),
    )
    assert small["confirmation_plan"]["n"] == 3
    assert 0.1 < small["confirmation_plan"]["share_of_budget"] < 0.3
    large = deliverable_advice(
        {}, OptimizationContext(n_dims=2, n_continuous=2, budget_total=120),
        Objective(name="y", direction="max"),
    )
    assert large["confirmation_plan"]["n"] == 3

def test_stopping_advice_accounts_for_the_delivery_slots():
    from condopt.prepare import stopping_advice
    from condopt.router import OptimizationContext

    small = stopping_advice(OptimizationContext(n_dims=2, n_continuous=2, budget_total=12))
    assert small["delivery_slots"] == 3
    assert small["effective_exploration_budget"] == 9
    assert "9 次" in small["delivery_note"]

    tiny = stopping_advice(OptimizationContext(n_dims=2, n_continuous=2, budget_total=4))
    assert tiny["delivery_slots"] == 3
    assert tiny["effective_exploration_budget"] == 1

    big = stopping_advice(OptimizationContext(n_dims=3, n_continuous=3, budget_total=120))
    assert big["recommended"] == "budget_first"
    assert big["effective_exploration_budget"] == 117
    assert big["delivery_tradeoff"] is not None
    assert "交付自查" in big["delivery_tradeoff"]
    assert small["delivery_tradeoff"] is None

def test_noise_advice_turns_the_level_into_actions():
    from condopt.prepare import noise_advice
    from condopt.router import OptimizationContext

    high = noise_advice(
        {"noise": "high"}, OptimizationContext(n_dims=3, n_continuous=3, budget_total=30, batch_size=12)
    )
    assert high["replicates"] == 3
    assert high["baseline_arm"] is True
    # measurement says UCB is *worse*, so the advice does not switch acquisition
    assert high["params"] == {}
    assert "UCB" in high["advice"] and "A/B" in high["advice"]
    assert "批量" in high["advice"]  # batch 12 is too large for noisy runs

    low = noise_advice(
        {"noise": "low"}, OptimizationContext(n_dims=3, n_continuous=3, budget_total=30)
    )
    assert low["replicates"] == 1
    assert low["baseline_arm"] is False
    assert low["params"]["alpha"] == 1.0

