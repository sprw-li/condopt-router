"""Tests for the optimizer contract and the built-in algorithms."""

import numpy as np
import pytest

from condopt import Objective, Observation, SearchSpace, registry


def simple_space() -> SearchSpace:
    return SearchSpace.from_dict(
        [
            {"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0},
            {"name": "y", "kind": "continuous", "low": 0.0, "high": 1.0},
        ]
    )


def mixed_space() -> SearchSpace:
    return SearchSpace.from_dict(
        [
            {"name": "T", "kind": "continuous", "low": 40.0, "high": 120.0},
            {"name": "conc", "kind": "log", "low": 0.01, "high": 1.0},
            {"name": "solvent", "kind": "categorical", "choices": ["MeCN", "EtOH", "water"]},
            {"name": "cycles", "kind": "integer", "low": 1, "high": 4},
        ]
    )


def test_registry_contents():
    names = registry.names()
    assert {"random", "sobol", "lhs", "grid", "bayes", "tpe", "parego", "constrained", "qehvi", "cqehvi"} <= set(names)
    for name in names:
        spec = registry.spec(name)
        assert spec.defaults()
        payload = spec.to_dict()
        assert payload["params"] and payload["summary"] and payload["family"]
        # defaults must validate against the schema itself
        assert registry.validate_params(name, {}) == spec.defaults()


def test_unknown_algorithm_and_param():
    with pytest.raises(KeyError):
        registry.get("no-such-algorithm")
    with pytest.raises(ValueError):
        registry.validate_params("bayes", {"bogus": 1})


def test_each_algorithm_ask_tell():
    space = mixed_space()
    objective = Objective(name="yield", direction="max")
    for name in registry.names():
        if name in ("qehvi", "qnehvi", "cqehvi", "cqnehvi", "qehvi_nd"):
            continue  # multi-objective/constraint-only; covered by their own test modules
        optimizer = registry.build(name, space, objective, seed=7)
        configs = optimizer.ask(3)
        assert len(configs) == 3
        for config in configs:
            space.validate_config(config)
        observations = [
            Observation(x=config, y=float(i))
            for i, config in enumerate(configs)
        ]
        optimizer.tell(observations)
        more = optimizer.ask(2)
        assert len(more) == 2
        for config in more:
            space.validate_config(config)


def test_bo_converges_on_quadratic():
    space = simple_space()
    objective = Objective(name="score", direction="max")

    def f(config: dict) -> float:
        return -((config["x"] - 0.3) ** 2) - ((config["y"] - 0.7) ** 2)

    optimizer = registry.build(
        "bayes",
        space,
        objective,
        params={"n_initial": 4, "n_candidates": 512},
        seed=0,
    )
    best = -np.inf
    for _ in range(20):
        for config in optimizer.ask(1):
            value = f(config)
            best = max(best, value)
            optimizer.tell([Observation(x=config, y=value)])
    assert best > -0.01, f"BO did not converge, best={best}"


def test_bo_handles_mixed_space():
    space = mixed_space()
    objective = Objective(name="yield", direction="max")
    optimizer = registry.build("bayes", space, objective, params={"n_initial": 3}, seed=1)
    told = 0
    for _ in range(4):
        for config in optimizer.ask(2):
            space.validate_config(config)
            optimizer.tell(
                [
                    Observation(
                        x=config,
                        y=float(np.random.default_rng(told).random()),
                    )
                ]
            )
            told += 1


def test_tpe_converges_on_mixed_space():
    space = SearchSpace.from_dict(
        [
            {"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0},
            {"name": "cat", "kind": "categorical", "choices": ["A", "B", "C", "D"]},
        ]
    )
    objective = Objective(name="y", direction="max")

    def f(config: dict) -> float:
        bonus = {"A": 0.0, "B": 1.0, "C": 0.3, "D": 0.6}[config["cat"]]
        return -((config["x"] - 0.7) ** 2) + bonus

    optimizer = registry.build("tpe", space, objective, seed=0)
    best = -np.inf
    for _ in range(24):
        config = optimizer.ask(1)[0]
        value = f(config)
        best = max(best, value)
        optimizer.tell([Observation(x=config, y=value)])
    assert best > 0.9, f"TPE did not converge, best={best}"


def test_bo_batch_is_distinct():
    space = simple_space()
    optimizer = registry.build(
        "bayes",
        space,
        Objective(),
        params={"n_initial": 2, "batch_strategy": "constant_liar"},
        seed=3,
    )
    optimizer.tell([Observation(x={"x": 0.2, "y": 0.1}, y=0.2)])
    batch = optimizer.ask(4)
    keys = {tuple(space.encode(c)) for c in batch}
    assert len(keys) == len(batch)


def test_minimize_direction():
    space = simple_space()
    objective = Objective(name="loss", direction="min")

    def loss(config: dict) -> float:
        return (config["x"] - 0.1) ** 2 + (config["y"] - 0.9) ** 2

    optimizer = registry.build("bayes", space, objective, params={"n_initial": 3}, seed=2)
    best = np.inf
    for _ in range(15):
        for config in optimizer.ask(1):
            value = loss(config)
            best = min(best, value)
            optimizer.tell([Observation(x=config, y=value)])
    assert best < 0.05, f"minimization failed, best={best}"


def test_log_ei_is_finite_for_far_candidates():
    from condopt.algorithms.bayes import _log_ei

    z = np.asarray([-60.0, -10.0, -1.0, 0.5])
    std = np.ones_like(z)
    imp = z * std
    values = _log_ei(imp, std, z)
    assert np.all(np.isfinite(values))
    # far candidates used to collapse to log(1e-300); they must differ now
    assert values[0] < values[1] < values[2]


def test_sobol_state_continues_sequence():
    space = simple_space()
    first = registry.build("sobol", space, seed=5)
    points_a = first.ask(3)
    state = first.state_dict()
    resumed = registry.build("sobol", space, seed=5)
    resumed.load_state_dict(state)
    points_b = resumed.ask(3)
    assert points_a != points_b
    # a fresh optimizer without state would repeat the first points
    fresh = registry.build("sobol", space, seed=5)
    assert fresh.ask(3) == points_a


def test_random_never_exceeds_space():
    space = SearchSpace.from_dict(
        [{"name": "n", "kind": "integer", "low": 0, "high": 2}]
    )
    optimizer = registry.build("random", space, seed=0)
    for config in optimizer.ask(20):
        assert 0 <= config["n"] <= 2


def test_ard_has_one_lengthscale_per_feature():
    space = mixed_space()
    optimizer = registry.build("bayes", space, Objective(), params={"ard": True}, seed=0)
    X = np.random.default_rng(0).random((8, space.feature_dim))
    y = np.random.default_rng(1).random(8)
    model = optimizer._fit(X, y)
    length_scales = model.kernel_.k2.length_scale
    assert np.ndim(length_scales) == 1
    assert len(length_scales) == space.feature_dim


def test_ard_frozen_when_few_observations():
    space = simple_space()  # feature_dim 2 -> threshold max(6, 4) = 6
    optimizer = registry.build("bayes", space, Objective(), params={"ard": True}, seed=0)
    X = np.random.default_rng(0).random((3, space.feature_dim))
    y = np.random.default_rng(1).random(3)
    model = optimizer._fit(X, y)
    assert model.optimizer is None
    X_many = np.random.default_rng(0).random((8, space.feature_dim))
    model_many = optimizer._fit(X_many, np.random.default_rng(1).random(8))
    assert model_many.optimizer is not None


def test_local_penalty_batch_is_spread_out():
    space = simple_space()
    optimizer = registry.build(
        "bayes",
        space,
        Objective(),
        params={"n_initial": 2, "batch_strategy": "local_penalty"},
        seed=0,
    )
    optimizer.tell([Observation(x={"x": 0.2, "y": 0.1}, y=0.2)])
    batch = optimizer.ask(4)
    keys = {tuple(space.encode(config)) for config in batch}
    assert len(keys) == len(batch)
    # not all points collapse onto one location
    points = np.vstack([space.encode(config) for config in batch])
    assert float(np.max(np.abs(points[:, None, :] - points[None, :, :]))) > 0.01


def test_ts_batch_uses_independent_draws():
    space = simple_space()
    optimizer = registry.build(
        "bayes",
        space,
        Objective(),
        params={"n_initial": 2, "acquisition": "ts"},
        seed=0,
    )
    optimizer.tell([Observation(x={"x": 0.2, "y": 0.1}, y=0.2)])
    batch = optimizer.ask(4)
    keys = {tuple(space.encode(config)) for config in batch}
    assert len(keys) == len(batch)


def test_bayes_initial_design_is_balanced_power_of_two():
    space = SearchSpace.from_dict(
        [{"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0}]
    )
    optimizer = registry.build("bayes", space, Objective(), params={"n_initial": 8}, seed=0)
    design = optimizer.ask(8)
    bins = sorted(int(config["x"] * 8) for config in design)
    assert bins == list(range(8))


def test_bayes_initial_queue_survives_state_roundtrip():
    space = SearchSpace.from_dict(
        [
            {"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0},
            {"name": "y", "kind": "continuous", "low": 0.0, "high": 1.0},
        ]
    )
    first = registry.build("bayes", space, Objective(), params={"n_initial": 8}, seed=3)
    first_batch = first.ask(4)
    state = first.state_dict()
    resumed = registry.build("bayes", space, Objective(), params={"n_initial": 8}, seed=3)
    resumed.load_state_dict(state)
    second_batch = resumed.ask(4)
    keys = {tuple(space.encode(config)) for config in first_batch + second_batch}
    assert len(keys) == 8
