"""Tests for the mixed search space encoding."""

import math

import numpy as np
import pytest

from condopt.space import SearchSpace, Variable


def mixed_space() -> SearchSpace:
    return SearchSpace.from_dict(
        [
            {"name": "T", "kind": "continuous", "low": 40.0, "high": 120.0, "unit": "degC"},
            {"name": "t", "kind": "integer", "low": 10, "high": 60},
            {"name": "conc", "kind": "log", "low": 0.01, "high": 1.0},
            {"name": "solvent", "kind": "categorical", "choices": ["MeCN", "EtOH", "water"]},
            {"name": "stage", "kind": "ordinal", "choices": ["early", "mid", "late"]},
            {"name": "inert", "kind": "bool"},
        ]
    )


def sample_config(space: SearchSpace) -> dict:
    return {
        "T": 87.5,
        "t": 33,
        "conc": 0.1,
        "solvent": "EtOH",
        "stage": "mid",
        "inert": True,
    }


def test_dimensions_and_counts():
    space = mixed_space()
    assert space.dim == 6
    assert space.feature_dim == 3 + 3 + 3 + 2  # 3 numeric + one-hot groups
    counts = space.kind_counts()
    assert counts["continuous"] == 1
    assert counts["integer"] == 1
    assert counts["log"] == 1
    assert counts["categorical"] == 1
    assert counts["ordinal"] == 1
    assert counts["bool"] == 1
    assert not space.all_discrete()


def test_encode_decode_roundtrip():
    space = mixed_space()
    config = sample_config(space)
    decoded = space.decode(space.encode(config))
    assert decoded == config or all(
        decoded[name] == pytest.approx(value)
        if isinstance(value, float)
        else decoded[name] == value
        for name, value in config.items()
    )


def test_encode_decode_exhaustive_discrete():
    space = SearchSpace.from_dict(
        [
            {"name": "solvent", "kind": "categorical", "choices": ["MeCN", "EtOH", "water"]},
            {"name": "stage", "kind": "ordinal", "choices": ["early", "mid", "late"]},
            {"name": "inert", "kind": "bool"},
        ]
    )
    for solvent in ("MeCN", "EtOH", "water"):
        for stage in ("early", "mid", "late"):
            for inert in (False, True):
                cfg = {"solvent": solvent, "stage": stage, "inert": inert}
                assert space.decode(space.encode(cfg)) == cfg


def test_validate_config_errors():
    space = mixed_space()
    config = sample_config(space)
    missing = dict(config)
    del missing["t"]
    with pytest.raises(ValueError, match="missing"):
        space.validate_config(missing)
    extra = dict(config)
    extra["bogus"] = 1
    with pytest.raises(ValueError, match="unknown"):
        space.validate_config(extra)
    bad = dict(config)
    bad["t"] = 61
    with pytest.raises(ValueError):
        space.validate_config(bad)
    bad_choice = dict(config)
    bad_choice["solvent"] = "hexane"
    with pytest.raises(ValueError):
        space.validate_config(bad_choice)


def test_variable_validation():
    with pytest.raises(ValueError):
        Variable("x", "log", low=0.0, high=1.0)
    with pytest.raises(ValueError):
        Variable("x", "categorical", choices=["a"])
    with pytest.raises(ValueError):
        Variable("x", "integer", low=0.5, high=3)
    with pytest.raises(ValueError):
        SearchSpace((Variable("x", "continuous", low=0, high=1), Variable("x", "continuous", low=0, high=2)))


def test_feature_encoding():
    space = mixed_space()
    config = sample_config(space)
    features = space.features(config)
    assert features.shape == (space.feature_dim,)
    assert np.all(features >= 0) and np.all(features <= 1)
    # T is first: (87.5 - 40) / 80 = 0.59375
    assert features[0] == pytest.approx(0.59375)
    # one-hot for solvent EtOH (index 1)
    assert features[3 + 3] == 0.0
    assert features[3 + 3 + 1] == 1.0


def test_sampling_shapes_and_types():
    space = mixed_space()
    rng = np.random.default_rng(0)
    for sample in space.sample_uniform(5, rng):
        space.validate_config(sample)
    for sampler in (space.sample_sobol, space.sample_lhs):
        points = sampler(8, rng=0)
        assert len(points) == 8
        keys = {tuple(space.encode(p)) for p in points}
        assert len(keys) == 8  # continuous space: no collisions


def test_sobol_base2_balance_and_prefix():
    space = SearchSpace.from_dict(
        [
            {"name": "a", "kind": "continuous", "low": 0.0, "high": 1.0},
            {"name": "b", "kind": "continuous", "low": 0.0, "high": 1.0},
        ]
    )
    design = space.sample_sobol_base2(8, seed=0)
    assert len(design) == 8
    # 8 balanced points: every 1/8 bin in each dimension has exactly one point
    for name in ("a", "b"):
        bins = np.floor(np.asarray([item[name] for item in design]) * 8).astype(int)
        assert sorted(bins.tolist()) == list(range(8))
    prefix = space.sample_sobol_base2(5, seed=0)
    assert len(prefix) == 5
    assert prefix == space.sample_sobol_base2(5, seed=0)


def test_sampling_is_deterministic():
    space = mixed_space()
    a = space.sample_sobol(6, rng=42)
    b = space.sample_sobol(6, rng=42)
    assert a == b


def test_grid_sample():
    space = SearchSpace.from_dict(
        [
            {"name": "a", "kind": "categorical", "choices": ["x", "y", "z"]},
            {"name": "b", "kind": "continuous", "low": 0.0, "high": 1.0},
        ]
    )
    grid = space.sample_grid(max_points=10, levels=5)
    assert 0 < len(grid) <= 10
    assert len({tuple(space.encode(g)) for g in grid}) == len(grid)
    for item in grid:
        space.validate_config(item)


def test_log_encoding_endpoints():
    space = mixed_space()
    conc = space.variable("conc")
    assert conc.decode(0.0) == pytest.approx(0.01)
    assert conc.decode(1.0) == pytest.approx(1.0)
    assert conc.decode(0.5) == pytest.approx(math.sqrt(0.01 * 1.0))


def test_space_json_roundtrip(tmp_path):
    space = mixed_space()
    path = tmp_path / "space.json"
    space.to_json(str(path))
    loaded = SearchSpace.from_json(str(path))
    assert loaded.to_dict() == space.to_dict()

def test_categorical_descriptors_replace_one_hot():
    space = SearchSpace.from_dict([
        {
            "name": "solvent",
            "kind": "categorical",
            "choices": ["DMF", "EtOH"],
            "descriptors": {"DMF": [73.09, -1.0, 36.7], "EtOH": [46.07, -0.3, 24.5]},
            "descriptor_names": ["mw", "logp", "dielectric"],
        },
        {"name": "T", "kind": "continuous", "low": 0.0, "high": 100.0},
    ])
    variable = space.variable("solvent")
    assert variable.feature_size() == 3
    assert space.feature_dim == 4  # 3 descriptors + 1 numeric
    # descriptor dimensions are min-max scaled onto [0, 1] for the GP
    features = space.features({"solvent": "DMF", "T": 50.0})
    assert features.tolist() == pytest.approx([1.0, 0.0, 1.0, 0.5])
    assert variable.descriptor_names == ("mw", "logp", "dielectric")
    assert variable.raw_descriptor_vector("DMF") == pytest.approx([73.09, -1.0, 36.7])

    restored = SearchSpace.from_dict(space.to_dict())
    assert restored.features({"solvent": "EtOH", "T": 25.0}).tolist() == pytest.approx(
        [0.0, 1.0, 0.0, 0.25]
    )
    assert restored.variable("solvent").raw_descriptor_vector("EtOH") == pytest.approx(
        [46.07, -0.3, 24.5]
    )


def test_categorical_descriptors_validation():
    # partial tables are allowed: unknown choices get zeros + indicator dim
    partial = SearchSpace.from_dict([
        {"name": "s", "kind": "categorical", "choices": ["a", "b"],
         "descriptors": {"a": [1.0]}}
    ])
    variable = partial.variable("s")
    assert variable.has_descriptor("a") is True
    assert variable.has_descriptor("b") is False
    assert variable.feature_size() == 2  # 1 descriptor + 1 unknown indicator
    assert partial.features({"s": "a"}).tolist() == pytest.approx([0.0, 0.0])
    assert partial.features({"s": "b"}).tolist() == pytest.approx([0.0, 1.0])
    with pytest.raises(ValueError, match="at least one choice"):
        SearchSpace.from_dict([
            {"name": "s", "kind": "categorical", "choices": ["a", "b"],
             "descriptors": {"z": [1.0]}}
        ])
    with pytest.raises(ValueError, match="share one length"):
        SearchSpace.from_dict([
            {"name": "s", "kind": "categorical", "choices": ["a", "b"],
             "descriptors": {"a": [1.0], "b": [1.0, 2.0]}}
        ])
    with pytest.raises(ValueError, match="descriptor_names"):
        SearchSpace.from_dict([
            {"name": "s", "kind": "categorical", "choices": ["a", "b"],
             "descriptors": {"a": [1.0, 2.0], "b": [3.0, 4.0]},
             "descriptor_names": ["only_one"]}
        ])


def test_descriptor_space_runs_through_study(tmp_path):
    from condopt.base import Objective
    from condopt.study import Study

    space = SearchSpace.from_dict([
        {"name": "solvent", "kind": "categorical", "choices": ["DMF", "EtOH", "THF"],
         "descriptors": {"DMF": [73.0, -1.0], "EtOH": [46.0, -0.3], "THF": [72.1, 0.5]}},
        {"name": "T", "kind": "continuous", "low": 20.0, "high": 120.0},
    ])
    study = Study.create(
        tmp_path / "descriptor_run",
        space,
        Objective(name="y", direction="max"),
        algorithm="bayes",
        budget=8,
        batch_size=4,
        seed=0,
    )
    pending = study.ask(4)
    study.tell([
        {"id": item["id"], "y": (1.0 if item["x"]["solvent"] == "DMF" else 0.2)}
        for item in pending
    ])
    assert len(study.dataset.observations) == 4
    prediction = study.predict({"solvent": "DMF", "T": 60.0})
    assert "y" in prediction["predictions"]

def test_benchmark_descriptors_script_runs(tmp_path):
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_descriptors.py"
    spec = importlib.util.spec_from_file_location("benchmark_descriptors", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    row = module.run_arm(True, budget=6, batch_size=3, seed=0)
    assert row["arm"] == "descriptors"
    assert row["best_yield"] is not None

def test_benchmark_descriptor_quality_script_runs():
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_descriptor_quality.py"
    spec = importlib.util.spec_from_file_location("benchmark_descriptor_quality", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    noisy = module.run_arm(0.5, budget=6, batch_size=3, seed=0)
    one_hot = module.run_arm(99.0, budget=6, batch_size=3, seed=0)
    assert noisy["best_yield"] is not None
    assert one_hot["best_yield"] is not None

def test_step_suggests_gp_family_for_descriptor_space_with_tpe(tmp_path):
    from condopt.base import Objective
    from condopt.study import Study

    space = SearchSpace.from_dict([
        {"name": "solvent", "kind": "categorical", "choices": ["a", "b", "c"],
         "descriptors": {"a": [0.0], "b": [0.5], "c": [1.0]}},
        {"name": "T", "kind": "continuous", "low": 0.0, "high": 100.0},
    ])
    study = Study.create(
        tmp_path / "descriptor_tpe",
        space,
        Objective(name="y", direction="max"),
        algorithm="tpe",
        budget=8,
        batch_size=4,
        seed=0,
    )
    study.import_observations([
        {"x": {"solvent": "a", "T": 10.0}, "y": 0.1},
        {"x": {"solvent": "b", "T": 40.0}, "y": 0.5},
        {"x": {"solvent": "c", "T": 70.0}, "y": 0.9},
        {"x": {"solvent": "a", "T": 90.0}, "y": 0.2},
    ])
    action = study.next_action()
    assert action["diagnostics"]["n_descriptor_vars"] == 1
    assert "review_algorithm_for_descriptors" in action["suggested_actions"]
    assert "review_algorithm_for_descriptors" in action["suggested_action_reasons"]

def test_solvent_constrained_descriptor_variant_shares_landscape():
    from condopt.benchmarks import builtin_problems

    base = builtin_problems()["solvent_constrained"]
    described = builtin_problems()["solvent_constrained_desc"]
    assert described.space.feature_dim < base.space.feature_dim
    assert base.fn(base.best_config) == described.fn(described.best_config)
    assert base.constraints_fn(base.best_config) == {"T_max": 80.0}
    # the unconstrained optimum sits outside the feasible region
    assert base.fn({"T": 95.0, "solvent": "S04"}) > base.fn(base.best_config)

