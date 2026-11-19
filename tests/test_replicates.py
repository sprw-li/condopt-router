"""Tests for replicate-aware observation noise (alpha) and data plumbing."""

import numpy as np
import pytest

from condopt import Objective, Observation, SearchSpace, Study
from condopt.dataset import Observation as ObservationClass
from condopt.registry import build


def test_observation_roundtrip_with_replicates():
    observation = Observation(
        x={"x": 0.5}, y=80.0, objective_std=1.5, n_replicates=3, fidelity=0.5
    )
    restored = ObservationClass.from_dict(observation.to_dict())
    assert restored.objective_std == pytest.approx(1.5)
    assert restored.n_replicates == 3
    assert restored.fidelity == pytest.approx(0.5)


def test_observation_validation():
    with pytest.raises(ValueError):
        Observation(x={"x": 0.5}, y=1.0, n_replicates=0)
    with pytest.raises(ValueError):
        Observation(x={"x": 0.5}, y=1.0, objective_std=-1.0)


def test_bayes_uses_replicate_noise_vector():
    space = SearchSpace.from_dict(
        [{"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0}]
    )
    optimizer = build("bayes", space, Objective(name="y", direction="max"), seed=0)
    observations = [
        Observation(x={"x": index / 10}, y=float(index), objective_std=0.5, n_replicates=4)
        for index in range(6)
    ]
    optimizer.tell(observations)
    model = optimizer._fit(optimizer._X, optimizer._y)
    assert np.ndim(model.alpha) == 1
    assert len(model.alpha) == len(optimizer._y)
    assert np.all(np.asarray(model.alpha) > 0)


def test_study_results_parse_std_and_replicates(tmp_path):
    space = SearchSpace.from_dict(
        [{"name": "x", "kind": "continuous", "low": 0.0, "high": 1.0}]
    )
    study = Study.create(tmp_path / "rep", space, Objective(name="y", direction="max"), budget=6, batch_size=2)
    pending = study.ask(2)
    study.tell(
        [
            {"id": item["id"], "y": 1.0, "std": 0.2, "k": 3}
            for item in pending
        ]
    )
    observations = study.dataset.observations
    assert observations[0].objective_std == pytest.approx(0.2)
    assert observations[0].n_replicates == 3
