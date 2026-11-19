"""Algorithm registry.

Registration is decentralised: every algorithm module decorates its
optimizer class with :func:`register`.  Importing ``condopt.algorithms``
loads the built-in families; third-party algorithms can be added with the
same decorator without touching this file.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .base import AlgorithmSpec, Objective, Optimizer
from .dataset import Observation
from .space import SearchSpace

_REGISTRY: dict[str, type[Optimizer]] = {}


def register(cls: type[Optimizer]) -> type[Optimizer]:
    """Register an optimizer class under ``cls.spec.name``."""
    spec = getattr(cls, "spec", None)
    if not isinstance(spec, AlgorithmSpec):
        raise TypeError(f"{cls.__name__} must declare an AlgorithmSpec as class attribute 'spec'")
    if spec.name in _REGISTRY and _REGISTRY[spec.name] is not cls:
        raise ValueError(f"algorithm {spec.name!r} is already registered by {_REGISTRY[spec.name]}")
    _REGISTRY[spec.name] = cls
    return cls


def get(name: str) -> type[Optimizer]:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"unknown algorithm {name!r}; available: {sorted(_REGISTRY)}") from exc


def spec(name: str) -> AlgorithmSpec:
    return get(name).spec


def specs() -> list[AlgorithmSpec]:
    return [_REGISTRY[name].spec for name in sorted(_REGISTRY)]


def names() -> list[str]:
    return sorted(_REGISTRY)


def describe(name: str) -> dict:
    return spec(name).to_dict()


def describe_all() -> list[dict]:
    return [s.to_dict() for s in specs()]


def validate_params(name: str, params: Mapping[str, Any] | None = None) -> dict:
    return spec(name).validate(params)


def build(
    name: str,
    space: SearchSpace,
    objective: Objective | None = None,
    params: Mapping[str, Any] | None = None,
    seed: int = 0,
    prior: Sequence[Observation] | None = None,
) -> Optimizer:
    """Instantiate a registered optimizer, applying defaults and validation."""
    cls = get(name)
    return cls(space=space, objective=objective, params=params, seed=seed, prior=prior)
