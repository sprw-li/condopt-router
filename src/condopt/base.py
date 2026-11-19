"""Optimizer interface, objective definition and algorithm metadata.

Every algorithm in ``condopt`` is an *ask/tell* optimizer:

* ``ask(n)`` proposes ``n`` new condition configurations;
* ``tell(observations)`` reports the measured outcomes of earlier proposals.

That contract keeps the algorithm independent of *how* an experiment is
executed (manual bench work, a robot, a simulator) and makes every
algorithm interchangeable behind the router in :mod:`condopt.router`.

Each optimizer class declares an :class:`AlgorithmSpec`: a machine
readable parameter schema (rendered as a UI form by the skill layer),
capability flags, and references.  The skill / AI layer is only allowed
to pick among registered algorithms and to set parameters that appear in
the spec -- this is the "hard algorithm, flexible parameters" split.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Mapping, Sequence

import numpy as np

from .dataset import Observation
from .space import SearchSpace


@dataclass(frozen=True)
class ParamSpec:
    """Schema of one tunable algorithm parameter."""

    name: str
    type: str  # "int" | "float" | "str" | "bool" | "choice"
    default: Any
    choices: tuple = ()
    low: float | None = None
    high: float | None = None
    description: str = ""
    unit: str = ""

    def __post_init__(self) -> None:
        if self.type not in ("int", "float", "str", "bool", "choice"):
            raise ValueError(f"param {self.name!r}: unsupported type {self.type!r}")
        object.__setattr__(self, "choices", tuple(self.choices))

    def coerce(self, value: Any) -> Any:
        if self.type == "bool":
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                low = value.strip().lower()
                if low in ("true", "1", "yes", "on"):
                    return True
                if low in ("false", "0", "no", "off"):
                    return False
            raise ValueError(f"param {self.name!r}: expected a boolean, got {value!r}")
        if self.type == "int":
            if isinstance(value, bool) or float(value) != int(value):
                raise ValueError(f"param {self.name!r}: expected an integer, got {value!r}")
            out: Any = int(value)
        elif self.type == "float":
            out = float(value)
        else:
            out = str(value)
        if self.type == "choice" and out not in [str(c) for c in self.choices]:
            raise ValueError(f"param {self.name!r}: expected one of {self.choices}, got {out!r}")
        if self.low is not None and out < self.low:
            raise ValueError(f"param {self.name!r}: {out} < lower bound {self.low}")
        if self.high is not None and out > self.high:
            raise ValueError(f"param {self.name!r}: {out} > upper bound {self.high}")
        return out

    def to_dict(self) -> dict:
        out: dict[str, object] = {
            "name": self.name,
            "type": self.type,
            "default": self.default,
        }
        if self.choices:
            out["choices"] = list(self.choices)
        if self.low is not None:
            out["low"] = self.low
        if self.high is not None:
            out["high"] = self.high
        if self.description:
            out["description"] = self.description
        if self.unit:
            out["unit"] = self.unit
        return out


@dataclass(frozen=True)
class AlgorithmSpec:
    """Static description of an algorithm family."""

    name: str
    family: str
    summary: str
    params: tuple[ParamSpec, ...] = ()
    capabilities: frozenset[str] = frozenset()
    references: tuple[str, ...] = ()
    #: When to prefer this algorithm, in one line (router documentation).
    when: str = ""

    def defaults(self) -> dict:
        return {p.name: p.default for p in self.params}

    def param(self, name: str) -> ParamSpec:
        for p in self.params:
            if p.name == name:
                return p
        raise KeyError(f"algorithm {self.name!r} has no parameter {name!r}")

    def validate(self, params: Mapping[str, Any] | None) -> dict:
        given = dict(params or {})
        unknown = [k for k in given if k not in [p.name for p in self.params]]
        if unknown:
            raise ValueError(f"algorithm {self.name!r}: unknown parameters {unknown}")
        merged = self.defaults()
        for name, value in given.items():
            merged[name] = self.param(name).coerce(value)
        return merged

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "family": self.family,
            "summary": self.summary,
            "when": self.when,
            "capabilities": sorted(self.capabilities),
            "params": [p.to_dict() for p in self.params],
            "references": list(self.references),
        }


@dataclass
class Objective:
    """What the study optimizes.

    Single-objective studies use ``name`` + ``direction``.  Multi-objective
    studies list every objective in ``objectives`` and may give a per-objective
    direction in ``directions`` (defaults to ``direction`` for all of them).
    ``name`` is the primary objective used for the single-best report.
    """

    name: str = "score"
    direction: str = "max"  # "max" | "min"
    objectives: tuple[str, ...] = ()
    #: Names of constraints measured alongside the objective.
    constraints: tuple[str, ...] = ()
    #: Per-objective directions; empty -> all equal to ``direction``.
    directions: tuple[str, ...] = ()
    #: Constraint definitions: name -> {"op": "<=" | ">=", "value": float}.
    constraint_limits: dict[str, dict] = field(default_factory=dict)
    #: Fixed raw-space reference point for hypervolume (one value per objective).
    ref_point: tuple[float, ...] = ()
    #: Fixed raw-space normalization bounds: name -> {"low": float, "high": float}.
    objective_bounds: dict[str, dict] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.direction not in ("max", "min"):
            raise ValueError("objective direction must be 'max' or 'min'")
        if not self.objectives:
            self.objectives = (self.name,)
        self.directions = tuple(self.directions) or tuple(self.direction for _ in self.objectives)
        if len(self.directions) != len(self.objectives):
            raise ValueError(
                f"directions length {len(self.directions)} != objectives length {len(self.objectives)}"
            )
        for direction in self.directions:
            if direction not in ("max", "min"):
                raise ValueError("each objective direction must be 'max' or 'min'")
        self.constraint_limits = {
            str(name): dict(spec) for name, spec in (self.constraint_limits or {}).items()
        }
        for name, spec in self.constraint_limits.items():
            if spec.get("op", "<=") not in ("<=", ">="):
                raise ValueError(f"constraint {name!r}: op must be '<=' or '>='")
            float(spec.get("value", 0.0))
        self.ref_point = tuple(float(value) for value in self.ref_point)
        if self.ref_point and len(self.ref_point) != self.n_objectives:
            raise ValueError(
                f"ref_point length {len(self.ref_point)} != objectives length {self.n_objectives}"
            )
        self.objective_bounds = {
            str(name): dict(spec) for name, spec in (self.objective_bounds or {}).items()
        }
        for name in self.objective_bounds:
            if name not in self.objectives:
                raise ValueError(f"objective_bounds references unknown objective {name!r}")
            low = float(self.objective_bounds[name]["low"])
            high = float(self.objective_bounds[name]["high"])
            if not high > low:
                raise ValueError(f"objective_bounds[{name!r}] requires high > low")
            self.objective_bounds[name] = {"low": low, "high": high}

    @property
    def sign(self) -> float:
        return 1.0 if self.direction == "max" else -1.0

    @property
    def n_objectives(self) -> int:
        return len(self.objectives)

    @property
    def is_multi_objective(self) -> bool:
        return self.n_objectives > 1

    def direction_for(self, objective: str) -> str:
        try:
            index = self.objectives.index(objective)
        except ValueError as exc:
            raise KeyError(f"unknown objective {objective!r}") from exc
        return self.directions[index]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "direction": self.direction,
            "objectives": list(self.objectives),
            "directions": list(self.directions),
            "constraints": list(self.constraints),
            "constraint_limits": self.constraint_limits,
            "ref_point": list(self.ref_point),
            "objective_bounds": self.objective_bounds,
        }

    @classmethod
    def from_dict(cls, payload: Mapping | None) -> "Objective":
        if not payload:
            return cls()
        return cls(
            name=payload.get("name", "score"),
            direction=payload.get("direction", "max"),
            objectives=tuple(payload.get("objectives", ())),
            constraints=tuple(payload.get("constraints", ())),
            directions=tuple(payload.get("directions", ())),
            constraint_limits=dict(payload.get("constraint_limits", {})),
            ref_point=tuple(payload.get("ref_point", ())),
            objective_bounds=dict(payload.get("objective_bounds", {})),
        )


class Optimizer(abc.ABC):
    """Base class for all ask/tell optimizers."""

    spec: ClassVar[AlgorithmSpec]
    #: Whether this optimizer can propose a per-point fidelity.
    supports_fidelity: ClassVar[bool] = False
    #: True when the optimizer's internal state is *derived from the observations*
    #: rather than an independent cursor.  ``Study`` rebuilds optimizers every round
    #: and restores ``opt_state.json`` on top; for these algorithms that persistent
    #: snapshot is stale (the Study never calls ``tell``), so the history must be
    #: replayed after loading.  Without it ``cmaes`` sampled its frozen initial
    #: ``N(0.5, sigma0^2)`` forever while the router kept recommending it.
    replay_history_on_load: ClassVar[bool] = False

    def __init__(
        self,
        space: SearchSpace,
        objective: Objective | None = None,
        params: Mapping[str, Any] | None = None,
        seed: int = 0,
        prior: Sequence[Observation] | None = None,
    ) -> None:
        self.space = space
        self.objective = objective or Objective()
        self.params = self.spec.validate(params)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.t = 0  # number of outcomes told so far
        #: Optional remaining cost budget injected by the Study before ``ask``.
        self.cost_remaining: float | None = None
        self._pending: list[dict] = []
        if prior:
            self.tell(list(prior))

    # -- contract ---------------------------------------------------------
    @abc.abstractmethod
    def ask(self, n: int = 1) -> list[dict]:
        """Propose ``n`` new configurations."""

    def ask_with_fidelity(self, n: int = 1, cost_remaining: float | None = None) -> list[tuple[dict, float]]:
        """Propose ``n`` configurations with a fidelity per point.

        Plain optimizers always return full fidelity (1.0); multi-fidelity
        optimizers override this and set ``supports_fidelity = True``.
        """
        return [(config, 1.0) for config in self.ask(n)]

    def tell(self, observations: Sequence[Observation]) -> None:
        """Report measured outcomes since the previous ``tell``."""
        observations = list(observations)
        for obs in observations:
            self.space.validate_config(obs.x)
        self.t += len(observations)

    # -- helpers ----------------------------------------------------------
    def params_dict(self) -> dict:
        return dict(self.params)

    def state_dict(self) -> dict:
        """Serializable state for resuming a study across processes.

        Only *counters* belong here: all algorithms rebuild their data from
        the study dataset, which is passed in as ``prior``.  Persisting the
        data twice would let a stale state file shadow newer observations.
        """
        return {"kind": self.spec.name}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore counters; subclasses override and call ``super()``."""
        _ = state

    def suggest(self, n: int = 1) -> list[dict]:
        return self.ask(n)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{type(self).__name__} {self.spec.name} t={self.t}>"


def make_objective(
    direction: str = "max",
    name: str = "score",
    objectives: Sequence[str] | None = None,
    constraints: Sequence[str] | None = None,
    directions: Sequence[str] | None = None,
) -> Objective:
    return Objective(
        name=name,
        direction=direction,
        objectives=tuple(objectives or ()),
        constraints=tuple(constraints or ()),
        directions=tuple(directions or ()),
    )


#: Type alias for a callable objective used by benchmarks and demos.
ObjectiveFn = Callable[[Mapping[str, Any]], float]
