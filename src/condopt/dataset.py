"""Observations and datasets.

An :class:`Observation` is one executed experiment: the condition
configuration plus the measured outcome.  ``condopt`` never invents
outcomes; every observation must come from a real measurement, a
simulation the user trusts, or a labeled benchmark function.

``Dataset`` is an append-only collection with JSONL persistence.  A single
study keeps one dataset file, so a run is always auditable and resumable.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

#: JSONL record schema version.  Bumped only when a field's *meaning* changes, so
#: an audit can tell which reader a stored line was written for (A9).
SCHEMA_VERSION = 1


def default_actor() -> str:
    """Who produced a record: ``CONDOPT_ACTOR`` when set, otherwise ``"unknown"``.

    Read at record-creation time instead of being threaded through every call
    site: an agent/robot wrapper can stamp its identity by exporting one
    environment variable, and a plain interactive run still gets a usable value.
    """
    return (os.environ.get("CONDOPT_ACTOR") or "").strip() or "unknown"


@dataclass
class Observation:
    """One measured condition.

    Parameters
    ----------
    x:
        Raw condition configuration (validated against the study's space).
    y:
        Primary objective value (for single-objective studies).
    objectives:
        Named objective values for multi-objective studies; may also be
        used to carry secondary metrics next to ``y``.
    constraints:
        Named constraint measurements (e.g. ``{"safety_margin": 12.0}``).
    fidelity:
        Fidelity of this evaluation, ``1.0`` = full experiment.
    cost:
        Resource cost (money / time / material) of this evaluation.
    source:
        Provenance: ``"manual"``, ``"robot"``, ``"simulation"``, ``"benchmark"``.
    actor:
        Who measured this observation (``CONDOPT_ACTOR``, else ``"unknown"``);
        explicit values are kept as given.
    """

    x: dict
    y: float | None = None
    objectives: dict[str, float] = field(default_factory=dict)
    constraints: dict[str, float] = field(default_factory=dict)
    fidelity: float = 1.0
    cost: float = 0.0
    #: Standard deviation of the primary objective measurement (replicates).
    objective_std: float = 0.0
    #: Per-objective measurement std, e.g. ``{"yield": 1.2, "cost": 0.05}``.  A single
    #: scalar cannot describe a heteroscedastic multi-objective problem (yield and
    #: cost rarely share a variance), and qNEHVI's nested MC needs the noise of *each*
    #: objective.  Values here take precedence over ``objective_std`` per metric.
    objectives_std: dict[str, float] = field(default_factory=dict)
    #: Number of replicate measurements aggregated into ``y`` (>=1).
    n_replicates: int = 1
    t: float = field(default_factory=time.time)
    source: str = "manual"
    meta: dict = field(default_factory=dict)
    #: Schema version of this record (see :data:`SCHEMA_VERSION`); legacy JSONL
    #: without the field reads as the current version rather than failing.
    schema_version: int = SCHEMA_VERSION
    #: Who measured this observation: ``CONDOPT_ACTOR`` when set, else ``"unknown"``.
    #: A stored record that never carried the field reads as ``"unknown"`` instead
    #: (see :meth:`from_dict`), so a later audit cannot invent an owner.
    actor: str = field(default_factory=default_actor)

    def __post_init__(self) -> None:
        self.x = dict(self.x)
        if self.y is not None:
            self.y = float(self.y)
        self.objectives = {k: float(v) for k, v in (self.objectives or {}).items()}
        self.constraints = {k: float(v) for k, v in (self.constraints or {}).items()}
        self.fidelity = float(self.fidelity)
        self.cost = float(self.cost)
        self.objective_std = float(self.objective_std)
        self.objectives_std = {
            str(key): float(value) for key, value in (self.objectives_std or {}).items()
        }
        for key, value in self.objectives_std.items():
            if value < 0:
                raise ValueError(f"objectives_std[{key!r}] must be >= 0")
        self.n_replicates = int(self.n_replicates)
        if self.n_replicates < 1:
            raise ValueError("n_replicates must be >= 1")
        if self.objective_std < 0:
            raise ValueError("objective_std must be >= 0")
        self.schema_version = int(self.schema_version)
        # An empty actor is not an identity: fall back instead of storing "".
        self.actor = str(self.actor or default_actor())
        if self.y is None and not self.objectives:
            raise ValueError("observation needs either y or objectives")
        if self.y is None and len(self.objectives) == 1:
            # Convenient normalisation: a single named objective also fills y.
            self.y = next(iter(self.objectives.values()))

    def std_for(self, objective: str, default: float = 0.0) -> float:
        """Measurement std for one objective (per-objective value wins)."""
        if objective in self.objectives_std:
            return float(self.objectives_std[objective])
        return float(self.objective_std or default)

    def score(self, objective: str | None = None, direction: str = "max") -> float:
        """Signed scalar score: higher is always better."""
        value: float | None = None
        if objective and self.objectives and objective in self.objectives:
            value = self.objectives[objective]
        if value is None:
            value = self.y
        if value is None and self.objectives:
            # Single named objective stored in ``objectives`` only.
            value = next(iter(self.objectives.values()))
        if value is None:
            raise ValueError(f"observation has no value for objective {objective!r}")
        return float(value) if direction == "max" else -float(value)

    def to_dict(self) -> dict:
        out: dict[str, object] = {"x": self.x}
        if self.y is not None:
            out["y"] = self.y
        if self.objectives:
            out["objectives"] = self.objectives
        if self.constraints:
            out["constraints"] = self.constraints
        out["fidelity"] = self.fidelity
        if self.cost:
            out["cost"] = self.cost
        if self.objective_std:
            out["objective_std"] = self.objective_std
        if self.objectives_std:
            out["objectives_std"] = self.objectives_std
        if self.n_replicates != 1:
            out["n_replicates"] = self.n_replicates
        out["t"] = self.t
        out["source"] = self.source
        # Always written (not only on non-default values): a reader that sees no
        # version cannot distinguish "old file" from "new file with defaults".
        out["schema_version"] = self.schema_version
        out["actor"] = self.actor
        if self.meta:
            out["meta"] = self.meta
        return out

    @classmethod
    def from_dict(cls, payload: Mapping) -> "Observation":
        return cls(
            x=payload["x"],
            y=payload.get("y"),
            objectives=payload.get("objectives", {}),
            constraints=payload.get("constraints", {}),
            fidelity=payload.get("fidelity", 1.0),
            cost=payload.get("cost", 0.0),
            objective_std=payload.get("objective_std", 0.0),
            objectives_std=payload.get("objectives_std", {}),
            n_replicates=payload.get("n_replicates", 1),
            t=payload.get("t", time.time()),
            source=payload.get("source", "manual"),
            meta=payload.get("meta", {}),
            # Legacy JSONL has neither key; reading it must succeed (A9).  A line
            # written before A9 has no recorded producer, so it reads as
            # "unknown" -- taking today's CONDOPT_ACTOR would attribute old
            # measurements to whoever happens to run the audit.
            schema_version=payload.get("schema_version", SCHEMA_VERSION),
            actor=payload.get("actor", "unknown"),
        )


class Dataset:
    """Append-only observation log backed by a JSONL file."""

    def __init__(self, observations: Iterable[Observation] | None = None) -> None:
        self._observations: list[Observation] = list(observations or [])

    # -- container protocol ----------------------------------------------
    def __len__(self) -> int:
        return len(self._observations)

    def __iter__(self):
        return iter(self._observations)

    def __getitem__(self, index: int) -> Observation:
        return self._observations[index]

    @property
    def observations(self) -> list[Observation]:
        return list(self._observations)

    def add(self, observation: Observation) -> None:
        self._observations.append(observation)

    def extend(self, observations: Iterable[Observation]) -> None:
        self._observations.extend(observations)

    # -- analytics --------------------------------------------------------
    def scores(self, objective: str | None = None, direction: str = "max") -> np.ndarray:
        return np.asarray([o.score(objective, direction) for o in self._observations], dtype=float)

    def best(
        self, objective: str | None = None, direction: str = "max"
    ) -> Observation | None:
        if not self._observations:
            return None
        scores = self.scores(objective, direction)
        return self._observations[int(np.argmax(scores))]

    def best_score(self, objective: str | None = None, direction: str = "max") -> float | None:
        best = self.best(objective, direction)
        return None if best is None else best.score(objective, direction)

    # -- persistence ------------------------------------------------------
    @classmethod
    def from_jsonl(cls, path: str | Path) -> "Dataset":
        p = Path(path)
        if not p.exists():
            return cls()
        observations = []
        with p.open("r", encoding="utf-8-sig") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    observations.append(Observation.from_dict(json.loads(line)))
        return cls(observations)

    def append_jsonl(self, path: str | Path, observations: Sequence[Observation]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as fh:
            for obs in observations:
                fh.write(json.dumps(obs.to_dict(), ensure_ascii=False) + "\n")

    def to_jsonl(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as fh:
            for obs in self._observations:
                fh.write(json.dumps(obs.to_dict(), ensure_ascii=False) + "\n")

