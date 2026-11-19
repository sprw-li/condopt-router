"""Search space definition for condition optimization.

A condition optimization problem is a black-box function over a *mixed*
search space.  Real laboratory condition spaces are rarely plain
``R^d``: they mix continuous knobs (temperature, time, flow rate),
integer knobs (number of equivalents, cycles), log-scaled knobs
(concentration spanning decades), categorical knobs (solvent, catalyst,
ligand) and booleans (inert atmosphere, addition order).

This module provides two encodings used throughout ``condopt``:

* ``Variable.encode`` / ``SearchSpace.encode`` -- map a raw configuration
  onto the unit hypercube ``[0, 1]^d``.  Unit-cube samplers (Sobol, Latin
  hypercube, random) live here.
* ``SearchSpace.features`` / ``SearchSpace.features_many`` -- map a raw
  configuration onto a numeric feature vector: numeric variables scaled
  to ``[0, 1]``, discrete variables one-hot encoded.  Surrogate models
  (GP in :mod:`condopt.algorithms.bayes`) consume this representation.

Both encodings are invertible for the discrete part only up to the
one-hot group; decoding is always done from the unit-cube encoding.
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

VALID_KINDS = ("continuous", "integer", "log", "categorical", "ordinal", "bool")
NUMERIC_KINDS = ("continuous", "integer", "log")
DISCRETE_KINDS = ("categorical", "ordinal", "bool")


def _as_rng(rng: np.random.Generator | int | None) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


@dataclass(frozen=True)
class Variable:
    """One knob of a condition space.

    Parameters
    ----------
    name:
        Unique variable name, also used as the column name in suggestions.
    kind:
        One of ``continuous``, ``integer``, ``log``, ``categorical``,
        ``ordinal``, ``bool``.
    low, high:
        Required for numeric kinds.  ``log`` requires ``low > 0``.
    choices:
        Required for discrete kinds.  Order matters for ``ordinal``
        (it is the semantic order); for ``categorical`` it is only the
        enumeration order used for one-hot encoding.
    unit:
        Physical unit shown to the experimentalist, e.g. ``"degC"``.
    description:
        Human readable note, e.g. ``"reaction temperature"``.
    """

    name: str
    kind: str = "continuous"
    low: float | None = None
    high: float | None = None
    choices: tuple[str, ...] = ()
    unit: str = ""
    description: str = ""
    #: optional per-choice numeric descriptors used instead of one-hot features
    descriptors: Mapping[str, tuple[float, ...]] | None = None
    #: optional labels for the descriptor dimensions
    descriptor_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("variable name must be a non-empty string")
        if self.kind not in VALID_KINDS:
            raise ValueError(
                f"variable {self.name!r}: kind must be one of {VALID_KINDS}, got {self.kind!r}"
            )
        object.__setattr__(self, "choices", tuple(str(c) for c in self.choices))
        if self.descriptors:
            normalized: dict[str, tuple[float, ...]] = {}
            lengths = set()
            for choice, vector in self.descriptors.items():
                values = tuple(float(item) for item in vector)
                if not values:
                    raise ValueError(
                        f"variable {self.name!r}: descriptor for {choice!r} is empty"
                    )
                normalized[str(choice)] = values
                lengths.add(len(values))
            missing = [c for c in self.choices if c not in normalized]
            known_choices = [c for c in self.choices if c in normalized]
            if not known_choices:
                raise ValueError(
                    f"variable {self.name!r}: descriptors must cover at least one choice"
                )
            if len(lengths) != 1:
                raise ValueError(
                    f"variable {self.name!r}: descriptor vectors must share one length"
                )
            object.__setattr__(self, "descriptors", normalized)
            size = lengths.pop()
            if self.descriptor_names and len(self.descriptor_names) != size:
                raise ValueError(
                    f"variable {self.name!r}: descriptor_names must match descriptor length"
                )
            # Min-max scale each descriptor dimension onto [0, 1] for the GP
            # features; raw values stay in ``descriptors`` for reporting.
            known = known_choices
            matrix = np.asarray([normalized[choice] for choice in known], dtype=float)
            lows = matrix.min(axis=0)
            spans = matrix.max(axis=0) - lows
            spans = np.where(spans > 1e-12, spans, 1.0)
            object.__setattr__(
                self,
                "_scaled_descriptors",
                {
                    choice: tuple(((matrix[index] - lows) / spans).tolist())
                    for index, choice in enumerate(known)
                },
            )
            # Partial tables are allowed: unknown choices get a zero descriptor
            # plus a dedicated indicator dimension so the GP can separate them.
            object.__setattr__(self, "_partial_descriptors", bool(missing))
            object.__setattr__(self, "_descriptor_width", len(next(iter(normalized.values()))))
        object.__setattr__(
            self, "descriptor_names", tuple(str(n) for n in self.descriptor_names)
        )
        if self.kind in ("categorical", "ordinal"):
            if len(self.choices) < 2:
                raise ValueError(
                    f"variable {self.name!r}: {self.kind} kind needs at least 2 choices"
                )
            if len(set(self.choices)) != len(self.choices):
                raise ValueError(f"variable {self.name!r}: choices must be unique")
        elif self.kind == "bool":
            if not self.choices:
                object.__setattr__(self, "choices", ("False", "True"))
            if len(self.choices) != 2:
                raise ValueError(f"variable {self.name!r}: bool kind needs exactly 2 choices")
        else:
            if self.low is None or self.high is None:
                raise ValueError(f"variable {self.name!r}: numeric kind needs low and high")
            low, high = float(self.low), float(self.high)
            if not (math.isfinite(low) and math.isfinite(high)) or not low < high:
                raise ValueError(f"variable {self.name!r}: require finite low < high")
            if self.kind == "log" and low <= 0:
                raise ValueError(f"variable {self.name!r}: log kind requires low > 0")
            if self.kind == "integer":
                if low != int(low) or high != int(high):
                    raise ValueError(f"variable {self.name!r}: integer bounds must be whole numbers")
                low, high = int(low), int(high)
            object.__setattr__(self, "low", low)
            object.__setattr__(self, "high", high)

    # -- basic properties -------------------------------------------------
    @property
    def is_numeric(self) -> bool:
        return self.kind in NUMERIC_KINDS

    @property
    def is_discrete(self) -> bool:
        return self.kind in DISCRETE_KINDS

    @property
    def cardinality(self) -> float | int:
        """Number of possible values; ``inf`` for continuous variables."""
        if self.kind == "continuous" or self.kind == "log":
            return math.inf
        if self.kind == "integer":
            return int(self.high) - int(self.low) + 1
        return len(self.choices)

    # -- sampling ---------------------------------------------------------
    def sample_uniform(self, rng: np.random.Generator) -> object:
        """Draw one value uniformly (in encoded space for ``log``)."""
        if self.kind == "continuous":
            return float(self.low + (self.high - self.low) * rng.random())
        if self.kind == "integer":
            return int(rng.integers(int(self.low), int(self.high) + 1))
        if self.kind == "log":
            lo, hi = math.log(self.low), math.log(self.high)
            return float(math.exp(lo + (hi - lo) * rng.random()))
        idx = int(rng.integers(len(self.choices)))
        return self._choice_value(idx)

    def _choice_value(self, idx: int) -> object:
        value = self.choices[idx]
        if self.kind == "bool":
            return value == "True"
        return value

    def _choice_index(self, value: object) -> int:
        if self.kind == "bool" and isinstance(value, bool):
            value = "True" if value else "False"
        text = str(value)
        try:
            return self.choices.index(text)
        except ValueError as exc:
            raise ValueError(
                f"variable {self.name!r}: {value!r} is not one of {self.choices}"
            ) from exc

    # -- encodings --------------------------------------------------------
    def encode(self, value: object) -> float:
        """Map a raw value onto ``[0, 1]``."""
        self.validate_value(value)
        if self.kind == "continuous":
            return float((float(value) - self.low) / (self.high - self.low))
        if self.kind == "integer":
            return float((int(value) - int(self.low)) / (int(self.high) - int(self.low)))
        if self.kind == "log":
            lo, hi = math.log(self.low), math.log(self.high)
            return float((math.log(float(value)) - lo) / (hi - lo))
        idx = self._choice_index(value)
        if len(self.choices) == 1:
            return 0.0
        return float(idx / (len(self.choices) - 1))

    def decode(self, u: float) -> object:
        """Map a unit-cube coordinate back to a raw value (clamped)."""
        u = min(max(float(u), 0.0), 1.0)
        if self.kind == "continuous":
            return float(self.low + (self.high - self.low) * u)
        if self.kind == "integer":
            raw = int(self.low) + int(round(u * (int(self.high) - int(self.low))))
            return int(min(max(raw, int(self.low)), int(self.high)))
        if self.kind == "log":
            lo, hi = math.log(self.low), math.log(self.high)
            return float(math.exp(lo + (hi - lo) * u))
        idx = min(int(u * len(self.choices)), len(self.choices) - 1)
        return self._choice_value(idx)

    def _within_bounds(self, value: float, tol: float = 0.0) -> bool:
        """Bounds check with a relative tolerance.

        A unit-cube round trip can land one ulp outside the bound (measured:
        ``decode`` gave 4.999999999999999 for a ``[5, 240]`` variable).  Rejecting that
        would make the library unusable, while still catching a real out-of-range value
        such as 150 in ``[60, 120]``.
        """
        low, high = float(self.low), float(self.high)
        span = max(abs(high - low), 1.0)
        margin = tol * span
        return (low - margin) <= float(value) <= (high + margin)

    def validate_value(self, value: object) -> None:
        if self.kind == "continuous":
            v = float(value)
            if not math.isfinite(v):
                raise ValueError(f"variable {self.name!r}: value must be finite")
            # Bounds used to be enforced only for integer variables, so a measured
            # 150 could be told into a [60, 120] space: ``encode`` then produced u > 1
            # and every downstream ``decode`` silently clipped it, i.e. the recorded
            # condition was not the one the optimizer actually saw.
            if not self._within_bounds(v, tol=1e-9):
                raise ValueError(
                    f"variable {self.name!r}: {v} outside [{self.low}, {self.high}]"
                )
            return
        if self.kind == "integer":
            if isinstance(value, bool) or float(value) != int(value):
                raise ValueError(f"variable {self.name!r}: value must be an integer")
            v = int(value)
            if not (int(self.low) <= v <= int(self.high)):
                raise ValueError(
                    f"variable {self.name!r}: {v} outside [{self.low}, {self.high}]"
                )
            return
        if self.kind == "log":
            v = float(value)
            if not math.isfinite(v) or v <= 0:
                raise ValueError(f"variable {self.name!r}: log value must be > 0")
            if not self._within_bounds(v, tol=1e-9):
                raise ValueError(
                    f"variable {self.name!r}: {v} outside [{self.low}, {self.high}]"
                )
            return
        self._choice_index(value)

    def one_hot_size(self) -> int:
        return len(self.choices) if self.is_discrete else 0

    def feature_size(self) -> int:
        """Feature width: descriptors replace one-hot when provided.

        A partial descriptor table adds one "unknown" indicator dimension.
        """
        if not self.is_numeric and self.descriptors:
            width = getattr(self, "_descriptor_width", None) or len(
                next(iter(self.descriptors.values()))
            )
            return width + (1 if getattr(self, "_partial_descriptors", False) else 0)
        return self.one_hot_size()

    def has_descriptor(self, value: object) -> bool:
        if not self.descriptors:
            return False
        choice = str(self._choice_value(self._choice_index(value)))
        return choice in self.descriptors

    def descriptor_vector(self, value: object) -> list[float]:
        """Feature vector for a choice (descriptors min-max scaled to [0, 1]).

        Unknown choices (partial tables) return zeros plus a 1.0 indicator in
        the extra last dimension.
        """
        if not self.descriptors:
            raise ValueError(f"variable {self.name!r} has no descriptors")
        choice = str(self._choice_value(self._choice_index(value)))
        scaled = getattr(self, "_scaled_descriptors", None) or self.descriptors
        if choice in scaled:
            vector = list(scaled[choice])
            if getattr(self, "_partial_descriptors", False):
                vector.append(0.0)
            return vector
        width = getattr(self, "_descriptor_width", None) or len(
            next(iter(self.descriptors.values()))
        )
        return [0.0] * width + [1.0]

    def raw_descriptor_vector(self, value: object) -> list[float]:
        """Original (unscaled) descriptor values, for reports."""
        if not self.descriptors:
            raise ValueError(f"variable {self.name!r} has no descriptors")
        choice = str(self._choice_value(self._choice_index(value)))
        if choice not in self.descriptors:
            raise ValueError(f"variable {self.name!r}: no descriptor for {choice!r}")
        return list(self.descriptors[choice])

    def to_dict(self) -> dict:
        out: dict[str, object] = {"name": self.name, "kind": self.kind}
        if self.is_numeric:
            out["low"] = self.low
            out["high"] = self.high
        else:
            out["choices"] = list(self.choices)
            if self.descriptors:
                out["descriptors"] = {
                    choice: list(vector) for choice, vector in self.descriptors.items()
                }
                if self.descriptor_names:
                    out["descriptor_names"] = list(self.descriptor_names)
        if self.unit:
            out["unit"] = self.unit
        if self.description:
            out["description"] = self.description
        return out

    @classmethod
    def from_dict(cls, payload: Mapping) -> "Variable":
        return cls(
            name=payload["name"],
            kind=payload.get("kind", "continuous"),
            low=payload.get("low"),
            high=payload.get("high"),
            choices=tuple(payload.get("choices", ())),
            unit=payload.get("unit", ""),
            description=payload.get("description", ""),
            descriptors={
                str(key): tuple(float(item) for item in vector)
                for key, vector in (payload.get("descriptors") or {}).items()
            }
            or None,
            descriptor_names=tuple(payload.get("descriptor_names") or ()),
        )


@dataclass
class SearchSpace:
    """An ordered collection of :class:`Variable` objects."""

    variables: tuple[Variable, ...]

    def __post_init__(self) -> None:
        if not self.variables:
            raise ValueError("search space must contain at least one variable")
        names = [v.name for v in self.variables]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate variable names: {names}")

    # -- construction -----------------------------------------------------
    @classmethod
    def from_dict(cls, payload: Mapping | Sequence) -> "SearchSpace":
        raw = payload.get("variables", payload) if isinstance(payload, Mapping) else payload
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raise ValueError("space payload must be a list of variables")
        return cls(tuple(Variable.from_dict(item) for item in raw))

    @classmethod
    def from_json(cls, path: str) -> "SearchSpace":
        with open(path, "r", encoding="utf-8-sig") as fh:
            return cls.from_dict(json.load(fh))

    def to_dict(self) -> dict:
        return {"variables": [v.to_dict() for v in self.variables]}

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False, indent=2)

    # -- structure --------------------------------------------------------
    def __len__(self) -> int:
        return len(self.variables)

    def __iter__(self):
        return iter(self.variables)

    @property
    def dim(self) -> int:
        """Number of variables (unit-cube dimension)."""
        return len(self.variables)

    @property
    def names(self) -> list[str]:
        return [v.name for v in self.variables]

    @property
    def numeric_variables(self) -> list[Variable]:
        return [v for v in self.variables if v.is_numeric]

    @property
    def discrete_variables(self) -> list[Variable]:
        return [v for v in self.variables if v.is_discrete]

    @property
    def feature_dim(self) -> int:
        return len(self.numeric_variables) + sum(v.feature_size() for v in self.discrete_variables)

    def variable(self, name: str) -> Variable:
        for v in self.variables:
            if v.name == name:
                return v
        raise KeyError(f"unknown variable {name!r}")

    def kind_counts(self) -> dict[str, int]:
        counts = {kind: 0 for kind in VALID_KINDS}
        for v in self.variables:
            counts[v.kind] += 1
        return counts

    def all_discrete(self) -> bool:
        return all(v.is_discrete for v in self.variables)

    def cardinality(self) -> float | int:
        total: float | int = 1
        for v in self.variables:
            card = v.cardinality
            total = math.inf if (card == math.inf or total == math.inf) else total * card
        return total

    # -- validation -------------------------------------------------------
    def _within_bounds(self, value: float, tol: float = 0.0) -> bool:
        """Bounds check with a relative tolerance.

        A unit-cube round trip can land one ulp outside the bound (measured:
        ``decode`` gave 4.999999999999999 for a ``[5, 240]`` variable).  Rejecting that
        would make the library unusable, while still catching a real out-of-range
        value such as 150 in ``[60, 120]``.
        """
        low, high = float(self.low), float(self.high)
        span = max(abs(high - low), 1.0)
        margin = tol * span
        return (low - margin) <= float(value) <= (high + margin)

    def coerce_bounds(
        self, config: Mapping[str, object]
    ) -> tuple[dict[str, object], dict[str, tuple[float, float]]]:
        """Clip numeric values into ``[low, high]`` and report what was changed.

        Returns ``(config, adjustments)`` where ``adjustments`` maps a variable name to
        ``(original, clipped)``.  Callers must *record* the adjustments: clipping
        without a trace is the silent behaviour this exists to replace.
        """
        out: dict[str, object] = dict(config)
        adjustments: dict[str, tuple[float, float]] = {}
        for variable in self.variables:
            if variable.name not in out:
                continue
            if variable.kind not in ("continuous", "log"):
                continue
            try:
                value = float(out[variable.name])
            except (TypeError, ValueError):
                continue
            if not math.isfinite(value):
                continue
            low, high = float(variable.low), float(variable.high)
            clipped = min(max(value, low), high)
            if clipped != value and not variable._within_bounds(value, tol=1e-9):
                adjustments[variable.name] = (value, clipped)
                out[variable.name] = clipped
            elif clipped != value:
                # within float tolerance: snap silently, it is not a real adjustment
                out[variable.name] = clipped
        return out, adjustments

    def validate_config(self, config: Mapping[str, object]) -> dict:
        """Validate a configuration dict; returns a fresh ordered dict."""
        if not isinstance(config, Mapping):
            raise TypeError("configuration must be a mapping")
        missing = [v.name for v in self.variables if v.name not in config]
        extra = [k for k in config if k not in self.names]
        if missing:
            raise ValueError(f"configuration missing variables: {missing}")
        if extra:
            raise ValueError(f"configuration contains unknown variables: {extra}")
        out: dict[str, object] = {}
        for v in self.variables:
            out[v.name] = self._coerce(v, config[v.name])
        return out

    def _coerce(self, variable: Variable, value: object) -> object:
        variable.validate_value(value)
        if variable.kind == "continuous":
            return float(value)
        if variable.kind in ("integer",):
            return int(value)
        if variable.kind == "log":
            return float(value)
        if variable.kind == "bool":
            if isinstance(value, bool):
                return value
            return str(value) == "True"
        return str(value)

    # -- encodings --------------------------------------------------------
    def encode(self, config: Mapping[str, object]) -> np.ndarray:
        return np.asarray([v.encode(config[v.name]) for v in self.variables], dtype=float)

    def decode(self, unit: Sequence[float]) -> dict:
        if len(unit) != self.dim:
            raise ValueError(f"expected {self.dim} coordinates, got {len(unit)}")
        return {v.name: v.decode(u) for v, u in zip(self.variables, unit)}

    def encode_many(self, configs: Iterable[Mapping[str, object]]) -> np.ndarray:
        rows = [self.encode(c) for c in configs]
        if not rows:
            return np.empty((0, self.dim))
        return np.vstack(rows)

    def decode_many(self, units: Iterable[Sequence[float]]) -> list[dict]:
        return [self.decode(u) for u in units]

    def features(self, config: Mapping[str, object]) -> np.ndarray:
        """Numeric feature vector used by surrogate models."""
        config = self.validate_config(config)
        parts: list[float] = []
        for v in self.variables:
            value = config[v.name]
            if v.is_numeric:
                parts.append(v.encode(value))
            elif v.descriptors:
                parts.extend(v.descriptor_vector(value))
            else:
                onehot = [0.0] * len(v.choices)
                onehot[v._choice_index(value)] = 1.0
                parts.extend(onehot)
        return np.asarray(parts, dtype=float)

    def features_many(self, configs: Iterable[Mapping[str, object]]) -> np.ndarray:
        rows = [self.features(c) for c in configs]
        if not rows:
            return np.empty((0, self.feature_dim))
        return np.vstack(rows)

    # -- sampling ---------------------------------------------------------
    def sample_uniform(self, n: int, rng: np.random.Generator | int | None = None) -> list[dict]:
        rng = _as_rng(rng)
        return [{v.name: v.sample_uniform(rng) for v in self.variables} for _ in range(n)]

    def sample_sobol(
        self,
        n: int,
        rng: np.random.Generator | int | None = None,
        scramble: bool = True,
    ) -> list[dict]:
        """Balanced space-filling points from a scrambled Sobol sequence."""
        from scipy.stats import qmc

        rng = _as_rng(rng)
        sampler = qmc.Sobol(d=self.dim, scramble=scramble, seed=int(rng.integers(0, 2**31 - 1)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            units = sampler.random(n)
        return self.decode_many(units)

    def sample_sobol_base2(
        self,
        n_total: int,
        seed: int | None = None,
        scramble: bool = True,
    ) -> list[dict]:
        """Balanced Sobol design of ``2**ceil(log2 n_total)`` points, prefix-cut.

        ``qmc.Sobol.random(n)`` only keeps its balance properties when ``n`` is
        a power of two; incremental calls also break them.  The robust recipe
        is to generate one balanced ``2**k`` design and slice a prefix.
        """
        from scipy.stats import qmc

        if n_total < 1:
            raise ValueError("n_total must be >= 1")
        k = max(1, int(math.ceil(math.log2(max(n_total, 2)))))
        if seed is None:
            seed = 0
        sampler = qmc.Sobol(d=self.dim, scramble=scramble, seed=int(seed))
        units = sampler.random_base2(k)
        return self.decode_many(units[:n_total])

    def sample_lhs(
        self,
        n: int,
        rng: np.random.Generator | int | None = None,
        optimization: str | None = None,
    ) -> list[dict]:
        """Latin hypercube sample of ``n`` points."""
        from scipy.stats import qmc

        rng = _as_rng(rng)
        sampler = qmc.LatinHypercube(
            d=self.dim,
            scramble=True,
            seed=int(rng.integers(0, 2**31 - 1)),
            optimization=optimization,
        )
        units = sampler.random(n)
        return self.decode_many(units)

    def sample_grid(self, max_points: int = 200, levels: int = 5) -> list[dict]:
        """Full factorial / gridded design, coarsened to fit ``max_points``."""
        if max_points < 1:
            raise ValueError("max_points must be >= 1")
        levels = max(int(levels), 2)
        import itertools

        grid_levels: list[list[object]] = []
        for v in self.variables:
            if v.kind == "continuous":
                grid_levels.append([float(x) for x in np.linspace(v.low, v.high, levels)])
            elif v.kind == "integer":
                raw = sorted({int(round(x)) for x in np.linspace(v.low, v.high, levels)})
                grid_levels.append(raw)
            elif v.kind == "log":
                grid_levels.append([float(x) for x in np.geomspace(v.low, v.high, levels)])
            else:
                grid_levels.append([v._choice_value(i) for i in range(len(v.choices))])
        level_lists = grid_levels
        def product() -> int:
            total = 1
            for lv in level_lists:
                total *= len(lv)
            return total

        while product() > max_points:
            candidates = [i for i, lv in enumerate(level_lists) if len(lv) > 2]
            if not candidates:
                break
            widest = max(candidates, key=lambda i: len(level_lists[i]))
            lv = level_lists[widest]
            keep = np.linspace(0, len(lv) - 1, len(lv) - 1).round().astype(int)
            level_lists[widest] = [lv[j] for j in keep]

        configs = [dict(zip(self.names, combo)) for combo in itertools.product(*level_lists)]
        configs = [self.validate_config(c) for c in configs]
        return configs[:max_points] if len(configs) > max_points else configs

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        parts = ", ".join(f"{v.name}:{v.kind}" for v in self.variables)
        return f"SearchSpace({parts})"

