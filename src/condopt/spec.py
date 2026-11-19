"""Specifications for real experimental goals.

A real condition-optimization problem is rarely "maximize one number".  It is a
set of *specifications*: targets, acceptable ranges, hard constraints, and
weights.  This module turns that decision layer into something the optimizers
can consume:

* :class:`Goal` -- one metric with a mode (max / min / target / range),
* :class:`Spec` -- goals + constraints + aggregation.

``Spec.scalar_objective()`` produces a normal :class:`condopt.base.Objective`
whose scalar score is ``spec.score(...)``; constraints become
``constraint_limits`` so the constrained optimizers can use them.  The original
measured metrics are kept in ``Observation.objectives``; ``Study`` recomputes
the scalar ``y`` from the spec, so the audit trail still contains raw data.

Example
-------
>>> spec = Spec(
...     goals=(
...         Goal("E_MPa", mode="target", target=4.5, scale=2.0),
...         Goal("UTS_MPa", mode="target", target=7.5, scale=2.5),
...         Goal("elongation_pct", mode="target", target=75, scale=25),
...     ),
...     constraints={"cure_time_s": {"op": "<=", "value": 10.0}},
... )
>>> objective = spec.scalar_objective()
>>> spec.score({"E_MPa": 4.4, "UTS_MPa": 7.6, "elongation_pct": 74})
-0.0...
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field, replace
from typing import Mapping

from .base import Objective

MODES = ("max", "min", "target", "range")


@dataclass
class Goal:
    metric: str
    mode: str = "max"
    target: float | None = None
    low: float | None = None
    high: float | None = None
    tolerance: float | None = None
    weight: float = 1.0
    scale: float | None = None

    def __post_init__(self) -> None:
        self.mode = str(self.mode).lower()
        if self.mode not in MODES:
            raise ValueError(f"goal {self.metric!r}: mode must be one of {MODES}")
        if self.mode == "target" and self.target is None:
            raise ValueError(f"goal {self.metric!r}: target mode needs target=")
        if self.mode == "range" and (self.low is None or self.high is None):
            raise ValueError(f"goal {self.metric!r}: range mode needs low= and high=")
        if self.low is not None and self.high is not None and self.low >= self.high:
            raise ValueError(f"goal {self.metric!r}: low must be < high")
        self.weight = float(self.weight)
        if self.weight <= 0:
            raise ValueError(f"goal {self.metric!r}: weight must be > 0")

    def _span(self) -> float:
        if self.scale is not None:
            return float(self.scale)
        if self.low is not None and self.high is not None:
            return max(float(self.high) - float(self.low), 1e-12)
        if self.tolerance is not None:
            return max(float(self.tolerance), 1e-12)
        return 1.0

    def score(self, value: float) -> float:
        value = float(value)
        span = self._span()
        if self.mode == "max":
            return (value - (self.low or 0.0)) / span if self.low is not None else value / span
        if self.mode == "min":
            return ((self.high or 0.0) - value) / span if self.high is not None else -value / span
        if self.mode == "target":
            return -abs(value - float(self.target)) / span
        # range: full score inside, negative distance outside
        if float(self.low) <= value <= float(self.high):
            return 0.0
        distance = float(self.low) - value if value < float(self.low) else value - float(self.high)
        return -distance / span

    def in_spec(self, value: float) -> bool:
        value = float(value)
        if self.mode in ("max", "min"):
            return True
        if self.mode == "target":
            if self.tolerance is None:
                # A target without an explicit tolerance is a soft preference:
                # the score still rewards proximity, but it is not a hard spec.
                return True
            return abs(value - float(self.target)) <= float(self.tolerance)
        return float(self.low) <= value <= float(self.high)

    def to_dict(self) -> dict:
        payload = {"metric": self.metric, "mode": self.mode, "weight": self.weight}
        for key in ("target", "low", "high", "tolerance", "scale"):
            value = getattr(self, key)
            if value is not None:
                payload[key] = float(value)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping) -> "Goal":
        return cls(**{key: payload.get(key) for key in (
            "metric", "mode", "target", "low", "high", "tolerance", "weight", "scale"
        ) if key in payload})


def _robust_scale(values: list[float]) -> float:
    """1.4826*MAD, falling back to population std then range."""
    if not values:
        return 0.0
    median = statistics.median(values)
    mad = statistics.median([abs(value - median) for value in values])
    robust = 1.4826 * mad
    if robust <= 0:
        robust = statistics.pstdev(values)
    if robust <= 0:
        robust = max(values) - min(values)
    return float(robust)


@dataclass
class Spec:
    goals: tuple[Goal, ...]
    constraints: dict[str, dict] = field(default_factory=dict)
    aggregation: str = "weighted_sum"

    def __post_init__(self) -> None:
        if not self.goals:
            raise ValueError("spec needs at least one goal")
        if self.aggregation not in ("weighted_sum", "worst_case"):
            raise ValueError(
                "aggregation must be 'weighted_sum' or 'worst_case'"
            )

    @property
    def metrics(self) -> tuple[str, ...]:
        return tuple(goal.metric for goal in self.goals)

    def score(self, values: Mapping[str, float]) -> float:
        missing = [goal.metric for goal in self.goals if goal.metric not in values]
        if missing:
            raise ValueError(f"spec metrics missing: {missing}")
        if self.aggregation == "worst_case":
            # Robust spec: the weakest goal dominates, so a single bad metric
            # cannot be compensated by gains elsewhere (weights are ignored).
            return min(goal.score(values[goal.metric]) for goal in self.goals)
        total_weight = sum(goal.weight for goal in self.goals)
        return sum(goal.weight * goal.score(values[goal.metric]) for goal in self.goals) / total_weight

    def compliance(self, values: Mapping[str, float]) -> dict:
        goals = {}
        in_spec = True
        for goal in self.goals:
            value = float(values.get(goal.metric, float("nan")))
            ok = goal.in_spec(value)
            in_spec = in_spec and ok
            goals[goal.metric] = {
                "value": value,
                "mode": goal.mode,
                "target": goal.target,
                "low": goal.low,
                "high": goal.high,
                "in_spec": ok,
                "score": goal.score(value) if value == value else None,
            }
        constraints = {}
        for name, limit in self.constraints.items():
            value = values.get(name)
            if value is None:
                constraints[name] = {"value": None, "in_spec": None}
                continue
            op = str(limit.get("op", "<="))
            threshold = float(limit.get("value", 0.0))
            ok = float(value) <= threshold if op == "<=" else float(value) >= threshold
            in_spec = in_spec and ok
            constraints[name] = {"value": float(value), "op": op, "threshold": threshold, "in_spec": ok}
        return {"in_spec": in_spec, "goals": goals, "constraints": constraints}

    def scalar_objective(self, name: str = "spec_score") -> Objective:
        return Objective(
            name=name,
            direction="max",
            # The scalar objective is single-objective by design: goal metrics
            # stay in the question/spec layer, not in Objective.objectives
            # (otherwise the router would treat the spec as multi-objective).
            constraints=tuple(self.constraints),
            constraint_limits=self.constraints,
        )

    def observation_values(self, observation) -> dict:
        values = dict(observation.objectives)
        values.update(observation.constraints)
        if not values and observation.y is not None and len(self.goals) == 1:
            values[self.goals[0].metric] = float(observation.y)
        return values

    def calibrate(
        self,
        records: list[Mapping[str, float]],
        min_samples: int = 3,
        tolerance_sigmas: float = 2.0,
    ) -> tuple["Spec", dict]:
        """Suggest scales/tolerances from historical measured values.

        Policy (deliberately conservative):
        * scale: robust scale (1.4826*MAD, fallback to range/std) when >= min_samples;
        * target tolerance: tolerance_sigmas * robust scale (only for target goals);
        * range bounds and modes are kept unchanged, never guessed from data.

        Returns ``(calibrated_spec, report)``. The input spec is not mutated.
        """
        goals = []
        report: dict = {"n_records": len(records), "min_samples": min_samples,
                        "tolerance_sigmas": tolerance_sigmas, "metrics": {}}
        for goal in self.goals:
            values = [
                float(record[goal.metric])
                for record in records
                if goal.metric in record
                and record[goal.metric] is not None
                and float(record[goal.metric]) == float(record[goal.metric])
            ]
            entry = {"n": len(values), "suggested_scale": None,
                     "suggested_tolerance": None, "confidence": "low"}
            if len(values) >= min_samples:
                robust = _robust_scale(values)
                entry["suggested_scale"] = float(robust)
                entry["confidence"] = "high" if len(values) >= 8 else "medium"
                updated = replace(goal, scale=float(robust))
                if goal.mode == "target":
                    tolerance = max(tolerance_sigmas * robust, 1e-12)
                    entry["suggested_tolerance"] = float(tolerance)
                    updated = replace(updated, tolerance=float(tolerance))
                goals.append(updated)
            else:
                goals.append(goal)
            report["metrics"][goal.metric] = entry
        calibrated = Spec(tuple(goals), constraints=dict(self.constraints),
                          aggregation=self.aggregation)
        return calibrated, report

    def calibrate_by_group(
        self,
        records: list[Mapping[str, float]],
        groups: list[str],
        min_samples: int = 3,
        tolerance_sigmas: float = 2.0,
    ) -> tuple["Spec", dict]:
        """Calibrate within group (e.g. plate), ignoring between-group shifts.

        Uses the median of per-group robust scales as the pooled scale, so a
        plate-to-plate offset does not inflate the tolerance.
        """
        report: dict = {"n_records": len(records), "min_samples": min_samples,
                        "tolerance_sigmas": tolerance_sigmas, "metrics": {}}
        goals = []
        for goal in self.goals:
            grouped: dict[str, list[float]] = {}
            for record, group in zip(records, groups):
                if goal.metric not in record or record[goal.metric] is None:
                    continue
                value = float(record[goal.metric])
                if value != value:
                    continue
                grouped.setdefault(str(group), []).append(value)
            per_group = {}
            scales = []
            for name, values in grouped.items():
                if len(values) >= 2:
                    scale = _robust_scale(values)
                    per_group[name] = {"n": len(values), "scale": float(scale)}
                    scales.append(scale)
                else:
                    per_group[name] = {"n": len(values), "scale": None}
            n_total = sum(len(values) for values in grouped.values())
            all_values = [value for values in grouped.values() for value in values]
            global_scale = _robust_scale(all_values) if len(all_values) >= 2 else None
            entry = {"n": n_total, "groups": per_group, "suggested_scale": None,
                     "suggested_tolerance": None, "global_scale": global_scale,
                     "global_tolerance": (
                         None if global_scale is None
                         else max(tolerance_sigmas * global_scale, 1e-12)
                     ),
                     "confidence": "low"}
            if scales and n_total >= min_samples:
                pooled = statistics.median(scales)
                entry["suggested_scale"] = float(pooled)
                entry["confidence"] = "high" if n_total >= 8 else "medium"
                updated = replace(goal, scale=float(pooled))
                if goal.mode == "target":
                    tolerance = max(tolerance_sigmas * pooled, 1e-12)
                    entry["suggested_tolerance"] = float(tolerance)
                    updated = replace(updated, tolerance=float(tolerance))
                goals.append(updated)
            else:
                goals.append(goal)
            report["metrics"][goal.metric] = entry
        calibrated = Spec(tuple(goals), constraints=dict(self.constraints),
                          aggregation=self.aggregation)
        return calibrated, report

    def calibrate_observations(
        self, observations, by_plate: bool = False, **kwargs
    ) -> tuple["Spec", dict]:
        records = [self.observation_values(observation) for observation in observations]
        if by_plate:
            groups = [str(obs.meta.get("plate") or "?") for obs in observations]
            if len(set(groups)) > 1:
                return self.calibrate_by_group(records, groups, **kwargs)
        return self.calibrate(records, **kwargs)

    def achievability(self, records: list[Mapping[str, float]]) -> dict:
        """Compare each goal with the observed value range.

        This is a diagnostic, not an automatic spec rewrite: if a target lies
        outside everything measured, the campaign should question the spec
        rather than spend the whole budget.
        """
        report: dict = {"n_records": len(records), "metrics": {}}
        for goal in self.goals:
            values = [
                float(record[goal.metric])
                for record in records
                if goal.metric in record
                and record[goal.metric] is not None
                and float(record[goal.metric]) == float(record[goal.metric])
            ]
            entry: dict = {
                "n": len(values),
                "mode": goal.mode,
                "observed": None,
                "target": goal.target,
                "target_within_observed": None,
                "suggested_target": None,
                "in_spec_count": 0,
            }
            if values:
                low, high = min(values), max(values)
                entry["observed"] = [low, high]
                entry["in_spec_count"] = sum(1 for value in values if goal.in_spec(value))
                if goal.mode == "target" and goal.target is not None:
                    inside = low - 1e-12 <= float(goal.target) <= high + 1e-12
                    entry["target_within_observed"] = inside
                    if not inside:
                        entry["suggested_target"] = low if float(goal.target) < low else high
                elif goal.mode == "range":
                    inside = any(goal.in_spec(value) for value in values)
                    entry["target_within_observed"] = inside
                    if not inside:
                        entry["suggested_range"] = [low, high]
            report["metrics"][goal.metric] = entry
        return report

    def suggest_spec(
        self,
        records: list[Mapping[str, float]],
        min_samples: int = 3,
        tolerance_sigmas: float = 2.0,
        groups: list[str] | None = None,
    ) -> tuple["Spec", dict]:
        """Suggest an *achievable* spec from measured history.

        Calibration first (scale/tolerance); then for target goals whose target
        lies outside the observed range, move the target to the nearest observed
        bound. Range goals with no observed point inside get the observed range.
        Never mutates the current spec; returns ``(suggested_spec, report)``.
        """
        if groups is not None:
            calibrated, calibration = self.calibrate_by_group(
                records, groups, min_samples=min_samples, tolerance_sigmas=tolerance_sigmas
            )
        else:
            calibrated, calibration = self.calibrate(
                records, min_samples=min_samples, tolerance_sigmas=tolerance_sigmas
            )
        achievability = self.achievability(records)
        goals = []
        changed = []
        for original, candidate in zip(self.goals, calibrated.goals):
            entry = achievability["metrics"][original.metric]
            updated = candidate
            if original.mode == "target" and entry.get("target_within_observed") is False:
                if entry.get("suggested_target") is not None:
                    updated = replace(candidate, target=float(entry["suggested_target"]))
                    changed.append({"metric": original.metric, "kind": "target",
                                    "from": original.target, "to": entry["suggested_target"]})
            elif original.mode == "range" and entry.get("target_within_observed") is False:
                suggested_range = entry.get("suggested_range")
                if suggested_range:
                    low, high = float(suggested_range[0]), float(suggested_range[1])
                    if high <= low:
                        high = low + 1e-6  # keep low < high valid for degenerate data
                    updated = replace(candidate, low=low, high=high)
                    changed.append({"metric": original.metric, "kind": "range",
                                    "from": [original.low, original.high],
                                    "to": [low, high]})
            goals.append(updated)
        suggested = Spec(tuple(goals), constraints=dict(self.constraints),
                          aggregation=self.aggregation)
        report = {
            "calibration": calibration,
            "achievability": achievability,
            "changed": changed,
        }
        return suggested, report

    def to_dict(self) -> dict:
        return {
            "goals": [goal.to_dict() for goal in self.goals],
            "constraints": self.constraints,
            "aggregation": self.aggregation,
        }

    @classmethod
    def from_dict(cls, payload: Mapping | None) -> "Spec | None":
        if not payload:
            return None
        goals = tuple(Goal.from_dict(item) for item in payload.get("goals", ()))
        return cls(goals=goals, constraints=dict(payload.get("constraints", {})),
                   aggregation=payload.get("aggregation", "weighted_sum"))
