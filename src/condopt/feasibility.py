"""Three-state feasibility semantics (backlog A12).

Before this module the same question -- *"is this measurement inside the declared
constraints?"* -- had three different answers depending on where you asked:

* ``study._is_feasible`` returned ``False`` when a constraint value was simply
  **missing** (so "not measured yet" was reported as "violated");
* ``benchmarks`` did the same, and additionally ignored observations whose
  ``constraints`` dict was empty altogether (so they were invisible rather than
  unknown);
* ``study.predict`` / the surrogate fit used ``np.nan`` (i.e. unknown) while the
  reported ``observed_feasible_fraction`` used the ``False`` rule.

A missing measurement is **not** evidence of violation: it is evidence of absence of
measurement.  This module makes that explicit with three states, and keeps the
conservative rule that only ``TRUE`` counts as feasible -- so no number moves upward
merely because something was unmeasured.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Iterable, Mapping, Sequence

__all__ = [
    "ConstraintState",
    "evaluate_constraint",
    "feasibility_state",
    "feasible_count",
    "coverage_note",
]


class ConstraintState(str, Enum):
    """Outcome of one constraint check against one measurement."""

    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return self.value


def _limit_of(limit: object) -> tuple[float, str] | None:
    if limit is None:
        return None
    if isinstance(limit, Mapping):
        if "value" not in limit:
            return None
        return float(limit.get("value", 0.0)), str(limit.get("op", "<="))
    # a bare number is accepted as a "<=" limit
    return float(limit), "<="


def evaluate_constraint(
    name: str, value: object, limit: object
) -> tuple[ConstraintState, str]:
    """Classify one constraint measurement.

    ``UNKNOWN`` covers: no measurement, a non-finite measurement, or a constraint
    without a usable limit (we cannot judge what we cannot compare against).
    """
    parsed = _limit_of(limit)
    if parsed is None:
        return ConstraintState.UNKNOWN, f"约束 {name!r} 未给出可用的限值"
    if value is None:
        return ConstraintState.UNKNOWN, f"未测到约束 {name!r}"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ConstraintState.UNKNOWN, f"约束 {name!r} 的测量值不是数字：{value!r}"
    if not math.isfinite(number):
        return ConstraintState.UNKNOWN, f"约束 {name!r} 的测量值非有限：{value!r}"
    threshold, op = parsed
    if op == ">=":
        ok = number >= threshold - 1e-9
    else:
        ok = number <= threshold + 1e-9
    if ok:
        return ConstraintState.TRUE, f"{name}={number:g} 满足 {op} {threshold:g}"
    return ConstraintState.FALSE, f"{name}={number:g} 违反 {op} {threshold:g}"


def _limits_for(objective: object) -> dict[str, object]:
    limits = dict(getattr(objective, "constraint_limits", {}) or {})
    for name in tuple(getattr(objective, "constraints", ()) or ()):
        limits.setdefault(name, None)
    return limits


def feasibility_state(observation: object, objective: object) -> dict:
    """Per-constraint and overall state for one observation.

    Overall rule: any ``FALSE`` -> ``FALSE``; else any ``UNKNOWN`` -> ``UNKNOWN``;
    else ``TRUE``.  With no declared constraints the state is ``TRUE`` (nothing to
    violate), which matches the previous behaviour.
    """
    limits = _limits_for(objective)
    if not limits:
        return {
            "state": ConstraintState.TRUE.value,
            "constraints": {},
            "violated": [],
            "unknown": [],
            "missing_names": [],
        }
    values = dict(getattr(observation, "constraints", {}) or {})
    details: dict[str, dict] = {}
    violated: list[str] = []
    unknown: list[str] = []
    missing: list[str] = []
    for name, limit in limits.items():
        raw = values.get(name)
        state, reason = evaluate_constraint(name, raw, limit)
        details[name] = {
            "state": state.value,
            "value": None if raw is None else float(raw),
            "limit": limit,
            "reason": reason,
        }
        if state is ConstraintState.FALSE:
            violated.append(name)
        elif state is ConstraintState.UNKNOWN:
            unknown.append(name)
            if raw is None:
                missing.append(name)
    if violated:
        overall = ConstraintState.FALSE
    elif unknown:
        overall = ConstraintState.UNKNOWN
    else:
        overall = ConstraintState.TRUE
    return {
        "state": overall.value,
        "constraints": details,
        "violated": violated,
        "unknown": unknown,
        "missing_names": missing,
    }


def feasible_count(
    observations: Iterable[object], objective: object
) -> dict[str, object]:
    """Count observations by state, with the *known* fraction made explicit.

    ``known_fraction`` is the share of observations whose constraints were actually
    measured; a low value means the feasibility numbers must not be trusted.
    """
    rows = list(observations)
    counts = {"true": 0, "false": 0, "unknown": 0}
    unknown_names: dict[str, int] = {}
    for observation in rows:
        report = feasibility_state(observation, objective)
        counts[report["state"]] += 1
        for name in report["missing_names"]:
            unknown_names[name] = unknown_names.get(name, 0) + 1
    total = len(rows)
    known = counts["true"] + counts["false"]
    return {
        "n": total,
        "true": counts["true"],
        "false": counts["false"],
        "unknown": counts["unknown"],
        "known_fraction": (known / total) if total else None,
        "unknown_by_constraint": unknown_names,
    }


def coverage_note(
    observations: Sequence[object], objective: object, threshold: float = 0.2
) -> dict | None:
    """Advisory when too many observations did not measure the constraints.

    Returns ``None`` when coverage is fine (or when there is nothing to cover), so the
    caller can simply inline it.
    """
    report = feasible_count(observations, objective)
    if report["n"] == 0 or report["unknown"] == 0:
        return None
    fraction = float(report["unknown"]) / float(report["n"])
    if fraction < threshold:
        return None
    names = sorted(report["unknown_by_constraint"])
    return {
        "unknown_share": round(fraction, 4),
        "unknown_by_constraint": report["unknown_by_constraint"],
        "advice": (
            f"有 {fraction:.0%} 的观测缺少约束测量"
            + (f"（{', '.join(names)}）" if names else "")
            + "：未测到 ≠ 满足，先补测这些约束，否则可行率与交付判断不可信。"
        ),
    }
