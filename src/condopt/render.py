"""Static, offline dashboard for a condopt study.

The renderer is strictly **read-only**: it never mutates the run directory and
never imports the optimizer stack, so it cannot change a campaign.  It reads
the files that :class:`condopt.study.Study` already writes:

``problem.json`` / ``config.json`` / ``dataset.jsonl`` / ``rounds.json`` /
``pending.json`` / ``decisions.jsonl`` (+ optional ``suggestions.csv`` as a
fallback when ``pending.json`` is empty).

It produces:

* ``dashboard.html`` -- one self-contained file (inline CSS + inline SVG,
  no CDN, no external request, UTF-8 without BOM) with the convergence curve,
  the space-coverage projection (first two variables, measured vs pending),
  the suggestion batch table, the router/override decision timeline and the
  budget progress;
* ``convergence.png`` -- optional matplotlib (Agg) rendering of the same
  convergence curve.

Deliberate limitations (v1, documented in ``docs/ui/PROTOTYPE_PLAN.md``):

* no prediction columns are invented: if ``pending.json`` has no
  ``predicted_mean`` / ``predicted_std`` the table shows an em dash;
* the coverage chart is a projection on the first two variables only;
* the parameter-importance panel is not part of this cut (needs >=20 points
  and a surrogate; see the plan).

CLI: ``py -3 scripts/render.py --run runs/demo``.
"""

from __future__ import annotations

import csv
import html
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

DASHBOARD_FILE = "dashboard.html"
CONVERGENCE_PNG = "convergence.png"
PNG_SIZE_INCH = (7.2, 4.2)
PNG_DPI = 110
MAX_DECISIONS = 12

# ---------------------------------------------------------------------------
# loading (tolerant: a partially written run must still render)
# ---------------------------------------------------------------------------


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (json.JSONDecodeError, OSError):
        return default


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError:
        return []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _read_suggestions_csv(path: Path) -> list[dict]:
    """Fallback when ``pending.json`` is empty.

    This is the *last issued* batch, which may already be measured; the
    dashboard labels it accordingly so it is not mistaken for new work.
    """
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except OSError:
        return []


def load_run(run_dir: str | Path) -> dict:
    """Load the read-only view of a run directory.

    Raises ``FileNotFoundError`` when ``problem.json`` is missing: a directory
    without a study is not renderable.
    """
    root = Path(run_dir)
    if not root.exists():
        raise FileNotFoundError(f"run directory does not exist: {root}")
    problem = _read_json(root / "problem.json")
    if not isinstance(problem, dict) or not problem.get("space"):
        raise FileNotFoundError(
            f"{root / 'problem.json'} is missing or invalid; run `condopt init` first"
        )
    config = _read_json(root / "config.json", {}) or {}
    dataset = _read_jsonl(root / "dataset.jsonl")
    rounds = _read_json(root / "rounds.json", []) or []
    pending = _read_json(root / "pending.json", []) or []
    decisions = _read_jsonl(root / "decisions.jsonl")
    suggestions = _read_suggestions_csv(root / "suggestions.csv")
    return {
        "run_dir": root,
        "problem": problem,
        "config": config,
        "dataset": dataset,
        "rounds": [r for r in rounds if isinstance(r, dict)],
        "pending": [p for p in pending if isinstance(p, dict)],
        "decisions": decisions,
        "decision": _read_json(root / "decision.json"),
        "spec_calibration": _read_json(root / "spec_calibration.json"),
        "step": _read_json(root / "step.json"),
        "suggestions": [s for s in suggestions if isinstance(s, dict)],
    }


# ---------------------------------------------------------------------------
# shaping
# ---------------------------------------------------------------------------


def _objective_target(problem: Mapping) -> tuple[str, str]:
    objective = problem.get("objective") or {}
    name = str(objective.get("name") or "score")
    direction = str(objective.get("direction") or "max")
    return name, ("min" if direction == "min" else "max")


def _objective_meta(problem: Mapping) -> dict:
    """Objectives + per-objective directions as stored by ``Objective.to_dict()``.

    Older ``problem.json`` files only have ``name/direction``; a single
    objective with that direction is synthesised then.
    """
    objective = problem.get("objective") or {}
    names = [str(name) for name in (objective.get("objectives") or [])]
    if not names:
        names = [str(objective.get("name") or "score")]
    base = "min" if str(objective.get("direction") or "max") == "min" else "max"
    directions = [str(value) for value in (objective.get("directions") or [])]
    if len(directions) != len(names):
        directions = [base for _ in names]
    directions = ["min" if value == "min" else "max" for value in directions]
    ref_point: list[float] = []
    for value in objective.get("ref_point") or []:
        try:
            ref_point.append(float(value))
        except (TypeError, ValueError):
            ref_point = []
            break
    if len(ref_point) != len(names):
        ref_point = []
    return {
        "name": str(objective.get("name") or names[0]),
        "direction": base,
        "objectives": names,
        "directions": directions,
        "ref_point": ref_point,
        "is_multi": len(names) > 1,
    }


def _objective_values(
    observation: Mapping,
    meta: Mapping,
    primary: str,
) -> tuple[list[float], list[float]] | None:
    """Raw values and signed scores for every objective; ``None`` if incomplete."""
    objectives = meta["objectives"]
    directions = meta["directions"]
    payload = observation.get("objectives") or {}
    if not isinstance(payload, Mapping):
        payload = {}
    values: list[float] = []
    for name in objectives:
        raw: Any = payload.get(name)
        if raw is None and name == primary:
            raw = observation.get("y")
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(value):
            return None
        values.append(value)
    signed = [value if directions[i] == "max" else -value for i, value in enumerate(values)]
    return values, signed


def _non_dominated_indices(points: Sequence[Sequence[float]]) -> list[int]:
    """Indices of non-dominated rows (all components maximised).

    Sorting each component descending guarantees that any dominator comes
    first, so a row only has to be compared against the current front.
    """
    if not points:
        return []
    width = len(points[0])
    order = sorted(range(len(points)), key=lambda index: tuple(-points[index][k] for k in range(width)))
    front: list[int] = []
    for index in order:
        candidate = points[index]
        dominated = False
        for other in front:
            reference = points[other]
            if all(reference[k] >= candidate[k] for k in range(width)) and any(
                reference[k] > candidate[k] for k in range(width)
            ):
                dominated = True
                break
        if not dominated:
            front.append(index)
    return front


def _hypervolume_2d(signed: Sequence[Sequence[float]], reference: Sequence[float]) -> float:
    """2D hypervolume in signed space (same formula as ``condopt.pareto``)."""
    ref_x, ref_y = float(reference[0]), float(reference[1])
    useful = [(float(x), float(y)) for x, y in signed if x > ref_x and y > ref_y]
    if not useful:
        return 0.0
    useful.sort()
    xs = [point[0] for point in useful]
    ys = [point[1] for point in useful]
    suffix_max: list[float] = []
    running = -math.inf
    for value in reversed(ys):
        running = max(running, value)
        suffix_max.append(running)
    suffix_max.reverse()
    area = 0.0
    previous_x = ref_x
    for index, x in enumerate(xs):
        if x <= previous_x:
            continue
        height = suffix_max[index] - ref_y
        if height > 0:
            area += (x - previous_x) * height
        previous_x = x
    return float(area)


def _pareto_summary(dataset: Sequence[Mapping], meta: Mapping) -> dict:
    """Non-dominated front + hypervolume, matching ``condopt.pareto.summarise``.

    Reference point (2 objectives) is the observed signed-score nadir shifted
    down by 5% of the observed span, exactly like ``pareto.summarise``.
    """
    primary = str(meta["name"])
    entries: list[dict] = []
    for index, observation in enumerate(dataset):
        parsed = _objective_values(observation, meta, primary)
        if parsed is None:
            continue
        values, signed = parsed
        entries.append(
            {
                "index": index,
                "x": observation.get("x") or {},
                "source": observation.get("source", ""),
                "values": values,
                "signed": signed,
            }
        )
    if not entries:
        return {
            "n_observations": 0,
            "n_front": 0,
            "front": [],
            "points": [],
            "hypervolume": None,
            "reference_signed": None,
            "reference_raw": None,
            "skipped": len(dataset),
        }
    points = [entry["signed"] for entry in entries]
    front_indices = _non_dominated_indices(points)
    front = [entries[index] for index in front_indices]
    front.sort(key=lambda entry: entry["signed"][0], reverse=True)
    hypervolume: float | None = None
    reference_signed: list[float] | None = None
    reference_raw: list[float] | None = None
    if len(meta["objectives"]) == 2:
        explicit = list(meta.get("ref_point") or [])
        if len(explicit) == 2:
            # Same precedence as ``condopt.pareto.summarise``: an explicit
            # ``Objective.ref_point`` (raw values) wins over the data-derived one.
            reference_raw = [float(value) for value in explicit]
            reference_signed = [
                value if meta["directions"][axis] == "max" else -value
                for axis, value in enumerate(explicit)
            ]
        else:
            lows = [min(point[axis] for point in points) for axis in range(2)]
            highs = [max(point[axis] for point in points) for axis in range(2)]
            span = [max(highs[axis] - lows[axis], 1e-9) for axis in range(2)]
            reference_signed = [lows[axis] - 0.05 * span[axis] for axis in range(2)]
            reference_raw = [
                reference_signed[axis] if meta["directions"][axis] == "max" else -reference_signed[axis]
                for axis in range(2)
            ]
        hypervolume = _hypervolume_2d(
            [entry["signed"] for entry in front], reference_signed
        )
    return {
        "n_observations": len(entries),
        "n_front": len(front),
        "front": front,
        "points": entries,
        "hypervolume": hypervolume,
        "reference_signed": reference_signed,
        "reference_raw": reference_raw,
        "skipped": len(dataset) - len(entries),
    }


# ---------------------------------------------------------------------------
# parameter importance (approximate, no sklearn)
# ---------------------------------------------------------------------------

IMPORTANCE_MIN_SAMPLES = 20


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
            end += 1
        average = (position + end) / 2.0 + 1.0
        for index in range(position, end + 1):
            ranks[order[index]] = average
        position = end + 1
    return ranks


def _spearman_abs(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) < 3 or len(ys) != len(xs):
        return None
    ranks_x = _average_ranks(xs)
    ranks_y = _average_ranks(ys)
    mean_x = sum(ranks_x) / len(ranks_x)
    mean_y = sum(ranks_y) / len(ranks_y)
    numerator = sum((a - mean_x) * (b - mean_y) for a, b in zip(ranks_x, ranks_y))
    norm_x = math.sqrt(sum((a - mean_x) ** 2 for a in ranks_x))
    norm_y = math.sqrt(sum((b - mean_y) ** 2 for b in ranks_y))
    if norm_x == 0 or norm_y == 0:
        return None
    return abs(numerator / (norm_x * norm_y))


def _eta_squared(keys: Sequence[str], ys: Sequence[float]) -> float | None:
    groups: dict[str, list[float]] = {}
    for key, value in zip(keys, ys):
        groups.setdefault(key, []).append(value)
    if len(groups) < 2:
        return None
    mean = sum(ys) / len(ys)
    total = sum((value - mean) ** 2 for value in ys)
    if total <= 0:
        return None
    between = 0.0
    for group in groups.values():
        group_mean = sum(group) / len(group)
        between += len(group) * (group_mean - mean) ** 2
    return max(0.0, min(1.0, between / total))


def _importance_summary(
    dataset: Sequence[Mapping],
    variables: Sequence[Mapping],
    objective_name: str,
    objective_direction: str,
) -> dict:
    """Approximate parameter importance for the primary objective.

    Numeric variables use |Spearman rho| with the signed objective; discrete
    variables use a one-way eta^2 (binned variance ratio).  No value is
    produced below ``IMPORTANCE_MIN_SAMPLES`` observations.
    """
    pairs: list[tuple[Mapping, float]] = []
    for observation in dataset:
        score, _ = _observation_score(observation, objective_name, objective_direction)
        x = observation.get("x")
        if score is None or not isinstance(x, Mapping):
            continue
        pairs.append((x, score))
    count = len(pairs)
    if count < IMPORTANCE_MIN_SAMPLES:
        return {"n": count, "insufficient": True, "items": [], "metric_mixed": False}
    items: list[dict] = []
    for variable in variables:
        name = str(variable.get("name"))
        kind = str(variable.get("kind") or "continuous")
        xs: list[Any] = []
        ys: list[float] = []
        for x, score in pairs:
            value = x.get(name)
            if value is None:
                continue
            xs.append(value)
            ys.append(score)
        if len(xs) < 3:
            continue
        if kind in ("categorical", "ordinal", "bool"):
            effect = _eta_squared([str(value) for value in xs], ys)
            metric = "η²"
        else:
            numeric: list[float] = []
            valid = True
            for value in xs:
                try:
                    numeric.append(float(value))
                except (TypeError, ValueError):
                    valid = False
                    break
            if not valid:
                continue
            effect = _spearman_abs(numeric, ys)
            metric = "|ρ|"
        if effect is None:
            continue
        items.append(
            {
                "var": name,
                "kind": kind,
                "effect": float(effect),
                "metric": metric,
                "n": len(xs),
            }
        )
    items.sort(key=lambda item: item["effect"], reverse=True)
    metrics = {item["metric"] for item in items}
    return {
        "n": count,
        "insufficient": False,
        "items": items[:8],
        "metric_mixed": len(metrics) > 1,
    }


def _observation_score(observation: Mapping, name: str, direction: str) -> tuple[float | None, Any]:
    """Return ``(signed_score, raw_value)``; signed score is always "higher is better"."""
    raw: Any = observation.get("y")
    if raw is None:
        objectives = observation.get("objectives") or {}
        if isinstance(objectives, Mapping):
            if name in objectives:
                raw = objectives[name]
            elif len(objectives) == 1:
                raw = next(iter(objectives.values()))
    if raw is None:
        return None, None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None, None
    if not math.isfinite(value):
        return None, None
    return (value if direction == "max" else -value), value


def _convergence_series(dataset: Sequence[Mapping], name: str, direction: str) -> dict:
    xs: list[int] = []
    scores: list[float] = []
    raw: list[float] = []
    for observation in dataset:
        score, value = _observation_score(observation, name, direction)
        if score is None:
            continue
        xs.append(len(xs) + 1)
        scores.append(score)
        raw.append(float(value))
    best: list[float] = []
    running = -math.inf
    for score in scores:
        running = max(running, score)
        best.append(running)
    return {"x": xs, "score": scores, "raw": raw, "best": best}


def _best_observation(dataset: Sequence[Mapping], name: str, direction: str) -> dict | None:
    best: dict | None = None
    best_score = -math.inf
    for observation in dataset:
        score, value = _observation_score(observation, name, direction)
        if score is None:
            continue
        if score > best_score:
            best_score = score
            best = {
                "score": score,
                "raw": value,
                "x": observation.get("x") or {},
                "source": observation.get("source", ""),
            }
    return best


def _stall_rounds(rounds: Sequence[Mapping]) -> int:
    """How many consecutive rounds without a new best (from ``rounds.json``)."""
    scores = [float(r["best"]) for r in rounds if isinstance(r.get("best"), (int, float))]
    stall = 0
    for value in reversed(scores):
        if stall == 0:
            peak = value
            stall = 1
            continue
        if value >= peak:
            break
        stall += 1
    return max(0, stall - 1)


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "—"
        if value == 0:
            return "0"
        magnitude = abs(value)
        if magnitude >= 1e5 or magnitude < 1e-3:
            return f"{value:.2e}"
        return f"{value:.{digits}g}"
    return str(value)


def _fmt_time(epoch: Any) -> str:
    try:
        stamp = float(epoch)
    except (TypeError, ValueError):
        return ""
    try:
        return datetime.fromtimestamp(stamp).strftime("%m-%d %H:%M")
    except (OverflowError, OSError, ValueError):
        return ""


def _escape(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


# ---------------------------------------------------------------------------
# axes / projection
# ---------------------------------------------------------------------------


def _nice_ticks(low: float, high: float, count: int = 5) -> list[float]:
    if not (math.isfinite(low) and math.isfinite(high)):
        return []
    if high == low:
        return [low]
    span = high - low
    raw_step = span / max(1, count - 1)
    magnitude = 10 ** math.floor(math.log10(raw_step))
    for factor in (1, 2, 2.5, 5, 10):
        step = factor * magnitude
        if span / step <= count:
            break
    start = math.ceil(low / step - 1e-9) * step
    ticks: list[float] = []
    value = start
    while value <= high + step * 1e-6 and len(ticks) < count + 2:
        ticks.append(value)
        value += step
    if not ticks:
        ticks = [low, high]
    if ticks[0] > low + 1e-12:
        ticks.insert(0, low)
    if ticks[-1] < high - 1e-12:
        ticks.append(high)
    return ticks


def _axis_for(variable: Mapping) -> dict:
    """Describe one variable as a normalized [0, 1] axis with tick labels."""
    name = str(variable.get("name", ""))
    kind = str(variable.get("kind", "continuous"))
    unit = str(variable.get("unit") or "")
    label = f"{name} / {unit}" if unit else name
    axis: dict = {"name": name, "kind": kind, "unit": unit, "label": label}

    if kind in ("categorical", "ordinal", "bool"):
        choices = [str(c) for c in (variable.get("choices") or [])]
        if kind == "bool" and not choices:
            choices = ["False", "True"]
        count = max(1, len(choices))
        positions = {choice: (index + 0.5) / count for index, choice in enumerate(choices)}
        axis["values"] = choices
        axis["positions"] = positions
        axis["ticks"] = [(positions[choice], choice) for choice in choices]
        axis["map"] = lambda value: (
            positions.get(str(value), 0.5)
            if not isinstance(value, bool)
            else positions.get("True" if value else "False", 0.5)
        )
        return axis

    try:
        low = float(variable.get("low"))
        high = float(variable.get("high"))
    except (TypeError, ValueError):
        low, high = 0.0, 1.0
    if high <= low:
        high = low + 1.0

    if kind == "log":
        low = max(low, 1e-12)
        log_low, log_high = math.log10(low), math.log10(high)

        def transform(value: float) -> float:
            return (math.log10(max(float(value), 1e-12)) - log_low) / (log_high - log_low)

        raw_ticks = [10 ** (log_low + (log_high - log_low) * i / 4) for i in range(5)]
        ticks = [(transform(value), _fmt(value)) for value in raw_ticks]
    else:
        def transform(value: float) -> float:
            return (float(value) - low) / (high - low)

        raw_ticks = _nice_ticks(low, high, 5)
        ticks = [(transform(value), _fmt(value)) for value in raw_ticks]

    axis.update({"low": low, "high": high, "ticks": ticks, "map": transform})
    return axis


def _project_point(x: Mapping, axis_x: dict, axis_y: dict | None) -> tuple[float, float] | None:
    value_x = x.get(axis_x["name"]) if isinstance(x, Mapping) else None
    if value_x is None:
        return None
    try:
        px = float(axis_x["map"](value_x))
    except (TypeError, ValueError):
        return None
    px = min(max(px, 0.0), 1.0)
    if axis_y is None:
        return px, 0.5
    value_y = x.get(axis_y["name"])
    if value_y is None:
        return None
    try:
        py = float(axis_y["map"](value_y))
    except (TypeError, ValueError):
        return None
    return px, min(max(py, 0.0), 1.0)


def _coverage(dataset: Sequence[Mapping], pending: Sequence[Mapping], variables: Sequence[Mapping]) -> dict:
    axis_x = _axis_for(variables[0])
    axis_y = _axis_for(variables[1]) if len(variables) > 1 else None
    measured: list[tuple[float, float]] = []
    proposed: list[tuple[float, float]] = []
    for observation in dataset:
        point = _project_point(observation.get("x") or {}, axis_x, axis_y)
        if point is not None:
            measured.append(point)
    for item in pending:
        point = _project_point(item.get("x") or {}, axis_x, axis_y)
        if point is not None:
            proposed.append(point)
    cells = {(min(int(x * 10), 9), min(int(y * 10), 9)) for x, y in measured}
    return {
        "axis_x": axis_x,
        "axis_y": axis_y,
        "measured": measured,
        "pending": proposed,
        "cells_occupied": len(cells),
        "cells_total": 100 if axis_y is not None else None,
    }


# ---------------------------------------------------------------------------
# SVG builders
# ---------------------------------------------------------------------------


def _svg_open(width: int, height: int, label: str) -> list[str]:
    # No xmlns on purpose: an inline <svg> in HTML5 needs none, and omitting it
    # guarantees the file contains no "http://" string (tested).
    return [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{_escape(label)}">'
    ]


def _svg_text(x: float, y: float, text: str, size: float = 10, anchor: str = "start", fill: str = "#69707d") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
        f'text-anchor="{anchor}" font-family="Segoe UI, Microsoft YaHei, sans-serif">{_escape(text)}</text>'
    )


def _convergence_svg(series: Mapping, objective_name: str, objective_direction: str) -> str:
    width, height = 760, 320
    left, right, top, bottom = 64, 16, 18, 40
    plot_w = width - left - right
    plot_h = height - top - bottom
    parts = _svg_open(width, height, "收敛曲线")
    xs: list[int] = series["x"]
    if not xs:
        parts.append(_svg_text(left, top + plot_h / 2, "暂无观测数据（先运行 condopt ask / tell）", 12, "start", "#69707d"))
        parts.append("</svg>")
        return "\n".join(parts)

    scores: list[float] = series["score"]
    best: list[float] = series["best"]
    low = min(scores + best)
    high = max(scores + best)
    pad = (high - low) * 0.08 or (abs(high) * 0.05 if high else 1.0)
    low -= pad
    high += pad
    span = high - low

    def px(index: int) -> float:
        return left + (index / max(1, len(xs) - 1)) * plot_w

    def py(value: float) -> float:
        return top + plot_h - ((value - low) / span) * plot_h

    # grid + y ticks
    for tick in _nice_ticks(low, high, 5):
        y = py(tick)
        if not (top - 1 <= y <= top + plot_h + 1):
            continue
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#eef1f5" stroke-width="1"/>')
        parts.append(_svg_text(left - 8, y + 4, _fmt(tick), 10, "end"))
    # x ticks
    tick_indexes = sorted({0, len(xs) - 1, *[round(i * (len(xs) - 1) / 4) for i in range(5)]})
    for index in tick_indexes:
        x = px(index)
        parts.append(f'<line x1="{x:.1f}" y1="{top + plot_h}" x2="{x:.1f}" y2="{top + plot_h + 4}" stroke="#c9cfd8"/>')
        parts.append(_svg_text(x, top + plot_h + 17, str(xs[index]), 10, "middle"))
    # axes
    parts.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#c9cfd8"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#c9cfd8"/>')
    # observations
    for index, value in zip(xs, scores):
        parts.append(f'<circle cx="{px(index):.1f}" cy="{py(value):.1f}" r="2.6" fill="#9aa3af"/>')
    # best-so-far step line
    points: list[str] = []
    for index, value in zip(xs, best):
        points.append(f"{px(index):.1f},{py(value):.1f}")
    parts.append(
        f'<polyline points="{" ".join(points)}" fill="none" stroke="#2f6feb" stroke-width="2.2" '
        'stroke-linejoin="round"/>'
    )
    # last best marker + label
    last_x, last_y = px(xs[-1]), py(best[-1])
    parts.append(f'<circle cx="{last_x:.1f}" cy="{last_y:.1f}" r="4" fill="#2f6feb" stroke="#fff" stroke-width="1.5"/>')
    direction_label = "越大越好" if objective_direction == "max" else "越小越好（已取负）"
    parts.append(
        _svg_text(
            last_x - 6,
            max(top + 12, last_y - 10),
            f"best {_fmt(best[-1])}",
            11,
            "end",
            "#2f6feb",
        )
    )
    # axis labels
    parts.append(_svg_text(left, height - 6, "评估序号（累计）", 10, "start"))
    parts.append(
        f'<text x="10" y="{top + plot_h / 2:.1f}" font-size="10" fill="#69707d" text-anchor="middle" '
        f'transform="rotate(-90 10 {top + plot_h / 2:.1f})" '
        f'font-family="Segoe UI, Microsoft YaHei, sans-serif">'
        f"{_escape(objective_name + '（' + direction_label + '）')}</text>"
    )
    parts.append("</svg>")
    return "\n".join(parts)


def _coverage_svg(coverage: Mapping, measured_label: str = "已测", pending_label: str = "待测") -> str:
    width, height = 760, 360
    axis_x = coverage["axis_x"]
    axis_y = coverage["axis_y"]
    left, right, top, bottom = 66, 150 if axis_y is not None else 20, 22, 52
    plot_w = width - left - right
    plot_h = height - top - bottom
    parts = _svg_open(width, height, "空间覆盖投影")

    def px(normalized: float) -> float:
        return left + normalized * plot_w

    def py(normalized: float) -> float:
        return top + plot_h - normalized * plot_h

    # frame
    parts.append(f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#fbfcfe" stroke="#d5dae3"/>')
    # x ticks
    for position, label in axis_x["ticks"]:
        x = px(position)
        parts.append(f'<line x1="{x:.1f}" y1="{top + plot_h}" x2="{x:.1f}" y2="{top + plot_h + 4}" stroke="#c9cfd8"/>')
        parts.append(_svg_text(x, top + plot_h + 17, label, 9, "middle"))
    # y ticks
    if axis_y is not None:
        for position, label in axis_y["ticks"]:
            y = py(position)
            parts.append(f'<line x1="{left - 4}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" stroke="#c9cfd8"/>')
            parts.append(_svg_text(left - 8, y + 3, label, 9, "end"))
        parts.append(_svg_text(left + plot_w / 2, height - 8, axis_x["label"], 10, "middle"))
        parts.append(
            f'<text x="14" y="{top + plot_h / 2:.1f}" font-size="10" fill="#69707d" text-anchor="middle" '
            f'transform="rotate(-90 14 {top + plot_h / 2:.1f})" '
            f'font-family="Segoe UI, Microsoft YaHei, sans-serif">{_escape(axis_y["label"])}</text>'
        )
    else:
        parts.append(_svg_text(left + plot_w / 2, height - 8, f"{axis_x['label']}（单变量投影）", 10, "middle"))

    # measured points
    for x, y in coverage["measured"]:
        parts.append(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="4" fill="#2f6feb" fill-opacity="0.75"/>')
    # pending points (hollow)
    for x, y in coverage["pending"]:
        parts.append(
            f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="5.2" fill="none" stroke="#b26a00" '
            'stroke-width="1.6" stroke-dasharray="3 2"/>'
        )

    # legend
    legend_x = left + plot_w + 18
    parts.append(f'<circle cx="{legend_x}" cy="{top + 14}" r="4" fill="#2f6feb" fill-opacity="0.75"/>')
    parts.append(_svg_text(legend_x + 10, top + 18, f"{measured_label} {len(coverage['measured'])}", 10))
    parts.append(
        f'<circle cx="{legend_x}" cy="{top + 38}" r="5.2" fill="none" stroke="#b26a00" stroke-width="1.6" stroke-dasharray="3 2"/>'
    )
    parts.append(_svg_text(legend_x + 10, top + 42, f"{pending_label} {len(coverage['pending'])}", 10))
    if coverage["cells_total"]:
        parts.append(
            _svg_text(legend_x, top + 72, f"10×10 网格命中 {coverage['cells_occupied']}/100", 9)
        )
    parts.append("</svg>")
    return "\n".join(parts)


def _pareto_svg(pareto: Mapping, meta: Mapping) -> str:
    """2D objective scatter (dominated vs front) + staircase frontier."""
    width, height = 760, 380
    left, right, top, bottom = 64, 150, 20, 52
    plot_w = width - left - right
    plot_h = height - top - bottom
    objectives = meta["objectives"]
    directions = meta["directions"]
    points = pareto["points"]
    front = pareto["front"]
    reference = pareto["reference_signed"]
    parts = _svg_open(width, height, "Pareto 前沿")

    signed_points = [entry["signed"] for entry in points]
    xs = [point[0] for point in signed_points]
    ys = [point[1] for point in signed_points]
    low_x, high_x = min(xs), max(xs)
    low_y, high_y = min(ys), max(ys)
    pad_x = (high_x - low_x) * 0.08 or 1.0
    pad_y = (high_y - low_y) * 0.08 or 1.0
    low_x -= pad_x
    high_x += pad_x
    low_y -= pad_y
    high_y += pad_y
    # The reference point can be far outside the observed range (e.g. a fixed
    # cross-run objective.ref_point); including it in the domain would squash
    # every observation into a corner, so it is drawn only when visible.
    ref_visible = (
        reference is not None
        and low_x <= float(reference[0]) <= high_x
        and low_y <= float(reference[1]) <= high_y
    )

    def px(value: float) -> float:
        return left + (value - low_x) / (high_x - low_x) * plot_w

    def py(value: float) -> float:
        return top + plot_h - (value - low_y) / (high_y - low_y) * plot_h

    def raw_label(value: float, axis: int) -> str:
        return _fmt(value if directions[axis] == "max" else -value)

    parts.append(f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#fbfcfe" stroke="#d5dae3"/>')
    for tick in _nice_ticks(low_x, high_x, 5):
        x = px(tick)
        if not (left - 1 <= x <= left + plot_w + 1):
            continue
        parts.append(f'<line x1="{x:.1f}" y1="{top + plot_h}" x2="{x:.1f}" y2="{top + plot_h + 4}" stroke="#c9cfd8"/>')
        parts.append(_svg_text(x, top + plot_h + 17, raw_label(tick, 0), 9, "middle"))
    for tick in _nice_ticks(low_y, high_y, 5):
        y = py(tick)
        if not (top - 1 <= y <= top + plot_h + 1):
            continue
        parts.append(f'<line x1="{left - 4}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" stroke="#c9cfd8"/>')
        parts.append(_svg_text(left - 8, y + 3, raw_label(tick, 1), 9, "end"))

    # dominated points then front points
    for entry in points:
        x, y = entry["signed"]
        parts.append(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="3" fill="#c3cad4" fill-opacity="0.8"/>')
    for entry in front:
        x, y = entry["signed"]
        parts.append(
            f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="4.6" fill="#2f6feb" '
            'stroke="#fff" stroke-width="1.2"/>'
        )

    # staircase frontier in signed space (ascending x, descending y)
    ordered = sorted(front, key=lambda entry: entry["signed"][0])
    if ordered:
        path: list[tuple[float, float]] = []
        for index, entry in enumerate(ordered):
            x, y = entry["signed"]
            if index == 0:
                path.append((px(x), py(y)))
                continue
            previous_x, previous_y = ordered[index - 1]["signed"]
            path.append((px(x), py(previous_y)))  # horizontal
            path.append((px(x), py(y)))  # vertical
        points_attr = " ".join(f"{x:.1f},{y:.1f}" for x, y in path)
        parts.append(
            f'<polyline points="{points_attr}" fill="none" stroke="#2f6feb" stroke-width="1.8" '
            'stroke-dasharray="5 3"/>'
        )

    # reference point (only when it is inside the visible data domain)
    if reference is not None:
        if ref_visible:
            ref_x, ref_y = px(float(reference[0])), py(float(reference[1]))
            parts.append(
                f'<line x1="{ref_x - 6:.1f}" y1="{ref_y:.1f}" x2="{ref_x + 6:.1f}" y2="{ref_y:.1f}" stroke="#c62828" stroke-width="1.5"/>'
            )
            parts.append(
                f'<line x1="{ref_x:.1f}" y1="{ref_y - 6:.1f}" x2="{ref_x:.1f}" y2="{ref_y + 6:.1f}" stroke="#c62828" stroke-width="1.5"/>'
            )
        else:
            reference_raw = pareto.get("reference_raw") or []
            label = "，".join(
                f"{name}={_fmt(value)}" for name, value in zip(objectives, reference_raw)
            )
            parts.append(
                _svg_text(
                    left + 8,
                    top + plot_h - 8,
                    f"参考点在图外：{label}（见下方 HV 说明）",
                    9,
                    "start",
                    "#c62828",
                )
            )

    # legend
    legend_x = left + plot_w + 16
    parts.append(f'<circle cx="{legend_x}" cy="{top + 14}" r="4" fill="#2f6feb"/>')
    parts.append(_svg_text(legend_x + 10, top + 18, f"非支配 {len(front)}", 10))
    parts.append(f'<circle cx="{legend_x}" cy="{top + 36}" r="3" fill="#c3cad4"/>')
    parts.append(_svg_text(legend_x + 10, top + 40, f"被支配 {len(points) - len(front)}", 10))
    if ref_visible:
        parts.append(
            f'<text x="{legend_x + 5}" y="{top + 62}" font-size="13" fill="#c62828" '
            f'font-family="Segoe UI, Microsoft YaHei, sans-serif">×</text>'
        )
        parts.append(_svg_text(legend_x + 18, top + 62, "参考点", 10))
    parts.append(
        _svg_text(
            left + plot_w / 2,
            height - 8,
            f"{objectives[0]}（{'越大越右' if directions[0] == 'max' else '越小越右'}）",
            10,
            "middle",
        )
    )
    parts.append(
        f'<text x="14" y="{top + plot_h / 2:.1f}" font-size="10" fill="#69707d" text-anchor="middle" '
        f'transform="rotate(-90 14 {top + plot_h / 2:.1f})" '
        f'font-family="Segoe UI, Microsoft YaHei, sans-serif">'
        f"{_escape(objectives[1] + ('（越大越上）' if directions[1] == 'max' else '（越小越上）'))}</text>"
    )
    parts.append("</svg>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# HTML sections
# ---------------------------------------------------------------------------


CSS = """
:root{--bg:#f6f7f9;--card:#fff;--ink:#1d2330;--muted:#69707d;--line:#e3e6eb;--brand:#2f6feb;--warn:#b26a00;--bad:#c62828}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 "Microsoft YaHei","Segoe UI",system-ui,sans-serif}
header{background:var(--card);border-bottom:1px solid var(--line);padding:14px 22px}
h1{font-size:17px;margin:0 0 8px}
h2{font-size:14px;margin:0 0 10px}
.row{display:flex;gap:14px;flex-wrap:wrap;align-items:center}
.chip{background:#eef2fb;color:#2f6feb;border-radius:10px;padding:1px 8px;font-size:12px;white-space:nowrap}
.chip.warn{background:#fff3e0;color:var(--warn)}
.muted{color:var(--muted)} .small{font-size:12px}
main{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;padding:14px 22px;max-width:1320px;margin:0 auto}
section{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;min-width:0}
.wide{grid-column:1/-1}
.bar{height:9px;background:#e9ecf1;border-radius:6px;overflow:hidden;flex:1;min-width:140px}
.bar>i{display:block;height:100%;background:var(--brand)}
.metric{margin:8px 0}
.metric .bar{height:10px;min-width:0;margin-top:3px}
.metric .bar>i{background:#7aa2f7}
table{width:100%;border-collapse:collapse;font-size:12.5px}
th,td{padding:6px 7px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
th{color:var(--muted);font-weight:600}
tbody tr:hover{background:#fafbfd}
.tag{display:inline-block;border-radius:9px;padding:0 8px;font-size:11px;background:#eef2fb;color:#2f6feb;white-space:nowrap}
.tag.override{background:#fff3e0;color:var(--warn)}
.tag.auto{background:#f3ecff;color:#6b3fc4}
ol.log{margin:4px 0 0;padding-left:18px;font-size:12.5px}
ol.log li{margin:7px 0}
ol.log p{margin:2px 0 0;color:#3a4150}
details summary{cursor:pointer;color:var(--muted);font-size:12px}
footer{max-width:1320px;margin:0 auto;padding:6px 22px 30px;color:var(--muted);font-size:12px}
.empty{padding:14px 6px;color:var(--muted);font-size:13px}
"""


def _objective_chip_text(meta: Mapping) -> str:
    arrows = " / ".join(
        f"{name}{'↑' if direction == 'max' else '↓'}"
        for name, direction in zip(meta["objectives"], meta["directions"])
    )
    suffix = "（多目标）" if meta["is_multi"] else ""
    return f"目标 {arrows}{suffix}"


def _importance_section(importance: Mapping) -> str:
    if importance["insufficient"]:
        return f"""
<section>
  <h2>参数重要性 <span class="muted small">近似</span></h2>
  <p class="empty">样本不足（&lt;{IMPORTANCE_MIN_SAMPLES}），暂不计算参数重要性（当前 n={importance['n']}）。</p>
  <p class="muted small">达到 {IMPORTANCE_MIN_SAMPLES} 条观测后，数值变量显示 |Spearman ρ|、分类变量显示 η²。</p>
</section>
"""
    items = importance["items"]
    if not items:
        return """
<section>
  <h2>参数重要性 <span class="muted small">近似</span></h2>
  <p class="empty">没有可分析的变量（全部缺失或常量）。</p>
</section>
"""
    rows: list[str] = []
    for item in items:
        width = max(0.0, min(1.0, float(item["effect"]))) * 100.0
        rows.append(
            f'<div class="metric"><span class="small">{_escape(item["var"])}</span> '
            f'<span class="muted small">{_escape(item["metric"])} {item["effect"]:.2f}（n={item["n"]}）</span>'
            f'<div class="bar"><i style="width:{width:.1f}%"></i></div></div>'
        )
    mixed = "；数值=|Spearman ρ|，分类=η²（组间方差比）" if importance["metric_mixed"] else ""
    return f"""
<section>
  <h2>参数重要性 <span class="muted small">近似，n={importance['n']}</span></h2>
  {''.join(rows)}
  <p class="muted small">相关 ≠ 因果，仅反映已评价区域{mixed}。样本 &lt;{IMPORTANCE_MIN_SAMPLES} 时不显示数值。</p>
</section>
"""


def _pareto_section(meta: Mapping, pareto: Mapping, variables: Sequence[Mapping]) -> str:
    if not meta["is_multi"]:
        return ""
    objectives = meta["objectives"]
    directions = meta["directions"]
    objective_heads = "".join(
        f"<th>{_escape(name)}{'↑' if direction == 'max' else '↓'}</th>"
        for name, direction in zip(objectives, directions)
    )
    if pareto["n_observations"] == 0:
        return f"""
<section class="wide">
  <h2>Pareto 前沿 <span class="muted small">{_escape(_objective_chip_text(meta))}</span></h2>
  <p class="empty">暂无完整多目标观测（需要 dataset.jsonl 中同时包含 {_escape(' / '.join(objectives))}）。</p>
</section>
"""
    front = pareto["front"]
    shown = front[:30]
    condition_heads = "".join(
        f"<th>{_escape(v.get('name'))}"
        + (f" / {_escape(v.get('unit'))}" if v.get("unit") else "")
        + "</th>"
        for v in variables
    )
    body: list[str] = []
    for rank, entry in enumerate(shown, start=1):
        cells = []
        for variable in variables:
            cells.append(f"<td>{_escape(_cell_text(entry['x'].get(str(variable.get('name')))))}</td>")
        values = "".join(f"<td>{_escape(_fmt(value))}</td>" for value in entry["values"])
        body.append(
            f"<tr><td>{rank}</td>" + "".join(cells) + values
            + f"<td>{_escape(entry.get('source') or '—')}</td></tr>"
        )
    hv_text = "—（仅 2 目标计算 HV）"
    reference_text = ""
    if pareto["hypervolume"] is not None:
        hv_text = _fmt(pareto["hypervolume"])
        reference_raw = pareto["reference_raw"] or []
        if reference_raw:
            reference_text = "；参考点：" + "，".join(
                f"{_escape(name)}={_escape(_fmt(value))}"
                for name, value in zip(objectives, reference_raw)
            )
    chart = ""
    if len(objectives) >= 2 and pareto["reference_signed"] is not None:
        chart = _pareto_svg(pareto, meta)
    elif len(objectives) > 2:
        chart = '<p class="muted small">目标数 &gt; 2：仅输出非支配点表（不画二维前沿图）。</p>'
    skipped = pareto["skipped"]
    return f"""
<section class="wide">
  <h2>Pareto 前沿 <span class="muted small">{_escape(_objective_chip_text(meta))} · 非支配 {pareto['n_front']} / {pareto['n_observations']} 点 · HV={_escape(hv_text)}</span></h2>
  {chart}
  <table>
    <thead><tr><th>#</th>{condition_heads}{objective_heads}<th>来源</th></tr></thead>
    <tbody>{''.join(body)}</tbody>
  </table>
  <p class="muted small">共 {pareto['n_front']} 个非支配点，表中显示前 {len(shown)} 个；HV 参考点口径与 Study 一致（显式 Objective.ref_point 优先，否则观测 nadir 下方 5%，signed 空间）{reference_text}
  {'' if skipped == 0 else f'；{skipped} 条观测缺少数值目标，已跳过'}。</p>
</section>
"""


def _header_html(info: Mapping, stats: Mapping) -> str:
    config = info["config"]
    algorithm = str(config.get("algorithm") or "—")
    context = config.get("context") or {}
    meta = stats.get("meta") or _objective_meta(info["problem"])
    run_name = info["run_dir"].name or str(info["run_dir"])
    params = config.get("params") or {}
    param_hint = ", ".join(f"{key}={_fmt(value)}" for key, value in list(params.items())[:6])
    dim = context.get("n_dims") or len((info["problem"].get("space") or {}).get("variables") or [])
    batch_size = config.get("batch_size") or (context.get("batch_size") if isinstance(context, Mapping) else None)
    budget = stats["budget"]
    used = stats["used"]
    ratio = (used / budget * 100.0) if budget else 0.0
    created = stats.get("generated_at", "")
    chips = [
        f'<span class="chip">算法 {_escape(algorithm)}</span>',
        f'<span class="chip">{_escape(_objective_chip_text(meta))}</span>',
        f'<span class="chip">维度 {_escape(dim)}</span>',
        f'<span class="chip">批量 {_escape(batch_size if batch_size is not None else "—")}</span>',
        f'<span class="chip">seed {_escape(info["problem"].get("seed", 0))}</span>',
    ]
    if config.get("manual_override"):
        chips.append('<span class="chip warn">manual override 中（auto_route 暂停）</span>')
    if stats["pending"]:
        chips.append(f'<span class="chip warn">待测建议 {stats["pending"]} 条</span>')
    return f"""
<header>
  <h1>condopt 看板 <span class="chip">离线静态</span> <span class="muted small">run: {_escape(run_name)}</span></h1>
  <div class="row small">
    {''.join(chips)}
    <span class="muted">生成于 {_escape(created)}</span>
  </div>
  <div class="row" style="margin-top:10px">
    <span class="small">预算 <b>{used}</b> / <b>{budget}</b> 已评价{('，另有 ' + str(stats['pending']) + ' 条待测') if stats['pending'] else ''}</span>
    <div class="bar"><i style="width:{min(100.0, ratio):.1f}%"></i></div>
    <span class="small muted">剩余 {max(0, budget - used - stats['pending'])} 次 · 轮次 {stats['rounds']}</span>
  </div>
  <div class="row small muted" style="margin-top:6px">
    <span>参数：{_escape(param_hint) if param_hint else '—'}</span>
  </div>
</header>
"""


def _convergence_section(stats: Mapping, series: Mapping, svg: str) -> str:
    meta = stats.get("meta") or {}
    multi_note = ""
    if meta.get("is_multi"):
        multi_note = (
            f" 多目标 study：曲线只画主目标 {meta['name']}，其余目标见 Pareto 区块。"
        )
    if not series["x"]:
        caption = "尚无观测；下方建议批次表可用于第一轮实验。"
    else:
        caption = (
            f"共 {len(series['x'])} 次观测 / {stats['rounds']} 轮；当前最优 {_fmt(series['best'][-1])}。"
            + (f" 已连续 {stats['stall_rounds']} 轮无改进（router 会在 patience 轮后插入探索批）。" if stats["stall_rounds"] >= 2 else "")
            + multi_note
        )
    return f"""
<section>
  <h2>收敛曲线 <span class="muted small">best-so-far + 观测散点</span></h2>
  {svg}
  <p class="muted small">{_escape(caption)}</p>
</section>
"""


def _coverage_section(coverage: Mapping, svg: str) -> str:
    axis_x = coverage["axis_x"]
    axis_y = coverage["axis_y"]
    if axis_y is None:
        scope = f"单变量 {axis_x['label']}"
    else:
        scope = f"前两维投影：{axis_x['label']} × {axis_y['label']}"
    cells = coverage["cells_occupied"]
    return f"""
<section>
  <h2>空间覆盖 <span class="muted small">{_escape(scope)}</span></h2>
  {svg}
  <p class="muted small">实心=已测，空心虚线=待测。前两维 10×10 网格命中 {cells} 个格子；仅反映前两维，其他维度未展开。</p>
</section>
"""


def _predicted_cell(item: Mapping, key: str) -> str:
    value = item.get(key)
    return _escape(_fmt(value)) if value is not None else "—"


def _cell_text(value: Any) -> str:
    """Format a suggestion cell; CSV values arrive as strings."""
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            return value
    return _fmt(value)


def _batch_table_html(info: Mapping, variables: Sequence[Mapping]) -> str:
    pending = info["pending"]
    rows_source = pending
    source_default = "pending"
    badge = '<span class="chip">待测（pending.json）</span>'
    if not pending and info["suggestions"]:
        rows_source = info["suggestions"]
        source_default = "suggestions.csv"
        badge = '<span class="chip warn">上一批（suggestions.csv，可能已 tell）</span>'
    if not rows_source:
        return f'<p class="empty">当前没有待测建议。运行 <b>condopt ask --run &lt;dir&gt;</b> 生成下一批条件。</p>{badge}'
    head_names = "".join(
        f"<th>{_escape(v.get('name'))}"
        + (f" / {_escape(v.get('unit'))}" if v.get("unit") else "")
        + "</th>"
        for v in variables
    )
    body: list[str] = []
    for row in rows_source:
        raw_x = row.get("x")
        x: Mapping = raw_x if isinstance(raw_x, Mapping) else {
            str(variable.get("name")): row.get(str(variable.get("name"))) for variable in variables
        }
        cells = []
        for variable in variables:
            name = str(variable.get("name"))
            cells.append(f"<td>{_escape(_cell_text(x.get(name)))}</td>")
        item_id = row.get("id", "")
        algorithm = row.get("algorithm", info["config"].get("algorithm", ""))
        source = row.get("source") or row.get("stage") or source_default
        body.append(
            "<tr>"
            f"<td>{_escape(item_id)}</td>"
            f"<td>{_escape(algorithm)}</td>"
            + "".join(cells)
            + f"<td>{_predicted_cell(row, 'predicted_mean')}</td>"
            f"<td>{_predicted_cell(row, 'predicted_std')}</td>"
            f"<td>{_predicted_cell(row, 'acquisition')}</td>"
            f"<td>{_escape(source)}</td>"
            "</tr>"
        )
    return f"""
<div class="row small" style="margin-bottom:8px">{badge}<span class="muted">预测列来自算法/代理输出；v1 的 Study.ask 不产出预测，因此显示「—」，不做伪造。</span></div>
<table>
  <thead><tr><th>id</th><th>算法</th>{head_names}<th>预测均值</th><th>预测σ</th><th>采集值</th><th>来源</th></tr></thead>
  <tbody>{''.join(body)}</tbody>
</table>
"""


_ACTION_LABELS = {
    "initial_route": ("初始路由", ""),
    "auto_reroute": ("自动重路由", "auto"),
    "override": ("覆写 override", "override"),
    "resume_auto_route": ("恢复自动路由", "auto"),
}


def _decision_html(info: Mapping) -> str:
    decisions = info["decisions"][-MAX_DECISIONS:]
    if not decisions:
        return '<p class="empty">没有决策记录（decisions.jsonl 为空或不存在）。</p>'
    items: list[str] = []
    for record in decisions:
        action = str(record.get("action") or "?")
        label, css = _ACTION_LABELS.get(action, (action, ""))
        payload = record.get("payload") or {}
        algorithm = str(record.get("algorithm") or "")
        reason = str(record.get("reason") or "")
        details: list[str] = []
        if isinstance(payload, Mapping):
            if payload.get("confidence"):
                details.append(f"confidence={payload['confidence']}")
            warnings = payload.get("warnings") or []
            if warnings:
                details.append("warnings: " + "；".join(str(w) for w in warnings[:3]))
            previous = payload.get("previous") or {}
            if isinstance(previous, Mapping) and previous.get("algorithm"):
                details.append(f"previous={previous['algorithm']}")
            if payload.get("from"):
                details.append(f"from={payload['from']}")
        if record.get("params"):
            params = record["params"]
            if isinstance(params, Mapping):
                keys = list(params)[:5]
                details.append("params: " + ", ".join(f"{k}={_fmt(params[k])}" for k in keys))
        item = [
            f'<li><span class="tag {css}">{_escape(label)}</span> '
            f'<b>{_escape(algorithm)}</b> <span class="muted small">{_escape(_fmt_time(record.get("t")))}</span>',
        ]
        if reason:
            item.append(f"<p>{_escape(reason)}</p>")
        if details:
            item.append(f'<p class="muted small">{_escape(" · ".join(details))}</p>')
        item.append("</li>")
        items.append("".join(item))
    return (
        f'<ol class="log">{"".join(items)}</ol>'
        '<p class="muted small">override 必须带理由（Study.override 强制），这里原样展示 decisions.jsonl。</p>'
    )


def _details_html(info: Mapping) -> str:
    files = ["problem.json", "config.json", "dataset.jsonl", "rounds.json", "pending.json", "decisions.jsonl"]
    summary = {
        "run_dir": str(info["run_dir"]),
        "algorithm": info["config"].get("algorithm"),
        "parameters": info["config"].get("params"),
        "context": info["config"].get("context"),
        "n_observations": len(info["dataset"]),
        "n_rounds": len(info["rounds"]),
        "n_pending": len(info["pending"]),
        "n_decisions": len(info["decisions"]),
    }
    raw = json.dumps(summary, ensure_ascii=False, indent=2)
    return (
        "<details><summary>原始摘要（JSON，只读）</summary>"
        f'<pre class="small" style="overflow:auto;max-height:280px">{_escape(raw)}</pre></details>'
        f'<p class="muted small">数据源：{_escape(" / ".join(files))}；只读渲染，不修改 run 目录。</p>'
    )


def _decision_card(decision: Any) -> str:
    """Recommendation verdict card (written by ``condopt decision``)."""
    if not isinstance(decision, Mapping):
        return ""
    basis = _escape(str(decision.get("selection_basis") or "—"))
    rows = [
        ("推荐依据", basis),
        ("推荐理由", _escape(str(decision.get("selection_reason") or "—"))),
        ("板效应", "是" if decision.get("plate_effect_detected") else "否"),
        ("规格符合性", "✓ in-spec" if decision.get("in_spec") else "✗ out-of-spec"),
        ("原始/校正", f"{decision.get('raw_in_spec')} / {decision.get('in_spec')}"),
    ]
    body = "".join(
        f"<tr><th>{_escape(name)}</th><td>{value}</td></tr>" for name, value in rows
    )
    explain = decision.get("explain") or []
    explain_html = (
        "<ul>" + "".join(f"<li>{_escape(str(line))}</li>" for line in explain) + "</ul>"
        if explain
        else ""
    )
    return f"""
<section>
  <h2>推荐判定 <span class="muted small">condopt decision</span></h2>
  <table><tbody>{body}</tbody></table>
  {explain_html}
</section>"""


def _calibration_card(calibration: Any) -> str:
    """Spec calibration report card (written by ``condopt calibrate-spec``)."""
    if not isinstance(calibration, Mapping):
        return ""
    metrics = calibration.get("metrics") or {}
    if not metrics:
        return ""
    rows = []
    for name, entry in metrics.items():
        scale = entry.get("suggested_scale")
        tolerance = entry.get("suggested_tolerance")
        groups = len(entry.get("groups") or {})
        global_scale = entry.get("global_scale")
        comparison = ""
        if global_scale is not None and scale is not None:
            comparison = f"（全局 scale={_fmt(global_scale)}）"
        rows.append(
            "<tr><th>{}</th><td>n={} scale={}{} tolerance={} groups={}</td></tr>".format(
                _escape(str(name)),
                _escape(str(entry.get("n"))),
                "—" if scale is None else _fmt(scale),
                comparison,
                "—" if tolerance is None else _fmt(tolerance),
                groups,
            )
        )
    return f"""
<section>
  <h2>规格校准 <span class="muted small">condopt calibrate-spec</span></h2>
  <table><tbody>{''.join(rows)}</tbody></table>
</section>"""


def _step_card(step: Any) -> str:
    """Next-action card (written by ``condopt step``)."""
    if not isinstance(step, Mapping) or not step.get("action"):
        return ""
    actions = step.get("suggested_actions") or []
    reasons = step.get("suggested_action_reasons") or {}
    items = []
    for code in actions:
        reason = reasons.get(code)
        text = f"<code>{_escape(str(code))}</code>"
        if reason:
            text += f" — {_escape(str(reason))}"
        items.append(f"<li>{text}</li>")
    action_list = f"<ul>{''.join(items)}</ul>" if items else ""
    return f"""
<section class="wide">
  <h2>下一步行动 <span class="muted small">condopt step</span></h2>
  <p><b>{_escape(str(step.get('action')))}</b> — {_escape(str(step.get('reason') or ''))}</p>
  <p class="muted small">命令：<code>{_escape(str(step.get('next_command') or ''))}</code></p>
  {action_list}
</section>"""


def build_html(info: Mapping) -> tuple[str, dict]:
    """Assemble the single-file dashboard; returns ``(html, stats)``."""
    meta = _objective_meta(info["problem"])
    objective_name, objective_direction = meta["name"], meta["direction"]
    variables = list((info["problem"].get("space") or {}).get("variables") or [])
    series = _convergence_series(info["dataset"], objective_name, objective_direction)
    best = _best_observation(info["dataset"], objective_name, objective_direction)
    importance = _importance_summary(
        info["dataset"], variables, objective_name, objective_direction
    )
    pareto = _pareto_summary(info["dataset"], meta) if meta["is_multi"] else None
    context = info["config"].get("context") or {}
    budget = int(info["problem"].get("budget") or context.get("budget_total") or 0)
    rounds = len(info["rounds"])
    stats = {
        "objective": (objective_name, objective_direction),
        "meta": meta,
        "n_objectives": len(meta["objectives"]),
        "pareto_front": pareto["n_front"] if pareto is not None else 0,
        "hypervolume": pareto["hypervolume"] if pareto is not None else None,
        "importance_n": importance["n"],
        "budget": budget,
        "used": len(info["dataset"]),
        "pending": len(info["pending"]),
        "rounds": rounds,
        "stall_rounds": _stall_rounds(info["rounds"]),
        "best": best,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    convergence_svg = _convergence_svg(series, objective_name, objective_direction)
    coverage = _coverage(info["dataset"], info["pending"], variables) if variables else None
    coverage_svg = (
        _coverage_svg(coverage)
        if coverage is not None
        else '<p class="empty">没有可投影的变量。</p>'
    )
    best_text = "—"
    if best is not None:
        conditions = "，".join(f"{_escape(k)}={_escape(_fmt(v))}" for k, v in list(best["x"].items())[:4])
        best_text = f"{_fmt(best['raw'])}（{conditions}）"
    pareto_section = _pareto_section(meta, pareto, variables) if pareto is not None else ""
    importance_section = _importance_section(importance)
    decision_section = _decision_card(info.get("decision"))
    calibration_section = _calibration_card(info.get("spec_calibration"))
    step_section = _step_card(info.get("step"))

    html_doc = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>condopt 看板 · {_escape(info["run_dir"].name or "run")}</title>
<style>{CSS}</style>
</head>
<body>
{_header_html(info, stats)}
<main>
  {_convergence_section(stats, series, convergence_svg)}
  {_coverage_section(coverage, coverage_svg) if coverage is not None else '<section><h2>空间覆盖</h2><p class="empty">没有可投影的变量。</p></section>'}
  {importance_section}
  {decision_section}
  {step_section}
  {calibration_section}
  {pareto_section}
  <section class="wide">
    <h2>建议批次 <span class="muted small">当前待测条件 + 单位</span></h2>
    {_batch_table_html(info, variables)}
  </section>
  <section class="wide">
    <h2>router 决策与 override 理由 <span class="muted small">最近 {min(len(info["decisions"]), MAX_DECISIONS)} / {len(info["decisions"])} 条</span></h2>
    {_decision_html(info)}
    <p class="muted small">当前最优（主目标）：{best_text}</p>
  </section>
  <section class="wide">
    <h2>数据摘要</h2>
    {_details_html(info)}
  </section>
</main>
<footer>离线单文件：无 CDN、无外部请求、UTF-8。由 condopt render 生成；数据只读。</footer>
</body>
</html>
"""
    return html_doc, stats


# ---------------------------------------------------------------------------
# PNG figure (matplotlib, lazy import, Agg)
# ---------------------------------------------------------------------------


def _write_convergence_png(path: Path, series: Mapping, objective: tuple[str, str]) -> list[str]:
    warnings: list[str] = []
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        from matplotlib import pyplot as plt
    except Exception as exc:  # pragma: no cover - matplotlib is a hard dependency in practice
        return [f"matplotlib 不可用，跳过 PNG：{type(exc).__name__}: {exc}"]

    objective_name, objective_direction = objective
    try:
        matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        matplotlib.rcParams["axes.unicode_minus"] = False
        fig, ax = plt.subplots(figsize=PNG_SIZE_INCH, dpi=PNG_DPI)
        if series["x"]:
            xs = series["x"]
            ax.scatter(xs, series["score"], s=14, c="#9aa3af", label="观测", zorder=2)
            ax.step(xs, series["best"], where="post", color="#2f6feb", linewidth=2.0, label="best-so-far", zorder=3)
            last_x, last_y = xs[-1], series["best"][-1]
            ax.scatter([last_x], [last_y], s=28, c="#2f6feb", zorder=4)
            ax.annotate(
                f"best {_fmt(last_y)}",
                xy=(last_x, last_y),
                xytext=(-8, 10),
                textcoords="offset points",
                ha="right",
                fontsize=9,
                color="#2f6feb",
            )
            ax.legend(loc="lower right", fontsize=9, frameon=False)
        else:
            ax.text(0.5, 0.5, "暂无观测数据", ha="center", va="center", transform=ax.transAxes, fontsize=12)
        direction_label = "越大越好" if objective_direction == "max" else "越小越好（已取负）"
        ax.set_xlabel("评估序号（累计）")
        ax.set_ylabel(f"{objective_name}（{direction_label}）")
        ax.set_title("condopt 收敛曲线（best-so-far）")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, format="png", facecolor="white")
        plt.close(fig)
    except Exception as exc:
        try:
            plt.close("all")
        except Exception:
            pass
        warnings.append(f"PNG 生成失败：{type(exc).__name__}: {exc}")
    return warnings


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def render_study(
    run_dir: str | Path,
    out_dir: str | Path | None = None,
    make_png: bool = True,
) -> dict:
    """Render a read-only dashboard for one study.

    Parameters
    ----------
    run_dir:
        Study directory written by :class:`condopt.study.Study`.
    out_dir:
        Where to write ``dashboard.html`` / ``convergence.png``; defaults to
        ``run_dir`` itself.
    make_png:
        Also write ``convergence.png`` via matplotlib Agg.

    Returns
    -------
    dict with the generated paths plus a few counters and any non-fatal
    warnings.  The run directory is never modified.
    """
    info = load_run(run_dir)
    target = Path(out_dir) if out_dir is not None else info["run_dir"]
    target.mkdir(parents=True, exist_ok=True)

    document, stats = build_html(info)
    dashboard = target / DASHBOARD_FILE
    dashboard.write_bytes(document.encode("utf-8"))  # UTF-8, no BOM

    warnings: list[str] = []
    png_path = target / CONVERGENCE_PNG
    if make_png:
        warnings.extend(_write_convergence_png(png_path, _convergence_series(
            info["dataset"],
            stats["objective"][0],
            stats["objective"][1],
        ), stats["objective"]))
        if not png_path.exists():
            png_path = None  # type: ignore[assignment]
    else:
        png_path = None  # type: ignore[assignment]

    return {
        "run_dir": str(info["run_dir"]),
        "dashboard": str(dashboard),
        "convergence_png": str(png_path) if png_path else None,
        "n_observations": len(info["dataset"]),
        "n_rounds": stats["rounds"],
        "n_pending": stats["pending"],
        "n_decisions": len(info["decisions"]),
        "n_objectives": stats["n_objectives"],
        "n_pareto_front": stats["pareto_front"],
        "hypervolume": stats["hypervolume"],
        "importance_n": stats["importance_n"],
        "warnings": warnings,
    }


__all__ = ["DASHBOARD_FILE", "CONVERGENCE_PNG", "build_html", "load_run", "render_study"]
