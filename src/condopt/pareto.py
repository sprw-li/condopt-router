"""Pareto-front utilities for multi-objective studies.

Everything internal is expressed as *signed scores*: for ``direction="max"``
the score is the value, for ``direction="min"`` it is the negated value, so
"larger is better" holds for every objective and dominance is the standard
component-wise comparison.

The reference point for hypervolume is also in signed-score space (a point
dominated by every front member); for a normalised value in ``[0, 1]`` it is
``(0, 0)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from .base import Objective
from .dataset import Dataset, Observation


def observation_values(observation: Observation, objective: Objective) -> np.ndarray:
    """Raw objective values of one observation, ordered by ``objective.objectives``."""
    values = []
    for name in objective.objectives:
        if name in observation.objectives:
            values.append(float(observation.objectives[name]))
        elif name == objective.name and observation.y is not None:
            values.append(float(observation.y))
        elif len(objective.objectives) == 1 and observation.y is not None:
            values.append(float(observation.y))
        else:
            raise ValueError(f"observation has no value for objective {name!r}")
    return np.asarray(values, dtype=float)


def signed_scores(values: np.ndarray, objective: Objective) -> np.ndarray:
    """Flip minimization objectives so that all objectives are maximized."""
    signs = np.asarray([1.0 if d == "max" else -1.0 for d in objective.directions], dtype=float)
    return values * signs


def score_matrix(dataset: Dataset | Sequence[Observation], objective: Objective) -> np.ndarray:
    observations = dataset.observations if isinstance(dataset, Dataset) else list(dataset)
    if not observations:
        return np.empty((0, objective.n_objectives))
    return np.vstack([signed_scores(observation_values(o, objective), objective) for o in observations])


def non_dominated_mask_2d(scores: np.ndarray) -> np.ndarray:
    """Fast O(n log n) non-dominated mask for two maximized objectives."""
    n = len(scores)
    mask = np.zeros(n, dtype=bool)
    if n == 0:
        return mask
    order = np.lexsort((-scores[:, 1], -scores[:, 0]))  # x desc, then y desc
    best_y = -np.inf
    for index in order:
        y = scores[index, 1]
        if y > best_y:
            mask[index] = True
            best_y = y
    return mask


def non_dominated_mask(scores: np.ndarray) -> np.ndarray:
    """Boolean mask of non-dominated rows (all objectives maximized)."""
    n = len(scores)
    if n == 0:
        return np.ones(0, dtype=bool)
    if scores.ndim == 2 and scores.shape[1] == 2:
        return non_dominated_mask_2d(scores)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        for j in range(n):
            if i == j or not mask[j]:
                continue
            # j dominates i: at least as good everywhere, strictly better somewhere
            if np.all(scores[j] >= scores[i]) and np.any(scores[j] > scores[i]):
                mask[i] = False
                break
    return mask


def pareto_front(
    dataset: Dataset | Sequence[Observation], objective: Objective
) -> list[tuple[Observation, np.ndarray]]:
    """Non-dominated observations and their signed score vectors."""
    observations = dataset.observations if isinstance(dataset, Dataset) else list(dataset)
    if len(observations) <= 1:
        return [(o, signed_scores(observation_values(o, objective), objective)) for o in observations]
    scores = score_matrix(observations, objective)
    mask = non_dominated_mask(scores)
    return [(observations[i], scores[i]) for i in range(len(observations)) if mask[i]]


def partition_2d(scores: np.ndarray, reference: Sequence[float]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Disjoint boxes covering the dominated region of a 2D non-dominated set.

    ``scores`` are signed (maximized) and must be non-dominated; the union of
    ``[reference, p]`` rectangles is partitioned into disjoint boxes so that
    ``hypervolume = sum(prod(upper - lower))``.
    """
    ref = np.asarray(reference, dtype=float)
    points = np.asarray(scores, dtype=float)
    points = points[np.all(points > ref, axis=1)]
    if len(points) == 0:
        return []
    order = np.argsort(-points[:, 0])
    points = points[order]
    boxes: list[tuple[np.ndarray, np.ndarray]] = []
    for index in range(len(points)):
        lower = np.asarray(
            [points[index + 1, 0] if index + 1 < len(points) else ref[0], ref[1]], dtype=float
        )
        upper = np.asarray([points[index, 0], points[index, 1]], dtype=float)
        if np.all(upper > lower):
            boxes.append((lower, upper))
    return boxes


def hypervolume_from_boxes(boxes: Sequence[tuple[np.ndarray, np.ndarray]]) -> float:
    return float(sum(np.prod(upper - lower) for lower, upper in boxes))


def dhv_2d_batch(
    points: np.ndarray,
    boxes: Sequence[tuple[np.ndarray, np.ndarray]],
    reference: Sequence[float],
) -> np.ndarray:
    """Vectorized 2D dominance hypervolume improvement for candidate draws.

    ``points`` has shape ``(..., 2)``; returns ``(...,)`` with
    ``DHV(y) = prod(y - r) - sum_boxes prod(clip(y, box) - lower)``.
    """
    ref = np.asarray(reference, dtype=float)
    values = np.asarray(points, dtype=float)
    total = np.prod(values - ref, axis=-1)
    overlap = np.zeros_like(total)
    for lower, upper in boxes:
        clipped = np.minimum(values, upper) - lower
        contained = np.all(clipped > 0, axis=-1)
        contribution = np.prod(np.where(contained[..., None], clipped, 0.0), axis=-1)
        overlap = overlap + contribution
    improvement = np.maximum(total - overlap, 0.0)
    valid = np.all(values > ref, axis=-1)
    return np.where(valid, improvement, 0.0)


def dhv_2d(
    point: Sequence[float],
    boxes: Sequence[tuple[np.ndarray, np.ndarray]],
    reference: Sequence[float],
) -> float:
    return float(dhv_2d_batch(np.asarray(point, dtype=float), boxes, reference))


@dataclass
class ParetoSummary:
    n_observations: int
    n_front: int
    scores: np.ndarray  # front signed scores, shape (n_front, m)
    indices: list[int]
    hypervolume: float | None = None
    reference_point: np.ndarray | None = None


def summarise(
    dataset: Dataset | Sequence[Observation],
    objective: Objective,
    reference: Sequence[float] | None = None,
) -> ParetoSummary:
    observations = dataset.observations if isinstance(dataset, Dataset) else list(dataset)
    if not observations:
        return ParetoSummary(0, 0, np.empty((0, objective.n_objectives)), [])
    scores = score_matrix(observations, objective)
    mask = non_dominated_mask(scores)
    indices = [int(i) for i in np.flatnonzero(mask)]
    front = scores[mask]
    hypervolume = None
    reference_point = None
    if objective.n_objectives in (2, 3, 4):
        if reference is not None:
            reference_point = np.asarray(reference, dtype=float)
        elif objective.ref_point:
            signs = np.asarray([1.0 if d == "max" else -1.0 for d in objective.directions])
            reference_point = np.asarray(objective.ref_point, dtype=float) * signs
        else:
            # Fallback: a data-derived reference slightly below the observed
            # nadir.  Comparable within a study, but not across studies -- pass
            # ``Objective.ref_point`` for cross-run comparability.
            lows = np.min(scores, axis=0)
            highs = np.max(scores, axis=0)
            span = np.maximum(highs - lows, 1e-9)
            reference_point = lows - 0.05 * span
        if objective.n_objectives == 2:
            hypervolume = hypervolume_2d(front, reference_point)
        else:
            hypervolume = hypervolume_nd(front, reference_point)
    return ParetoSummary(
        len(observations), len(indices), front, indices, hypervolume, reference_point
    )


def hypervolume_2d(scores: np.ndarray, reference: Sequence[float]) -> float:
    """Hypervolume of a set of 2D signed scores w.r.t. a reference point."""
    if len(scores) == 0:
        return 0.0
    ref = np.asarray(reference, dtype=float)
    points = np.asarray(scores, dtype=float)
    useful = points[np.all(points > ref, axis=1)]
    if len(useful) == 0:
        return 0.0
    order = np.argsort(useful[:, 0])
    useful = useful[order]
    xs = useful[:, 0]
    # height of the strip ending at x_i is the max y among points with x >= x_i
    suffix_max = np.maximum.accumulate(useful[::-1, 1])[::-1]
    area = 0.0
    previous_x = ref[0]
    for index, x in enumerate(xs):
        if x <= previous_x:
            continue
        height = suffix_max[index] - ref[1]
        if height > 0:
            area += (x - previous_x) * height
        previous_x = x
    return float(area)


def hypervolume_nd(scores: np.ndarray, reference: Sequence[float]) -> float:
    """Hypervolume of signed scores in any dimension (recursive slicing).

    Maximization convention: ``scores`` must dominate ``reference`` to count.
    Works for 2-4 objectives on modest fronts; for 2D prefer the closed-form
    :func:`hypervolume_2d`.
    """
    points = np.asarray(scores, dtype=float)
    if points.size == 0:
        return 0.0
    if points.ndim == 1:
        points = points.reshape(1, -1)
    ref = np.asarray(reference, dtype=float)
    if points.shape[1] == 0:
        return 0.0
    if points.shape[1] == 1:
        useful_1d = points[:, 0][points[:, 0] > ref[0]]
        return float(max(useful_1d) - ref[0]) if len(useful_1d) else 0.0
    useful = points[np.all(points > ref, axis=1)]
    if len(useful) == 0:
        return 0.0
    coordinates = sorted({float(value) for value in useful[:, -1]}, reverse=True)
    boundaries = coordinates + [float(ref[-1])]
    total = 0.0
    for index in range(len(boundaries) - 1):
        high, low = boundaries[index], boundaries[index + 1]
        if low >= high:
            continue
        active = useful[useful[:, -1] >= high]
        if len(active) == 0:
            continue
        total += (high - low) * hypervolume_nd(active[:, :-1], ref[:-1])
    return float(total)


def hypervolume_contribution(
    point: Sequence[float], front: np.ndarray, reference: Sequence[float]
) -> float:
    """Exclusive hypervolume contribution of ``point`` w.r.t. ``front``.

    Cheaper than ``hypervolume_nd(front ∪ {point}) - hypervolume_nd(front)``:
    the point's box is first reduced by clipping every front member into it, so
    the recursive HV only sees points that actually overlap that box.
    """
    point_array = np.asarray(point, dtype=float)
    ref = np.asarray(reference, dtype=float)
    if point_array.shape[0] != ref.shape[0]:
        return 0.0
    if not bool(np.all(point_array > ref)):
        return 0.0
    box = float(np.prod(point_array - ref))
    candidates = np.asarray(front, dtype=float)
    if candidates.size == 0:
        return box
    if candidates.ndim == 1:
        candidates = candidates.reshape(1, -1)
    keep = np.all(candidates > ref, axis=1)
    if not bool(np.any(keep)):
        return box
    candidates = candidates[keep]
    if bool(np.any(np.all(candidates >= point_array, axis=1))):
        return 0.0  # point is dominated by an existing front member
    clipped = np.minimum(candidates, point_array)
    clipped = clipped[np.all(clipped > ref, axis=1)]
    if len(clipped):
        clipped = clipped[non_dominated_mask(clipped)]
        covered = hypervolume_nd(clipped, ref)
    else:
        covered = 0.0
    return float(max(0.0, box - covered))


def hypervolume_ratio(
    candidate_scores: np.ndarray,
    reference_scores: np.ndarray,
    reference_point: Sequence[float],
) -> float | None:
    """HV(candidate) / HV(reference front), ``None`` when the reference HV is 0."""
    ref_hv = hypervolume_2d(reference_scores, reference_point)
    if ref_hv <= 0:
        return None
    cand_hv = hypervolume_2d(candidate_scores, reference_point)
    return float(cand_hv / ref_hv)
