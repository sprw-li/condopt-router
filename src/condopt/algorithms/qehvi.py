"""2D Expected Hypervolume Improvement (qEHVI, single-candidate MC).

For two objectives the dominance hypervolume improvement of a candidate has
an exact partitioned form (Daulton et al., qEHVI, NeurIPS 2020):

    DHV(y) = prod_j(y_j - r_j) - sum_boxes prod_j max(0, min(y_j, u_j) - l_j)

where the boxes partition the already-dominated region.  EHVI is the
expectation of ``max(0, DHV(f(x)))`` over the independent-GP posteriors of the
two objectives; we estimate it with Monte Carlo draws.  Batches are filled
greedily with a local-penalization factor (no fake observations).

This is a deliberately simple 2-objective implementation: 3+ objectives need a
recursive partitioning, which is on the roadmap.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from ..base import AlgorithmSpec, ParamSpec
from ..dataset import Observation
from ..pareto import dhv_2d_batch, non_dominated_mask, partition_2d, signed_scores
from ..registry import register
from .bayes import GaussianProcessBO

_EPS = 1e-12


@register
class QEHVI(GaussianProcessBO):
    """Exact 2D EHVI surrogate (independent GPs, MC expectation)."""

    spec = AlgorithmSpec(
        name="qehvi",
        family="model_based",
        summary="2 目标 qEHVI：用分区公式精确算 EHVI，MC 估计超体积改进，直接优化 Pareto 前沿覆盖。",
        when="恰好 2 个目标、预算 ≥40、噪声低/无；比 ParEGO 的前沿覆盖与 HV 更好时。",
        params=GaussianProcessBO.spec.params
        + (
            ParamSpec(
                "mc_draws",
                "int",
                128,
                low=16,
                high=512,
                description="EHVI 的 Monte Carlo 采样数",
            ),
        ),
        capabilities=GaussianProcessBO.spec.capabilities | {"multi_objective"},
        references=(
            "Daulton, Balandat & Bakshy, Differentiable Expected Hypervolume Improvement for Parallel Multi-Objective Bayesian Optimization (qEHVI), NeurIPS 2020",
            "Daulton, Balandat & Bakshy, Parallel Bayesian Optimization of Multiple Noisy Objectives with qNEHVI, NeurIPS 2021",
        ),
    )

    def __init__(self, *args, **kwargs) -> None:
        self._xs: list[dict] = []
        self._Y = np.empty((0, 0))
        self._ref: np.ndarray | None = None
        self._skipped = 0
        super().__init__(*args, **kwargs)
        if self.objective.n_objectives != 2:
            raise ValueError("qehvi currently supports exactly 2 objectives")

    # -- data -------------------------------------------------------------
    def tell(self, observations: Sequence[Observation]) -> None:
        observations = list(observations)
        if not observations:
            return
        rows: list[np.ndarray] = []
        for observation in observations:
            values = self._objective_values(observation)
            rows.append(values if values is not None else np.full(2, np.nan))
        # Always tell the parent: ``t`` / candidate-pool data must stay in sync
        # with the study dataset even when an objective value is missing.
        super().tell(observations)
        matrix = np.vstack(rows)
        self._Y = matrix if len(self._Y) == 0 else np.vstack([self._Y, matrix])
        self._xs.extend(dict(obs.x) for obs in observations)
        self._skipped = int(np.sum(~np.all(np.isfinite(self._Y), axis=1)))

    def _objective_values(self, observation: Observation) -> np.ndarray | None:
        names = self.objective.objectives
        if len(names) != 2:
            raise ValueError("qehvi currently supports exactly 2 objectives")
        values = []
        for name in names:
            if name in observation.objectives:
                values.append(float(observation.objectives[name]))
            elif observation.y is not None and (name == self.objective.name or len(names) == 1):
                values.append(float(observation.y))
            else:
                return None
        return np.asarray(values, dtype=float)

    # -- shared helpers (also used by the constrained variant) -------------
    def _complete_rows(self) -> tuple[np.ndarray, np.ndarray]:
        """X/Y rows with a complete objective vector (no NaN)."""
        valid = np.all(np.isfinite(self._Y), axis=1)
        if not np.any(valid):
            raise ValueError(
                "qEHVI: no observation has a complete 2-objective vector; "
                "check the results file / objective names"
            )
        return self._X[valid], self._Y[valid]

    def _front_boxes(self, signed: np.ndarray, ref: np.ndarray, label: str = "qEHVI") -> list:
        """Disjoint 2D partition of the dominated region of one front."""
        mask = non_dominated_mask(signed)
        boxes = partition_2d(signed[mask], ref)
        if not boxes:
            raise ValueError(
                f"{label}: no observation dominates objective.ref_point="
                f"{self.objective.ref_point}; lower the reference point or "
                "provide observations above it"
            )
        return boxes

    def _posterior_draws(
        self, X: np.ndarray, Y: np.ndarray, features: np.ndarray, draws: int
    ) -> tuple[np.ndarray, list]:
        """Joint MC draws of the two objective GPs: ``(draws, n_candidates, 2)``."""
        ensemble = []
        models = []
        for index in range(2):
            model = self._fit(X, Y[:, index])
            models.append(model)
            draw = self._sample_posterior_multi(model, features, draws)  # (m, S)
            ensemble.append(draw.T)
        return np.stack(ensemble, axis=-1), models

    def _greedy_configs(
        self, acquisition: np.ndarray, candidates: np.ndarray, scale: float, n: int
    ) -> list[dict]:
        """Greedy batch with a local-penalization factor (no fake observations)."""
        chosen: list[int] = []
        chosen_units: list[np.ndarray] = []
        available = np.ones(len(candidates), dtype=bool)
        for _ in range(n):
            if not available.any():
                break
            scores = np.asarray(acquisition, dtype=float).copy()
            if chosen_units:
                distances = np.sum(
                    (candidates[:, None, :] - np.asarray(chosen_units)[None, :, :]) ** 2, axis=-1
                )
                penalty = np.prod(1.0 - np.exp(-distances / (2.0 * scale * scale)), axis=1)
                scores = scores * penalty
            scores = np.where(available, scores, -np.inf)
            pick = int(np.argmax(scores))
            chosen.append(pick)
            chosen_units.append(candidates[pick])
            available[pick] = False
        configs = [self.space.decode(candidates[i]) for i in chosen]
        while len(configs) < n:
            configs.extend(self.space.sample_sobol(n - len(configs), rng=self.rng))
        return configs

    def _signed_references(self) -> np.ndarray:
        if self.objective.ref_point:
            raw = np.asarray(self.objective.ref_point, dtype=float)
            signs = np.asarray([1.0 if d == "max" else -1.0 for d in self.objective.directions])
            return raw * signs
        if self._ref is None:
            # Fallback: freeze a data-derived reference from COMPLETE rows only
            # (NaN rows from missing objectives must not poison the min/max).
            valid = np.all(np.isfinite(self._Y), axis=1)
            rows = self._Y[valid]
            if len(rows) == 0:
                raise ValueError("qEHVI: no complete 2-objective observation for a data-derived ref")
            signed = signed_scores(rows, self.objective)
            lows = np.min(signed, axis=0)
            highs = np.max(signed, axis=0)
            span = np.maximum(highs - lows, 1e-9)
            self._ref = lows - 0.05 * span
        return self._ref

    # -- propose ----------------------------------------------------------
    def _model_ask(self, n: int) -> list[dict]:
        if self._X is None or len(self._y) < 3:
            return self.space.sample_sobol(n, rng=self.rng)
        candidates, features = self._candidate_pool()
        if len(candidates) == 0:
            return self.space.sample_sobol(n, rng=self.rng)

        draws = int(self.params["mc_draws"])
        ref = self._signed_references()
        X, Y = self._complete_rows()
        signed = signed_scores(Y, self.objective)
        boxes = self._front_boxes(signed, ref)

        # One independent-GP MC ensemble: shapes (n_draws, n_candidates, 2)
        points, models = self._posterior_draws(X, Y, features, draws)
        improvement = dhv_2d_batch(points, boxes, ref)  # (S, m)
        acquisition = improvement.mean(axis=0)  # (m,)

        # Reuse the first objective GP for the local-penalization scale instead
        # of fitting a third model (review v0.5 P1).
        scale = self._local_penalty_scale(models[0])
        return self._greedy_configs(acquisition, candidates, scale, n)

    # -- state ------------------------------------------------------------
    def state_dict(self) -> dict:
        state = super().state_dict()
        state["ref"] = None if self._ref is None else self._ref.tolist()
        return state

    def load_state_dict(self, state: Mapping) -> None:
        super().load_state_dict(state)
        if getattr(self, "_ref", None) is not None:
            return  # a reference point was already derived from the dataset
        ref = state.get("ref")
        self._ref = None if ref is None else np.asarray(ref, dtype=float)
