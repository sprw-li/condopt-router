"""Multi-objective multi-fidelity Bayesian optimization (momfbo).

Combines ParEGO-style random scalarization with the multi-fidelity GP of
``mfbo``: every proposal draws a weight vector on the simplex, scalarizes the
observed objective vectors (augmented Tchebycheff), fits a fidelity-aware GP on
that scalar, and maximizes ``acquisition / cost(s)**cost_exponent``.  Hard
constraints reuse the per-constraint feasibility probability.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import norm

from ..base import Optimizer, ParamSpec
from ..dataset import Observation
from ..pareto import observation_values
from ..registry import register
from .bayes import _log_ei
from .mfbo import MultiFidelityBO

_EPS = 1e-12


@register
class MultiObjectiveMultiFidelityBO(MultiFidelityBO):
    """Random-scalarization multi-objective BO over a fidelity channel."""

    supports_fidelity = True

    spec = replace(
        MultiFidelityBO.spec,
        name="momfbo",
        summary=(
            "多目标 + 多保真：随机加权（增广 Tchebycheff）标量化后就保真度维做 GP+EI/成本，"
            "并支持硬约束 P(feasible)。"
        ),
        when="有低成本粗筛通道，同时需要 2+ 目标折中解（Pareto）时。",
        params=MultiFidelityBO.spec.params + (
            ParamSpec("rho", "float", 0.05, low=0.0, high=1.0,
                      description="增广 Tchebycheff 的增广系数"),
            ParamSpec("weight_alpha", "float", 1.0, low=0.01, high=10.0,
                      description="随机权重 Dirichlet 的集中度"),
            ParamSpec(
                "noise_aware",
                "bool",
                True,
                description="把 objective_std / n_replicates 估计的逐点噪声传入 GP alpha",
            ),
        ),
        capabilities=frozenset(
            set(MultiFidelityBO.spec.capabilities) | {"multi_objective"}
        ),
        references=(
            "Knowles, ParEGO, IEEE TEC 2006",
            "Kandasamy et al., BOCA, ICML 2017",
        ),
    )

    def __init__(self, *args, **kwargs) -> None:
        # The base constructor replays ``prior`` through ``tell``, so the data
        # buffers must exist *before* calling it (a resumed study passes prior and
        # used to raise AttributeError here).
        objective = kwargs.get("objective")
        n_objectives = int(getattr(objective, "n_objectives", 1) or 1)
        self._Y = np.empty((0, n_objectives))
        self._Ystd = np.empty((0, n_objectives))
        super().__init__(*args, **kwargs)
        if len(self._Y) == 0:
            self._Y = np.empty((0, self.objective.n_objectives))
            self._Ystd = np.empty((0, self.objective.n_objectives))

    # -- data -------------------------------------------------------------
    def tell(self, observations: Sequence[Observation]) -> None:
        observations = list(observations)
        rows = []
        for observation in observations:
            try:
                rows.append(observation_values(observation, self.objective))
            except ValueError:
                rows.append(np.full(self.objective.n_objectives, np.nan))
        stds = np.asarray(
            [
                [float(obs.objective_std) if obs.objective_std else 0.0
                 for _ in self.objective.objectives]
                for obs in observations
            ],
            dtype=float,
        )
        super().tell(observations)
        if not observations:
            return
        matrix = np.vstack(rows)
        if self._Y is None or len(self._Y) == 0:
            self._Y = matrix
            self._Ystd = stds
        else:
            self._Y = np.vstack([self._Y, matrix])
            self._Ystd = np.vstack([self._Ystd, stds])

    def _scalar_matrix(self, weights: np.ndarray) -> np.ndarray:
        """Augmented Tchebycheff scalarization for every complete row."""
        m = self.objective.n_objectives
        rows = self._Y
        if len(rows) == 0:
            return np.zeros(0)
        finite = np.all(np.isfinite(rows), axis=1)
        values = np.full(len(rows), np.nan)
        if not np.any(finite):
            return values
        usable = rows[finite]
        lows = np.min(usable, axis=0)
        highs = np.max(usable, axis=0)
        spans = np.maximum(highs - lows, _EPS)
        normalized = np.zeros_like(usable)
        for axis in range(m):
            if self.objective.directions[axis] == "max":
                normalized[:, axis] = (usable[:, axis] - lows[axis]) / spans[axis]
            else:
                normalized[:, axis] = (highs[axis] - usable[:, axis]) / spans[axis]
        weighted = normalized * weights
        values[finite] = np.max(weighted, axis=1) + float(self.params["rho"]) * np.sum(
            weighted, axis=1
        )
        return values

    def _scalar_std(self, weights: np.ndarray) -> np.ndarray:
        """Conservative scalarized noise (same normalization as the scalar)."""
        if len(self._Ystd) == 0:
            return np.zeros(0)
        rows = self._Y
        finite = np.all(np.isfinite(rows), axis=1)
        out = np.full(len(rows), np.nan)
        if not np.any(finite):
            return out
        usable = rows[finite]
        lows = np.min(usable, axis=0)
        highs = np.max(usable, axis=0)
        spans = np.maximum(highs - lows, _EPS)
        coefficients = (
            np.asarray(weights, dtype=float) * (1.0 + float(self.params["rho"])) / spans
        )
        std_finite = np.all(np.isfinite(self._Ystd[finite]), axis=1)
        values = np.full(int(np.sum(finite)), np.nan)
        values[std_finite] = self._Ystd[finite][std_finite] @ coefficients
        out[finite] = values
        return out

    def _best_scalar(self, weights: np.ndarray | None = None) -> float:
        """Best scalarised value *under the same weights as the fitted model*.

        The GP is fitted on ``_scalar_matrix(weights)`` with the round's Dirichlet
        weights; scoring the incumbent with a *uniform* vector instead makes
        ``imp = mu - best`` inconsistent (measured: model max 0.9421 vs incumbent
        0.525), which inflates the apparent improvement.  ``parego``/``mfparego``
        already keep the two consistent.
        """
        if weights is None:
            weights = np.full(self.objective.n_objectives, 1.0 / self.objective.n_objectives)
        weights = np.asarray(weights, dtype=float)
        scalar = self._scalar_matrix(weights)
        if len(scalar) == 0:
            return 0.0
        full = self._fidelity >= 1.0 - 1e-9
        mask = np.isfinite(scalar) & (full if bool(np.any(full)) else True)
        if not bool(np.any(mask)):
            mask = np.isfinite(scalar)
        if not bool(np.any(mask)):
            return 0.0
        return float(np.max(scalar[mask]))

    # -- propose ----------------------------------------------------------
    def _propose(
        self,
        batch: Sequence[tuple[dict, float]],
        force_full: bool = False,
        min_feasibility: float = 0.0,
    ) -> tuple[dict, float]:
        levels = [1.0] if force_full else self._levels()
        # (constraint prefilter is handled by the shared feasibility-aware logic
        # in the parent class; momfbo keeps its own candidate scoring below)
        if self._X is None or len(self._X) < 3:
            config = self.space.sample_uniform(1, self.rng)[0]
            return config, float(self.rng.choice(levels))
        candidates = self._candidate_pool()
        if not candidates:
            return self.space.sample_uniform(1, self.rng)[0], 1.0
        weights = self.rng.dirichlet(
            np.full(self.objective.n_objectives, float(self.params["weight_alpha"]))
        )
        scalar = self._scalar_matrix(weights)
        usable = np.isfinite(scalar)
        if int(np.sum(usable)) < 3:
            config = self.space.sample_uniform(1, self.rng)[0]
            return config, float(self.rng.choice(levels))
        scalar_std = self._scalar_std(weights)
        backup_noise = getattr(self, "_noise_var", None)
        if (
            bool(self.params["noise_aware"])
            and len(scalar_std) == len(scalar)
            and np.any(np.isfinite(scalar_std[usable]))
        ):
            self._noise_var = np.where(np.isfinite(scalar_std[usable]), scalar_std[usable], np.nan)
        model = self._fit(self._X[usable], scalar[usable])
        self._noise_var = backup_noise
        best = self._best_scalar(weights)
        features = np.vstack(
            [self._features(config, level) for config in candidates for level in levels]
        )
        mu, std = model.predict(features, return_std=True)
        mu = np.asarray(mu, dtype=float)
        std = np.maximum(np.asarray(std, dtype=float), 1e-12)
        mode = str(self.params["acquisition"])
        if mode == "ucb":
            acquisition = mu + float(self.params["kappa"]) * std
        elif mode == "logei":
            imp = mu - best
            acquisition = _log_ei(imp, std, imp / std)
        else:
            imp = mu - best
            acquisition = imp * norm.cdf(imp / std) + std * norm.pdf(imp / std)
        acquisition = acquisition * self._feasibility_probability(features)
        costs = np.asarray(
            [self._cost(level) for _ in candidates for level in levels], dtype=float
        )
        scores = acquisition / np.maximum(costs, 1e-6)
        pair_index = int(np.argmax(scores))
        candidate_index = pair_index // len(levels)
        level_index = pair_index % len(levels)
        config = candidates[candidate_index]
        chosen_units = [self.space.encode(item) for item, _ in batch]
        if chosen_units:
            scale = 0.15
            for _ in range(5):
                unit = self.space.encode(config)
                distances = np.linalg.norm(np.asarray(chosen_units) - unit, axis=1)
                if float(np.min(distances)) > scale:
                    break
                candidate_index = (candidate_index + 1) % len(candidates)
                config = candidates[candidate_index]
        return config, float(levels[level_index])

    # -- state ------------------------------------------------------------
    def state_dict(self) -> dict:
        state = super().state_dict()
        # ``Y`` is data, not a counter: on resume it is rebuilt from the study
        # dataset (``prior``).  Persisting it would let a state written earlier
        # shadow the newer, longer matrix (the bug fixed in mfbo).
        return state

    def load_state_dict(self, state: Mapping) -> None:
        super().load_state_dict(state)
        if getattr(self, "_Y", None) is not None and len(self._Y):
            return  # already rebuilt from the dataset
        matrix = state.get("Y")
        self._Y = (
            np.empty((0, self.objective.n_objectives))
            if matrix is None
            else np.asarray(matrix, dtype=float)
        )