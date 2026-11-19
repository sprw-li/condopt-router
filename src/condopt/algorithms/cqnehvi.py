"""Noisy 2-objective BO with hard constraints (``cqnehvi``).

Combines the nested-MC noisy EHVI of ``qnehvi`` with the feasibility
probability of ``cqehvi``: constrained candidates are scored by
``E[DHV] · P(feasible)`` where the baseline front is resampled from the
observation-noise posterior.  With no feasible observation yet it falls back to
pure feasibility search (phase A), exactly like ``cqehvi``.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..pareto import dhv_2d_batch, non_dominated_mask, partition_2d, signed_scores
from ..registry import register
from .cqehvi import ConstrainedQEHVI
from .qnehvi import QNEHVI

_EPS = 1e-12


@register
class ConstrainedQNEHVI(ConstrainedQEHVI):
    """Noisy constrained EHVI (2 objectives + inequality constraints)."""

    spec = replace(
        ConstrainedQEHVI.spec,
        name="cqnehvi",
        params=ConstrainedQEHVI.spec.params
        + tuple(
            parameter
            for parameter in QNEHVI.spec.params
            if parameter.name in ("n_base", "n_y", "noise_std")
        ),
        summary=(
            "2 目标 + 硬约束 + 噪声：嵌套 MC 估计带噪 EHVI，再乘逐候选 P(feasible)；"
            "无可行点时先做可行性搜索。"
        ),
        when="2 个目标、硬约束、且观测噪声中高（复孔/测量波动明显）时。",
        capabilities=ConstrainedQEHVI.spec.capabilities | {"noisy"},
        references=(
            "Daulton, Balandat & Bakshy, Parallel Bayesian Optimization of Multiple Noisy Objectives with qNEHVI, NeurIPS 2021",
            "Gardner et al., Bayesian Optimization with Inequality Constraints, ICML 2014",
        ),
    )

    def _constraint_probability(self, features: np.ndarray) -> np.ndarray:
        draws = int(self.params["mc_draws"])
        tolerance = float(self.params["constraint_margin"])
        probability = np.ones(len(features))
        for index, name in enumerate(self._constraint_names):
            column = self._C[:, index]
            usable = np.isfinite(column)
            if not np.any(usable):
                raise ValueError(f"cqnehvi: constraint {name!r} has no measured value")
            model = self._fit(self._X[usable], column[usable])
            constraint_draws = self._sample_posterior_multi(model, features, draws).T
            limit = self.objective.constraint_limits[name]
            threshold = float(limit.get("value", 0.0))
            if str(limit.get("op", "<=")) == "<=":
                probability *= (constraint_draws <= threshold - tolerance).mean(axis=0)
            else:
                probability *= (constraint_draws >= threshold + tolerance).mean(axis=0)
        return probability

    def _model_ask(self, n: int) -> list[dict]:
        if self._X is None or len(self._y) < 3:
            return self.space.sample_sobol(n, rng=self.rng)
        candidates, features = self._candidate_pool()
        if len(candidates) == 0:
            return self.space.sample_sobol(n, rng=self.rng)

        C = self._C
        complete = np.all(np.isfinite(self._Y), axis=1)
        feasible_obs = complete & self._feasibility(C)
        probability = self._constraint_probability(features)

        if not bool(np.any(feasible_obs)):
            acquisition = probability
        else:
            ref = self._signed_references()
            X = self._X[feasible_obs]
            Y = self._Y[feasible_obs]
            configured = float(self.params.get("noise_std", 0.0) or 0.0)
            if configured > 0:
                noise = np.full(len(X), configured)
            else:
                variance = getattr(self, "_noise_var", None)
                if variance is None or len(variance) != len(self._Y):
                    noise = np.zeros(len(X))
                else:
                    noise = np.sqrt(np.nan_to_num(variance[feasible_obs], nan=0.0))
            noise = np.asarray(noise, dtype=float)
            if noise.ndim == 1:
                noise = noise[:, None]  # per-observation std -> per-objective column
            signed = signed_scores(Y, self.objective)
            if not np.any(np.all(signed > ref, axis=1)):
                acquisition = probability
            else:
                n_base = int(self.params["n_base"])
                n_y = int(self.params["n_y"])
                ensemble = []
                for index in range(2):
                    model = self._fit(X, Y[:, index])
                    draw = self._sample_posterior_multi(model, features, n_y)
                    ensemble.append(draw.T)
                points = np.stack(ensemble, axis=-1)  # (n_y, m, 2)
                acquisition = np.zeros(len(candidates))
                effective = 0
                for _ in range(n_base):
                    drawn = signed + noise * self.rng.standard_normal(signed.shape)
                    mask = non_dominated_mask(drawn)
                    boxes = partition_2d(drawn[mask], ref)
                    if not boxes:
                        continue
                    acquisition += dhv_2d_batch(points, boxes, ref).mean(axis=0)
                    effective += 1
                if effective == 0:
                    acquisition = probability
                else:
                    acquisition = acquisition / effective * probability

        acquisition = np.asarray(acquisition, dtype=float)
        threshold = float(self.params["feasibility_threshold"])
        masked = np.where(probability >= threshold, acquisition, -np.inf)
        if np.any(np.isfinite(masked)):
            acquisition = masked
        else:
            acquisition = probability

        if bool(np.any(complete)):
            scale = self._local_penalty_scale(
                self._fit(self._X[complete], self._Y[complete][:, 0])
            )
        else:
            scale = self._local_penalty_scale(self._fit(self._X, self._y))
        return self._greedy_configs(acquisition, candidates, scale, n)
