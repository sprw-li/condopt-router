"""Monte-Carlo EHVI for 3-4 objectives (``qehvi_nd``).

For each candidate the posterior of every objective is sampled independently
(``mc_draws`` draws); the improvement is the hypervolume gain of the observed
front after adding the sample, averaged over draws and multiplied by the
probability of satisfying all hard constraints.  Hypervolume is exact (up to
front size) through :func:`condopt.pareto.hypervolume_nd`, so this replaces the
2-objective partitioning used by ``qehvi`` with a dimension-agnostic MC form.

It is intentionally conservative: independent per-objective GPs (no cross-
covariance) and a greedy batch rule with a local penalty.  Use it when 3-4
objectives are optimized under a moderate budget and noise; ``parego`` remains
cheaper and more robust for very small budgets.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor

from ..base import AlgorithmSpec, Optimizer, ParamSpec
from ..dataset import Observation
from ..pareto import (
    hypervolume_contribution,
    hypervolume_nd,
    non_dominated_mask,
    observation_values,
)
from ..registry import register
from .bayes import _ard_initial_lengthscale, _make_kernel

_EPS = 1e-12


@register
class MultiObjectiveEHVI(Optimizer):
    """Dimension-agnostic MC expected hypervolume improvement."""

    spec = AlgorithmSpec(
        name="qehvi_nd",
        family="model_based",
        summary="3–4 目标 MC-EHVI：逐目标 GP 后验采样，用递归切片 HV 估计期望超体积改进（P(feasible) 过滤）。",
        when="3–4 个目标、预算 >=40、噪声低到中等时；预算很小或目标 >4 时用 parego。",
        params=(
            ParamSpec("kernel", "choice", "matern52", choices=("matern52", "matern32", "rbf")),
            ParamSpec("length_scale", "float", 0.3, low=0.01, high=10.0),
            ParamSpec("ard", "bool", True),
            ParamSpec("alpha", "float", 1e-4, low=1e-12, high=1.0),
            ParamSpec("n_initial", "int", 0, low=0, high=200),
            ParamSpec("n_candidates", "int", 256, low=64, high=4096),
            ParamSpec("mc_draws", "int", 32, low=4, high=256,
                      description="每个候选的后验采样数"),
            ParamSpec("mc_prefilter", "int", 64, low=0, high=2048,
                      description="MC 前用廉价 PI 预筛的候选数（0 = 不筛）"),
            ParamSpec("local_frac", "float", 0.3, low=0.0, high=0.9),
            ParamSpec("local_sigma", "float", 0.15, low=0.01, high=1.0),
        ),
        capabilities=frozenset(
            {"batch", "mixed", "categorical", "continuous", "noisy",
             "model_based", "multi_objective", "constraints"}
        ),
        references=(
            "Daulton, Balandat & Bakshy, Differentiable Expected Hypervolume Improvement for Parallel Multi-Objective Bayesian Optimization, NeurIPS 2020",
            "Emmerich & Fonseca, Computing Hypervolume Contributions in Low Dimensions, 2011",
        ),
    )

    def __init__(self, *args, **kwargs) -> None:
        self._X: np.ndarray | None = None
        self._Y: np.ndarray | None = None
        self._C: np.ndarray | None = None
        self._noise: np.ndarray | None = None
        self._units: np.ndarray | None = None
        super().__init__(*args, **kwargs)

    # -- data -------------------------------------------------------------
    def tell(self, observations: Sequence[Observation]) -> None:
        observations = list(observations)
        super().tell(observations)
        if not observations:
            return
        features = self.space.features_many([obs.x for obs in observations])
        units = np.vstack([self.space.encode(obs.x) for obs in observations])
        rows, constraints, noise = [], [], []
        for observation in observations:
            try:
                rows.append(observation_values(observation, self.objective))
            except ValueError:
                rows.append(np.full(self.objective.n_objectives, np.nan))
            constraints.append([
                float(observation.constraints.get(name, np.nan))
                if observation.constraints else np.nan
                for name in self.objective.constraints
            ])
            variance = (
                (observation.objective_std ** 2) / observation.n_replicates
                if observation.objective_std > 0 else np.nan
            )
            noise.append(variance)
        matrix = np.vstack(rows)
        c_matrix = np.asarray(constraints, dtype=float)
        noise_var = np.asarray(noise, dtype=float)
        if self._X is None:
            self._X, self._Y, self._C = features, matrix, c_matrix
            self._noise, self._units = noise_var, units
        else:
            self._X = np.vstack([self._X, features])
            self._Y = np.vstack([self._Y, matrix])
            self._C = np.vstack([self._C, c_matrix])
            self._noise = np.concatenate([self._noise, noise_var])
            self._units = np.vstack([self._units, units])

    # -- GP helpers --------------------------------------------------------
    def _model_feature_dim(self) -> int:
        """GP input width (subclasses append fidelity as an extra feature)."""
        return int(self.space.feature_dim)

    def _fit(self, y: np.ndarray) -> GaussianProcessRegressor:
        finite = np.isfinite(y)
        X, target = self._X[finite], y[finite]
        noise = None if self._noise is None else self._noise[finite]
        base = float(self.params["alpha"])
        if noise is not None and np.any(np.isfinite(noise) & (noise > 0)):
            y_var = max(float(np.var(target)) if len(target) > 1 else 1.0, _EPS)
            scaled = np.where(np.isfinite(noise) & (noise > 0), noise / y_var, base)
            alpha = np.clip(scaled, 1e-12, 0.5)
        else:
            alpha = base
        kernel = _make_kernel(
            str(self.params["kernel"]),
            float(self.params["length_scale"]),
            feature_dim=self._model_feature_dim(),
            ard=bool(self.params["ard"]),
        )
        model = GaussianProcessRegressor(
            kernel=kernel, alpha=alpha, normalize_y=True, optimizer=None,
            n_restarts_optimizer=0, random_state=self.seed,
        )
        return model.fit(X, target)

    def _signed_reference(self) -> np.ndarray:
        signs = np.asarray(
            [1.0 if direction == "max" else -1.0 for direction in self.objective.directions]
        )
        if self.objective.ref_point:
            return np.asarray(self.objective.ref_point, dtype=float) * signs
        signed = self._signed_observations()
        lows = np.min(signed, axis=0)
        highs = np.max(signed, axis=0)
        span = np.maximum(highs - lows, 1e-9)
        return lows - 0.05 * span

    def _signed_observations(self) -> np.ndarray:
        assert self._Y is not None
        usable = self._Y[np.all(np.isfinite(self._Y), axis=1)]
        if len(usable) == 0:
            return np.empty((0, self.objective.n_objectives))
        signs = np.asarray(
            [1.0 if direction == "max" else -1.0 for direction in self.objective.directions]
        )
        return usable * signs

    def _feasibility(self, features: np.ndarray) -> np.ndarray:
        if self._C is None or self._C.shape[1] == 0:
            return np.ones(len(features))
        probability = np.ones(len(features))
        limits = self.objective.constraint_limits or {}
        for index, name in enumerate(self.objective.constraints):
            values = self._C[:, index]
            finite = np.isfinite(values)
            if int(np.sum(finite)) < 2:
                continue
            model = self._fit(values)
            mu, std = model.predict(features, return_std=True)
            std = np.maximum(np.asarray(std, dtype=float), 1e-9)
            limit = limits.get(name) or {}
            threshold = float(limit.get("value", 0.0))
            op = str(limit.get("op", "<="))
            z = (threshold - mu) / std if op == "<=" else (mu - threshold) / std
            probability *= norm.cdf(z)
        return probability

    # -- propose -----------------------------------------------------------
    def _initial_design_left(self) -> int:
        n_initial = int(self.params["n_initial"])
        if n_initial <= 0:
            n_initial = max(2 * (self.space.feature_dim + 1), 2 * self.objective.n_objectives)
            n_initial = min(n_initial, 20)
        return max(0, n_initial - int(self.t))

    def ask(self, n: int = 1) -> list[dict]:
        if n <= 0:
            return []
        out: list[dict] = []
        design_left = self._initial_design_left()
        if design_left > 0:
            k = min(n, design_left)
            out.extend(self.space.sample_sobol(k, rng=self.rng))
            n -= k
        if n > 0 and (self._X is None or len(self._Y) < 3):
            out.extend(self.space.sample_uniform(n, self.rng))
            return out
        proposals: list[dict] = []
        for _ in range(n):
            proposals.append(self._propose(out + proposals))
        return out + proposals

    def _candidate_pool(self) -> list[dict]:
        count = int(self.params["n_candidates"])
        local_frac = float(self.params["local_frac"])
        rng = self.rng
        pool = self.space.sample_uniform(max(1, int(count * (1 - local_frac))), rng)
        if local_frac > 0 and self._units is not None and len(self._units):
            best_index = int(np.argmax(self._scalar_incumbent()))
            center = self._units[best_index]
            sigma = float(self.params["local_sigma"])
            local = self.space.decode_many(
                np.clip(center + sigma * rng.standard_normal(
                    (max(1, int(count * local_frac)), self.space.dim)
                ), 0.0, 1.0)
            )
            pool.extend(local)
        return pool

    def _scalar_incumbent(self) -> np.ndarray:
        signed = self._signed_observations()
        if len(signed) == 0:
            return np.zeros(0)
        spans = np.maximum(signed.max(axis=0) - signed.min(axis=0), _EPS)
        return np.sum((signed - signed.min(axis=0)) / spans, axis=1)

    def _propose(self, batch: Sequence[dict]) -> dict:
        if self._X is None or len(self._Y) < 3:
            return self.space.sample_uniform(1, self.rng)[0]
        signed = self._signed_observations()
        reference = self._signed_reference()
        mask = non_dominated_mask(signed)
        front = signed[mask]
        candidates = self._candidate_pool()
        features = self.space.features_many(candidates)
        rng = np.random.default_rng(self.seed + 7919 * (int(self.t) + len(batch) + 1))
        acquisition, keep = self._ehvi_acquisition(features, front, reference, rng)
        candidates = [candidates[index] for index in keep]
        # local penalty against conditions already chosen in this batch
        if batch:
            chosen = np.vstack([self.space.encode(item) for item in batch])
            units = np.vstack([self.space.encode(item) for item in candidates])
            for index in range(len(candidates)):
                distance = float(np.min(np.linalg.norm(chosen - units[index], axis=1)))
                acquisition[index] *= 1.0 - np.exp(-distance / 0.1)
        return candidates[int(np.argmax(acquisition))]

    def _ehvi_acquisition(
        self,
        features: np.ndarray,
        front: np.ndarray,
        reference: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Monte-Carlo expected hypervolume improvement for prebuilt features.

        Shared by the fidelity-agnostic path and subclasses that append
        additional GP inputs (e.g. multi-fidelity ``mqehvi_nd``).
        """
        models = [
            self._fit(self._Y[:, index]) for index in range(self.objective.n_objectives)
        ]
        means, stds = [], []
        for model in models:
            mu, std = model.predict(features, return_std=True)
            means.append(np.asarray(mu, dtype=float))
            stds.append(np.maximum(np.asarray(std, dtype=float), 1e-9))
        means = np.vstack(means).T
        stds = np.vstack(stds).T
        feasible = self._feasibility(features)
        draws = int(self.params["mc_draws"])
        prefilter = int(self.params["mc_prefilter"])
        if 0 < prefilter < len(features):
            # Cheap product-of-PI screen against the per-axis front extents:
            # keeps the expensive HV loop on the most promising candidates.
            front_best = front.max(axis=0) if len(front) else reference
            z = (means - front_best) / stds
            screen = np.prod(norm.cdf(z), axis=1) * feasible
            keep = np.argsort(screen)[::-1][:prefilter]
        else:
            keep = np.arange(len(features))
        means, stds, feasible = means[keep], stds[keep], feasible[keep]
        acquisition = np.zeros(len(feasible))
        front_array = front if len(front) else np.empty((0, self.objective.n_objectives))
        for _ in range(draws):
            sample = means + stds * rng.standard_normal(means.shape)
            for index in range(len(feasible)):
                improved = hypervolume_contribution(
                    sample[index], front_array, reference
                )
                if improved > 0:
                    acquisition[index] += improved
        return acquisition / draws * feasible, keep

    # -- state -------------------------------------------------------------
    def state_dict(self) -> dict:
        # The observation matrices are *data*: on resume the instance replays the
        # study dataset (``prior``).  Persisting them let an older, shorter state
        # shadow the rebuilt matrices (same bug class as mfbo/momfbo).
        return super().state_dict()

    def load_state_dict(self, state: Mapping) -> None:
        super().load_state_dict(state)
        if getattr(self, "_X", None) is not None and np.asarray(self._X).size:
            return  # already rebuilt from the dataset
        for key, attribute in (
            ("X", "_X"), ("Y", "_Y"), ("C", "_C"), ("noise", "_noise"), ("units", "_units")
        ):
            value = state.get(key)
            if value is not None:
                setattr(self, attribute, np.asarray(value, dtype=float))