"""ParEGO: multi-objective Bayesian optimization via random scalarization.

ParEGO turns a multi-objective problem into a sequence of single-objective
ones: every proposal draws a weight vector ``lambda`` on the simplex,
scalarizes *all* observed objective vectors with an augmented Tchebycheff
function, fits the ordinary GP + EI machinery on that scalar, and proposes
the next point.  Repeating with fresh weights rotates the search direction,
which recovers non-convex Pareto fronts that a plain weighted sum misses.

References: Knowles, "ParEGO: A Hybrid Algorithm with On-Line Landscape
Approximation for Expensive Multiobjective Optimization Problems",
IEEE TEC 2006 [R11].
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

from ..base import AlgorithmSpec, Objective, Optimizer, ParamSpec
from ..dataset import Observation
from ..pareto import observation_values
from ..registry import register
from .bayes import GaussianProcessBO

_EPS = 1e-12


@register
class ParEGO(Optimizer):
    """Random-scalarization multi-objective BO (reuses the GP/EI stack)."""

    spec = AlgorithmSpec(
        name="parego",
        family="model_based",
        summary="ParEGO 多目标优化：每轮随机加权（增广 Tchebycheff）把多目标压成单目标，复用 GP+EI。",
        when="2–3 个目标（如收率 vs 成本/纯度）、目标昂贵或带噪声、需要一组折中解（Pareto 前沿）时。",
        params=(
            ParamSpec(
                "rho",
                "float",
                0.05,
                low=0.0,
                high=0.5,
                description="增广 Tchebycheff 的 ρ；太小前沿覆盖差，太大退化为加权和",
            ),
            ParamSpec(
                "weight_alpha",
                "float",
                1.0,
                low=0.1,
                high=10.0,
                description="Dirichlet 权重浓度；<1 强调目标端点，>1 更均匀",
            ),
            ParamSpec(
                "acquisition",
                "choice",
                "logei",
                choices=("ei", "logei", "ucb"),
                description="标量化后单目标上使用的采集函数",
            ),
            ParamSpec(
                "n_initial",
                "int",
                0,
                low=0,
                high=200,
                description="初始设计点数；0 表示自动 = max(2*(feature_dim+1), 2*目标数)，上限 20",
            ),
            ParamSpec(
                "n_candidates",
                "int",
                1024,
                low=64,
                high=65536,
                description="每次采集评估的候选点数",
            ),
            ParamSpec(
                "alpha",
                "float",
                1e-4,
                low=1e-12,
                high=1.0,
                description="观测噪声方差（测量噪声）",
            ),
            ParamSpec(
                "length_scale",
                "float",
                0.3,
                low=0.01,
                high=10.0,
                description="GP 初始长度尺度（单位立方体）",
            ),
            ParamSpec(
                "noise_aware",
                "bool",
                True,
                description="把 objective_std / n_replicates 估计的逐点噪声传入 GP alpha",
            ),
        ),
        capabilities=frozenset(
            {
                "batch",
                "mixed",
                "categorical",
                "continuous",
                "noisy",
                "expensive",
                "model_based",
                "multi_objective",
                "constraints",
            }
        ),
        references=(
            "Knowles, ParEGO: A Hybrid Algorithm with On-Line Landscape Approximation for Expensive Multiobjective Optimization Problems, IEEE TEC 2006",
            "Daulton, Balandat & Bakshy, Parallel Bayesian Optimization of Multiple Noisy Objectives with qNEHVI, NeurIPS 2021",
        ),
    )

    def __init__(self, *args, **kwargs) -> None:
        self._xs: list[dict] = []
        self._Y: np.ndarray = np.empty((0, 0))
        self._Ystd: np.ndarray = np.empty((0, 0))
        self._bounds: dict[str, tuple[float, float]] = {}
        self._skipped = 0
        self._constraint_names: tuple[str, ...] = ()
        self._C: np.ndarray | None = None
        self._Xf = np.empty((0, 0))
        super().__init__(*args, **kwargs)
        self._constraint_names = tuple(getattr(self.objective, "constraints", ()) or ())

    # -- data -------------------------------------------------------------
    def tell(self, observations: Sequence[Observation]) -> None:
        observations = list(observations)
        valid: list[Observation] = []
        values: list[np.ndarray] = []
        for observation in observations:
            try:
                values.append(observation_values(observation, self.objective))
            except ValueError:
                # An incomplete objective vector must not kill the campaign; the
                # study keeps it, ParEGO simply cannot use it (review P0-2).
                self._skipped += 1
                continue
            valid.append(observation)
        # Parent tell sees every observation so ``t`` matches the study dataset.
        super().tell(observations)
        if not valid:
            return
        matrix = np.vstack(values)
        stds = np.asarray(
            [
                [float(obs.objective_std) if obs.objective_std else 0.0
                 for _ in self.objective.objectives]
                for obs in valid
            ],
            dtype=float,
        )
        features = np.vstack([self.space.features(obs.x) for obs in valid])
        constraints = (
            np.asarray(
                [
                    [float(obs.constraints.get(name, np.nan)) if obs.constraints else np.nan
                     for name in self._constraint_names]
                    for obs in valid
                ],
                dtype=float,
            )
            if self._constraint_names
            else np.empty((len(valid), 0))
        )
        if len(self._Y) == 0:
            self._Y = matrix
            self._Ystd = stds
            self._Xf = features
            self._C = constraints
        else:
            self._Y = np.vstack([self._Y, matrix])
            self._Ystd = np.vstack([self._Ystd, stds])
            self._Xf = np.vstack([self._Xf, features])
            self._C = np.vstack([self._C, constraints])
        self._xs.extend(dict(obs.x) for obs in valid)

    def _window(self) -> tuple[np.ndarray, np.ndarray]:
        """Fixed normalization window: explicit bounds, else frozen first rows."""
        m = self.objective.n_objectives
        if len(self._bounds) == m:
            lows = np.asarray([self._bounds[name][0] for name in self.objective.objectives])
            highs = np.asarray([self._bounds[name][1] for name in self.objective.objectives])
            return lows, highs
        explicit = all(name in self.objective.objective_bounds for name in self.objective.objectives)
        if explicit:
            lows = np.asarray([self.objective.objective_bounds[name]["low"] for name in self.objective.objectives])
            highs = np.asarray([self.objective.objective_bounds[name]["high"] for name in self.objective.objectives])
        else:
            freeze_n = max(3, 2 * m)
            if len(self._Y) < freeze_n:
                # Not enough observations to freeze the window yet: keep using
                # the current (growing) range instead of freezing it permanently
                # at a partial batch (review v0.4 P0-1).
                rows = self._Y
                lows = np.min(rows, axis=0)
                highs = np.max(rows, axis=0)
                return lows, highs
            rows = self._Y[:freeze_n]
            lows = np.min(rows, axis=0)
            highs = np.max(rows, axis=0)
        self._bounds = {
            name: (float(lows[index]), float(highs[index]))
            for index, name in enumerate(self.objective.objectives)
        }
        return lows, highs

    def _normalized(self) -> np.ndarray:
        """Normalize with a FROZEN window so historical scalar values stay comparable."""
        lows, highs = self._window()
        Y = self._Y
        m = Y.shape[1]
        out = np.zeros_like(Y, dtype=float)
        for j in range(m):
            span = highs[j] - lows[j]
            if span < _EPS:
                out[:, j] = 0.0
                continue
            if self.objective.directions[j] == "max":
                out[:, j] = (Y[:, j] - lows[j]) / span
            else:
                out[:, j] = (highs[j] - Y[:, j]) / span
        return out

    # -- propose ----------------------------------------------------------
    def ask(self, n: int = 1) -> list[dict]:
        if n <= 0:
            return []
        out: list[dict] = []
        design_left = self._initial_design_left()
        if design_left > 0:
            k = min(n, design_left)
            out.extend(self.space.sample_sobol(k, rng=self.rng))
            n -= k
        m = self.objective.n_objectives
        for _ in range(n):
            out.append(self._propose(out, m))
        return out

    def _initial_design_left(self) -> int:
        n_initial = int(self.params["n_initial"])
        if n_initial <= 0:
            n_initial = max(2 * (self.space.feature_dim + 1), 2 * self.objective.n_objectives)
            n_initial = min(n_initial, 20)
        return max(0, n_initial - int(self.t))

    def _propose(self, batch: Sequence[dict], m: int) -> dict:
        if len(self._xs) < 2:
            return self.space.sample_uniform(1, self.rng)[0]
        weights = self.rng.dirichlet(np.full(m, float(self.params["weight_alpha"])))
        scalar = self._scalar_values(weights)
        scalar_std = self._scalar_std(weights)
        scalar_objective = Objective(name="parego_scalar", direction="max")
        prior = [
            Observation(
                x=x,
                y=float(value),
                objective_std=(
                    float(scalar_std[index])
                    if bool(self.params["noise_aware"])
                    and np.isfinite(scalar_std[index])
                    and scalar_std[index] > 0
                    else 0.0
                ),
            )
            for index, (x, value) in enumerate(zip(self._xs, scalar))
        ]
        inner = GaussianProcessBO(
            space=self.space,
            objective=scalar_objective,
            params={
                "n_initial": 1,
                "n_candidates": int(self.params["n_candidates"]),
                "acquisition": str(self.params["acquisition"]),
                "alpha": float(self.params["alpha"]),
                "length_scale": float(self.params["length_scale"]),
                "batch_strategy": "greedy",
            },
            seed=self.seed + 104729 * (len(self._xs) + 1) + len(batch),
            prior=prior,
        )
        existing = {self._key(x) for x in self._xs}
        batch_keys = {self._key(x) for x in batch}
        pool: list[dict] = []
        for _ in range(20):
            config = inner.ask(1)[0]
            key = self._key(config)
            if key not in existing and key not in batch_keys and key not in {
                self._key(item) for item in pool
            }:
                pool.append(config)
                if not self._constraint_names:
                    return config
                if len(pool) >= 8:
                    break
            # A duplicate is unlikely for continuous spaces; resample the GP
            # proposal with a fresh seed for discrete/degenerate spaces.
            inner.rng = np.random.default_rng(self.seed + 7919 * (len(self._xs) + len(batch) + 1))
        if pool and self._constraint_names:
            probability = self._feasibility_probability(pool)
            return pool[int(np.argmax(probability))]
        return pool[0] if pool else self.space.sample_uniform(1, self.rng)[0]

    def _feasibility_probability(self, configs: Sequence[dict]) -> np.ndarray:
        if (
            not self._constraint_names
            or self._C is None
            or self._C.shape[1] == 0
            or len(self._Xf) == 0
        ):
            return np.ones(len(configs))
        from scipy.stats import norm
        from sklearn.gaussian_process import GaussianProcessRegressor

        from .bayes import _make_kernel

        features = np.vstack([self.space.features(config) for config in configs])
        limits = getattr(self.objective, "constraint_limits", {}) or {}
        probability = np.ones(len(configs))
        modelled = False
        for index, name in enumerate(self._constraint_names):
            values = self._C[:, index]
            usable = np.isfinite(values)
            if int(np.sum(usable)) < 2:
                continue
            kernel = _make_kernel(
                "matern52", 0.3, feature_dim=self.space.feature_dim, ard=True
            )
            model = GaussianProcessRegressor(
                kernel=kernel, alpha=1e-4, normalize_y=True,
                optimizer=None, random_state=self.seed,
            )
            model.fit(self._Xf[usable], values[usable])
            mu, std = model.predict(features, return_std=True)
            mu = np.asarray(mu, dtype=float)
            std = np.maximum(np.asarray(std, dtype=float), 1e-9)
            limit = limits.get(name) or {}
            threshold = float(limit.get("value", 0.0))
            op = str(limit.get("op", "<="))
            z = (threshold - mu) / std if op == "<=" else (mu - threshold) / std
            probability *= norm.cdf(z)
            modelled = True
        return probability if modelled else np.ones(len(configs))

    def _scalar_std(self, weights: np.ndarray) -> np.ndarray:
        """Conservative scalar noise: sum_j w_j(1+rho)/span_j * std_j."""
        if len(self._Ystd) == 0:
            return np.zeros(0)
        lows, highs = self._window()
        spans = np.maximum(np.asarray(highs) - np.asarray(lows), _EPS)
        coefficients = (
            np.asarray(weights, dtype=float) * (1.0 + float(self.params["rho"])) / spans
        )
        finite = np.all(np.isfinite(self._Ystd), axis=1)
        out = np.full(len(self._Ystd), np.nan)
        out[finite] = self._Ystd[finite] @ coefficients
        return out

    def _scalar_values(self, weights: np.ndarray) -> np.ndarray:
        normalized = self._normalized()
        weighted = normalized * weights
        return np.max(weighted, axis=1) + float(self.params["rho"]) * np.sum(weighted, axis=1)

    def _key(self, config: Mapping) -> tuple:
        return tuple(np.round(self.space.encode(config), 9).tolist())
