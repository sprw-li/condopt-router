"""Gaussian-process Bayesian optimization (GP-BO).

A compact, dependency-light implementation of the classic
"surrogate + acquisition" loop:

1. fit a GP surrogate to all observations (mixed space features: numeric
   scaled to [0, 1], categorical one-hot);
2. score a large candidate pool with an acquisition function
   (EI / log-EI / UCB / PI / Thompson sampling);
3. propose the best candidate(s), with a batched greedy strategy
   (Kriging-believer or constant-liar) for parallel experiments.

The algorithm is intentionally explicit rather than maximally clever:
every knob that matters in practice is a validated parameter, so the
router in :mod:`condopt.router` (or the user) can adapt it per problem
without editing code.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern

from ..base import AlgorithmSpec, Optimizer, ParamSpec
from ..dataset import Observation
from ..registry import register

_JITTER = 1e-9


def _ard_initial_lengthscale(feature_dim: int) -> float:
    """Dimension-scaled length-scale prior (R15): longer scales in high dims."""
    d = max(1, int(feature_dim))
    if d <= 8:
        return 0.3
    return float(np.exp(np.sqrt(2.0) + 0.5 * np.log(d)))


def _make_kernel(kind: str, length_scale: float, feature_dim: int, ard: bool):
    """Build the GP kernel; with ARD every feature gets its own length-scale."""
    d = max(1, int(feature_dim))
    if ard:
        ls_init = _ard_initial_lengthscale(d)
        ls = np.full(d, ls_init if length_scale is None else float(length_scale))
        upper = 1e3 if d > 20 else 1e2
        bounds = np.tile(np.asarray([1e-2, upper], dtype=float), (d, 1))
    else:
        ls = float(length_scale)
        bounds = (1e-2, 1e2)
    if kind == "matern52":
        base = Matern(length_scale=ls, length_scale_bounds=bounds, nu=2.5)
    elif kind == "matern32":
        base = Matern(length_scale=ls, length_scale_bounds=bounds, nu=1.5)
    elif kind == "rbf":
        base = RBF(length_scale=ls, length_scale_bounds=bounds)
    else:  # pragma: no cover - guarded by ParamSpec
        raise ValueError(f"unknown kernel {kind!r}")
    return ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3)) * base


def _log_ei(imp: np.ndarray, std: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Numerically stable log-EI.

    ``log(imp*Phi(z) + std*phi(z))`` underflows for far-away candidates; the
    asymptotic form keeps their ranking meaningful instead of collapsing all
    of them to the same floor value.
    """
    out = np.empty_like(z, dtype=float)
    moderate = z > -6.0
    ei = imp[moderate] * norm.cdf(z[moderate]) + std[moderate] * norm.pdf(z[moderate])
    out[moderate] = np.log(np.maximum(ei, 1e-300))
    far = ~moderate
    if np.any(far):
        z_far = z[far]
        out[far] = (
            np.log(std[far])
            - 0.5 * np.log(2.0 * np.pi)
            - 0.5 * z_far**2
            - 2.0 * np.log(-z_far)
        )
    return out


@register
class GaussianProcessBO(Optimizer):
    """GP surrogate + acquisition function Bayesian optimization."""

    spec = AlgorithmSpec(
        name="bayes",
        family="model_based",
        summary="GP 贝叶斯优化：用高斯过程代理 + 采集函数（EI/UCB/PI/TS）平衡探索与利用，样本效率最高。",
        when="目标昂贵（每次实验成本高）、预算 10–200、维度 ≤15、目标较光滑、连续/顺序型条件为主时。",
        params=(
            ParamSpec(
                "kernel",
                "choice",
                "matern52",
                choices=("matern52", "matern32", "rbf"),
                description="GP 核函数；matern52 通用，matern32 更粗糙，rbf 最光滑但易过拟合",
            ),
            ParamSpec(
                "length_scale",
                "float",
                0.3,
                low=0.01,
                high=10.0,
                description="初始特征长度尺度（单位立方体坐标）；ARD 开启时为所有特征的初值",
            ),
            ParamSpec(
                "ard",
                "bool",
                True,
                description="是否使用 ARD（每特征一个 length_scale）；混合/高基数空间建议开启",
            ),
            ParamSpec(
                "alpha",
                "float",
                1e-6,
                low=1e-12,
                high=1.0,
                description="观测噪声方差（测量噪声的方差，不是标准差）；噪声大时用 0.01–0.1",
            ),
            ParamSpec(
                "n_initial",
                "int",
                0,
                low=0,
                high=200,
                description="初始设计点数；0 表示自动 = max(3, min(feature_dim+1, 20))",
            ),
            ParamSpec(
                "n_candidates",
                "int",
                1024,
                low=64,
                high=65536,
                description="每次采集时评估的候选点数；越大越准越慢",
            ),
            ParamSpec(
                "acquisition",
                "choice",
                "ei",
                choices=("ei", "logei", "ucb", "pi", "ts"),
                description="采集函数：ei/logei 平衡探索利用，ucb 更偏探索，pi 偏利用，ts 用后验采样",
            ),
            ParamSpec("kappa", "float", 2.0, low=0.0, high=10.0, description="UCB 的探索系数"),
            ParamSpec("xi", "float", 0.01, low=0.0, high=1.0, description="EI/PI 的改进阈值"),
            ParamSpec(
                "batch_strategy",
                "choice",
                "local_penalty",
                choices=("greedy", "kriging_believer", "constant_liar", "local_penalty"),
                description="批量选点：local_penalty（推荐）/贪心/信念值/常数谎言；批量≥4 用前者避免扎堆",
            ),
            ParamSpec(
                "local_frac",
                "float",
                0.3,
                low=0.0,
                high=0.9,
                description="候选池中围绕当前最优点的局部采样比例",
            ),
            ParamSpec(
                "local_sigma",
                "float",
                0.15,
                low=0.01,
                high=1.0,
                description="局部候选的单位立方体扰动标准差",
            ),
        ),
        capabilities=frozenset(
            {"batch", "mixed", "categorical", "continuous", "noisy", "expensive", "model_based"}
        ),
        references=(
            "Jones, Schonlau & Welch, Efficient Global Optimization of Expensive Black-Box Functions, JOGO 1998",
            "Srinivas, Krause, Kakade & Seeger, Gaussian Process Optimization in the Bandit Setting (GP-UCB), ICML 2010",
            "Rasmussen & Williams, Gaussian Processes for Machine Learning, MIT Press 2006",
            "Balandat et al., BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization, NeurIPS 2020",
        ),
    )

    def __init__(self, *args, **kwargs) -> None:
        self._X: np.ndarray | None = None
        self._y = np.zeros(0)
        self._units = np.zeros((0, 0))
        self._noise_var = np.zeros(0)
        self._init_queue: list[dict] | None = None
        super().__init__(*args, **kwargs)

    # -- data -------------------------------------------------------------
    def tell(self, observations: Sequence[Observation]) -> None:
        observations = list(observations)
        super().tell(observations)
        if not observations:
            return
        X = self.space.features_many([o.x for o in observations])
        units = np.vstack([self.space.encode(o.x) for o in observations])
        y = np.asarray(
            [o.score(self.objective.name, self.objective.direction) for o in observations],
            dtype=float,
        )
        # Per-point observation-noise variance from replicates (0 = unknown).
        noise = np.asarray(
            [
                (o.objective_std**2 / o.n_replicates) if o.objective_std > 0 else np.nan
                for o in observations
            ],
            dtype=float,
        )
        if self._X is None:
            self._X, self._y, self._units = X, y, units
            self._noise_var = noise
        else:
            self._X = np.vstack([self._X, X])
            self._y = np.concatenate([self._y, y])
            self._units = np.vstack([self._units, units])
            self._noise_var = np.concatenate([self._noise_var, noise])

    # -- propose ----------------------------------------------------------
    def ask(self, n: int = 1) -> list[dict]:
        if n <= 0:
            return []
        out: list[dict] = []
        design_left = self._initial_design_left()
        if design_left > 0:
            k = min(n, design_left)
            out.extend(self._initial_design(k))
            n -= k
        if n > 0:
            out.extend(self._model_ask(n))
        return out

    def _initial_design(self, k: int) -> list[dict]:
        """Balanced 2^k Sobol design, generated once and sliced on demand."""
        if self._init_queue is None:
            n_initial = int(self.params["n_initial"])
            if n_initial <= 0:
                n_initial = max(3, min(self.space.feature_dim + 1, 20))
            total = max(n_initial, self.t + k, 2)
            design = self.space.sample_sobol_base2(total, seed=self.seed + 104729)
            seen = {self._key(unit) for unit in self._units}
            self._init_queue = [
                config for config in design if self._key(self.space.encode(config)) not in seen
            ]
        take = self._init_queue[:k]
        self._init_queue = self._init_queue[k:]
        if not take:
            take = self.space.sample_sobol(k, rng=self.rng)
        return take

    def _initial_design_left(self) -> int:
        n_initial = int(self.params["n_initial"])
        if n_initial <= 0:
            # Use the *feature* dimension (numeric + one-hot), not the number of
            # variables: 3 categorical knobs with 20 levels each are 60 features.
            n_initial = max(3, min(self.space.feature_dim + 1, 20))
        return max(0, n_initial - int(self.t))

    # -- model-based proposal --------------------------------------------
    def _model_ask(self, n: int) -> list[dict]:
        if self._X is None or len(self._y) < 2:
            return self.space.sample_sobol(n, rng=self.rng) if n else []

        candidates, features = self._candidate_pool()
        if len(candidates) == 0:
            return self.space.sample_sobol(n, rng=self.rng)

        model = self._fit(self._X, self._y)
        mode = str(self.params["acquisition"])
        strategy = str(self.params["batch_strategy"])

        if mode == "ts":
            return self._thompson_batch(model, candidates, features, n)
        if strategy == "local_penalty":
            return self._local_penalty_batch(model, candidates, features, n)

        chosen: list[int] = []
        available = np.ones(len(candidates), dtype=bool)
        X_aug, y_aug = self._X.copy(), self._y.copy()
        for i in range(n):
            if not available.any():
                break
            acq = self._acquisition(model, features)
            acq = np.where(available, acq, -np.inf)
            pick = int(np.argmax(acq))
            chosen.append(pick)
            available[pick] = False
            if i < n - 1 and strategy != "greedy":
                x_new = features[pick : pick + 1]
                mu_new = float(model.predict(x_new)[0])
                if strategy == "constant_liar":
                    y_lie = float(np.max(self._y)) - float(np.std(self._y))
                else:
                    y_lie = mu_new
                X_aug = np.vstack([X_aug, x_new])
                y_aug = np.concatenate([y_aug, [y_lie]])
                model = self._fit(X_aug, y_aug)

        configs = [self.space.decode(candidates[i]) for i in chosen]
        while len(configs) < n:  # pragma: no cover - tiny candidate pools only
            configs.extend(self.space.sample_sobol(n - len(configs), rng=self.rng))
        return configs

    def _local_penalty_scale(self, model: GaussianProcessRegressor) -> float:
        """Local-penalization length scale: half the median ARD length-scale."""
        kernel = model.kernel_
        length_scale = getattr(getattr(kernel, "k2", None), "length_scale", None)
        if length_scale is None:
            length_scale = self.params["length_scale"]
        if np.ndim(length_scale) == 0:
            value = float(length_scale)
        else:
            value = float(np.median(np.asarray(length_scale, dtype=float)))
        return float(np.clip(0.5 * value, 0.03, 0.25))

    def _local_penalty_batch(
        self, model: GaussianProcessRegressor, candidates: np.ndarray, features: np.ndarray, n: int
    ) -> list[dict]:
        """Greedy batch with a local-penalization factor (no fake observations)."""
        scale = self._local_penalty_scale(model)
        chosen: list[int] = []
        chosen_units: list[np.ndarray] = []
        available = np.ones(len(candidates), dtype=bool)
        for _ in range(n):
            if not available.any():
                break
            acq = self._acquisition(model, features)
            if chosen_units:
                distances = np.sum(
                    (candidates[:, None, :] - np.asarray(chosen_units)[None, :, :]) ** 2, axis=-1
                )
                penalty = np.prod(1.0 - np.exp(-distances / (2.0 * scale * scale)), axis=1)
                acq = acq * penalty
            acq = np.where(available, acq, -np.inf)
            pick = int(np.argmax(acq))
            chosen.append(pick)
            chosen_units.append(candidates[pick])
            available[pick] = False
        configs = [self.space.decode(candidates[i]) for i in chosen]
        while len(configs) < n:
            configs.extend(self.space.sample_sobol(n - len(configs), rng=self.rng))
        return configs

    def _thompson_batch(
        self,
        model: GaussianProcessRegressor,
        candidates: np.ndarray,
        features: np.ndarray,
        n: int,
    ) -> list[dict]:
        """Parallel Thompson sampling: one independent posterior draw per point."""
        draws = self._sample_posterior_multi(model, features, max(n, 1))
        picks: list[int] = []
        for index in range(draws.shape[1]):
            column = draws[:, index].copy()
            if picks:
                column[picks] = -np.inf
            picks.append(int(np.argmax(column)))
        return [self.space.decode(candidates[i]) for i in picks]

    def _candidate_pool(self) -> tuple[np.ndarray, np.ndarray]:
        m = int(self.params["n_candidates"])
        dim = self.space.dim
        rng = self.rng

        n_local = int(m * float(self.params["local_frac"])) if self._X is not None else 0
        n_rest = max(m - n_local, 2)
        n_uniform = n_rest // 2
        n_sobol = n_rest - n_uniform

        parts = [rng.random((n_uniform, dim))]
        if n_sobol > 0:
            import warnings

            from scipy.stats import qmc

            sobol = qmc.Sobol(d=dim, scramble=True, seed=int(rng.integers(0, 2**31 - 1)))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                parts.append(sobol.random(n_sobol))
        if n_local > 0:
            top = np.argsort(-self._y)[: min(5, len(self._y))]
            centers = self._units[top]
            base = centers[rng.integers(0, len(centers), size=n_local)]
            parts.append(np.clip(base + rng.normal(0.0, float(self.params["local_sigma"]), (n_local, dim)), 0.0, 1.0))

        raw = np.vstack(parts)

        # Canonicalise: decode to a real configuration, then re-encode, so that
        # categorical/ordinal coordinates collapse to the same point.
        configs = [self.space.decode(u) for u in raw]
        units = np.vstack([self.space.encode(c) for c in configs])

        # Drop points that were already evaluated or proposed.
        existing = {self._key(u) for u in self._units} if len(self._units) else set()
        keep_units: list[np.ndarray] = []
        keep_features: list[np.ndarray] = []
        seen: set[tuple] = set()
        for cfg, u in zip(configs, units):
            key = self._key(u)
            if key in existing or key in seen:
                continue
            seen.add(key)
            keep_units.append(u)
            keep_features.append(self.space.features(cfg))
        if not keep_units:  # pragma: no cover - exhaustive discrete spaces
            return np.zeros((0, dim)), np.zeros((0, self.space.feature_dim))
        return np.vstack(keep_units), np.vstack(keep_features)

    @staticmethod
    def _key(unit: np.ndarray) -> tuple:
        return tuple(np.round(unit, 9).tolist())

    def _fit(self, X: np.ndarray, y: np.ndarray) -> GaussianProcessRegressor:
        import warnings

        from sklearn.exceptions import ConvergenceWarning

        kernel = _make_kernel(
            str(self.params["kernel"]),
            float(self.params["length_scale"]),
            feature_dim=self.space.feature_dim,
            ard=bool(self.params["ard"]),
        )
        # Freeze hyper-parameters when the data cannot identify them (n < 2d):
        # optimizing ~feature_dim length-scales on fewer points overfits badly.
        frozen = len(y) < max(6, 2 * self.space.feature_dim)
        alpha: float | np.ndarray = float(self.params["alpha"])
        noise = getattr(self, "_noise_var", None)
        if noise is not None and len(noise):
            base = float(self.params["alpha"])
            values = np.asarray(noise, dtype=float)
            y_var = max(float(np.var(y)) if len(y) > 1 else 1.0, 1e-12)
            scaled = np.where(np.isfinite(values) & (values > 0), values / y_var, base)
            if len(scaled) < len(y):  # fantasy/augmented points in a batch refit
                scaled = np.concatenate([scaled, np.full(len(y) - len(scaled), base)])
            elif len(scaled) > len(y):
                scaled = scaled[: len(y)]
            alpha = np.clip(scaled, 1e-12, 0.5)
        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            normalize_y=True,
            optimizer=None if frozen else "fmin_l_bfgs_b",
            n_restarts_optimizer=0 if frozen else 1,
            random_state=self.seed,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(X, y)
        return model

    def lengthscale_diagnostics(self) -> dict | None:
        """ARD diagnostics: median length-scale per space variable.

        Returns ``None`` when no fitted model is available.  The mapping back
        to variable names follows the feature layout (numeric scalars, then
        one block per discrete variable).
        """
        if self._X is None or len(self._y) < 2:
            return None
        model = self._fit(self._X, self._y)
        length_scales = getattr(getattr(model.kernel_, "k2", None), "length_scale", None)
        if length_scales is None:
            return None
        values = np.atleast_1d(np.asarray(length_scales, dtype=float))
        per_variable: list[dict] = []
        index = 0
        for variable in self.space.variables:
            width = 1 if variable.is_numeric else len(variable.choices)
            block = values[index : index + width]
            per_variable.append(
                {
                    "name": variable.name,
                    "median_length_scale": float(np.median(block)),
                    "min": float(np.min(block)),
                    "max": float(np.max(block)),
                }
            )
            index += width
        return {"per_variable": per_variable, "n_features": int(len(values))}

    # -- acquisition ------------------------------------------------------
    def _acquisition(self, model: GaussianProcessRegressor, features: np.ndarray) -> np.ndarray:
        mode = str(self.params["acquisition"])
        mu, std = model.predict(features, return_std=True)
        mu = np.asarray(mu, dtype=float)
        std = np.maximum(np.asarray(std, dtype=float), 1e-12)
        best = float(np.max(self._y))
        if mode in ("ei", "logei"):
            xi = float(self.params["xi"])
            imp = mu - best - xi
            z = imp / std
            if mode == "logei":
                return _log_ei(imp, std, z)
            return imp * norm.cdf(z) + std * norm.pdf(z)
        if mode == "ucb":
            return mu + float(self.params["kappa"]) * std
        if mode == "pi":
            xi = float(self.params["xi"])
            return norm.cdf((mu - best - xi) / std)
        raise ValueError(f"unknown acquisition {mode!r}")  # pragma: no cover

    def _sample_posterior_multi(
        self, model: GaussianProcessRegressor, features: np.ndarray, n_draws: int
    ) -> np.ndarray:
        """``n_draws`` independent draws from the GP posterior at ``features``."""
        mu, cov = model.predict(features, return_cov=True)
        mu = np.asarray(mu, dtype=float)
        cov = np.asarray(cov, dtype=float)
        cov = 0.5 * (cov + cov.T) + _JITTER * np.eye(len(features))
        try:
            chol = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:  # pragma: no cover - fall back to eigen decomposition
            vals, vecs = np.linalg.eigh(cov)
            chol = vecs @ np.diag(np.sqrt(np.maximum(vals, 0.0)))
        z = self.rng.standard_normal((len(features), max(1, int(n_draws))))
        return mu[:, None] + chol @ z

    # -- state ------------------------------------------------------------
    # The GP data is rebuilt from the study dataset on every ask; only the
    # initial-design queue needs to survive a process restart.
    def state_dict(self) -> dict:
        state = super().state_dict()
        state["init_queue"] = self._init_queue
        return state

    def load_state_dict(self, state: Mapping) -> None:
        super().load_state_dict(state)
        queue = state.get("init_queue")
        self._init_queue = None if queue is None else [dict(item) for item in queue]
