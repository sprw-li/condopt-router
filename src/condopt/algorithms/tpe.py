"""Tree-structured Parzen Estimator (TPE).

TPE replaces the GP surrogate with two Parzen density estimates:

* ``l(x) = p(x | y < y*)``  -- the "good" observations,
* ``g(x) = p(x | y >= y*)`` -- the rest,

and proposes the point that maximises the density ratio ``l/g`` (which is
proportional to EI up to a constant).  Because the densities are built
per-dimension with simple kernels, TPE handles mixed, categorical and
conditional-ish spaces without a GP and with very few observations -- the
regime where a GP's hyper-parameters are hopelessly under-determined.

Our implementation is deliberately independent per dimension (no
tree/conditional structure yet) with:
* Gaussian KDE on the unit-cube coordinate for numeric variables,
* Laplace-smoothed categorical distributions for discrete variables,
* optional uniform exploration candidates.

References: Bergstra et al., "Algorithms for Hyper-Parameter Optimization"
(NeurIPS 2011) [R4]; Optuna's TPE sampler.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np
from scipy.special import logsumexp

from ..base import AlgorithmSpec, Optimizer, ParamSpec
from ..dataset import Observation
from ..registry import register
from ..space import SearchSpace

_LOG_2PI = math.log(2.0 * math.pi)


def _numeric_bandwidth(points: np.ndarray, factor: float) -> float:
    if len(points) <= 1:
        return max(0.1, factor * 0.2)
    spread = max(float(np.std(points)), 0.05)
    return max(0.05, factor * spread * len(points) ** (-0.2))


def _kde_logpdf(points: np.ndarray, x: float, bandwidth: float) -> float:
    z = (x - points) / bandwidth
    return float(logsumexp(-0.5 * z * z) - math.log(len(points)) - math.log(bandwidth) - 0.5 * _LOG_2PI)


@register
class TreeParzenEstimator(Optimizer):
    """Independent-dimension TPE for mixed search spaces."""

    spec = AlgorithmSpec(
        name="tpe",
        family="model_based",
        summary="TPE：用 l/g 两组核密度比代替 GP，天然支持分类/混合空间，小样本也能用。",
        when="空间含较多分类变量或 one-hot 后特征维很高（GP 不可靠）、观测很少、目标不要求光滑时。",
        params=(
            ParamSpec(
                "gamma",
                "float",
                0.25,
                low=0.02,
                high=0.9,
                description="好样本分位数（y<y* 的比例）；越小越利用，越大越探索",
            ),
            ParamSpec(
                "n_candidates",
                "int",
                256,
                low=32,
                high=4096,
                description="每次提议时从 l(x) 采样的候选数；越大越准越慢",
            ),
            ParamSpec(
                "n_initial",
                "int",
                0,
                low=0,
                high=200,
                description="初始设计点数；0 表示自动 = max(3, min(feature_dim+1, 15))",
            ),
            ParamSpec(
                "bandwidth_factor",
                "float",
                1.0,
                low=0.1,
                high=5.0,
                description="数值维高斯核带宽的缩放系数；数据稀疏时调大",
            ),
            ParamSpec(
                "exploration_frac",
                "float",
                0.1,
                low=0.0,
                high=0.5,
                description="每次提议直接均匀随机探索的概率（防止密度模型锁死在局部）",
            ),
            ParamSpec(
                "min_distance",
                "float",
                0.02,
                low=0.0,
                high=0.5,
                description="新提议与已测点的最小单位立方体距离；太小会反复提同一点",
            ),
            ParamSpec(
                "prior_weight",
                "float",
                1.0,
                low=0.1,
                high=10.0,
                description="分类变量 Laplace 平滑强度",
            ),
        ),
        capabilities=frozenset(
            {"batch", "mixed", "categorical", "continuous", "noisy", "cheap", "model_based"}
        ),
        references=(
            "Bergstra, Bardenet, Bengio & Kegl, Algorithms for Hyper-Parameter Optimization, NeurIPS 2011",
            "Akiba et al., Optuna: A Next-generation Hyperparameter Optimization Framework, KDD 2019",
        ),
    )

    def __init__(self, *args, **kwargs) -> None:
        self._units = np.zeros((0, 0))
        self._y = np.zeros(0)
        super().__init__(*args, **kwargs)

    # -- data -------------------------------------------------------------
    def tell(self, observations: Sequence[Observation]) -> None:
        observations = list(observations)
        super().tell(observations)
        if not observations:
            return
        units = np.vstack([self.space.encode(o.x) for o in observations])
        y = np.asarray(
            [o.score(self.objective.name, self.objective.direction) for o in observations],
            dtype=float,
        )
        if len(self._units) == 0:
            self._units, self._y = units, y
        else:
            self._units = np.vstack([self._units, units])
            self._y = np.concatenate([self._y, y])

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
        while n > 0:
            out.append(self._propose(out))
            n -= 1
        return out

    def _initial_design_left(self) -> int:
        n_initial = int(self.params["n_initial"])
        if n_initial <= 0:
            n_initial = max(3, min(self.space.feature_dim + 1, 15))
        return max(0, n_initial - int(self.t))

    def _propose(self, batch: Sequence[dict]) -> dict:
        if len(self._units) < 2:
            return self._random_unique(batch)
        # Explicit global exploration: density models on few points lock onto
        # one basin, which is fatal on multimodal landscapes.
        if self.rng.random() < float(self.params["exploration_frac"]):
            return self._random_unique(batch)

        candidates = self._candidate_pool()
        existing_keys = {self._key(unit) for unit in self._units}
        batch_units = [self.space.encode(item) for item in batch]
        min_distance = float(self.params["min_distance"])
        log_ratio = self._log_ratio(candidates)
        order = np.argsort(-log_ratio)
        for index in order:
            unit = candidates[index]
            config = self.space.decode(unit)
            canonical = self.space.encode(config)
            key = self._key(canonical)
            if key in existing_keys:
                continue
            if min_distance > 0 and batch_units:
                distances = np.max(np.abs(np.vstack(batch_units) - canonical), axis=1)
                if float(np.min(distances)) < min_distance:
                    continue
            return config
        # Degenerate: everything was already evaluated or too close.
        return self._random_unique(batch)

    def _random_unique(self, batch: Sequence[dict]) -> dict:
        existing = {self._key(unit) for unit in self._units}
        batch_keys = {self._key(self.space.encode(item)) for item in batch}
        for _ in range(100):
            config = self.space.sample_uniform(1, self.rng)[0]
            key = self._key(self.space.encode(config))
            if key not in existing and key not in batch_keys:
                return config
        return self.space.sample_uniform(1, self.rng)[0]

    # -- density model ----------------------------------------------------
    def _split(self) -> tuple[np.ndarray, np.ndarray]:
        n = len(self._y)
        gamma = float(self.params["gamma"])
        n_good = min(max(1, int(math.ceil(gamma * n))), max(1, n - 1))
        order = np.argsort(-self._y)
        good = self._units[order[:n_good]]
        bad = self._units[order[n_good:]]
        return good, bad

    def _candidate_pool(self) -> np.ndarray:
        m = int(self.params["n_candidates"])
        good, _ = self._split()
        samples = np.empty((m, self.space.dim), dtype=float)

        bandwidths = self._bandwidths(good)
        numeric = [i for i, var in enumerate(self.space.variables) if var.is_numeric]
        discrete = [i for i, var in enumerate(self.space.variables) if var.is_discrete]
        for dim in numeric:
            centers = good[self.rng.integers(0, len(good), size=m), dim]
            samples[:, dim] = np.clip(centers + self.rng.normal(0.0, bandwidths[dim], m), 0.0, 1.0)
        for dim in discrete:
            probs = self._categorical_probs(dim, good)
            indices = self.rng.choice(len(probs), size=m, p=probs)
            samples[:, dim] = self._discrete_unit(indices, dim)
        return samples

    def _bandwidths(self, good: np.ndarray) -> dict[int, float]:
        factor = float(self.params["bandwidth_factor"])
        return {
            dim: _numeric_bandwidth(good[:, dim], factor)
            for dim, var in enumerate(self.space.variables)
            if var.is_numeric
        }

    def _categorical_probs(self, dim: int, group: np.ndarray) -> np.ndarray:
        variable = self.space.variables[dim]
        k = len(variable.choices)
        counts = np.bincount(self._discrete_index(group[:, dim], dim), minlength=k).astype(float)
        prior = max(float(self.params["prior_weight"]), 1e-9)
        probs = (counts + prior / k) / (counts.sum() + prior)
        return probs / probs.sum()

    def _discrete_index(self, unit_column: np.ndarray, dim: int) -> np.ndarray:
        k = len(self.space.variables[dim].choices)
        return np.clip(np.round(unit_column * (k - 1)).astype(int), 0, k - 1)

    def _discrete_unit(self, indices: np.ndarray, dim: int) -> np.ndarray:
        k = len(self.space.variables[dim].choices)
        return indices / max(1, k - 1)

    def _log_ratio(self, candidates: np.ndarray) -> np.ndarray:
        good, bad = self._split()
        log_l = np.zeros(len(candidates))
        log_g = np.zeros(len(candidates))
        bandwidths = self._bandwidths(good)
        for dim, variable in enumerate(self.space.variables):
            column = candidates[:, dim]
            if variable.is_numeric:
                h_good = bandwidths[dim]
                h_bad = _numeric_bandwidth(bad[:, dim], float(self.params["bandwidth_factor"])) if len(bad) else h_good
                log_l += np.asarray([_kde_logpdf(good[:, dim], x, h_good) for x in column])
                if len(bad):
                    log_g += np.asarray([_kde_logpdf(bad[:, dim], x, h_bad) for x in column])
                else:  # no bad group -> uniform
                    log_g += 0.0
            else:
                p_l = self._categorical_probs(dim, good)
                p_g = self._categorical_probs(dim, bad) if len(bad) else np.full(len(p_l), 1.0 / len(p_l))
                indices = self._discrete_index(column, dim)
                log_l += np.log(p_l[indices])
                log_g += np.log(p_g[indices])
        return log_l - log_g

    @staticmethod
    def _key(unit: np.ndarray) -> tuple:
        return tuple(np.round(unit, 9).tolist())
