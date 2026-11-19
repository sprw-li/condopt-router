"""Compact CMA-ES for smooth mid/high-dimensional continuous spaces.

Implements a (mu/mu_w, lambda)-ES with rank-mu covariance adaptation: it keeps a
multivariate normal ``N(m, sigma^2 C)`` on the unit cube, samples a population,
ranks it by the objective, and moves the mean to the weighted best while
updating ``C`` from the successful steps.  No evolution paths, so it is a
simplified CMA-ES, but it behaves like the classic algorithm on smooth
quadratic/valley landscapes where GP surrogates (dimension > ~15) stop being
reliable.

Defaults (``sigma0=0.1``, ``c_mu=0.5``) come from the 10-D sweep in
``scripts/sweep_cmaes.py``: on rosenbrock10/ackley10 they beat the textbook
``sigma0=0.3`` by several times in best score under an 80-evaluation budget.
A Sobol-seeded variant was tried and rejected (it made both benchmarks worse).
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

from ..base import AlgorithmSpec, Optimizer, ParamSpec
from ..dataset import Observation
from ..registry import register

_EPS = 1e-12


@register
class CMAES(Optimizer):
    #: mean/covariance are a function of the ranked history, not a cursor
    replay_history_on_load = True
    """Covariance-matrix adaptation evolution strategy (rank-mu update)."""

    spec = AlgorithmSpec(
        name="cmaes",
        family="evolutionary",
        summary="CMA-ES：在单位立方体上维护 N(m, sigma^2 C)，按排名更新均值与协方差；中高维光滑连续问题可替代 GP。",
        when="连续、光滑、维度偏高（>10）或 GP 代理不可靠、预算 ≥ 种群规模的若干倍时。",
        params=(
            ParamSpec("sigma0", "float", 0.1, low=0.01, high=1.0,
                      description="初始步长（单位立方体比例；10 维扫描建议 0.05–0.1）"),
            ParamSpec("population_size", "int", 0, low=0, high=200,
                      description="0 = 自动 4 + floor(3 ln d)；实际每代规模由 ask/batch 决定"),
            ParamSpec("c_mu", "float", 0.5, low=0.01, high=1.0,
                      description="协方差学习率"),
        ),
        capabilities=frozenset(
            {"batch", "continuous", "mixed", "noisy", "evolutionary"}
        ),
        references=(
            "Hansen & Ostermeier, Completely Derandomized Self-Adaptation in Evolution Strategies, Evolutionary Computation 9(2), 2001",
        ),
    )

    def __init__(self, *args, **kwargs) -> None:
        self._m: np.ndarray | None = None
        self._C: np.ndarray | None = None
        self._sigma = 0.1
        self._pending: list[tuple[np.ndarray, dict]] = []
        super().__init__(*args, **kwargs)
        self._sigma = float(self.params["sigma0"])

    # -- population --------------------------------------------------------
    def _population_size(self) -> int:
        configured = int(self.params["population_size"])
        if configured > 0:
            return configured
        return max(4, int(4 + 3 * math.log(max(2, self.space.dim))))

    def _ensure(self) -> None:
        if self._m is None:
            self._m = np.full(self.space.dim, 0.5, dtype=float)
            self._C = np.eye(self.space.dim, dtype=float)

    def ask(self, n: int = 1) -> list[dict]:
        if n <= 0:
            return []
        self._ensure()
        assert self._m is not None and self._C is not None
        try:
            chol = np.linalg.cholesky(self._C + 1e-10 * np.eye(self.space.dim))
        except np.linalg.LinAlgError:  # pragma: no cover - numerical fallback
            self._C = np.eye(self.space.dim, dtype=float)
            chol = np.eye(self.space.dim)
        configs: list[dict] = []
        for _ in range(n):
            z = self._m + self._sigma * (chol @ self.rng.standard_normal(self.space.dim))
            z = np.clip(z, 0.0, 1.0)
            config = self.space.decode(z.tolist())
            self._pending.append((z, config))
            configs.append(config)
        return configs

    def tell(self, observations: Sequence[Observation]) -> None:
        observations = list(observations)
        super().tell(observations)
        if not observations:
            return
        self._ensure()
        assert self._m is not None and self._C is not None
        count = min(len(observations), len(self._pending))
        if count == 0:
            # A *prior replay*: ``Study`` rebuilds the optimizer every round with
            # ``prior=<all observations>`` and never calls ``tell`` itself, so
            # ``_pending`` is empty and the history used to be dropped entirely --
            # the mean/covariance then never moved and cmaes degenerated into
            # sampling its initial N(0.5, 0.1^2), although the router recommends it
            # for >=8-D smooth problems.  Feed the history through the same rank-mu
            # update instead (there are no pending z-values to pair with, so the
            # recorded configurations are encoded back into the unit cube).
            self._update_from_history(observations)
            return
        entries = self._pending[-count:]
        del self._pending[-count:]
        scored: list[tuple[float, np.ndarray]] = []
        for (z, _), observation in zip(entries, observations):
            scored.append(
                (float(observation.score(self.objective.name, self.objective.direction)), z)
            )
        scored.sort(key=lambda item: item[0], reverse=True)
        self._apply_rank_mu(scored)

    def _update_from_history(self, observations: Sequence[Observation]) -> None:
        """Rank-mu update from observations the optimizer did not propose itself."""
        scored: list[tuple[float, np.ndarray]] = []
        for observation in observations:
            if observation.y is None and not observation.objectives:
                continue
            z = np.clip(np.asarray(self.space.encode(observation.x), dtype=float), 0.0, 1.0)
            scored.append(
                (float(observation.score(self.objective.name, self.objective.direction)), z)
            )
        if not scored:
            return
        scored.sort(key=lambda item: item[0], reverse=True)
        self._apply_rank_mu(scored)

    def _apply_rank_mu(self, scored: list[tuple[float, np.ndarray]]) -> None:
        """One rank-mu + covariance/sigma step from scored (value, z) pairs."""
        assert self._m is not None and self._C is not None
        if len(scored) == 1:
            self._m = 0.9 * self._m + 0.1 * scored[0][1]
            return
        mu = max(1, len(scored) // 2)
        weights = np.asarray(
            [math.log(mu + 0.5) - math.log(index + 1) for index in range(mu)], dtype=float
        )
        weights /= float(np.sum(weights))
        best = np.vstack([item[1] for item in scored[:mu]])
        old_mean = self._m.copy()
        self._m = weights @ best
        steps = (best - old_mean) / max(self._sigma, _EPS)
        c_mu = float(self.params["c_mu"])
        self._C = (1.0 - c_mu) * self._C + c_mu * (steps.T @ (steps * weights[:, None]))
        self._C = 0.5 * (self._C + self._C.T) + 1e-10 * np.eye(self.space.dim)
        chi_n = math.sqrt(self.space.dim) * (
            1.0 - 1.0 / (4.0 * self.space.dim) + 1.0 / (21.0 * self.space.dim**2)
        )
        step_norm = float(np.linalg.norm(self._m - old_mean)) / max(
            self._sigma * chi_n, _EPS
        )
        self._sigma = float(
            np.clip(self._sigma * math.exp(0.2 * (step_norm - 1.0)), 1e-5, 2.0)
        )

    # -- state -------------------------------------------------------------
    def state_dict(self) -> dict:
        state = super().state_dict()
        state.update({
            "m": None if self._m is None else self._m.tolist(),
            "C": None if self._C is None else self._C.tolist(),
            "sigma": self._sigma,
        })
        return state

    def load_state_dict(self, state: Mapping) -> None:
        super().load_state_dict(state)
        m = state.get("m")
        c = state.get("C")
        if m is not None:
            self._m = np.asarray(m, dtype=float)
        if c is not None:
            self._C = np.asarray(c, dtype=float)
        if state.get("sigma") is not None:
            self._sigma = float(state["sigma"])