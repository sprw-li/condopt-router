"""Space-filling samplers: random, Sobol, Latin hypercube, grid.

These are the cheap, model-free workhorses.  They are not "baselines to
throw away": for very small budgets, very high dimensions, categorical
spaces or coarse discrete spaces they often beat model-based methods,
and Bayesian optimization uses them for the initial design.

Registered algorithms
---------------------
* ``random``  -- i.i.d. uniform sampling (random search).
* ``sobol``   -- scrambled Sobol low-discrepancy sequence.
* ``lhs``     -- Latin hypercube sampling.
* ``grid``    -- full factorial design, coarsened to a point budget.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from ..base import AlgorithmSpec, Optimizer, ParamSpec
from ..registry import register
from ..space import SearchSpace


@register
class RandomSearch(Optimizer):
    """Uniform random sampling in the condition space."""

    spec = AlgorithmSpec(
        name="random",
        family="sampling",
        summary="均匀随机采样（random search）：无模型、极稳健，高维/极小预算的默认基线。",
        when="维度很高 (>20)、预算很小 (<10)、空间含大量分类变量、或不确定目标是否光滑时。",
        params=(
            ParamSpec("dedupe", "bool", True, description="是否尽量避开已经评价过的配置"),
            ParamSpec(
                "max_attempts_factor",
                "int",
                20,
                low=1,
                high=1000,
                description="去重时的尝试倍数（仅离散空间容易撞点）",
            ),
        ),
        capabilities=frozenset({"batch", "mixed", "categorical", "high_dim", "cheap", "model_free"}),
        references=(
            "Bergstra & Bengio, Random Search for Hyper-Parameter Optimization, JMLR 2012",
        ),
    )

    def __init__(self, *args, **kwargs) -> None:
        self._seen: set[tuple] = set()
        super().__init__(*args, **kwargs)

    def tell(self, observations: Sequence) -> None:
        super().tell(observations)
        for obs in observations:
            self._seen.add(self._key(obs.x))

    def _key(self, config: Mapping) -> tuple:
        return tuple(np.round(self.space.encode(config), 9).tolist())

    def ask(self, n: int = 1) -> list[dict]:
        if n <= 0:
            return []
        out: list[dict] = []
        attempts = 0
        limit = n * int(self.params["max_attempts_factor"])
        while len(out) < n and attempts < limit:
            attempts += 1
            config = self.space.sample_uniform(1, self.rng)[0]
            if self.params["dedupe"] and self._key(config) in self._seen:
                continue
            out.append(config)
            self._seen.add(self._key(config))
        while len(out) < n:  # tiny discrete space: allow duplicates rather than hang
            out.append(self.space.sample_uniform(1, self.rng)[0])
        return out


class _SequenceOptimizer(Optimizer):
    """Shared logic for algorithms that walk a deterministic sequence."""

    def __init__(self, *args, **kwargs) -> None:
        self._asked = 0
        super().__init__(*args, **kwargs)

    def state_dict(self) -> dict:
        state = super().state_dict()
        state["asked"] = self._asked
        return state

    def load_state_dict(self, state: Mapping) -> None:
        super().load_state_dict(state)
        self._asked = int(state.get("asked", 0))


@register
class SobolSearch(_SequenceOptimizer):
    """Scrambled Sobol low-discrepancy sequence."""

    spec = AlgorithmSpec(
        name="sobol",
        family="sampling",
        summary="Sobol 低差异序列：空间填充比随机更均匀，适合批量初始设计与中等维度。",
        when="需要均匀覆盖空间（初始设计、批量并行实验）、预算 8–50、维度 ≤20。",
        params=(
            ParamSpec("scramble", "bool", True, description="是否使用随机化（scrambled）Sobol"),
            ParamSpec(
                "dedupe",
                "bool",
                True,
                description="是否避开已经评价过的配置（序列重启时可能撞点）",
            ),
        ),
        capabilities=frozenset({"batch", "mixed", "categorical", "space_filling", "cheap", "model_free"}),
        references=(
            "Sobol, On the distribution of points in a cube and the approximate evaluation of integrals, 1967",
            "scipy.stats.qmc.Sobol documentation",
        ),
    )

    def __init__(self, *args, **kwargs) -> None:
        self._sampler = None
        self._seen: set[tuple] = set()
        super().__init__(*args, **kwargs)

    def _ensure_sampler(self):
        if self._sampler is None:
            from scipy.stats import qmc

            self._sampler = qmc.Sobol(
                d=self.space.dim,
                scramble=bool(self.params["scramble"]),
                seed=self.seed,
            )
        return self._sampler

    def tell(self, observations: Sequence) -> None:
        super().tell(observations)
        for obs in observations:
            self._seen.add(tuple(np.round(self.space.encode(obs.x), 9).tolist()))

    def ask(self, n: int = 1) -> list[dict]:
        if n <= 0:
            return []
        import warnings

        sampler = self._ensure_sampler()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            units = sampler.random(max(n, 1))
        configs = [self.space.decode(u) for u in units]
        self._asked += len(configs)
        if self.params["dedupe"]:
            out: list[dict] = []
            for cfg in configs:
                key = tuple(np.round(self.space.encode(cfg), 9).tolist())
                if key in self._seen:
                    continue
                self._seen.add(key)
                out.append(cfg)
            return out
        return configs

    def state_dict(self) -> dict:
        state = super().state_dict()
        state["asked"] = self._asked
        return state

    def load_state_dict(self, state: Mapping) -> None:
        super().load_state_dict(state)
        self._asked = int(state.get("asked", 0))
        from scipy.stats import qmc

        self._sampler = qmc.Sobol(
            d=self.space.dim,
            scramble=bool(self.params["scramble"]),
            seed=self.seed,
        )
        if self._asked:
            self._sampler.fast_forward(self._asked)


@register
class LatinHypercube(_SequenceOptimizer):
    """Latin hypercube sampling."""

    spec = AlgorithmSpec(
        name="lhs",
        family="sampling",
        summary="拉丁超立方：每维分层抽样，单批内边缘分布均匀，适合一次性批量实验。",
        when="一次性给一大批（8–48）候选、空间维度 ≤15、希望每维都被覆盖。",
        params=(
            ParamSpec(
                "optimization",
                "choice",
                "random-cd",
                choices=("none", "random-cd", "lloyd"),
                description="scipy 的 LHS 优化准则；random-cd 降低相关性，lloyd 更均匀但更慢",
            ),
        ),
        capabilities=frozenset({"batch", "mixed", "categorical", "space_filling", "cheap", "model_free"}),
        references=(
            "McKay, Beckman & Conover, A Comparison of Three Methods for Selecting Values of Input Variables, Technometrics 1979",
            "scipy.stats.qmc.LatinHypercube documentation",
        ),
    )

    def ask(self, n: int = 1) -> list[dict]:
        if n <= 0:
            return []
        opt = self.params["optimization"]
        return self.space.sample_lhs(
            max(n, 1), rng=self.rng, optimization=None if opt == "none" else opt
        )


@register
class GridSearch(Optimizer):
    """Full factorial grid over a coarsened discrete design."""

    spec = AlgorithmSpec(
        name="grid",
        family="sampling",
        summary="网格/析因设计：按层枚举全部组合，适合少量离散或低维粗粒度条件。",
        when="总组合数 ≤ ~200 且以分类/整数条件为主，需要穷举或做效应对比时。",
        params=(
            ParamSpec("max_points", "int", 200, low=2, high=100000, description="最多枚举的点数"),
            ParamSpec("levels", "int", 5, low=2, high=20, description="连续/整数变量每维取几层"),
        ),
        capabilities=frozenset({"batch", "mixed", "categorical", "exhaustive", "cheap", "model_free"}),
        references=(
            "Montgomery, Design and Analysis of Experiments (factorial designs)",
        ),
    )

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._grid: list[dict] = []
        self._index = 0

    def _ensure_grid(self) -> list[dict]:
        if not self._grid:
            self._grid = self.space.sample_grid(
                max_points=int(self.params["max_points"]),
                levels=int(self.params["levels"]),
            )
        return self._grid

    def ask(self, n: int = 1) -> list[dict]:
        if n <= 0:
            return []
        grid = self._ensure_grid()
        out: list[dict] = []
        for _ in range(n):
            out.append(grid[self._index % len(grid)])
            self._index += 1
        return out

    def state_dict(self) -> dict:
        state = super().state_dict()
        state["index"] = self._index
        state["grid_size"] = len(self._grid)
        return state

    def load_state_dict(self, state: Mapping) -> None:
        super().load_state_dict(state)
        self._index = int(state.get("index", 0))
