"""Constrained Bayesian optimization (EI under feasibility probability).

The objective is optimized subject to observed constraints such as
``T_max <= 110`` or ``dissolved >= 0.5``.  Two regimes:

* **no feasible point yet** -- maximize the posterior probability of
  feasibility (PoF), i.e. find *any* safe condition first;
* **at least one feasible point** -- maximize the constrained expected
  improvement ``E[max(0, f(x) - f*) * 1{all constraints feasible}]`` estimated
  by Monte Carlo over independent GP posteriors (an independent-GP
  approximation to the Letham-style noisy-constraint EI; observation noise is
  not modelled explicitly, so aggregate replicates before ``tell`` when the
  measurement noise is large).

This is still an independent-GP approximation (no joint objective/constraint
posterior), which is adequate for a handful of constraints and much more
robust than post-hoc filtering with noisy measurements.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from ..base import AlgorithmSpec, Objective, ParamSpec
from ..dataset import Observation
from ..registry import register
from .bayes import GaussianProcessBO

_INFEASIBLE = 1e300


@register
class ConstrainedEI(GaussianProcessBO):
    """GP-BO with feasibility-aware acquisition (PoF then constrained EI)."""

    spec = AlgorithmSpec(
        name="constrained",
        family="model_based",
        summary="约束贝叶斯优化：先找可行点（PoF），再做带可行性的期望改进（EI_c，MC 估计）。",
        when="有安全/可行性边界（温度上限、溶解度、放热、成本上限）且实验昂贵/带噪声时。",
        params=GaussianProcessBO.spec.params
        + (
            ParamSpec(
                "mc_draws",
                "int",
                128,
                low=16,
                high=512,
                description="约束 EI 的 Monte Carlo 采样数；越大越稳越慢",
            ),
            ParamSpec(
                "constraint_margin",
                "float",
                0.0,
                low=0.0,
                high=10.0,
                description="安全裕量：<= 约束按 value-margin 判可行，>= 约束按 value+margin",
            ),
            ParamSpec(
                "feasibility_threshold",
                "float",
                0.02,
                low=0.0,
                high=0.5,
                description="候选点的最低可行性概率；低于此值不参与采集",
            ),
        ),
        capabilities=GaussianProcessBO.spec.capabilities | {"constraints"},
        references=(
            "Gardner et al., Bayesian Optimization with Inequality Constraints (EI-CDF), ICML 2014",
            "Letham, Karrer, Ottoni & Bakshy, Constrained Bayesian Optimization with Noisy Experiments, Bayesian Analysis 2019",
        ),
    )

    def __init__(self, *args, **kwargs) -> None:
        self._constraint_names: tuple[str, ...] = ()
        self._C = np.empty((0, 0))
        self._feasible = np.zeros(0, dtype=bool)
        super().__init__(*args, **kwargs)

    @property
    def objective(self) -> Objective:
        return self._objective

    @objective.setter
    def objective(self, value: Objective) -> None:
        self._objective = value
        self._constraint_names = tuple(value.constraints)
        missing = [name for name in self._constraint_names if name not in value.constraint_limits]
        if self._constraint_names and missing:
            raise ValueError(
                f"constrained optimizer needs constraint_limits for {missing}; "
                'e.g. {"T_max": {"op": "<=", "value": 110.0}}'
            )
        if value.is_multi_objective and self._constraint_names:
            raise ValueError("constrained + multi-objective is not supported yet")

    # -- data -------------------------------------------------------------
    def tell(self, observations: Sequence[Observation]) -> None:
        observations = list(observations)
        super().tell(observations)
        if not observations:
            return
        rows = []
        for obs in observations:
            rows.append(
                [float(obs.constraints.get(name, np.nan)) for name in self._constraint_names]
            )
        matrix = np.asarray(rows, dtype=float)
        self._C = matrix if len(self._C) == 0 else np.vstack([self._C, matrix])
        self._feasible = self._feasibility(self._C)

    def _feasibility(self, matrix: np.ndarray) -> np.ndarray:
        margin = float(self.params["constraint_margin"])
        feasible = np.all(~np.isnan(matrix), axis=1) if matrix.size else np.zeros(0, dtype=bool)
        for index, name in enumerate(self._constraint_names):
            spec = self.objective.constraint_limits[name]
            threshold = float(spec.get("value", 0.0))
            column = matrix[:, index]
            if spec.get("op", "<=") == "<=":
                feasible &= column <= threshold - margin
            else:
                feasible &= column >= threshold + margin
        return feasible

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
            out.extend(self._constrained_ask(n, out))
        return out

    def _constrained_ask(self, n: int, batch: Sequence[dict]) -> list[dict]:
        if self._X is None or len(self._y) < 3:
            return self.space.sample_sobol(n, rng=self.rng)
        candidates, features = self._candidate_pool()
        if len(candidates) == 0:
            return self.space.sample_sobol(n, rng=self.rng)

        objective_model = self._fit(self._X, self._y)
        constraint_models = [self._fit(self._X, self._C[:, j]) for j in range(len(self._constraint_names))]
        draws = int(self.params["mc_draws"])
        objective_draws = self._sample_posterior_multi(objective_model, features, draws)
        feasible = np.ones_like(objective_draws, dtype=bool)
        for index, model in enumerate(constraint_models):
            constraint_draws = self._sample_posterior_multi(model, features, draws)
            spec = self.objective.constraint_limits[self._constraint_names[index]]
            threshold = float(spec.get("value", 0.0))
            margin = float(self.params["constraint_margin"])
            if spec.get("op", "<=") == "<=":
                feasible &= constraint_draws <= threshold - margin
            else:
                feasible &= constraint_draws >= threshold + margin
        feasibility = feasible.mean(axis=1)
        feasible_observations = self._feasible if len(self._feasible) else np.zeros(0, dtype=bool)

        if bool(np.any(feasible_observations)):
            best = float(np.max(self._y[feasible_observations]))
            improvement = np.maximum(objective_draws - best, 0.0)
            acquisition = (improvement * feasible).mean(axis=1)
        else:
            # Phase A: no safe point yet -> find feasibility first.
            acquisition = feasibility
        acquisition = np.asarray(acquisition, dtype=float)
        threshold = float(self.params["feasibility_threshold"])
        acquisition = np.where(feasibility >= threshold, acquisition, -np.inf)
        if not np.any(np.isfinite(acquisition)):
            acquisition = feasibility

        scale = self._local_penalty_scale(objective_model)
        chosen: list[int] = []
        chosen_units: list[np.ndarray] = []
        available = np.ones(len(candidates), dtype=bool)
        batch_units = [self.space.encode(config) for config in batch]
        for _ in range(n):
            if not available.any():
                break
            scores = acquisition.copy()
            if chosen_units or batch_units:
                centers = np.asarray(chosen_units + batch_units, dtype=float)
                distances = np.sum((candidates[:, None, :] - centers[None, :, :]) ** 2, axis=-1)
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
