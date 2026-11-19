"""Constrained Expected Hypervolume Improvement (feasibility-aware qEHVI).

Most real multi-objective campaigns also have a hard feasibility boundary: a
safety temperature limit, a solubility floor, a cost cap.  Optimizing the
unconstrained Pareto front spends half of the budget on conditions that can
never be run, and the "best" reported trade-off is then infeasible.  This
module adds the missing combination of the two existing building blocks:

* the exact 2D EHVI partition + Monte Carlo expectation from
  :class:`~condopt.algorithms.qehvi.QEHVI`;
* the two-phase feasibility handling from
  :class:`~condopt.algorithms.constrained.ConstrainedEI`.

Acquisition
-----------
The constrained EHVI is the expectation of the dominance hypervolume
improvement over the *joint* posterior of the two objective GPs and the
constraint GPs::

    cEHVI(x) = E[ DHV(f(x)) * 1{all g_j(x) <= 0} ]

which we estimate with independent-GP Monte Carlo draws (the same
independent-GP approximation as ``constrained``; joint objective/constraint
correlation is not modelled, so aggregate replicates before ``tell`` when the
measurement noise is large).

Two phases, exactly like ``constrained``:

1. **no feasible observation yet** -> maximize the posterior feasibility
   probability (PoF), so the campaign finds *any* safe condition first;
2. **at least one feasible point** -> maximize ``cEHVI`` above, i.e. only
   hypervolume improvements inside the feasible region are rewarded.

The Pareto partition is rebuilt from feasible observations only: an
infeasible point must not "block" volume that can never be used.

References: Daulton et al., qEHVI (NeurIPS 2020); Gardner et al., EI-CDF
(ICML 2014); Letham et al., Constrained Bayesian Optimization with Noisy
Experiments (Bayesian Analysis 2019).
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from ..base import AlgorithmSpec, Objective, ParamSpec
from ..dataset import Observation
from ..pareto import dhv_2d_batch, signed_scores
from ..registry import register
from .qehvi import QEHVI


@register
class ConstrainedQEHVI(QEHVI):
    """2-objective qEHVI with a feasibility-aware (two-phase) acquisition."""

    spec = AlgorithmSpec(
        name="cqehvi",
        family="model_based",
        summary=(
            "2 目标 + 硬约束：可行域内精确 EHVI（MC 估计），无可行点时先最大化可行性概率 PoF，"
            "有可行点后只奖励可行域内的超体积改进。"
        ),
        when=(
            "恰好 2 个目标且同时有安全/可行性硬约束（温度上限、溶解度、成本上限），"
            "噪声低/中；比 ParEGO+事后过滤更省实验。"
        ),
        params=QEHVI.spec.params
        + (
            ParamSpec(
                "constraint_margin",
                "float",
                0.0,
                low=0.0,
                high=10.0,
                description="安全裕量：<= 约束按 value-margin 判可行，>= 约束按 value+margin",
            ),
            ParamSpec(
                "batch_strategy",
                "choice",
                "local_penalty",
                choices=("local_penalty", "kriging_believer", "constant_liar"),
                description="批内选点策略；kriging_believer 用后验均值、constant_liar 用悲观常数做伪观测重算",
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
        capabilities=QEHVI.spec.capabilities | {"constraints"},
        references=(
            "Daulton, Balandat & Bakshy, Differentiable Expected Hypervolume Improvement for Parallel Multi-Objective Bayesian Optimization (qEHVI), NeurIPS 2020",
            "Gardner, Kusner, Xu, Weinberger & Cunningham, Bayesian Optimization with Inequality Constraints (EI-CDF), ICML 2014",
            "Letham, Karrer, Ottoni & Bakshy, Constrained Bayesian Optimization with Noisy Experiments, Bayesian Analysis 2019",
        ),
    )

    def __init__(self, *args, **kwargs) -> None:
        self._constraint_names: tuple[str, ...] = ()
        self._C = np.empty((0, 0))
        self._feasible = np.zeros(0, dtype=bool)
        super().__init__(*args, **kwargs)
        if not self._constraint_names:
            raise ValueError(
                "cqehvi needs at least one constraint on Objective.constraints; "
                "use qehvi for unconstrained 2-objective problems"
            )

    # -- objective / constraints -----------------------------------------
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
                f"cqehvi needs constraint_limits for {missing}; "
                'e.g. {"T_max": {"op": "<=", "value": 110.0}}'
            )

    # -- data -------------------------------------------------------------
    def tell(self, observations: Sequence[Observation]) -> None:
        observations = list(observations)
        super().tell(observations)
        if not observations:
            return
        rows = [
            [float(obs.constraints.get(name, np.nan)) for name in self._constraint_names]
            for obs in observations
        ]
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
    def _acquisition(self, candidates, features, X, Y, C):
        """Constrained EHVI acquisition for the given (possibly augmented) data."""
        draws = int(self.params["mc_draws"])
        ref = self._signed_references()
        complete = np.all(np.isfinite(Y), axis=1)
        if not np.any(complete):
            raise ValueError("cqehvi: no complete 2-objective observation")
        Xc, Yc = X[complete], Y[complete]
        points, objective_models = self._posterior_draws(Xc, Yc, features, draws)

        feasible = np.ones(points.shape[:2], dtype=bool)
        tolerance = float(self.params["constraint_margin"])
        constraint_models = []
        for index, name in enumerate(self._constraint_names):
            column = C[:, index]
            usable = np.isfinite(column)
            if not np.any(usable):
                raise ValueError(f"cqehvi: constraint {name!r} has no measured value")
            model = self._fit(X[usable], column[usable])
            constraint_models.append(model)
            constraint_draws = self._sample_posterior_multi(model, features, draws).T
            spec = self.objective.constraint_limits[name]
            threshold = float(spec.get("value", 0.0))
            if spec.get("op", "<=") == "<=":
                feasible &= constraint_draws <= threshold - tolerance
            else:
                feasible &= constraint_draws >= threshold + tolerance
        probability = feasible.mean(axis=0)

        feasible_obs = self._feasibility(C)
        feasible_rows = complete & feasible_obs
        if bool(np.any(feasible_rows)):
            # Phase B: feasible EHVI -- the partition only contains feasible
            # observations, so infeasible points never block usable volume.
            front = signed_scores(Y[feasible_rows], self.objective)
            boxes = self._front_boxes(front, ref, label="cqehvi")
            improvement = dhv_2d_batch(points, boxes, ref)  # (S, m)
            acquisition = (improvement * feasible).mean(axis=0)
        else:
            # Phase A: no safe condition yet -> find one before optimizing.
            acquisition = probability

        acquisition = np.asarray(acquisition, dtype=float)
        threshold = float(self.params["feasibility_threshold"])
        masked = np.where(probability >= threshold, acquisition, -np.inf)
        if np.any(np.isfinite(masked)):
            acquisition = masked
        else:
            acquisition = probability
        return acquisition, probability, objective_models, constraint_models

    def _model_ask(self, n: int) -> list[dict]:
        if self._X is None or len(self._y) < 3:
            return self.space.sample_sobol(n, rng=self.rng)
        candidates, features = self._candidate_pool()
        if len(candidates) == 0:
            return self.space.sample_sobol(n, rng=self.rng)
        if len(self._feasible) != len(self._Y):
            raise ValueError("cqehvi: constraint rows and objective rows are out of sync")

        strategy = str(self.params.get("batch_strategy", "local_penalty"))
        X, Y, C = self._X, self._Y, self._C
        if strategy not in ("kriging_believer", "constant_liar") or n == 1:
            acquisition, _prob, models, _cons = self._acquisition(candidates, features, X, Y, C)
            scale = self._local_penalty_scale(models[0])
            return self._greedy_configs(acquisition, candidates, scale, n)

        # Sequential batch fantasies: after each pick, append a pseudo-observation
        # and re-fit/re-score.  kriging_believer uses the posterior mean;
        # constant_liar uses a pessimistic constant (worst feasible objective
        # vector) to discourage piling points onto the same region.
        feasible_rows = np.all(np.isfinite(Y), axis=1) & self._feasibility(C)
        if strategy == "constant_liar":
            if bool(np.any(feasible_rows)):
                lie_objectives = np.min(Y[feasible_rows], axis=0)
            else:
                lie_objectives = np.zeros(2)
        else:
            lie_objectives = None
        chosen: list[int] = []
        chosen_units: list[np.ndarray] = []
        available = np.ones(len(candidates), dtype=bool)
        Xa, Ya, Ca = X.copy(), Y.copy(), C.copy()
        for step in range(n):
            acquisition, probability, objective_models, constraint_models = self._acquisition(
                candidates, features, Xa, Ya, Ca
            )
            scale = self._local_penalty_scale(objective_models[0])
            scores = acquisition.copy()
            if chosen_units:
                distances = np.sum(
                    (candidates[:, None, :] - np.asarray(chosen_units)[None, :, :]) ** 2, axis=-1
                )
                penalty = np.prod(1.0 - np.exp(-distances / (2.0 * scale * scale)), axis=1)
                scores = scores * penalty
            scores = np.where(available, scores, -np.inf)
            if not np.any(np.isfinite(scores)):
                scores = np.where(available, probability, -np.inf)
            pick = int(np.argmax(scores))
            chosen.append(pick)
            chosen_units.append(candidates[pick])
            available[pick] = False
            if step < n - 1:
                feature = features[pick].reshape(1, -1)
                if strategy == "constant_liar":
                    pseudo_objectives = np.asarray(lie_objectives, dtype=float)
                else:
                    pseudo_objectives = np.asarray(
                        [float(model.predict(feature)[0]) for model in objective_models],
                        dtype=float,
                    )
                mu_constraints = [float(model.predict(feature)[0]) for model in constraint_models]
                Xa = np.vstack([Xa, features[pick]])
                Ya = np.vstack([Ya, pseudo_objectives])
                Ca = np.vstack([Ca, mu_constraints])
        configs = [self.space.decode(candidates[i]) for i in chosen]
        while len(configs) < n:
            configs.extend(self.space.sample_sobol(n - len(configs), rng=self.rng))
        return configs

    # -- state ------------------------------------------------------------
    def state_dict(self) -> dict:
        state = super().state_dict()
        state["constraint_names"] = list(self._constraint_names)
        return state

    def load_state_dict(self, state: Mapping) -> None:
        super().load_state_dict(state)
        names = state.get("constraint_names")
        if names:
            self._constraint_names = tuple(str(name) for name in names)
