"""Multi-fidelity ParEGO (``mfparego``): random scalarization + fidelity GP.

Combines the two mechanisms that previously had no shared implementation:

* ParEGO's per-iteration augmented-Tchebycheff scalarization (so a multi-objective
  problem is handled as a rotating sequence of single-objective ones), and
* ``mfbo``'s fidelity-as-GP-feature surrogate with ``acquisition / cost^alpha``
  selection of both condition and fidelity.

This gives the multi-fidelity multi-objective regime a second candidate besides
``momfbo`` (which uses a scalarization + fidelity GP too, but with a different
exploration/feasibility treatment).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import numpy as np

from ..base import ParamSpec
from ..dataset import Observation
from ..pareto import observation_values
from ..registry import register
from .mfbo import MultiFidelityBO

_EPS = 1e-12


@register
class MultiFidelityParEGO(MultiFidelityBO):
    """ParEGO scalarization on top of the multi-fidelity GP stack."""

    spec = replace(
        MultiFidelityBO.spec,
        name="mfparego",
        summary=(
            "多保真 + 多目标：每轮随机加权（增广 Tchebycheff）标量化，再把 fidelity 作为 GP "
            "输入，按 EI/成本^α 同时选条件与保真度。"
        ),
        when="2+ 个目标且存在低成本粗筛（低保真仿真/短时试跑）时；momfbo 之外的备选。",
        params=MultiFidelityBO.spec.params
        + (
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
        ),
        capabilities=MultiFidelityBO.spec.capabilities | {"multi_objective"},
        references=(
            "Knowles, ParEGO, IEEE TEC 2006",
            "Kandasamy et al., Multi-fidelity Bayesian Optimisation with Continuous Approximations (BOCA), ICML 2017",
        ),
    )

    def __init__(self, *args, **kwargs) -> None:
        self._Ymatrix: np.ndarray = np.empty((0, 0))
        self._Ystd: np.ndarray = np.empty((0, 0))
        self._bounds: dict[str, tuple[float, float]] = {}
        # Valid history: (x, fidelity, constraints, signed objective values).
        # Rebuilt into the parent arrays on every tell so observations with an
        # incomplete objective vector can never desynchronise ``_X`` from ``_y``.
        self._history: list[tuple[dict, float, dict, np.ndarray]] = []
        super().__init__(*args, **kwargs)

    # -- data -------------------------------------------------------------
    def tell(self, observations: Sequence[Observation]) -> None:
        observations = list(observations)
        for observation in observations:
            try:
                values = observation_values(observation, self.objective)
            except ValueError:
                continue  # incomplete vector: keep it in the study, not the model
            self._history.append(
                (
                    dict(observation.x),
                    float(observation.fidelity),
                    dict(observation.constraints or {}),
                    np.asarray(values, dtype=float),
                )
            )
        if observations:
            super().tell(observations)  # keeps ``t`` and space bookkeeping
        self._rebuild()

    def _rebuild(self) -> None:
        if not self._history:
            return
        constraint_names = tuple(getattr(self.objective, "constraints", ()) or ())
        self._X = np.vstack(
            [self._features(x, fidelity) for x, fidelity, _, _ in self._history]
        )
        self._units = np.vstack([self.space.encode(x) for x, _, _, _ in self._history])
        self._fidelity = np.asarray([fidelity for _, fidelity, _, _ in self._history])
        self._Ymatrix = np.vstack([values for _, _, _, values in self._history])
        self._y = self._Ymatrix[:, 0].copy()
        if constraint_names:
            self._C = np.asarray(
                [
                    [float(constraints.get(name, np.nan)) for name in constraint_names]
                    for _, _, constraints, _ in self._history
                ],
                dtype=float,
            )
        else:
            self._C = np.empty((len(self._history), 0))

    # -- scalarization ----------------------------------------------------
    def _window(self) -> tuple[np.ndarray, np.ndarray]:
        m = self.objective.n_objectives
        if len(self._bounds) == m:
            lows = np.asarray([self._bounds[name][0] for name in self.objective.objectives])
            highs = np.asarray([self._bounds[name][1] for name in self.objective.objectives])
            return lows, highs
        explicit = all(
            name in self.objective.objective_bounds for name in self.objective.objectives
        )
        if explicit:
            lows = np.asarray(
                [self.objective.objective_bounds[name]["low"] for name in self.objective.objectives]
            )
            highs = np.asarray(
                [self.objective.objective_bounds[name]["high"] for name in self.objective.objectives]
            )
        else:
            freeze_n = max(3, 2 * m)
            rows = (
                self._Ymatrix
                if len(self._Ymatrix) < freeze_n
                else self._Ymatrix[:freeze_n]
            )
            lows = np.min(rows, axis=0)
            highs = np.max(rows, axis=0)
            if len(self._Ymatrix) < freeze_n:
                return lows, highs
        self._bounds = {
            name: (float(lows[index]), float(highs[index]))
            for index, name in enumerate(self.objective.objectives)
        }
        return lows, highs

    def _normalized(self) -> np.ndarray:
        lows, highs = self._window()
        out = np.zeros_like(self._Ymatrix, dtype=float)
        for index in range(self._Ymatrix.shape[1]):
            span = highs[index] - lows[index]
            if span < _EPS:
                continue
            if self.objective.directions[index] == "max":
                out[:, index] = (self._Ymatrix[:, index] - lows[index]) / span
            else:
                out[:, index] = (highs[index] - self._Ymatrix[:, index]) / span
        return out

    def _scalar_values(self, weights: np.ndarray) -> np.ndarray:
        weighted = self._normalized() * weights
        return np.max(weighted, axis=1) + float(self.params["rho"]) * np.sum(weighted, axis=1)

    # -- propose ----------------------------------------------------------
    def _propose(
        self,
        batch: Sequence[tuple[dict, float]],
        force_full: bool = False,
        min_feasibility: float = 0.0,
    ) -> tuple[dict, float]:
        if len(self._Ymatrix) >= 2 and self._X is not None and len(self._y) >= 3:
            m = self.objective.n_objectives
            weights = self.rng.dirichlet(np.full(m, float(self.params["weight_alpha"])))
            scalar = self._scalar_values(weights)
            if len(scalar) == len(self._y):
                self._y = scalar
        return super()._propose(
            batch, force_full=force_full, min_feasibility=min_feasibility
        )
