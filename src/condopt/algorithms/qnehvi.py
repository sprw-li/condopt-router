"""Noisy multi-objective BO (qNEHVI, 2 objectives).

qNEHVI extends EHVI to noisy observations by integrating over the uncertainty
of the *current Pareto front*: each Monte-Carlo baseline set draws objective
values from their observation-noise posteriors, rebuilds the front, and the
candidate's EHVI is averaged over those baselines (nested MC).  This is a
2-objective implementation reusing :mod:`condopt.pareto`'s exact partition.

Defaults follow research/qNEHVI_SPEC_V2.md: ``n_base=64``, ``n_y=128``.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from ..base import AlgorithmSpec, ParamSpec
from ..dataset import Observation
from ..pareto import (
    dhv_2d_batch,
    non_dominated_mask,
    partition_2d,
    signed_scores,
)
from ..registry import register
from .qehvi import QEHVI

_EPS = 1e-12


@register
class QNEHVI(QEHVI):
    """Nested-MC noisy EHVI for two objectives."""

    spec = AlgorithmSpec(
        name="qnehvi",
        family="model_based",
        summary="2 目标带噪 qNEHVI：对 Pareto 前沿做嵌套 MC，噪声越大越偏探索。",
        when="恰好 2 个目标、预算 ≥40、噪声 medium/high；低噪声用 qEHVI 更省算力。",
        params=QEHVI.spec.params
        + (
            ParamSpec(
                "n_base",
                "int",
                64,
                low=4,
                high=256,
                description="baseline（前沿不确定性）MC 样本数",
            ),
            ParamSpec(
                "n_y",
                "int",
                128,
                low=16,
                high=512,
                description="候选点后验 MC 样本数",
            ),
            ParamSpec(
                "noise_std",
                "float",
                0.0,
                low=0.0,
                high=1.0,
                description="观测噪声标准差（归一化前）；0 表示用 observation.objective_std",
            ),
        ),
        capabilities=QEHVI.spec.capabilities | {"noisy"},
        references=(
            "Daulton, Balandat & Bakshy, Parallel Bayesian Optimization of Multiple Noisy Objectives with Expected Hypervolume Improvement (qNEHVI), NeurIPS 2021",
        ),
    )

    def __init__(self, *args, **kwargs) -> None:
        self._noise_std = np.zeros((0, 2))
        self._noise_missing = False
        self._warned_noise_missing = False
        super().__init__(*args, **kwargs)

    def tell(self, observations: Sequence[Observation]) -> None:
        observations = list(observations)
        if not observations:
            return
        super().tell(observations)
        names = list(self.objective.objectives) or [self.objective.name]
        default = float(self.params["noise_std"])
        rows = []
        for observation in observations:
            # A single scalar cannot describe heteroscedastic objectives: yield might
            # have std 1.2 while cost has 0.05.  ``objectives_std`` supplies the
            # per-objective values; ``objective_std``/``noise_std`` remain the fallback.
            rows.append([
                observation.std_for(names[index], default) if index < len(names) else default
                for index in range(2)
            ])
        noise = np.asarray(rows, dtype=float)
        # ``super().tell`` also appends missing-objective rows as NaN; keep noise
        # aligned with self._Y.
        if len(self._noise_std) == 0:
            self._noise_std = noise
        else:
            self._noise_std = np.vstack([self._noise_std, noise])
        if len(self._noise_std) > len(self._Y):
            self._noise_std = self._noise_std[: len(self._Y)]
        # qNEHVI only differs from qEHVI when noise actually enters the nested
        # MC; warn instead of silently degrading (review v0.6 P0-2).
        if float(np.max(self._noise_std)) <= 0.0:
            self._noise_missing = True

    def _model_ask(self, n: int) -> list[dict]:
        if self._X is None or len(self._y) < 3:
            return self.space.sample_sobol(n, rng=self.rng)
        candidates, features = self._candidate_pool()
        if len(candidates) == 0:
            return self.space.sample_sobol(n, rng=self.rng)

        ref = self._signed_references()
        valid = np.all(np.isfinite(self._Y), axis=1)
        if not np.any(valid):
            raise ValueError("qNEHVI: no complete 2-objective observation available")
        X = self._X[valid]
        Y = self._Y[valid]
        noise = self._noise_std[valid]
        signed = signed_scores(Y, self.objective)
        if not np.any(np.all(signed > ref, axis=1)):
            raise ValueError(
                f"qNEHVI: no observation dominates objective.ref_point={self.objective.ref_point}"
            )

        n_base = int(self.params["n_base"])
        n_y = int(self.params["n_y"])
        if self._noise_missing and not self._warned_noise_missing:
            import warnings

            warnings.warn(
                "qNEHVI: no observation-noise information (objective_std / noise_std are 0); "
                "the nested MC reduces to the noiseless qEHVI - set noise_std or record "
                "objective_std to get the noisy behaviour",
                RuntimeWarning,
                stacklevel=2,
            )
            self._warned_noise_missing = True
        # Candidate posterior draws: (n_y, m_candidates, 2)
        ensemble = []
        for index in range(2):
            model = self._fit(X, Y[:, index])
            draw = self._sample_posterior_multi(model, features, n_y)  # (m, n_y)
            ensemble.append(draw.T)
        points = np.stack(ensemble, axis=-1)  # (n_y, m, 2)

        # Nested MC: each baseline draw is a possible *current front*.  BoTorch's
        # qNEHVI samples the posterior at ``X_baseline``; perturbing the observed
        # labels with i.i.d. noise instead ignores the model's uncertainty structure
        # (it cannot represent "this region is poorly explored").  Reviewed
        # externally as "baseline posterior = False"; implemented here.
        signs = np.asarray(
            [1.0 if direction == "max" else -1.0 for direction in self.objective.directions],
            dtype=float,
        )
        baseline = np.empty((n_base, len(X), 2), dtype=float)
        backup_noise = getattr(self, "_noise_var", None)
        for index in range(2):
            column = None
            if noise.ndim == 2 and noise.shape[1] > index:
                column = noise[:, index]
            self._noise_var = column
            model_index = self._fit(X, Y[:, index])
            draws = self._sample_posterior_multi(model_index, X, n_base)
            baseline[:, :, index] = np.asarray(draws, dtype=float).T
        self._noise_var = backup_noise
        baseline_signed = baseline * signs[None, None, :]

        acquisition = np.zeros(len(candidates))
        effective = 0
        for index in range(n_base):
            drawn = baseline_signed[index]
            mask = non_dominated_mask(drawn)
            boxes = partition_2d(drawn[mask], ref)
            if not boxes:
                continue
            acquisition += dhv_2d_batch(points, boxes, ref).mean(axis=0)
            effective += 1
        if effective == 0:
            raise ValueError("qNEHVI: no baseline draw produced a valid front above ref_point")
        acquisition /= effective

        scale = self._local_penalty_scale(self._fit(X, Y[:, 0]))
        chosen: list[int] = []
        chosen_units: list[np.ndarray] = []
        available = np.ones(len(candidates), dtype=bool)
        for _ in range(n):
            if not available.any():
                break
            scores = acquisition.copy()
            if chosen_units:
                distances = np.sum(
                    (candidates[:, None, :] - np.asarray(chosen_units)[None, :, :]) ** 2, axis=-1
                )
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
