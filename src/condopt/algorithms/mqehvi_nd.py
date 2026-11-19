"""Multi-fidelity, multi-objective MC-EHVI (``mqehvi_nd``).

Hypervolume-based multi-fidelity BO for 2–4 objectives: the fidelity level is
fed to every per-objective GP as an extra input feature, and each round scores
``(condition, fidelity)`` pairs by ``E[hypervolume improvement] / cost(s)**alpha``.
A configurable fraction of proposals is forced to full fidelity so the delivered
recommendation rests on real measurements.

This complements the scalarization-based ``momfbo`` / ``mfparego`` with a
hypervolume-based option for the same regime.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping, Sequence

import numpy as np

from ..base import ParamSpec
from ..dataset import Observation
from ..pareto import non_dominated_mask, observation_values
from ..registry import register
from .ehvi_nd import MultiObjectiveEHVI

_EPS = 1e-12


@register
class MultiFidelityEHVI(MultiObjectiveEHVI):
    """Fidelity-aware MC-EHVI for 2–4 objectives."""

    supports_fidelity = True

    spec = replace(
        MultiObjectiveEHVI.spec,
        name="mqehvi_nd",
        summary=(
            "多保真 + 2–4 目标 MC-EHVI：fidelity 作为逐目标 GP 输入，"
            "按 E[HV 改进]/成本^α 同时选条件与保真度。"
        ),
        when="2–4 个目标且有低保真粗筛通道、预算中等以上时；momfbo/mfparego 的 HV 型备选。",
        params=MultiObjectiveEHVI.spec.params
        + (
            ParamSpec(
                "cost_exponent",
                "float",
                1.0,
                low=0.0,
                high=3.0,
                description="成本模型 cost(s)=s**exp；1 表示与 fidelity 线性",
            ),
            ParamSpec(
                "fidelity_levels",
                "str",
                "0.25,0.5,1.0",
                description="可选的保真度档位（逗号分隔，0–1，必须含 1.0）",
            ),
            ParamSpec(
                "fidelity_costs",
                "str",
                "",
                description="可选每档真实成本，如 '0.25:0.05,0.5:0.2,1.0:1.0'；空=用 cost_exponent",
            ),
            ParamSpec(
                "min_full_fraction",
                "float",
                0.25,
                low=0.0,
                high=1.0,
                description="至少这么多比例的提议点使用全保真",
            ),
        ),
        capabilities=MultiObjectiveEHVI.spec.capabilities | {"multi_fidelity"},
        references=(
            "Daulton, Balandat & Bakshy, Differentiable Expected Hypervolume Improvement (qEHVI), NeurIPS 2020",
            "Kandasamy et al., Multi-fidelity Bayesian Optimisation with Continuous Approximations (BOCA), ICML 2017",
        ),
    )

    def __init__(self, *args, **kwargs) -> None:
        self._fidelity = np.zeros(0)
        self.last_cost_trimmed = 0
        self.last_full_requested = 0
        self.last_full_returned = 0
        super().__init__(*args, **kwargs)

    # -- feature space -----------------------------------------------------
    def _model_feature_dim(self) -> int:
        return int(self.space.feature_dim) + 1

    def _features(self, config: Mapping, fidelity: float) -> np.ndarray:
        return np.concatenate([self.space.features(config), [float(fidelity)]])

    # -- fidelity levels ---------------------------------------------------
    def _levels(self) -> list[float]:
        levels = []
        for chunk in str(self.params["fidelity_levels"]).split(","):
            try:
                value = float(chunk.strip())
            except ValueError:
                continue
            if 0.0 < value <= 1.0:
                levels.append(value)
        if 1.0 not in levels:
            levels.append(1.0)
        return sorted(set(levels))

    def _cost(self, level: float) -> float:
        table: dict[float, float] = {}
        for chunk in str(self.params.get("fidelity_costs", "") or "").split(","):
            if ":" not in chunk:
                continue
            key, value = chunk.split(":", 1)
            try:
                table[float(key.strip())] = float(value.strip())
            except ValueError:
                continue
        if table:
            level = float(level)
            if level in table:
                return float(table[level])
            keys = sorted(table)
            if level < keys[0] or level > keys[-1]:
                raise ValueError(
                    f"fidelity_costs has no cost for level {level:.3g} (have {keys})"
                )
            upper = min(key for key in keys if key > level)
            lower = max(key for key in keys if key < level)
            ratio = (level - lower) / (upper - lower)
            return float(table[lower] + ratio * (table[upper] - table[lower]))
        return float(level) ** float(self.params["cost_exponent"])

    # -- data --------------------------------------------------------------
    def tell(self, observations: Sequence[Observation]) -> None:
        # Full override: the parent stacks fidelity-free features, which would
        # desynchronise ``_X`` from the augmented GP input width.
        observations = list(observations)
        for observation in observations:
            self.space.validate_config(observation.x)
        self.t += len(observations)
        if not observations:
            return
        from ..pareto import observation_values as _observation_values

        features = np.vstack(
            [self._features(o.x, float(o.fidelity)) for o in observations]
        )
        units = np.vstack([self.space.encode(o.x) for o in observations])
        rows, constraints, noise = [], [], []
        for observation in observations:
            try:
                rows.append(_observation_values(observation, self.objective))
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
        fidelity = np.asarray([o.fidelity for o in observations], dtype=float)
        if self._X is None:
            self._X, self._Y, self._C = features, matrix, c_matrix
            self._noise, self._units, self._fidelity = noise_var, units, fidelity
        else:
            self._X = np.vstack([self._X, features])
            self._Y = np.vstack([self._Y, matrix])
            self._C = np.vstack([self._C, c_matrix])
            self._noise = np.concatenate([self._noise, noise_var])
            self._units = np.vstack([self._units, units])
            self._fidelity = np.concatenate([self._fidelity, fidelity])

    # -- propose -----------------------------------------------------------
    def ask(self, n: int = 1) -> list[dict]:
        return [config for config, _ in self.ask_with_fidelity(n)]

    def ask_with_fidelity(
        self, n: int = 1, cost_remaining: float | None = None
    ) -> list[tuple[dict, float]]:
        if cost_remaining is not None:
            self.cost_remaining = float(cost_remaining)
        if n <= 0:
            return []
        requested = n
        levels = self._levels()
        out: list[tuple[dict, float]] = []
        design_left = self._initial_design_left()
        if design_left > 0:
            k = min(n, design_left)
            configs = self.space.sample_sobol(k, rng=self.rng)
            out.extend((config, levels[0]) for config in configs)
            n -= k
        n_full = 0
        if n > 0 and float(self.params["min_full_fraction"]) > 0:
            n_full = max(1, int(np.ceil(float(self.params["min_full_fraction"]) * n)))
        if n > 0 and (self._X is None or len(self._Y) < 3):
            configs = self.space.sample_uniform(n, self.rng)
            out.extend(
                (config, 1.0 if index < n_full else levels[0])
                for index, config in enumerate(configs)
            )
        else:
            for index in range(n):
                config, level = self._propose_pair(out)
                if index < n_full:
                    level = 1.0
                out.append((config, level))

        budget = self.cost_remaining
        if budget is not None:
            if budget <= 0:
                self.last_cost_trimmed = requested
                self.last_full_requested = n_full
                self.last_full_returned = 0
                return []
            affordable = [level for level in levels if self._cost(level) <= budget + 1e-12]
            cap = affordable[-1] if affordable else levels[0]
            out = [(config, min(level, cap)) for config, level in out]
            spent = 0.0
            trimmed: list[tuple[dict, float]] = []
            for config, level in out:
                price = self._cost(level)
                if spent + price <= budget + 1e-12:
                    trimmed.append((config, level))
                    spent += price
            if not trimmed:
                self.last_cost_trimmed = requested
                self.last_full_requested = n_full
                self.last_full_returned = 0
                return []
            out = trimmed
        self.last_cost_trimmed = max(0, requested - len(out))
        self.last_full_requested = n_full
        self.last_full_returned = sum(1 for _, level in out if level >= 1.0 - 1e-9)
        return out

    def _propose_pair(
        self, batch: Sequence[tuple[dict, float]]
    ) -> tuple[dict, float]:
        levels = self._levels()
        if self._X is None or len(self._Y) < 3:
            return self.space.sample_uniform(1, self.rng)[0], float(levels[0])
        signed = self._signed_observations()
        reference = self._signed_reference()
        front = signed[non_dominated_mask(signed)]
        candidates = self._candidate_pool()
        if not candidates:
            return self.space.sample_uniform(1, self.rng)[0], 1.0
        base = self.space.features_many(candidates)
        rng = np.random.default_rng(self.seed + 7919 * (int(self.t) + len(batch) + 1))
        best_score = -np.inf
        best_pair = (candidates[0], levels[0])
        for level in levels:
            features = np.hstack(
                [base, np.full((len(base), 1), float(level), dtype=float)]
            )
            acquisition, keep = self._ehvi_acquisition(features, front, reference, rng)
            price = max(self._cost(level), 1e-6)
            scores = acquisition / price
            if len(scores) == 0:
                continue
            index = int(np.argmax(scores))
            if float(scores[index]) > best_score:
                best_score = float(scores[index])
                best_pair = (candidates[keep[index]], float(level))
        config, level = best_pair
        # local penalty against conditions already chosen in this batch
        if batch:
            chosen = np.vstack([self.space.encode(item) for item, _ in batch])
            unit = self.space.encode(config)
            distance = float(np.min(np.linalg.norm(chosen - unit, axis=1)))
            if distance <= 0.05:  # too close: take the next distinct candidate
                for candidate in candidates:
                    other = self.space.encode(candidate)
                    if float(np.min(np.linalg.norm(chosen - other, axis=1))) > 0.05:
                        config = candidate
                        break
        return config, level

    # -- state -------------------------------------------------------------
    def state_dict(self) -> dict:
        state = super().state_dict()
        # data, not a counter -- rebuilt from the dataset on resume
        return state

    def load_state_dict(self, state: Mapping) -> None:
        super().load_state_dict(state)
        existing = np.asarray(getattr(self, "_fidelity", []))
        if existing.size:
            return  # already rebuilt from the dataset
        fidelity = state.get("fidelity")
        if fidelity is not None:
            self._fidelity = np.asarray(fidelity, dtype=float)
