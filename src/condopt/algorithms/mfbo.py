"""Multi-fidelity Bayesian optimization (mfbo).

A cheap experiment at fidelity ``s ∈ (0, 1]`` costs roughly ``s**cost_exponent``
of a full run, but carries a fidelity-dependent bias.  ``mfbo``:
* feeds ``fidelity`` as one extra GP input feature (so the surrogate learns the
  bias empirically instead of assuming a correction formula),
* chooses both the condition **and** the fidelity that maximizes
  ``acquisition / cost(s)**cost_exponent``,
* always keeps a fraction of proposals at full fidelity so the final answer is
  based on real full-fidelity evidence.

The optimizer exposes :meth:`ask_with_fidelity`; :class:`condopt.study.Study`
records the proposed fidelity in ``pending.json`` / ``suggestions.csv`` and
``Observation.fidelity`` carries it back.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from scipy.stats import norm
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor

from ..base import AlgorithmSpec, Objective, Optimizer, ParamSpec
from ..dataset import Observation
from ..registry import register
from .bayes import _ard_initial_lengthscale, _log_ei, _make_kernel

_EPS = 1e-12


@register
class MultiFidelityBO(Optimizer):
    """Fidelity-aware GP-BO with cost-aware acquisition."""

    supports_fidelity = True

    spec = AlgorithmSpec(
        name="mfbo",
        family="model_based",
        summary="多保真贝叶斯优化：把 fidelity 作为 GP 输入，按 EI/成本^α 选择条件与保真度。",
        when="有低成本粗筛（短时试跑、低精度仿真）且最终需要全保真验证时。",
        params=(
            ParamSpec("kernel", "choice", "matern52", choices=("matern52", "matern32", "rbf")),
            ParamSpec("length_scale", "float", 0.3, low=0.01, high=10.0),
            ParamSpec("ard", "bool", True),
            ParamSpec("alpha", "float", 1e-4, low=1e-12, high=1.0),
            ParamSpec("n_initial", "int", 0, low=0, high=200),
            ParamSpec("n_candidates", "int", 512, low=64, high=16384),
            ParamSpec(
                "acquisition",
                "choice",
                "ei",
                choices=("ei", "logei", "ucb"),
                description="扫描显示 ei 在带约束多保真上更稳（logei 偶发整轮无可行点）",
            ),
            ParamSpec("kappa", "float", 2.0, low=0.0, high=10.0),
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
            ParamSpec("local_frac", "float", 0.3, low=0.0, high=0.9),
            ParamSpec("local_sigma", "float", 0.15, low=0.01, high=1.0),
            ParamSpec(
                "min_full_fraction",
                "float",
                0.25,
                low=0.0,
                high=1.0,
                description="至少这么多比例的提议点使用全保真",
            ),
            ParamSpec(
                "two_phase_feasibility",
                "bool",
                True,
                description=(
                    "（仅带约束时生效）尚无可行观测时先最大化 P(feasible)（phase A），"
                    "找到可行点后再回到 EI/成本；关闭则始终 EI×P(feasible)"
                ),
            ),
            ParamSpec(
                "feasibility_full_threshold",
                "float",
                0.0,
                low=0.0,
                high=1.0,
                description=(
                    "（仅带约束时生效）全保真候选要求 P(feasible) ≥ 该值；"
                    "0 = 不预筛；若没有候选达标则回退到全体候选"
                ),
            ),
            ParamSpec(
                "feasibility_aware_full",
                "bool",
                True,
                description=(
                    "（仅带约束时生效）全保真槽位只在高保真档位里按 acquisition×P(feasible) "
                    "选点；关闭则沿用普通选点再强制全保真（旧行为）"
                ),
            ),
        ),
        capabilities=frozenset(
            {
                "batch",
                "mixed",
                "categorical",
                "continuous",
                "noisy",
                "model_based",
                "multi_fidelity",
                "constraints",
            }
        ),
        references=(
            "Kandasamy et al., Multi-fidelity Bayesian Optimisation with Continuous Approximations (BOCA), ICML 2017",
            "Jamieson & Talwalkar, Non-stochastic Best Arm Identification and Hyperparameter Optimization, AISTATS 2016",
        ),
    )

    def __init__(self, *args, **kwargs) -> None:
        self._X: np.ndarray | None = None
        self._y = np.zeros(0)
        self._units = np.zeros((0, 0))
        self._fidelity = np.zeros(0)
        self._init_queue: list[tuple[dict, float]] | None = None
        self._C: np.ndarray | None = None
        self._noise_var: np.ndarray | None = None
        self.last_cost_trimmed = 0
        self.last_full_requested = 0
        self.last_full_returned = 0
        super().__init__(*args, **kwargs)

    # -- fidelity levels --------------------------------------------------
    def _levels(self) -> list[float]:
        raw = str(self.params["fidelity_levels"])
        levels = []
        for chunk in raw.split(","):
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
        """Cost model: per-level table when configured, else level**exponent.

        Both the acquisition and the budget check call this single method, so
        the two can never disagree.
        """
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
                    f"fidelity_costs has no cost for level {level:.3g}; "
                    f"provide an entry for every fidelity level (have {keys})"
                )
            # Linear interpolation between neighbouring table entries; never
            # silently reuse a different level's cost.
            upper = min(key for key in keys if key > level)
            lower = max(key for key in keys if key < level)
            ratio = (level - lower) / (upper - lower)
            return float(table[lower] + ratio * (table[upper] - table[lower]))
        return float(level) ** float(self.params["cost_exponent"])

    def _features(self, config: Mapping, fidelity: float) -> np.ndarray:
        return np.concatenate([self.space.features(config), [float(fidelity)]])

    # -- data -------------------------------------------------------------
    def tell(self, observations: Sequence[Observation]) -> None:
        observations = list(observations)
        super().tell(observations)
        if not observations:
            return
        constraint_names = tuple(getattr(self.objective, "constraints", ()) or ())
        constraint_rows = np.asarray(
            [
                [float(o.constraints.get(name, np.nan)) if o.constraints else np.nan
                 for name in constraint_names]
                for o in observations
            ],
            dtype=float,
        ) if constraint_names else np.empty((len(observations), 0))
        X = np.vstack([self._features(o.x, o.fidelity) for o in observations])
        units = np.vstack([self.space.encode(o.x) for o in observations])
        y = np.asarray(
            [o.score(self.objective.name, self.objective.direction) for o in observations],
            dtype=float,
        )
        # Per-point observation-noise variance (0/nan = unknown).  ``transfer_from``
        # raises ``objective_std`` for a doubtful source; without this the MF GP
        # ignored it and ``prior_reliability`` was a no-op here.
        noise_var = np.asarray(
            [
                (o.objective_std**2 / o.n_replicates) if o.objective_std > 0 else np.nan
                for o in observations
            ],
            dtype=float,
        )
        if self._X is None:
            self._X, self._y, self._units = X, y, units
            self._fidelity = np.asarray([o.fidelity for o in observations], dtype=float)
            self._C = constraint_rows
            self._noise_var = noise_var
        else:
            self._X = np.vstack([self._X, X])
            self._y = np.concatenate([self._y, y])
            self._units = np.vstack([self._units, units])
            self._fidelity = np.concatenate(
                [self._fidelity, np.asarray([o.fidelity for o in observations], dtype=float)]
            )
            self._C = np.vstack([self._C, constraint_rows])
            self._noise_var = np.concatenate([self._noise_var, noise_var])

    # -- propose ----------------------------------------------------------
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
        out: list[tuple[dict, float]] = []
        design_left = self._initial_design_left()
        if design_left > 0:
            k = min(n, design_left)
            out.extend(self._initial_design(k))
            n -= k
        n_full = 0
        if n > 0 and float(self.params["min_full_fraction"]) > 0:
            n_full = max(1, int(np.ceil(float(self.params["min_full_fraction"]) * n)))
        for index in range(n):
            forced_full = index < n_full
            constrained = bool(getattr(self.objective, "constraints", ()) or ())
            if (
                forced_full
                and constrained
                and bool(self.params.get("feasibility_aware_full", True))
            ):
                # Constrained + multi-fidelity: choose the full-fidelity point
                # itself (acquisition x P(feasible) at s=1) instead of upgrading
                # a cheap-fidelity proposal, which could sit outside the feasible
                # region and waste the expensive run.
                threshold = float(self.params.get("feasibility_full_threshold", 0.0) or 0.0)
                config, fidelity = self._propose(
                    out, force_full=True, min_feasibility=threshold
                )
            else:
                config, fidelity = self._propose(out)
            if forced_full:
                fidelity = 1.0
            out.append((config, fidelity))
        budget = self.cost_remaining
        if budget is not None:
            if budget <= 0:
                self.last_cost_trimmed = requested
                self.last_full_requested = n_full
                self.last_full_returned = 0
                return []
            levels = self._levels()
            affordable = [level for level in levels if self._cost(level) <= budget + 1e-12]
            cap = affordable[-1] if affordable else levels[0]
            # Downgrade expensive proposals, then trim to what the budget affords.
            out = [(config, min(fidelity, cap)) for config, fidelity in out]
            spent = 0.0
            trimmed: list[tuple[dict, float]] = []
            for config, fidelity in out:
                price = self._cost(fidelity)
                if spent + price <= budget + 1e-12:
                    trimmed.append((config, fidelity))
                    spent += price
            if not trimmed:
                # Remaining budget cannot afford even the cheapest level: return
                # no suggestion instead of raising, so an agent loop can stop.
                self.last_cost_trimmed = requested
                self.last_full_requested = n_full
                self.last_full_returned = 0
                return []
            out = trimmed
        self.last_cost_trimmed = max(0, requested - len(out))
        self.last_full_requested = n_full
        self.last_full_returned = sum(1 for _, fidelity in out if fidelity >= 1.0 - 1e-9)
        return out

    def _initial_design_left(self) -> int:
        n_initial = int(self.params["n_initial"])
        if n_initial <= 0:
            n_initial = max(4, min(2 * (self.space.feature_dim + 1), 20))
        return max(0, n_initial - int(self.t))

    def _initial_design(self, k: int) -> list[tuple[dict, float]]:
        if self._init_queue is None:
            total = max(
                int(self.params["n_initial"]) or max(4, 2 * (self.space.feature_dim + 1)), k
            )
            configs = self.space.sample_sobol_base2(min(total, 20), seed=self.seed + 17)
            levels = self._levels()
            seen = {self._key(unit) for unit in self._units}
            queue = []
            for index, config in enumerate(configs):
                key = self._key(self.space.encode(config))
                if key in seen:
                    continue
                # Initial design should be CHEAP: explore at the lowest fidelity
                # first (review v0.7 P1: initial design contained full runs).
                queue.append((config, levels[0]))
            self._init_queue = queue
        take = self._init_queue[:k]
        self._init_queue = self._init_queue[k:]
        return take

    def _propose(
        self,
        batch: Sequence[tuple[dict, float]],
        force_full: bool = False,
        min_feasibility: float = 0.0,
    ) -> tuple[dict, float]:
        levels = [1.0] if force_full else self._levels()
        if self._X is None or len(self._y) < 3:
            config = self.space.sample_uniform(1, self.rng)[0]
            return config, float(self.rng.choice(levels))
        candidates = self._candidate_pool()
        if not candidates:
            config = self.space.sample_uniform(1, self.rng)[0]
            return config, 1.0
        if min_feasibility > 0 and bool(getattr(self.objective, "constraints", ()) or ()):
            base = self.space.features_many(candidates)
            probe = np.hstack(
                [base, np.full((len(base), 1), float(levels[-1]), dtype=float)]
            )
            probability = self._feasibility_probability(probe)
            keep = [index for index, value in enumerate(probability) if value >= min_feasibility]
            if keep:  # otherwise fall back to the full pool instead of deadlocking
                candidates = [candidates[index] for index in keep]
        model = self._fit(self._X, self._y)
        best = self._best_scalar()

        # Score every candidate × fidelity pair by acq / cost.
        features = np.vstack(
            [self._features(config, level) for config in candidates for level in levels]
        )
        mu, std = model.predict(features, return_std=True)
        mu = np.asarray(mu, dtype=float)
        std = np.maximum(np.asarray(std, dtype=float), 1e-12)
        mode = str(self.params["acquisition"])
        if mode == "ucb":
            acquisition = mu + float(self.params["kappa"]) * std
        elif mode == "logei":
            imp = mu - best
            acquisition = _log_ei(imp, std, imp / std)
        else:
            imp = mu - best
            acquisition = imp * norm.cdf(imp / std) + std * norm.pdf(imp / std)
        probability = self._feasibility_probability(features)
        if (
            bool(getattr(self.objective, "constraints", ()) or ())
            and bool(self.params.get("two_phase_feasibility", True))
            and self._C is not None
            and self._C.shape[1] > 0
            and not np.any(
                self._feasible_mask(np.ones(len(self._y), dtype=bool))
            )
        ):
            # Phase A: nothing feasible has been measured yet, so spending the
            # budget on the feasibility probability beats chasing EI on a model
            # that has never seen a feasible point.
            acquisition = probability
        else:
            acquisition = acquisition * probability
        costs = np.asarray(
            [self._cost(level) for _ in candidates for level in levels], dtype=float
        )
        scores = acquisition / np.maximum(costs, 1e-6)

        chosen_units = [self.space.encode(config) for config, _ in batch]
        pair_index = int(np.argmax(scores))
        candidate_index = pair_index // len(levels)
        level_index = pair_index % len(levels)
        config = candidates[candidate_index]
        # Local penalty against already chosen conditions (batch spread).
        if chosen_units:
            scale = 0.15
            for _ in range(5):
                unit = self.space.encode(config)
                distances = np.linalg.norm(np.asarray(chosen_units) - unit, axis=1)
                if float(np.min(distances)) > scale:
                    break
                candidate_index = (candidate_index + 1) % len(candidates)
                config = candidates[candidate_index]
        return config, float(levels[level_index])

    def _feasible_mask(self, rows: np.ndarray) -> np.ndarray | None:
        if self._C is None or self._C.shape[1] == 0:
            return None
        rows = np.asarray(rows)
        if rows.dtype == bool and rows.shape[0] != self._C.shape[0]:
            # internal memory is out of sync (should not happen after the fix in
            # ``load_state_dict``); prefer "unknown feasibility" over a crash
            return None
        mask = np.ones(int(np.sum(rows)), dtype=bool)
        limits = getattr(self.objective, "constraint_limits", {}) or {}
        for index, name in enumerate(self.objective.constraints):
            values = self._C[rows, index]
            limit = limits.get(name) or {}
            threshold = float(limit.get("value", 0.0))
            op = str(limit.get("op", "<="))
            ok = (values <= threshold) if op == "<=" else (values >= threshold)
            ok = np.where(np.isfinite(values), ok, True)
            mask &= ok
        return mask

    def _feasibility_probability(self, features: np.ndarray) -> np.ndarray:
        if self._C is None or self._C.shape[1] == 0 or self._X is None:
            return np.ones(len(features))
        limits = getattr(self.objective, "constraint_limits", {}) or {}
        probability = np.ones(len(features))
        if self._X is None or self._C.shape[0] != self._X.shape[0]:
            return probability  # out-of-sync memory: no feasibility model
        modelled = False
        for index, name in enumerate(self.objective.constraints):
            values = self._C[:, index]
            usable = np.isfinite(values)
            if int(np.sum(usable)) < 2:
                continue
            model = self._fit(self._X[usable], values[usable])
            mu, std = model.predict(features, return_std=True)
            mu = np.asarray(mu, dtype=float)
            std = np.maximum(np.asarray(std, dtype=float), 1e-9)
            limit = limits.get(name) or {}
            threshold = float(limit.get("value", 0.0))
            op = str(limit.get("op", "<="))
            z = (threshold - mu) / std if op == "<=" else (mu - threshold) / std
            probability *= norm.cdf(z)
            modelled = True
        return probability if modelled else np.ones(len(features))

    def _best_scalar(self) -> float:
        """Incumbent for EI: FULL-fidelity observations only when available."""
        if len(self._y) == 0:
            return 0.0
        full = self._fidelity >= 1.0 - 1e-9
        values = self._y[full] if bool(np.any(full)) else self._y
        feasible = self._feasible_mask(full if bool(np.any(full)) else np.ones(len(self._y), bool))
        if feasible is not None and bool(np.any(feasible)):
            values = values[feasible]
        return float(np.max(values))

    def _candidate_pool(self) -> list[dict]:
        m = int(self.params["n_candidates"])
        dim = self.space.dim
        rng = self.rng
        n_local = int(m * float(self.params["local_frac"])) if self._X is not None else 0
        n_rest = max(m - n_local, 2)
        parts = [rng.random((n_rest // 2, dim))]
        import warnings

        from scipy.stats import qmc

        sobol = qmc.Sobol(d=dim, scramble=True, seed=int(rng.integers(0, 2**31 - 1)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            parts.append(sobol.random(n_rest - n_rest // 2))
        if n_local > 0:
            top = np.argsort(-self._y)[: min(5, len(self._y))]
            centers = self._units[top]
            base = centers[rng.integers(0, len(centers), size=n_local)]
            parts.append(
                np.clip(
                    base + rng.normal(0.0, float(self.params["local_sigma"]), (n_local, dim)),
                    0.0,
                    1.0,
                )
            )
        raw = np.vstack(parts)
        configs = [self.space.decode(unit) for unit in raw]
        seen: set[tuple] = set()
        existing = {self._key(unit) for unit in self._units}
        out = []
        for config in configs:
            key = self._key(self.space.encode(config))
            if key in seen or key in existing:
                continue
            seen.add(key)
            out.append(config)
        return out

    @staticmethod
    def _key(unit: np.ndarray) -> tuple:
        return tuple(np.round(unit, 9).tolist())

    def _fit(self, X: np.ndarray, y: np.ndarray) -> GaussianProcessRegressor:
        import warnings

        kernel = _make_kernel(
            str(self.params["kernel"]),
            float(self.params["length_scale"]),
            feature_dim=self.space.feature_dim + 1,
            ard=bool(self.params["ard"]),
        )
        frozen = len(y) < max(6, 2 * (self.space.feature_dim + 1))
        alpha: float | np.ndarray = float(self.params["alpha"])
        noise = getattr(self, "_noise_var", None)
        if noise is not None and len(noise):
            base = float(self.params["alpha"])
            values = np.asarray(noise, dtype=float)
            y_var = max(float(np.var(y)) if len(y) > 1 else 1.0, 1e-12)
            scaled = np.where(np.isfinite(values) & (values > 0), values / y_var, base)
            if len(scaled) < len(y):
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

    # -- state ------------------------------------------------------------
    def state_dict(self) -> dict:
        state = super().state_dict()
        state["init_queue"] = self._init_queue
        # Constraint values are *data*, not counters: on resume the instance is
        # rebuilt from the study dataset (``prior``).  Persisting them here used to
        # shadow that rebuild — a state written after the first batch (1 row) then
        # overwrote the rebuilt matrix (N rows), and ``_feasible_mask`` raised
        # ``IndexError`` on warm-started campaigns.
        return state

    def load_state_dict(self, state: Mapping) -> None:
        super().load_state_dict(state)
        queue = state.get("init_queue")
        self._init_queue = None if queue is None else [(dict(c), float(f)) for c, f in queue]
        if getattr(self, "_suppress_persisted_constraints", True):
            # Legacy/defensive: never let a stale persisted matrix replace data that
            # was already rebuilt from the dataset.
            return
        matrix = state.get("C")
        self._C = None if matrix is None else np.asarray(matrix, dtype=float)
