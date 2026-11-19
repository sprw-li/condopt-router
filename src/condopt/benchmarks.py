"""Synthetic benchmark problems and a small benchmark runner.

The point is not to prove any single algorithm "best" -- no free lunch.  It
is to (a) catch regressions in the algorithms, (b) calibrate the router's
rule weights with evidence, and (c) give the skill an honest sanity check
before it is pointed at a real lab.

All problems are deterministic and dependency-free so they run on a laptop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from .feasibility import feasibility_state as _feasibility_state
from .base import Objective
from .dataset import Observation
from . import registry
from .space import SearchSpace


@dataclass
class BenchmarkProblem:
    name: str
    space: SearchSpace
    objective: Objective
    fn: Callable[[dict], float]
    description: str = ""
    #: reference optimum (raw objective value), for normalised regret
    optimum: float | None = None
    #: a configuration that attains ``optimum`` (used by self-checks)
    best_config: dict | None = None
    #: objective-vector function for multi-objective problems
    objectives_fn: Callable[[dict], dict] | None = None
    #: constraint-vector function for constrained problems
    constraints_fn: Callable[[dict], dict] | None = None
    #: fidelity-aware evaluation for multi-fidelity problems
    fidelity_fn: Callable[[dict, float], float] | None = None
    #: fidelity-aware objective vectors for multi-objective + multi-fidelity
    objectives_fidelity_fn: Callable[[dict, float], dict] | None = None
    #: free-form capability tags, e.g. ``("noisy", "constrained")``
    tags: tuple[str, ...] = ()
    #: observation-noise std used by noise-aware calibration/reporting
    noise_std: float = 0.0

    def evaluate(self, config: dict) -> float:
        return float(self.fn(config))

    def make_observation(self, config: dict, source: str = "benchmark") -> Observation:
        constraints = self.constraints_fn(config) if self.constraints_fn is not None else {}
        if self.objective.is_multi_objective:
            if self.objectives_fn is None:
                raise ValueError(f"multi-objective problem {self.name!r} has no objectives_fn")
            return Observation(
                x=config,
                objectives=self.objectives_fn(config),
                constraints=constraints,
                source=source,
            )
        return Observation(x=config, y=self.evaluate(config), constraints=constraints, source=source)

    def regret(self, best_score: float) -> float | None:
        """Signed gap to the optimum (>=0, lower is better)."""
        if self.optimum is None:
            return None
        sign = 1.0 if self.objective.direction == "max" else -1.0
        return max(0.0, sign * self.optimum - best_score)


def _unit_space(names: Sequence[str]) -> SearchSpace:
    return SearchSpace.from_dict(
        [{"name": name, "kind": "continuous", "low": 0.0, "high": 1.0} for name in names]
    )


def _branin(config: dict) -> float:
    x1 = config["x1"] * 15.0 - 5.0
    x2 = config["x2"] * 15.0
    a, b, c = 1.0, 5.1 / (4.0 * math.pi**2), 5.0 / math.pi
    r, s, t = 6.0, 10.0, 1.0 / (8.0 * math.pi)
    return a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * math.cos(x1) + s


def _sphere(config: dict) -> float:
    return -sum((value - 0.3) ** 2 for value in config.values())


def _hartmann3(config: dict) -> float:
    alpha = np.asarray([1.0, 1.2, 3.0, 3.2])
    a = np.asarray(
        [[3.0, 10, 30], [0.1, 10, 35], [3.0, 10, 30], [0.1, 10, 35]]
    )
    p = np.asarray(
        [[0.3689, 0.1170, 0.2673], [0.4699, 0.4387, 0.7470], [0.1091, 0.8732, 0.5547], [0.0381, 0.5743, 0.8828]]
    )
    x = np.asarray([config["x1"], config["x2"], config["x3"]])
    outer = np.sum(a * (x - p) ** 2, axis=1)
    return float(np.sum(alpha * np.exp(-outer)) * -1.0)


def _mixed_reaction(config: dict) -> float:
    t = (config["T"] - 82.0) / 18.0
    dur = (config["time"] - 55.0) / 25.0
    conc = (math.log10(config["conc"]) + 0.8) / 0.9
    solvent_bonus = {"MeCN": 4.0, "EtOH": 1.0, "water": -3.0, "THF": 2.0}[config["solvent"]]
    equiv_bonus = -0.8 * (config["equiv"] - 2) ** 2
    inert_bonus = 2.0 if config["inert"] else 0.0
    return (
        70.0
        + 12.0 * math.exp(-(t**2))
        + 8.0 * math.exp(-(dur**2))
        + 6.0 * math.exp(-(conc**2))
        + solvent_bonus
        + equiv_bonus
        + inert_bonus
    )


def _discrete_catalyst(config: dict) -> float:
    base = {"Ni": 60.0, "Pd": 72.0, "Pt": 68.0}[config["metal"]]
    ligand = {"A": 3.0, "B": 1.0, "C": 6.0, "D": 2.0}[config["ligand"]]
    temp = -0.002 * (config["temp"] - 80) ** 2
    return base + ligand + temp


def _moo_yield(config: dict) -> float:
    temperature = config["T"]
    time = config["time"]
    return 70.0 + 12.0 * math.exp(-(((temperature - 82.0) / 18.0) ** 2)) + 8.0 * math.exp(
        -(((time - 55.0) / 25.0) ** 2)
    )


def _moo_objectives(config: dict) -> dict:
    temperature = config["T"]
    time = config["time"]
    return {"yield": _moo_yield(config), "cost": 0.1 * temperature + 0.05 * time}


def moo_reference_front(n_temperature: int = 221, n_time: int = 161) -> np.ndarray:
    """Signed reference Pareto front for ``moo_yield_cost`` (yield max, cost min)."""
    from .pareto import non_dominated_mask

    temperatures = np.linspace(60.0, 115.0, n_temperature)
    times = np.linspace(5.0, 120.0, n_time)
    grid_t, grid_time = np.meshgrid(temperatures, times, indexing="ij")
    yield_values = 70.0 + 12.0 * np.exp(-(((grid_t - 82.0) / 18.0) ** 2)) + 8.0 * np.exp(
        -(((grid_time - 55.0) / 25.0) ** 2)
    )
    costs = 0.1 * grid_t + 0.05 * grid_time
    scores = np.column_stack([yield_values.ravel(), -costs.ravel()])
    return scores[non_dominated_mask(scores)]


#: Safety limits of the ``moo_constrained`` benchmark.  The unconstrained yield
#: optimum (T ~ 82 C, time ~ 55 min) violates both, so feasibility is binding.
MOO_T_MAX = 78.0
MOO_TIME_MAX = 50.0


def _moo_constrained_constraint(config: dict) -> dict:
    """Reactor safety limits: ``T <= MOO_T_MAX`` and ``time <= MOO_TIME_MAX``."""
    return {"T_max": float(config["T"]), "time_max": float(config["time"])}


def moo_constrained_reference_front(n_temperature: int = 181, n_time: int = 181) -> np.ndarray:
    """Signed *feasible* reference Pareto front for ``moo_constrained``.

    Dense grid over the feasible box ``T in [60, 78]`` x ``time in [5, 50]``,
    so an infeasible point can never contribute to the reference hypervolume.
    """
    from .pareto import non_dominated_mask

    temperatures = np.linspace(60.0, MOO_T_MAX, n_temperature)
    times = np.linspace(5.0, MOO_TIME_MAX, n_time)
    grid_t, grid_time = np.meshgrid(temperatures, times, indexing="ij")
    yield_values = 70.0 + 12.0 * np.exp(-(((grid_t - 82.0) / 18.0) ** 2)) + 8.0 * np.exp(
        -(((grid_time - 55.0) / 25.0) ** 2)
    )
    costs = 0.1 * grid_t + 0.05 * grid_time
    scores = np.column_stack([yield_values.ravel(), -costs.ravel()])
    return scores[non_dominated_mask(scores)]


_SUZUKI_SOLVENT = {"MeCN": 5.0, "EtOH": 3.0, "THF": 2.0, "water": -2.0, "DMF": 4.0}
_SUZUKI_LIGAND = {"L1": 4.0, "L2": 2.0, "L3": 3.0, "L4": 1.0, "L5": 0.0, "L6": 2.0}
_SUZUKI_BASE = {1: 0.0, 2: 1.0, 3: 0.0, 4: -1.0}


def _solvent_constrained_yield(config: dict) -> float:
    """Binding-constraint variant of the Tier C solvent landscape (T <= 80)."""
    polarity, protic = SOLVENT_DESCRIPTORS[config["solvent"]]
    temperature = float(config["T"])
    return (
        50.0
        + 28.0 * math.exp(-(((polarity - 0.80) / 0.35) ** 2) - (((protic - 0.20) / 0.60) ** 2))
        + 22.0 * math.exp(-(((temperature - 95.0) / 25.0) ** 2))
    )


def _mf_solvent_objectives(config: dict, fidelity: float = 1.0) -> dict:
    """Two-objective multi-fidelity landscape over a 12-level solvent choice.

    ``solvent`` is looked up in ``SOLVENT_DESCRIPTORS`` (polarity, protic), so
    the same function backs both the one-hot and the descriptor problem; only
    the feature encoding differs.
    """
    polarity, protic = SOLVENT_DESCRIPTORS[config["solvent"]]
    temperature = float(config["T"])
    yield_value = (
        50.0
        + 32.0 * math.exp(-(((polarity - 0.80) / 0.35) ** 2) - (((protic - 0.20) / 0.60) ** 2))
        + 12.0 * math.exp(-(((temperature - 95.0) / 25.0) ** 2))
    )
    cost = 6.0 + 0.06 * (temperature - 60.0) + 6.0 * protic
    bias = 14.0 * ((1.0 - float(fidelity)) ** 1.5)
    return {"yield": yield_value - bias, "cost": cost}


def _suzuki_like(config: dict) -> float:
    temperature = config["T"]
    time = config["time"]
    loading = config["cat_loading"]
    core = 20.0 + 80.0 * math.exp(
        -(((temperature - 85.0) / 20.0) ** 2)
        - ((math.log(time / 60.0) / 0.9) ** 2)
        - ((math.log(loading / 1.5) / 0.7) ** 2)
    )
    return (
        core
        + _SUZUKI_SOLVENT[config["solvent"]]
        + _SUZUKI_LIGAND[config["ligand"]]
        + _SUZUKI_BASE[config["base_eq"]]
    )


def mf_suzuki(config: dict, fidelity: float = 1.0) -> float:
    """Multi-fidelity Suzuki benchmark: unbiased full fidelity, biased cheap runs."""
    bias = 15.0 * ((1.0 - float(fidelity)) ** 1.5)
    return _suzuki_like(config) - bias


def mf_suzuki_cost(fidelity: float) -> float:
    """Cost model for ``mf_suzuki``: cost equals fidelity (full run = 1.0)."""
    return float(fidelity)


def mf_suzuki_problem() -> BenchmarkProblem:
    problem = builtin_problems()["suzuki_like"]
    return BenchmarkProblem(
        name="mf_suzuki",
        space=problem.space,
        objective=problem.objective,
        fn=lambda config: mf_suzuki(config, 1.0),
        description="Tier F：多保真 Suzuki（s=0.5 最优约 104.7，全保真 110.0，cost=s）",
        fidelity_fn=mf_suzuki,
        tags=("multi_fidelity",),
        optimum=110.0,
        best_config=dict(problem.best_config or {}),
    )


def cost_aware_best(observations, fidelity_key: str = "fidelity") -> dict:
    """Best full-fidelity value and the best prefix value-per-cost of a run."""
    best_full = None
    best_so_far = -math.inf
    spent = 0.0
    best_per_cost = 0.0
    for observation in observations:
        value = float(observation.y)
        fidelity = float(getattr(observation, fidelity_key, 1.0))
        spent += mf_suzuki_cost(fidelity)
        best_so_far = max(best_so_far, value)
        if spent > 0:
            best_per_cost = max(best_per_cost, best_so_far / spent)
        if fidelity >= 1.0 - 1e-9:
            best_full = value if best_full is None else max(best_full, value)
    return {
        "best_full_fidelity": best_full,
        "total_cost": spent,
        "best_value_per_cost": best_per_cost,
    }


def _discrete_catalyst_v2(config: dict) -> float:
    base = {"Ni": 60.0, "Pd": 72.0, "Pt": 68.0}[config["metal"]]
    ligand = {"A": 3.0, "B": 1.0, "C": 6.0, "D": 2.0}[config["ligand"]]
    additive = {"X1": 2.0, "X2": -1.0, "X3": 4.0, "X4": 0.0}[config["additive"]]
    interaction = 3.0 if (config["metal"] == "Pt" and config["additive"] == "X3") else 0.0
    temperature = -0.002 * (config["temp"] - 80) ** 2
    return base + ligand + additive + interaction + temperature


#: 12 solvents with two physics-descriptor coordinates (polarity, proticity).
SOLVENT_DESCRIPTORS = {
    "S01": (0.10, 0.90),
    "S02": (0.25, 0.75),
    "S03": (0.45, 0.55),
    "S04": (0.80, 0.00),
    "S05": (0.60, 0.30),
    "S06": (0.35, 0.10),
    "S07": (0.90, 0.40),
    "S08": (0.55, 0.80),
    "S09": (0.70, 0.60),
    "S10": (0.20, 0.35),
    "S11": (0.65, 0.15),
    "S12": (0.05, 0.20),
}


def _high_cardinality_solvent(config: dict) -> float:
    polarity, protic = SOLVENT_DESCRIPTORS[config["solvent"]]
    temperature = config["T"]
    return (
        50.0
        + 25.0 * math.exp(-(((polarity - 0.8) / 0.35) ** 2) - (((protic - 0.2) / 0.6) ** 2))
        + 20.0 * math.exp(-(((temperature - 90.0) / 25.0) ** 2))
    )


def _constraint_yield(config: dict) -> float:
    temperature = config["T"]
    time = config["time"]
    return 60.0 + 35.0 * math.exp(
        -(((temperature - 110.0) / 20.0) ** 2) - (((time - 100.0) / 45.0) ** 2)
    )


def _constraint_easy_g(config: dict) -> float:
    return ((config["T"] - 95.0) / 25.0) ** 2 + ((config["time"] - 80.0) / 60.0) ** 2 - 1.0


def _constraint_hard_g(config: dict) -> float:
    return ((config["T"] - 95.0) / 12.5) ** 2 + ((config["time"] - 80.0) / 30.0) ** 2 - 1.0


def _rosenbrock_high_dim(config: dict) -> float:
    """10-D Rosenbrock on x in [-2, 2]; optimum 0 at x = 1 (unit 0.75)."""
    values = [4.0 * float(config[f"x{index}"]) - 2.0 for index in range(10)]
    total = 0.0
    for index in range(9):
        total += 100.0 * (values[index + 1] - values[index] ** 2) ** 2 + (1.0 - values[index]) ** 2
    return -total


def _ackley_high_dim(config: dict) -> float:
    """10-D Ackley on x in [-5, 5]; optimum 0 at x = 0 (unit 0.5)."""
    values = [10.0 * float(config[f"x{index}"]) - 5.0 for index in range(10)]
    n = len(values)
    sum_sq = sum(value * value for value in values) / n
    sum_cos = sum(math.cos(2.0 * math.pi * value) for value in values) / n
    return -(
        -20.0 * math.exp(-0.2 * math.sqrt(sum_sq))
        - math.exp(sum_cos)
        + 20.0
        + math.e
    )



def builtin_problems() -> dict[str, BenchmarkProblem]:
    branin_space = _unit_space(["x1", "x2"])
    return {
        "branin": BenchmarkProblem(
            name="branin",
            space=branin_space,
            objective=Objective(name="branin", direction="min"),
            fn=_branin,
            description="经典 2D 多峰函数（最小化）",
            optimum=0.397887,
            best_config={"x1": (5.0 - math.pi) / 15.0, "x2": 12.275 / 15.0},
        ),
        "sphere2d": BenchmarkProblem(
            name="sphere2d",
            space=_unit_space(["x1", "x2"]),
            objective=Objective(name="sphere", direction="max"),
            fn=_sphere,
            description="2D 单峰（最大化，最优 0 在 (0.3,0.3)）",
            optimum=0.0,
            best_config={"x1": 0.3, "x2": 0.3},
        ),
        "hartmann3": BenchmarkProblem(
            name="hartmann3",
            space=_unit_space(["x1", "x2", "x3"]),
            objective=Objective(name="hartmann3", direction="max"),
            fn=_hartmann3,
            description="3D 多峰（最大化，负值）",
            optimum=-3.86278,
            best_config={"x1": 0.114614, "x2": 0.555649, "x3": 0.852547},
        ),
        "mixed_reaction": BenchmarkProblem(
            name="mixed_reaction",
            space=SearchSpace.from_dict(
                [
                    {"name": "T", "kind": "continuous", "low": 40.0, "high": 120.0, "unit": "degC"},
                    {"name": "time", "kind": "continuous", "low": 5.0, "high": 120.0, "unit": "min"},
                    {"name": "conc", "kind": "log", "low": 0.02, "high": 1.0, "unit": "M"},
                    {
                        "name": "solvent",
                        "kind": "categorical",
                        "choices": ["MeCN", "EtOH", "water", "THF"],
                    },
                    {"name": "equiv", "kind": "integer", "low": 1, "high": 4},
                    {"name": "inert", "kind": "bool"},
                ]
            ),
            objective=Objective(name="yield", direction="max"),
            fn=_mixed_reaction,
            description="6D 混合空间，仿反应收率",
            optimum=102.0,
            best_config={
                "T": 82.0,
                "time": 55.0,
                "conc": 10**-0.8,
                "solvent": "MeCN",
                "equiv": 2,
                "inert": True,
            },
        ),
        "discrete_catalyst": BenchmarkProblem(
            name="discrete_catalyst",
            space=SearchSpace.from_dict(
                [
                    {"name": "metal", "kind": "categorical", "choices": ["Ni", "Pd", "Pt"]},
                    {"name": "ligand", "kind": "categorical", "choices": ["A", "B", "C", "D"]},
                    {"name": "temp", "kind": "integer", "low": 40, "high": 120},
                ]
            ),
            objective=Objective(name="yield", direction="max"),
            fn=_discrete_catalyst,
            description="3D 全离散（网格可穷举，12×81 组合）",
            optimum=78.0,
            best_config={"metal": "Pd", "ligand": "C", "temp": 80},
        ),
        "suzuki_like": BenchmarkProblem(
            name="suzuki_like",
            space=SearchSpace.from_dict(
                [
                    {"name": "T", "kind": "continuous", "low": 40.0, "high": 120.0, "unit": "degC"},
                    {"name": "time", "kind": "log", "low": 5.0, "high": 240.0, "unit": "min"},
                    {"name": "cat_loading", "kind": "log", "low": 0.1, "high": 5.0, "unit": "mol%"},
                    {
                        "name": "solvent",
                        "kind": "categorical",
                        "choices": ["MeCN", "EtOH", "THF", "water", "DMF"],
                    },
                    {
                        "name": "ligand",
                        "kind": "categorical",
                        "choices": ["L1", "L2", "L3", "L4", "L5", "L6"],
                    },
                    {"name": "base_eq", "kind": "integer", "low": 1, "high": 4},
                ]
            ),
            objective=Objective(name="yield", direction="max"),
            fn=_suzuki_like,
            description="Tier B：Suzuki 风格 6D 混合空间（log 变量中心最优 + 5/6 级分类）",
            optimum=110.0,
            best_config={
                "T": 85.0,
                "time": 60.0,
                "cat_loading": 1.5,
                "solvent": "MeCN",
                "ligand": "L1",
                "base_eq": 2,
            },
        ),
        "discrete_catalyst_v2": BenchmarkProblem(
            name="discrete_catalyst_v2",
            space=SearchSpace.from_dict(
                [
                    {"name": "metal", "kind": "categorical", "choices": ["Ni", "Pd", "Pt"]},
                    {"name": "ligand", "kind": "categorical", "choices": ["A", "B", "C", "D"]},
                    {"name": "additive", "kind": "categorical", "choices": ["X1", "X2", "X3", "X4"]},
                    {"name": "temp", "kind": "integer", "low": 40, "high": 120},
                ]
            ),
            objective=Objective(name="yield", direction="max"),
            fn=_discrete_catalyst_v2,
            description="Tier C：4 维全离散 + 交互项（穷举 3×4×4×81）",
            optimum=82.0,
            best_config={"metal": "Pd", "ligand": "C", "additive": "X3", "temp": 80},
        ),
        "high_cardinality_solvent": BenchmarkProblem(
            name="high_cardinality_solvent",
            space=SearchSpace.from_dict(
                [
                    {"name": "T", "kind": "continuous", "low": 60.0, "high": 120.0, "unit": "degC"},
                    {
                        "name": "solvent",
                        "kind": "categorical",
                        "choices": list(SOLVENT_DESCRIPTORS),
                    },
                ]
            ),
            objective=Objective(name="yield", direction="max"),
            fn=_high_cardinality_solvent,
            description="Tier C：12 级高基数分类溶剂（one-hot 会让 GP 维度膨胀）",
            optimum=92.371,
            best_config={"T": 90.0, "solvent": "S04"},
        ),
        "constraint_easy": BenchmarkProblem(
            name="constraint_easy",
            space=SearchSpace.from_dict(
                [
                    {"name": "T", "kind": "continuous", "low": 40.0, "high": 120.0, "unit": "degC"},
                    {"name": "time", "kind": "continuous", "low": 5.0, "high": 120.0, "unit": "min"},
                ]
            ),
            objective=Objective(
                name="yield",
                direction="max",
                constraints=("g",),
                constraint_limits={"g": {"op": "<=", "value": 0.0}},
            ),
            fn=_constraint_yield,
            constraints_fn=lambda config: {"g": _constraint_easy_g(config)},
            description="Tier E：椭圆可行域（约 25% 可行），约束不在最优点 binding",
            optimum=95.0,
            best_config={"T": 110.0, "time": 100.0},
        ),
        "constraint_hard": BenchmarkProblem(
            name="constraint_hard",
            space=SearchSpace.from_dict(
                [
                    {"name": "T", "kind": "continuous", "low": 40.0, "high": 120.0, "unit": "degC"},
                    {"name": "time", "kind": "continuous", "low": 5.0, "high": 120.0, "unit": "min"},
                ]
            ),
            objective=Objective(
                name="yield",
                direction="max",
                constraints=("g",),
                constraint_limits={"g": {"op": "<=", "value": 0.0}},
            ),
            fn=_constraint_yield,
            constraints_fn=lambda config: {"g": _constraint_hard_g(config)},
            description="Tier E：小可行域（约 6% 可行），约束 binding，需可行性感知搜索",
            optimum=93.096,
            best_config={"T": 105.9, "time": 94.8},
        ),
        "moo_yield_cost": BenchmarkProblem(
            name="moo_yield_cost",
            space=SearchSpace.from_dict(
                [
                    {"name": "T", "kind": "continuous", "low": 60.0, "high": 115.0, "unit": "degC"},
                    {"name": "time", "kind": "continuous", "low": 5.0, "high": 120.0, "unit": "min"},
                ]
            ),
            objective=Objective(
                name="yield",
                direction="max",
                objectives=("yield", "cost"),
                directions=("max", "min"),
                # raw-space reference: yield 60, cost 30 -> signed (60, -30)
                ref_point=(60.0, 30.0),
                objective_bounds={
                    "yield": {"low": 60.0, "high": 95.0},
                    "cost": {"low": 6.0, "high": 18.0},
                },
            ),
            fn=_moo_yield,
            objectives_fn=_moo_objectives,
            description="2 目标：max 收率 vs min 成本（T 与 time 的折中）",
        ),
        "moo_constrained": BenchmarkProblem(
            name="moo_constrained",
            space=SearchSpace.from_dict(
                [
                    {"name": "T", "kind": "continuous", "low": 60.0, "high": 115.0, "unit": "degC"},
                    {"name": "time", "kind": "continuous", "low": 5.0, "high": 120.0, "unit": "min"},
                ]
            ),
            objective=Objective(
                name="yield",
                direction="max",
                objectives=("yield", "cost"),
                directions=("max", "min"),
                constraints=("T_max", "time_max"),
                constraint_limits={
                    "T_max": {"op": "<=", "value": MOO_T_MAX},
                    "time_max": {"op": "<=", "value": MOO_TIME_MAX},
                },
                ref_point=(60.0, 30.0),
                objective_bounds={
                    "yield": {"low": 60.0, "high": 95.0},
                    "cost": {"low": 6.0, "high": 18.0},
                },
            ),
            fn=_moo_yield,
            objectives_fn=_moo_objectives,
            constraints_fn=_moo_constrained_constraint,
            description=(
                "2 目标 + 双硬约束：max 收率 vs min 成本，T ≤ 78 °C 且 time ≤ 50 min；"
                "可行域仅约 13%，无约束收率最优不可行，事后过滤会浪费预算"
            ),
            tags=("multi_objective", "constrained"),
        ),
        "moo3_constrained": BenchmarkProblem(
            name="moo3_constrained",
            space=SearchSpace.from_dict(
                [
                    {"name": "T", "kind": "continuous", "low": 60.0, "high": 115.0, "unit": "degC"},
                    {"name": "time", "kind": "continuous", "low": 5.0, "high": 120.0, "unit": "min"},
                ]
            ),
            objective=Objective(
                name="yield",
                direction="max",
                objectives=("yield", "cost", "purity"),
                directions=("max", "min", "max"),
                constraints=("T_max",),
                constraint_limits={"T_max": {"op": "<=", "value": 78.0}},
            ),
            fn=lambda config: 70.0
            + 12.0 * math.exp(-(((config["T"] - 82.0) / 18.0) ** 2))
            + 8.0 * math.exp(-(((config["time"] - 55.0) / 30.0) ** 2)),
            objectives_fn=lambda config: {
                "yield": 70.0
                + 12.0 * math.exp(-(((config["T"] - 82.0) / 18.0) ** 2))
                + 8.0 * math.exp(-(((config["time"] - 55.0) / 30.0) ** 2)),
                "cost": 6.0 + 0.10 * (config["T"] - 60.0) + 0.05 * (config["time"] - 5.0),
                "purity": 88.0 + 9.0 * math.exp(-(((config["T"] - 70.0) / 25.0) ** 2)),
            },
            constraints_fn=lambda config: {"T_max": float(config["T"])},
            description="3 目标 + 硬约束：max yield / min cost / max purity，T ≤ 78 °C",
            tags=("multi_objective", "constrained"),
        ),
        "moo4_constrained": BenchmarkProblem(
            name="moo4_constrained",
            space=SearchSpace.from_dict(
                [
                    {"name": "T", "kind": "continuous", "low": 60.0, "high": 115.0, "unit": "degC"},
                    {"name": "time", "kind": "continuous", "low": 5.0, "high": 120.0, "unit": "min"},
                ]
            ),
            objective=Objective(
                name="yield",
                direction="max",
                objectives=("yield", "cost", "purity", "selectivity"),
                directions=("max", "min", "max", "max"),
                constraints=("T_max",),
                constraint_limits={"T_max": {"op": "<=", "value": 78.0}},
                ref_point=(60.0, 30.0, 88.0, 80.0),
            ),
            fn=lambda config: 70.0
            + 12.0 * math.exp(-(((config["T"] - 82.0) / 18.0) ** 2))
            + 8.0 * math.exp(-(((config["time"] - 55.0) / 30.0) ** 2)),
            objectives_fn=lambda config: {
                "yield": 70.0
                + 12.0 * math.exp(-(((config["T"] - 82.0) / 18.0) ** 2))
                + 8.0 * math.exp(-(((config["time"] - 55.0) / 30.0) ** 2)),
                "cost": 6.0 + 0.10 * (config["T"] - 60.0) + 0.05 * (config["time"] - 5.0),
                "purity": 88.0 + 9.0 * math.exp(-(((config["T"] - 70.0) / 25.0) ** 2)),
                "selectivity": 82.0
                + 10.0 * math.exp(-(((config["T"] - 78.0) / 20.0) ** 2))
                + 4.0 * math.exp(-(((config["time"] - 45.0) / 25.0) ** 2)),
            },
            constraints_fn=lambda config: {"T_max": float(config["T"])},
            description="4 目标 + 硬约束：max yield / min cost / max purity / max selectivity，T ≤ 78 °C",
            tags=("multi_objective", "constrained"),
        ),
        "high_cardinality_solvent_desc": BenchmarkProblem(
            name="high_cardinality_solvent_desc",
            space=SearchSpace.from_dict(
                [
                    {"name": "T", "kind": "continuous", "low": 60.0, "high": 120.0, "unit": "degC"},
                    {
                        "name": "solvent",
                        "kind": "categorical",
                        "choices": list(SOLVENT_DESCRIPTORS),
                        "descriptors": {
                            name: list(vector) for name, vector in SOLVENT_DESCRIPTORS.items()
                        },
                        "descriptor_names": ["polarity", "protic"],
                    },
                ]
            ),
            objective=Objective(name="yield", direction="max"),
            fn=_high_cardinality_solvent,
            description=(
                "Tier C（描述符版）：同一 12 级溶剂空间，但用 polarity/protic 描述符替代 one-hot"
            ),
            optimum=92.371,
            best_config={"T": 90.0, "solvent": "S04"},
            tags=("categorical_high",),
        ),
        "rosenbrock10": BenchmarkProblem(
            name="rosenbrock10",
            space=SearchSpace.from_dict(
                [{"name": f"x{index}", "kind": "continuous", "low": 0.0, "high": 1.0}
                 for index in range(10)]
            ),
            objective=Objective(name="y", direction="max"),
            fn=_rosenbrock_high_dim,
            description="Tier G：10 维 Rosenbrock（光滑窄谷，测试中高维优化）",
            optimum=0.0,
            best_config={f"x{index}": 0.75 for index in range(10)},
            tags=("smooth", "continuous", "high_dim"),
        ),
        "ackley10": BenchmarkProblem(
            name="ackley10",
            space=SearchSpace.from_dict(
                [{"name": f"x{index}", "kind": "continuous", "low": 0.0, "high": 1.0}
                 for index in range(10)]
            ),
            objective=Objective(name="y", direction="max"),
            fn=_ackley_high_dim,
            description="Tier G：10 维 Ackley（多峰，全局最优在中心）",
            optimum=0.0,
            best_config={f"x{index}": 0.5 for index in range(10)},
            tags=("smooth", "continuous", "high_dim", "multimodal"),
        ),
        "mf_moo": BenchmarkProblem(
            name="mf_moo",
            space=SearchSpace.from_dict(
                [
                    {"name": "T", "kind": "continuous", "low": 60.0, "high": 115.0, "unit": "degC"},
                    {"name": "time", "kind": "continuous", "low": 5.0, "high": 120.0, "unit": "min"},
                ]
            ),
            objective=Objective(
                name="yield",
                direction="max",
                objectives=("yield", "cost"),
                directions=("max", "min"),
                ref_point=(60.0, 30.0),
                objective_bounds={
                    "yield": {"low": 60.0, "high": 95.0},
                    "cost": {"low": 6.0, "high": 18.0},
                },
            ),
            fn=lambda config: 70.0
            + 12.0 * math.exp(-(((config["T"] - 82.0) / 18.0) ** 2))
            + 8.0 * math.exp(-(((config["time"] - 55.0) / 30.0) ** 2)),
            objectives_fn=lambda config: {
                "yield": 70.0
                + 12.0 * math.exp(-(((config["T"] - 82.0) / 18.0) ** 2))
                + 8.0 * math.exp(-(((config["time"] - 55.0) / 30.0) ** 2)),
                "cost": 6.0 + 0.10 * (config["T"] - 60.0) + 0.05 * (config["time"] - 5.0),
            },
            fidelity_fn=lambda config, fidelity: (
                70.0
                + 12.0 * math.exp(-(((config["T"] - 82.0) / 18.0) ** 2))
                + 8.0 * math.exp(-(((config["time"] - 55.0) / 30.0) ** 2))
                - 15.0 * ((1.0 - float(fidelity)) ** 1.5)
            ),
            objectives_fidelity_fn=lambda config, fidelity: {
                "yield": 70.0
                + 12.0 * math.exp(-(((config["T"] - 82.0) / 18.0) ** 2))
                + 8.0 * math.exp(-(((config["time"] - 55.0) / 30.0) ** 2))
                - 15.0 * ((1.0 - float(fidelity)) ** 1.5),
                "cost": 6.0 + 0.10 * (config["T"] - 60.0) + 0.05 * (config["time"] - 5.0),
            },
            description="多目标 + 多保真：max yield / min cost，低保真对 yield 有负偏",
            tags=("multi_objective", "multi_fidelity"),
        ),
        "mf_moo_constrained": BenchmarkProblem(
            name="mf_moo_constrained",
            space=SearchSpace.from_dict(
                [
                    {"name": "T", "kind": "continuous", "low": 60.0, "high": 115.0, "unit": "degC"},
                    {"name": "time", "kind": "continuous", "low": 5.0, "high": 120.0, "unit": "min"},
                ]
            ),
            objective=Objective(
                name="yield",
                direction="max",
                objectives=("yield", "cost"),
                directions=("max", "min"),
                ref_point=(60.0, 30.0),
                constraints=("T_max",),
                constraint_limits={"T_max": {"op": "<=", "value": 75.0}},
                objective_bounds={
                    "yield": {"low": 60.0, "high": 95.0},
                    "cost": {"low": 6.0, "high": 18.0},
                },
            ),
            fn=lambda config: (
                70.0
                + 12.0 * math.exp(-(((config["T"] - 82.0) / 18.0) ** 2))
                + 8.0 * math.exp(-(((config["time"] - 55.0) / 30.0) ** 2))
            ),
            objectives_fn=lambda config: {
                "yield": 70.0
                + 12.0 * math.exp(-(((config["T"] - 82.0) / 18.0) ** 2))
                + 8.0 * math.exp(-(((config["time"] - 55.0) / 30.0) ** 2)),
                "cost": 6.0 + 0.10 * (config["T"] - 60.0) + 0.05 * (config["time"] - 5.0),
            },
            constraints_fn=lambda config: {"T_max": float(config["T"])},
            fidelity_fn=lambda config, fidelity: (
                70.0
                + 12.0 * math.exp(-(((config["T"] - 82.0) / 18.0) ** 2))
                + 8.0 * math.exp(-(((config["time"] - 55.0) / 30.0) ** 2))
                - 15.0 * ((1.0 - float(fidelity)) ** 1.5)
            ),
            objectives_fidelity_fn=lambda config, fidelity: {
                "yield": 70.0
                + 12.0 * math.exp(-(((config["T"] - 82.0) / 18.0) ** 2))
                + 8.0 * math.exp(-(((config["time"] - 55.0) / 30.0) ** 2))
                - 15.0 * ((1.0 - float(fidelity)) ** 1.5),
                "cost": 6.0 + 0.10 * (config["T"] - 60.0) + 0.05 * (config["time"] - 5.0),
            },
            description=(
                "多目标 + 多保真 + 硬约束：max yield / min cost，T ≤ 75 °C（约束 binding，"
                "无约束最优 T=82）"
            ),
            optimum=88.3159,
            best_config={"T": 75.0, "time": 55.0},
            tags=("multi_objective", "multi_fidelity", "constraints"),
        ),
        "solvent_constrained": BenchmarkProblem(
            name="solvent_constrained",
            space=SearchSpace.from_dict(
                [
                    {"name": "T", "kind": "continuous", "low": 60.0, "high": 120.0, "unit": "degC"},
                    {"name": "solvent", "kind": "categorical",
                     "choices": list(SOLVENT_DESCRIPTORS)},
                ]
            ),
            objective=Objective(
                name="yield",
                direction="max",
                constraints=("T_max",),
                constraint_limits={"T_max": {"op": "<=", "value": 80.0}},
            ),
            fn=lambda config: _solvent_constrained_yield(config),
            constraints_fn=lambda config: {"T_max": float(config["T"])},
            description=(
                "Tier C + 硬约束：12 级分类溶剂（one-hot），T ≤ 80 °C 且无约束最优在 T≈95"
            ),
            optimum=90.40438004436503,
            best_config={"T": 80.0, "solvent": "S04"},
            tags=("categorical_high", "constraints"),
        ),
        "solvent_constrained_desc": BenchmarkProblem(
            name="solvent_constrained_desc",
            space=SearchSpace.from_dict(
                [
                    {"name": "T", "kind": "continuous", "low": 60.0, "high": 120.0, "unit": "degC"},
                    {
                        "name": "solvent",
                        "kind": "categorical",
                        "choices": list(SOLVENT_DESCRIPTORS),
                        "descriptors": {
                            name: list(vector) for name, vector in SOLVENT_DESCRIPTORS.items()
                        },
                        "descriptor_names": ["polarity", "protic"],
                    },
                ]
            ),
            objective=Objective(
                name="yield",
                direction="max",
                constraints=("T_max",),
                constraint_limits={"T_max": {"op": "<=", "value": 80.0}},
            ),
            fn=lambda config: _solvent_constrained_yield(config),
            constraints_fn=lambda config: {"T_max": float(config["T"])},
            description=(
                "Tier C + 硬约束（描述符版）：polarity/protic 替代 one-hot，同一 landscape"
            ),
            optimum=90.40438004436503,
            best_config={"T": 80.0, "solvent": "S04"},
            tags=("categorical_high", "constraints"),
        ),
        "moo_solvent_constrained": BenchmarkProblem(
            name="moo_solvent_constrained",
            space=SearchSpace.from_dict(
                [
                    {"name": "T", "kind": "continuous", "low": 60.0, "high": 120.0, "unit": "degC"},
                    {"name": "solvent", "kind": "categorical",
                     "choices": list(SOLVENT_DESCRIPTORS)},
                ]
            ),
            objective=Objective(
                name="yield",
                direction="max",
                objectives=("yield", "cost"),
                directions=("max", "min"),
                constraints=("T_max",),
                constraint_limits={"T_max": {"op": "<=", "value": 80.0}},
                ref_point=(55.0, 40.0),
                objective_bounds={
                    "yield": {"low": 55.0, "high": 92.0},
                    "cost": {"low": 5.0, "high": 22.0},
                },
            ),
            fn=lambda config: _mf_solvent_objectives(config, 1.0)["yield"],
            objectives_fn=lambda config: _mf_solvent_objectives(config, 1.0),
            constraints_fn=lambda config: {"T_max": float(config["T"])},
            description=(
                "2 目标 + 硬约束 + 12 级分类溶剂（one-hot）：T ≤ 80 °C 时可行前沿缩小，"
                "用于验证“描述符 × 约束感知算法”的配对建议"
            ),
            best_config={"T": 80.0, "solvent": "S04"},
            tags=("multi_objective", "constraints", "categorical_high"),
        ),
        "moo_solvent_constrained_desc": BenchmarkProblem(
            name="moo_solvent_constrained_desc",
            space=SearchSpace.from_dict(
                [
                    {"name": "T", "kind": "continuous", "low": 60.0, "high": 120.0, "unit": "degC"},
                    {
                        "name": "solvent",
                        "kind": "categorical",
                        "choices": list(SOLVENT_DESCRIPTORS),
                        "descriptors": {
                            name: list(vector) for name, vector in SOLVENT_DESCRIPTORS.items()
                        },
                        "descriptor_names": ["polarity", "protic"],
                    },
                ]
            ),
            objective=Objective(
                name="yield",
                direction="max",
                objectives=("yield", "cost"),
                directions=("max", "min"),
                constraints=("T_max",),
                constraint_limits={"T_max": {"op": "<=", "value": 80.0}},
                ref_point=(55.0, 40.0),
                objective_bounds={
                    "yield": {"low": 55.0, "high": 92.0},
                    "cost": {"low": 5.0, "high": 22.0},
                },
            ),
            fn=lambda config: _mf_solvent_objectives(config, 1.0)["yield"],
            objectives_fn=lambda config: _mf_solvent_objectives(config, 1.0),
            constraints_fn=lambda config: {"T_max": float(config["T"])},
            description="2 目标 + 硬约束（描述符版）：polarity/protic 替代 one-hot，同一 landscape",
            best_config={"T": 80.0, "solvent": "S04"},
            tags=("multi_objective", "constraints", "categorical_high"),
        ),
        "mf_solvent": BenchmarkProblem(
            name="mf_solvent",
            space=SearchSpace.from_dict(
                [
                    {"name": "T", "kind": "continuous", "low": 60.0, "high": 120.0, "unit": "degC"},
                    {"name": "solvent", "kind": "categorical",
                     "choices": list(SOLVENT_DESCRIPTORS)},
                ]
            ),
            objective=Objective(
                name="yield",
                direction="max",
                objectives=("yield", "cost"),
                directions=("max", "min"),
                ref_point=(55.0, 40.0),
                objective_bounds={
                    "yield": {"low": 55.0, "high": 92.0},
                    "cost": {"low": 5.0, "high": 22.0},
                },
            ),
            fn=lambda config: _mf_solvent_objectives(config, 1.0)["yield"],
            objectives_fn=lambda config: _mf_solvent_objectives(config, 1.0),
            fidelity_fn=lambda config, fidelity: _mf_solvent_objectives(config, fidelity)["yield"],
            objectives_fidelity_fn=_mf_solvent_objectives,
            description=(
                "多目标 + 多保真 + 12 级分类溶剂（one-hot）：max yield / min cost，"
                "低保真对 yield 负偏"
            ),
            best_config={"T": 95.0, "solvent": "S04"},
            tags=("multi_objective", "multi_fidelity", "categorical_high"),
        ),
        "mf_solvent_desc": BenchmarkProblem(
            name="mf_solvent_desc",
            space=SearchSpace.from_dict(
                [
                    {"name": "T", "kind": "continuous", "low": 60.0, "high": 120.0, "unit": "degC"},
                    {
                        "name": "solvent",
                        "kind": "categorical",
                        "choices": list(SOLVENT_DESCRIPTORS),
                        "descriptors": {
                            name: list(vector) for name, vector in SOLVENT_DESCRIPTORS.items()
                        },
                        "descriptor_names": ["polarity", "protic"],
                    },
                ]
            ),
            objective=Objective(
                name="yield",
                direction="max",
                objectives=("yield", "cost"),
                directions=("max", "min"),
                ref_point=(55.0, 40.0),
                objective_bounds={
                    "yield": {"low": 55.0, "high": 92.0},
                    "cost": {"low": 5.0, "high": 22.0},
                },
            ),
            fn=lambda config: _mf_solvent_objectives(config, 1.0)["yield"],
            objectives_fn=lambda config: _mf_solvent_objectives(config, 1.0),
            fidelity_fn=lambda config, fidelity: _mf_solvent_objectives(config, fidelity)["yield"],
            objectives_fidelity_fn=_mf_solvent_objectives,
            description=(
                "多目标 + 多保真 + 12 级分类溶剂（polarity/protic 描述符）：同一 landscape"
            ),
            best_config={"T": 95.0, "solvent": "S04"},
            tags=("multi_objective", "multi_fidelity", "categorical_high"),
        ),
        "mf_moo4": BenchmarkProblem(
            name="mf_moo4",
            space=SearchSpace.from_dict(
                [
                    {"name": "T", "kind": "continuous", "low": 60.0, "high": 115.0, "unit": "degC"},
                    {"name": "time", "kind": "continuous", "low": 5.0, "high": 120.0, "unit": "min"},
                ]
            ),
            objective=Objective(
                name="yield",
                direction="max",
                objectives=("yield", "cost", "selectivity", "purity"),
                directions=("max", "min", "max", "max"),
                ref_point=(60.0, 30.0, 50.0, 80.0),
                objective_bounds={
                    "yield": {"low": 60.0, "high": 95.0},
                    "cost": {"low": 6.0, "high": 18.0},
                    "selectivity": {"low": 55.0, "high": 92.0},
                    "purity": {"low": 82.0, "high": 99.0},
                },
            ),
            fn=lambda config: (
                70.0
                + 12.0 * math.exp(-(((config["T"] - 82.0) / 18.0) ** 2))
                + 8.0 * math.exp(-(((config["time"] - 55.0) / 30.0) ** 2))
            ),
            objectives_fn=lambda config: {
                "yield": 70.0
                + 12.0 * math.exp(-(((config["T"] - 82.0) / 18.0) ** 2))
                + 8.0 * math.exp(-(((config["time"] - 55.0) / 30.0) ** 2)),
                "cost": 6.0 + 0.10 * (config["T"] - 60.0) + 0.05 * (config["time"] - 5.0),
                "selectivity": 55.0
                + 30.0 * math.exp(-(((config["T"] - 74.0) / 22.0) ** 2))
                + 6.0 * math.exp(-(((config["time"] - 40.0) / 35.0) ** 2)),
                "purity": 82.0
                + 15.0 * math.exp(-(((config["T"] - 88.0) / 20.0) ** 2))
                + 4.0 * math.exp(-(((config["time"] - 70.0) / 40.0) ** 2)),
            },
            fidelity_fn=lambda config, fidelity: (
                70.0
                + 12.0 * math.exp(-(((config["T"] - 82.0) / 18.0) ** 2))
                + 8.0 * math.exp(-(((config["time"] - 55.0) / 30.0) ** 2))
                - 15.0 * ((1.0 - float(fidelity)) ** 1.5)
            ),
            objectives_fidelity_fn=lambda config, fidelity: {
                "yield": 70.0
                + 12.0 * math.exp(-(((config["T"] - 82.0) / 18.0) ** 2))
                + 8.0 * math.exp(-(((config["time"] - 55.0) / 30.0) ** 2))
                - 15.0 * ((1.0 - float(fidelity)) ** 1.5),
                "cost": 6.0 + 0.10 * (config["T"] - 60.0) + 0.05 * (config["time"] - 5.0),
                "selectivity": 55.0
                + 30.0 * math.exp(-(((config["T"] - 74.0) / 22.0) ** 2))
                + 6.0 * math.exp(-(((config["time"] - 40.0) / 35.0) ** 2))
                - 9.0 * ((1.0 - float(fidelity)) ** 1.5),
                "purity": 82.0
                + 15.0 * math.exp(-(((config["T"] - 88.0) / 20.0) ** 2))
                + 4.0 * math.exp(-(((config["time"] - 70.0) / 40.0) ** 2))
                - 6.0 * ((1.0 - float(fidelity)) ** 1.5),
            },
            description=(
                "多目标（4）+ 多保真：max yield / min cost / max selectivity / max purity，"
                "低保真对三个“越大越好”的目标都有负偏"
            ),
            best_config={"T": 82.0, "time": 55.0},
            tags=("multi_objective", "multi_fidelity"),
        ),
        "mf_moo3": BenchmarkProblem(
            name="mf_moo3",
            space=SearchSpace.from_dict(
                [
                    {"name": "T", "kind": "continuous", "low": 60.0, "high": 115.0, "unit": "degC"},
                    {"name": "time", "kind": "continuous", "low": 5.0, "high": 120.0, "unit": "min"},
                ]
            ),
            objective=Objective(
                name="yield",
                direction="max",
                objectives=("yield", "cost", "selectivity"),
                directions=("max", "min", "max"),
                ref_point=(60.0, 30.0, 50.0),
                objective_bounds={
                    "yield": {"low": 60.0, "high": 95.0},
                    "cost": {"low": 6.0, "high": 18.0},
                    "selectivity": {"low": 55.0, "high": 92.0},
                },
            ),
            fn=lambda config: (
                70.0
                + 12.0 * math.exp(-(((config["T"] - 82.0) / 18.0) ** 2))
                + 8.0 * math.exp(-(((config["time"] - 55.0) / 30.0) ** 2))
            ),
            objectives_fn=lambda config: {
                "yield": 70.0
                + 12.0 * math.exp(-(((config["T"] - 82.0) / 18.0) ** 2))
                + 8.0 * math.exp(-(((config["time"] - 55.0) / 30.0) ** 2)),
                "cost": 6.0 + 0.10 * (config["T"] - 60.0) + 0.05 * (config["time"] - 5.0),
                "selectivity": 55.0
                + 30.0 * math.exp(-(((config["T"] - 74.0) / 22.0) ** 2))
                + 6.0 * math.exp(-(((config["time"] - 40.0) / 35.0) ** 2)),
            },
            fidelity_fn=lambda config, fidelity: (
                70.0
                + 12.0 * math.exp(-(((config["T"] - 82.0) / 18.0) ** 2))
                + 8.0 * math.exp(-(((config["time"] - 55.0) / 30.0) ** 2))
                - 15.0 * ((1.0 - float(fidelity)) ** 1.5)
            ),
            objectives_fidelity_fn=lambda config, fidelity: {
                "yield": 70.0
                + 12.0 * math.exp(-(((config["T"] - 82.0) / 18.0) ** 2))
                + 8.0 * math.exp(-(((config["time"] - 55.0) / 30.0) ** 2))
                - 15.0 * ((1.0 - float(fidelity)) ** 1.5),
                "cost": 6.0 + 0.10 * (config["T"] - 60.0) + 0.05 * (config["time"] - 5.0),
                "selectivity": 55.0
                + 30.0 * math.exp(-(((config["T"] - 74.0) / 22.0) ** 2))
                + 6.0 * math.exp(-(((config["time"] - 40.0) / 35.0) ** 2))
                - 9.0 * ((1.0 - float(fidelity)) ** 1.5),
            },
            description=(
                "多目标（3）+ 多保真：max yield / min cost / max selectivity，"
                "低保真对 yield 与 selectivity 都有负偏"
            ),
            best_config={"T": 80.0, "time": 45.0},
            tags=("multi_objective", "multi_fidelity"),
        ),
        "mf_constrained": BenchmarkProblem(
            name="mf_constrained",
            space=SearchSpace.from_dict(
                [
                    {"name": "T", "kind": "continuous", "low": 40.0, "high": 120.0, "unit": "degC"},
                    {"name": "time", "kind": "log", "low": 5.0, "high": 240.0, "unit": "min"},
                    {"name": "cat_loading", "kind": "log", "low": 0.1, "high": 5.0, "unit": "mol%"},
                    {"name": "solvent", "kind": "categorical",
                     "choices": ["MeCN", "EtOH", "THF", "water", "DMF"]},
                    {"name": "ligand", "kind": "categorical",
                     "choices": ["L1", "L2", "L3", "L4", "L5", "L6"]},
                    {"name": "base_eq", "kind": "integer", "low": 1, "high": 4},
                ]
            ),
            objective=Objective(
                name="yield",
                direction="max",
                constraints=("T_max",),
                constraint_limits={"T_max": {"op": "<=", "value": 80.0}},
            ),
            fn=lambda config: mf_suzuki(config, 1.0),
            constraints_fn=lambda config: {"T_max": float(config["T"])},
            fidelity_fn=mf_suzuki,
            description=(
                "多保真 + 硬约束：Suzuki 风格产率（低保真有偏），T ≤ 80 °C；"
                "无约束最优约 T=85，约束 binding"
            ),
            optimum=105.15304502507806,
            best_config={
                "T": 80.0,
                "time": 60.0,
                "cat_loading": 1.5,
                "solvent": "MeCN",
                "ligand": "L1",
                "base_eq": 2,
            },
            tags=("multi_fidelity", "constrained"),
        ),
    }


@dataclass
class BenchmarkResult:
    problem: str
    algorithm: str
    seed: int
    budget: int
    best_score: float
    best_value: float
    trajectory: list[float]
    #: raw objective vectors from every evaluated point (multi-objective /
    #: records-v2 true-HV metrics); empty for single-objective runs.
    objective_vectors: list[dict] = field(default_factory=list)
    #: raw constraint vectors, one per evaluated point when the problem
    #: declares ``constraints_fn`` (constrained problems / records-v2 metrics)
    constraint_vectors: list[dict] = field(default_factory=list)
    #: best TRUE score among feasible observations (constrained problems)
    best_feasible_score: float | None = None

    def to_dict(self) -> dict:
        return {
            "problem": self.problem,
            "algorithm": self.algorithm,
            "seed": self.seed,
            "budget": self.budget,
            "best_score": self.best_score,
            "best_value": self.best_value,
            "trajectory": list(self.trajectory),
        }


def run_algorithm(
    problem: BenchmarkProblem,
    algorithm: str,
    budget: int = 20,
    batch_size: int = 1,
    seed: int = 0,
    params: dict | None = None,
    noise_std: float = 0.0,
) -> BenchmarkResult:
    """Run one algorithm on one benchmark problem and return its trajectory."""
    if params is None:
        # Use the router's parameter policy so model-based algorithms do not
        # waste 80% of a small budget on the initial design.  This mirrors what
        # the product actually does (`init --algorithm auto`).
        try:
            from .router import OptimizationContext, _params_for

            context = OptimizationContext.from_space(
                problem.space,
                budget_total=budget,
                batch_size=batch_size,
                n_objectives=problem.objective.n_objectives,
            )
            params = _params_for(algorithm, context)
        except Exception:  # pragma: no cover - benchmark-only convenience
            params = None
    optimizer = registry.build(
        algorithm,
        problem.space,
        problem.objective,
        params=params,
        seed=seed,
    )
    noise_rng = np.random.default_rng(seed + 9973)
    best_score = -math.inf
    best_value = math.nan
    trajectory: list[float] = []
    objective_vectors: list[dict] = []
    constraint_vectors: list[dict] = []
    best_feasible_score: float | None = None
    done = 0
    while done < budget:
        n = min(batch_size, budget - done)
        configs = optimizer.ask(n)
        observations = []
        for config in configs:
            observation = problem.make_observation(config)
            told = observation
            if noise_std > 0:
                # The optimizer sees noisy measurements while the metric still
                # uses the noiseless truth (records-v2 noise tasks).
                told = Observation(
                    x=config,
                    y=(
                        observation.y + float(noise_rng.normal(0.0, noise_std))
                        if observation.y is not None
                        else None
                    ),
                    objectives={
                        key: value + float(noise_rng.normal(0.0, noise_std))
                        for key, value in observation.objectives.items()
                    },
                    constraints=dict(observation.constraints),
                    source="benchmark_noisy",
                )
            observations.append(told)
            if observation.objectives:
                objective_vectors.append(dict(observation.objectives))
            if problem.constraints_fn is not None:
                constraint_vectors.append(dict(observation.constraints))
            if problem.objective.constraint_limits:
                # Three-state rule (A12): a missing measurement is UNKNOWN, not a
                # violation, and it never counts as feasible.  The previous code
                # additionally skipped observations with an empty ``constraints`` dict
                # entirely (they were invisible instead of unknown).
                state = _feasibility_state(observation, problem.objective)["state"]
                if state == "true":
                    feasible_score = observation.score(
                        problem.objective.name,
                        problem.objective.direction_for(problem.objective.name),
                    )
                    best_feasible_score = (
                        feasible_score
                        if best_feasible_score is None
                        else max(best_feasible_score, feasible_score)
                    )
            score = observation.score(
                problem.objective.name,
                problem.objective.direction_for(problem.objective.name),
            )
            if score > best_score:
                best_score = score
                best_value = observation.y if observation.y is not None else best_value
        optimizer.tell(observations)
        done += len(observations)
        trajectory.append(best_score)
    return BenchmarkResult(
        problem=problem.name,
        algorithm=algorithm,
        seed=seed,
        budget=done,
        best_score=best_score,
        best_value=best_value,
        trajectory=trajectory,
        objective_vectors=objective_vectors,
        constraint_vectors=constraint_vectors,
        best_feasible_score=best_feasible_score,
    )


def run_suite(
    problems: Sequence[str] | None = None,
    algorithms: Sequence[str] | None = None,
    budget: int = 20,
    batch_size: int = 1,
    seeds: Sequence[int] = (0, 1, 2),
) -> list[BenchmarkResult]:
    available = builtin_problems()
    problem_names = list(problems or available)
    results: list[BenchmarkResult] = []
    for problem_name in problem_names:
        problem = available[problem_name]
        if algorithms is None:
            if problem.objective.is_multi_objective:
                algorithm_names = ["random", "sobol", "parego"]
                if problem.objective.constraints and problem.objective.n_objectives == 2:
                    algorithm_names.append("cqehvi")
            else:
                algorithm_names = ["random", "sobol", "lhs", "bayes", "tpe"]
        else:
            algorithm_names = list(algorithms)
        for algorithm in algorithm_names:
            for seed in seeds:
                results.append(
                    run_algorithm(
                        problem,
                        algorithm,
                        budget=budget,
                        batch_size=batch_size,
                        seed=seed,
                    )
                )
    return results


def summarize(results: Sequence[BenchmarkResult]) -> list[dict]:
    """Aggregate mean best score per (problem, algorithm)."""
    grouped: dict[tuple[str, str], list[BenchmarkResult]] = {}
    for result in results:
        grouped.setdefault((result.problem, result.algorithm), []).append(result)
    summary = []
    for (problem, algorithm), runs in sorted(grouped.items()):
        scores = [run.best_score for run in runs]
        summary.append(
            {
                "problem": problem,
                "algorithm": algorithm,
                "n_runs": len(runs),
                "mean_best_score": float(np.mean(scores)),
                "std_best_score": float(np.std(scores)),
                "mean_final_trajectory": float(np.mean([run.trajectory[-1] for run in runs])),
            }
        )
    return summary
