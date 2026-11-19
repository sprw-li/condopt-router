"""condopt: condition optimization algorithms with a transparent router.

``condopt`` is the algorithm layer of a condition-optimization skill: a
set of interchangeable ask/tell optimizers over a mixed search space, a
registry with machine-readable parameter schemas, and a rule-based router
that picks an algorithm and its parameters from the problem context.

Quick start
-----------
>>> from condopt import SearchSpace, Objective, Study
>>> space = SearchSpace.from_dict([{"name": "T", "kind": "continuous", "low": 40, "high": 120}])
>>> study = Study.create("runs/demo", space, algorithm="auto", budget=20)
>>> suggestions = study.ask(4)
>>> # ... run the experiments, then:
>>> study.tell([{"id": s["id"], "y": 0.5} for s in suggestions])
"""

from __future__ import annotations

from . import algorithms  # noqa: F401  (registers built-in algorithms)
from .base import AlgorithmSpec, Objective, Optimizer, ParamSpec
from .dataset import Dataset, Observation
from .registry import build, describe, describe_all, names, spec, specs, validate_params
from .router import OptimizationContext, Recommendation, route, route_round
from .space import SearchSpace, Variable
from .study import PendingError, Study
from .spec import Goal, Spec

__version__ = "0.1.0"

__all__ = [
    "AlgorithmSpec",
    "Dataset",
    "Objective",
    "Observation",
    "OptimizationContext",
    "Optimizer",
    "ParamSpec",
    "PendingError",
    "Recommendation",
    "SearchSpace",
    "Goal",
    "Spec",
    "Study",
    "Variable",
    "build",
    "describe",
    "describe_all",
    "names",
    "route",
    "route_round",
    "spec",
    "specs",
    "validate_params",
    "__version__",
]
