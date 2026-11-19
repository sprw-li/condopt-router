"""Built-in algorithm families.

Importing this package registers every built-in optimizer.  Keep the
import list explicit: registration is a side effect, and a missing import
is a silently missing algorithm in the router.
"""

from . import (  # noqa: F401
    bayes,
    cmaes,
    constrained,
    cqehvi,
    cqnehvi,
    ehvi_nd,
    mfbo,
    mfparego,
    momfbo,
    mqehvi_nd,
    parego,
    qehvi,
    qnehvi,
    sampling,
    tpe,
)
from .bayes import GaussianProcessBO
from .cmaes import CMAES
from .constrained import ConstrainedEI
from .cqehvi import ConstrainedQEHVI
from .cqnehvi import ConstrainedQNEHVI
from .ehvi_nd import MultiObjectiveEHVI
from .mfbo import MultiFidelityBO
from .mfparego import MultiFidelityParEGO
from .momfbo import MultiObjectiveMultiFidelityBO
from .mqehvi_nd import MultiFidelityEHVI
from .parego import ParEGO
from .qehvi import QEHVI
from .qnehvi import QNEHVI
from .sampling import GridSearch, LatinHypercube, RandomSearch, SobolSearch
from .tpe import TreeParzenEstimator

__all__ = [
    "CMAES",
    "ConstrainedEI",
    "ConstrainedQEHVI",
    "ConstrainedQNEHVI",
    "GaussianProcessBO",
    "GridSearch",
    "LatinHypercube",
    "MultiFidelityBO",
    "MultiFidelityParEGO",
    "MultiObjectiveEHVI",
    "MultiObjectiveMultiFidelityBO",
    "MultiFidelityEHVI",
    "ParEGO",
    "QEHVI",
    "QNEHVI",
    "RandomSearch",
    "SobolSearch",
    "TreeParzenEstimator",
]
