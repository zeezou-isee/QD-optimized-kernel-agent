"""Inner parameter search (Workflow §6): analytic prune + coarse grid + hill climb."""

from .hardware_specs import HardwareSpecs, detect
from .constraint_engine import ConstraintEngine, FeasibilityReport, safe_eval
from .coarse_grid import coarse_points
from .hill_climb import hill_climb
from .inner_search import inner_search

__all__ = [
    "HardwareSpecs", "detect",
    "ConstraintEngine", "FeasibilityReport", "safe_eval",
    "coarse_points", "hill_climb", "inner_search",
]
