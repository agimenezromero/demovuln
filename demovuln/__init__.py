"""Demographic vulnerability analysis for matrix population models.

``demovuln`` provides tools to simulate temporally structured perturbations of
matrix population models and to summarize their demographic consequences as
population reduction and integrated vulnerability metrics.

The core workflow is:

1. define an unperturbed projection matrix with :class:`MatrixPopulationModel`;
2. simulate a single perturbation regime with :func:`simulate_dynamics`; or
3. evaluate a perturbation space with :class:`PerturbationGrid` and
   :func:`run_grid`.
"""

from .models import MatrixPopulationModel, dominant_eigenvalue, stable_stage_distribution
from .perturbations import Target, build_target_mask, apply_perturbation
from .simulation import SimulationResult, simulate_dynamics, population_reduction
from .vulnerability import PerturbationGrid, GridResult, run_grid, compute_vulnerability

__version__ = "0.1.0"

__all__ = [
    "MatrixPopulationModel",
    "dominant_eigenvalue",
    "stable_stage_distribution",
    "Target",
    "build_target_mask",
    "apply_perturbation",
    "SimulationResult",
    "simulate_dynamics",
    "population_reduction",
    "PerturbationGrid",
    "GridResult",
    "run_grid",
    "compute_vulnerability",
]
