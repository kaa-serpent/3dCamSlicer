"""Discrete cylindrical stock simulation."""

from rotarycam.simulation.cutter_kernel import (
    ball_cutter_surface,
    cutter_kernel,
    flat_cutter_surface,
    lateral_distance,
    tapered_cutter_surface,
)
from rotarycam.simulation.models import SimulationResult
from rotarycam.simulation.residual import ResidualResult, compute_residual
from rotarycam.simulation.stock_simulator import removed_volume, simulate_toolpath

__all__ = [
    "ResidualResult",
    "SimulationResult",
    "ball_cutter_surface",
    "compute_residual",
    "cutter_kernel",
    "flat_cutter_surface",
    "lateral_distance",
    "removed_volume",
    "simulate_toolpath",
    "tapered_cutter_surface",
]
