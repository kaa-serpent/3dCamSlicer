"""Cross-object validation for automatic planning inputs."""

from __future__ import annotations

import numpy as np

from rotarycam.geometry.models import RotaryGrid
from rotarycam.tools.models import Tool


def validate_planning_inputs(target: RotaryGrid, stock: RotaryGrid, tools: list[Tool]) -> None:
    """Reject incompatible grids and invalid cutter collections before planning."""
    if target.shape != stock.shape:
        raise ValueError("target and stock grids must have the same shape")
    if not np.array_equal(target.x_values, stock.x_values) or not np.array_equal(
        target.angles_deg,
        stock.angles_deg,
    ):
        raise ValueError("target and stock grids must use the same X/A coordinates")
    common_valid = target.valid & stock.valid
    if np.any(target.radius[common_valid] > stock.radius[common_valid] + 1e-9):
        raise ValueError("target radius must not exceed initial stock radius")
    if not tools:
        raise ValueError("at least one tool is required")
