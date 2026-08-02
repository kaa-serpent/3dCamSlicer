"""Conservative first-pass tool accessibility on cylindrical grids."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from rotarycam.geometry.models import RotaryGrid
from rotarycam.tools.compensation import (
    compute_compensated_tool_grid,
    conservative_lateral_distance,
)
from rotarycam.tools.models import Tool

BoolArray = npt.NDArray[np.bool_]


def _same_layout(left: RotaryGrid, right: RotaryGrid) -> bool:
    return np.array_equal(left.x_values, right.x_values) and np.array_equal(
        left.angles_deg, right.angles_deg
    )


def _shank_clearance_mask(target: RotaryGrid, tcp: RotaryGrid, tool: Tool) -> BoolArray:
    """Reject target geometry colliding with the shank above flute length."""

    accessible = np.ones(target.shape, dtype=np.bool_)
    shank_radius = tool.shank_diameter / 2.0
    if shank_radius <= tool.diameter / 2.0:
        return accessible
    for x_index, angle_index in np.ndindex(target.shape):
        if not target.valid[x_index, angle_index]:
            accessible[x_index, angle_index] = False
            continue
        x_delta = np.abs(target.x_values - target.x_values[x_index])
        x_candidates = np.flatnonzero(x_delta <= shank_radius)
        neighbors = target.radius[x_candidates]
        distance = conservative_lateral_distance(
            target.x_values[x_candidates],
            target.angles_deg,
            neighbors,
            center_x=float(target.x_values[x_index]),
            center_angle_deg=float(target.angles_deg[angle_index]),
            center_radius=float(target.radius[x_index, angle_index]),
        )
        shank_zone = (distance <= shank_radius) & target.valid[x_candidates]
        shank_floor = tcp.radius[x_index, angle_index] + tool.flute_length
        if np.any(neighbors[shank_zone] > shank_floor):
            accessible[x_index, angle_index] = False
    return accessible


def compute_accessibility_mask(
    target: RotaryGrid,
    current_stock: RotaryGrid,
    tool: Tool,
) -> BoolArray:
    """Return cells reachable within cutting length and simplified shank clearance."""

    if not _same_layout(target, current_stock):
        raise ValueError("target and current_stock must use the same X/A layout")
    common_valid = target.valid & current_stock.valid
    if np.any(target.radius[common_valid] > current_stock.radius[common_valid]):
        raise ValueError("target radius must not exceed current stock radius")
    tcp = compute_compensated_tool_grid(target, tool)
    required_reach = np.maximum(0.0, current_stock.radius - target.radius)
    length_ok = required_reach <= tool.cutting_length + 1e-12
    shank_ok = _shank_clearance_mask(target, tcp, tool)
    result = common_valid & length_ok & shank_ok
    result.setflags(write=False)
    return result
