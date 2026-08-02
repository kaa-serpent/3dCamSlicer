"""Cylindrical-grid tool compensation without target mutation."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from rotarycam.geometry.models import RotaryGrid
from rotarycam.tools.cutter_kernel import CutterKernel, kernel_for_tool
from rotarycam.tools.models import Tool


def periodic_angle_distance_deg(
    angles_deg: npt.ArrayLike,
    center_deg: float,
) -> npt.NDArray[np.float64]:
    """Return unsigned shortest angular distances in degrees."""

    angles = np.asarray(angles_deg, dtype=np.float64)
    return np.abs((angles - center_deg + 180.0) % 360.0 - 180.0)


def conservative_lateral_distance(
    x_values: npt.ArrayLike,
    angles_deg: npt.ArrayLike,
    radii: npt.ArrayLike,
    *,
    center_x: float,
    center_angle_deg: float,
    center_radius: float,
) -> npt.NDArray[np.float64]:
    """Underestimate local surface distance for safe discrete compensation."""

    x_axis = np.asarray(x_values, dtype=np.float64)
    angle_axis = np.asarray(angles_deg, dtype=np.float64)
    neighbor_radius = np.asarray(radii, dtype=np.float64)
    if neighbor_radius.shape != (x_axis.size, angle_axis.size):
        raise ValueError("radii shape must match x_values and angles_deg")
    dx = x_axis[:, None] - center_x
    delta_angle = np.radians(periodic_angle_distance_deg(angle_axis, center_angle_deg))[None, :]
    # A chord at the smaller endpoint radius is no longer than the actual local
    # surface path. Underestimating distance lowers the kernel and therefore moves
    # the compensated TCP outward, which is the safe direction.
    chord_radius = np.minimum(np.maximum(neighbor_radius, 0.0), max(center_radius, 0.0))
    tangent = 2.0 * chord_radius * np.sin(delta_angle / 2.0)
    return np.hypot(dx, tangent)


def _compensate_cell(
    target: RotaryGrid,
    kernel: CutterKernel,
    x_index: int,
    angle_index: int,
) -> float:
    center_radius = float(target.radius[x_index, angle_index])
    x_delta = np.abs(target.x_values - target.x_values[x_index])
    x_candidates = np.flatnonzero(x_delta <= kernel.footprint_radius)
    neighbor_radii = target.radius[x_candidates]
    distance = conservative_lateral_distance(
        target.x_values[x_candidates],
        target.angles_deg,
        neighbor_radii,
        center_x=float(target.x_values[x_index]),
        center_angle_deg=float(target.angles_deg[angle_index]),
        center_radius=center_radius,
    )
    in_footprint = (distance <= kernel.footprint_radius) & target.valid[x_candidates]
    heights = kernel.height_at(distance)
    candidates = np.where(in_footprint, neighbor_radii - heights, -np.inf)
    return max(center_radius, float(np.max(candidates)))


def compute_compensated_tool_grid(target: RotaryGrid, tool: Tool) -> RotaryGrid:
    """Compute a conservative TCP radius envelope for a flat or ball end mill.

    The returned radius is the radial position of the tool tip. For every valid
    grid sample covered by the discrete cutter footprint, the kernel's lower
    surface remains at or outside the target radius.
    """

    kernel = kernel_for_tool(tool)
    compensated = np.zeros(target.shape, dtype=np.float64)
    for x_index, angle_index in np.ndindex(target.shape):
        if target.valid[x_index, angle_index]:
            compensated[x_index, angle_index] = _compensate_cell(
                target, kernel, x_index, angle_index
            )
    return RotaryGrid(
        target.x_values,
        target.angles_deg,
        compensated,
        target.valid,
        target.undercut_status,
        target.warnings,
    )
