"""Discrete cutter footprints for the cylindrical stock grid."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from rotarycam.geometry.models import RotaryGrid
from rotarycam.toolpath.models import ToolpathPoint
from rotarycam.tools.models import Tool, ToolType

FloatArray = npt.NDArray[np.float64]


def lateral_distance(grid: RotaryGrid, point: ToolpathPoint) -> FloatArray:
    """Return unwrapped-surface distance from a cutter axis to every grid cell."""

    delta_x = np.asarray(grid.x_values)[:, None] - point.x
    delta_angle_deg = (
        (np.asarray(grid.angles_deg)[None, :] - point.a + 180.0) % 360.0
    ) - 180.0
    tangent_distance = np.deg2rad(delta_angle_deg) * np.asarray(grid.radius)
    return np.asarray(np.hypot(delta_x, tangent_distance), dtype=np.float64)


def flat_cutter_surface(
    tip_radius: float,
    distance: npt.ArrayLike,
    cutter_radius: float,
) -> FloatArray:
    """Radial swept surface of a flat cutter; infinity is outside its footprint."""

    distances = np.asarray(distance, dtype=np.float64)
    return np.where(distances <= cutter_radius, max(0.0, tip_radius), np.inf)


def ball_cutter_surface(
    tip_radius: float,
    distance: npt.ArrayLike,
    cutter_radius: float,
) -> FloatArray:
    """Radial swept surface of a ball cutter whose path Z denotes its tip."""

    distances = np.asarray(distance, dtype=np.float64)
    inside = distances <= cutter_radius
    squared_height = np.maximum(0.0, cutter_radius**2 - distances**2)
    rise = cutter_radius - np.sqrt(squared_height)
    return np.where(inside, np.maximum(0.0, tip_radius + rise), np.inf)


def tapered_cutter_surface(
    tip_radius: float,
    distance: npt.ArrayLike,
    narrow_radius: float,
    maximum_radius: float,
    taper_length: float,
) -> FloatArray:
    """Radial swept surface of a cutter widening linearly from its narrow tip."""

    distances = np.asarray(distance, dtype=np.float64)
    inside = distances <= maximum_radius
    rise = taper_length * np.maximum(0.0, distances - narrow_radius)
    rise /= maximum_radius - narrow_radius
    return np.where(inside, np.maximum(0.0, tip_radius + rise), np.inf)


def cutter_kernel(
    grid: RotaryGrid,
    point: ToolpathPoint,
    tool: Tool,
) -> FloatArray:
    """Return the minimum radial stock surface swept at one tool position."""

    distance = lateral_distance(grid, point)
    radius = tool.diameter / 2.0
    if tool.tool_type is ToolType.FLAT:
        return flat_cutter_surface(point.z, distance, radius)
    if tool.tool_type is ToolType.BALL:
        return ball_cutter_surface(point.z, distance, radius)
    if tool.tool_type is ToolType.TAPERED:
        assert tool.tip_diameter is not None and tool.taper_length is not None
        return tapered_cutter_surface(
            point.z,
            distance,
            tool.tip_diameter / 2.0,
            radius,
            tool.taper_length,
        )
    raise ValueError(f"unsupported tool type: {tool.tool_type!r}")
