"""Numerical helpers shared by support footprint implementations."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def angular_distance_deg(angles_deg: npt.ArrayLike, center_deg: float) -> FloatArray:
    """Return the shortest signed periodic angular distance in degrees."""

    angles = np.asarray(angles_deg, dtype=np.float64)
    return ((angles - center_deg + 180.0) % 360.0) - 180.0


def smooth_footprint(distance: FloatArray, core_extent: float, transition: float) -> FloatArray:
    """Return a cubic smoothstep weight inside a core plus feather distance."""

    absolute_distance = np.abs(distance)
    if transition == 0.0:
        return (absolute_distance <= core_extent).astype(np.float64)

    normalized = np.clip(
        1.0 - ((absolute_distance - core_extent) / transition),
        0.0,
        1.0,
    )
    return normalized * normalized * (3.0 - 2.0 * normalized)


def surface_angular_distance(
    angles_deg: npt.ArrayLike,
    center_deg: float,
    local_radius: FloatArray,
) -> FloatArray:
    """Convert periodic angular separation to tangent distance in millimetres."""

    delta_rad = np.deg2rad(angular_distance_deg(angles_deg, center_deg))[None, :]
    return np.abs(delta_rad) * np.maximum(np.abs(local_radius), 0.0)
