"""Periodic interpolation helpers for cylindrical grids."""

from __future__ import annotations

import numpy as np

from rotarycam.geometry.models import RotaryGrid


def interpolate_radius(grid: RotaryGrid, x: float, angle_deg: float) -> float:
    """Bilinearly interpolate radius with a periodic angular seam.

    A query outside the X domain or touching an invalid interpolation cell raises
    ``ValueError`` instead of silently manufacturing geometry.
    """

    if not np.isfinite(x) or not np.isfinite(angle_deg):
        raise ValueError("x and angle_deg must be finite")
    if x < grid.x_values[0] or x > grid.x_values[-1]:
        raise ValueError("x lies outside the grid")
    wrapped_angle = angle_deg % 360.0
    x_hi = int(np.searchsorted(grid.x_values, x, side="right"))
    x_hi = min(max(x_hi, 1), len(grid.x_values) - 1)
    x_lo = x_hi - 1
    if len(grid.x_values) == 1:  # RotaryGrid currently requires only non-empty axes
        x_lo = x_hi = 0

    periodic_angles = np.concatenate((grid.angles_deg, (grid.angles_deg[0] + 360.0,)))
    query_angle = wrapped_angle
    if query_angle < grid.angles_deg[0]:
        query_angle += 360.0
    a_hi_ext = int(np.searchsorted(periodic_angles, query_angle, side="right"))
    a_hi_ext = min(max(a_hi_ext, 1), len(periodic_angles) - 1)
    a_lo = a_hi_ext - 1
    a_hi = a_hi_ext % len(grid.angles_deg)

    indices = ((x_lo, a_lo), (x_lo, a_hi), (x_hi, a_lo), (x_hi, a_hi))
    if not all(bool(grid.valid[index]) for index in indices):
        raise ValueError("interpolation neighborhood contains invalid geometry")
    x0, x1 = grid.x_values[x_lo], grid.x_values[x_hi]
    a0, a1 = periodic_angles[a_lo], periodic_angles[a_hi_ext]
    tx = 0.0 if x1 == x0 else float((x - x0) / (x1 - x0))
    ta = 0.0 if a1 == a0 else float((query_angle - a0) / (a1 - a0))
    r00 = grid.radius[x_lo, a_lo]
    r01 = grid.radius[x_lo, a_hi]
    r10 = grid.radius[x_hi, a_lo]
    r11 = grid.radius[x_hi, a_hi]
    low = (1.0 - ta) * r00 + ta * r01
    high = (1.0 - ta) * r10 + ta * r11
    return float((1.0 - tx) * low + tx * high)
