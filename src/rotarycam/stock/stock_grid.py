"""Construction of initial-stock radius grids."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from rotarycam.geometry.models import RotaryGrid
from rotarycam.stock.base import Stock


def build_initial_stock_grid(
    stock: Stock,
    x_values: npt.ArrayLike,
    angles_deg: npt.ArrayLike,
) -> RotaryGrid:
    """Evaluate an analytical stock on an existing X/A sampling layout."""

    x_axis = np.asarray(x_values, dtype=np.float64)
    angle_axis = np.asarray(angles_deg, dtype=np.float64)
    if x_axis.ndim != 1 or angle_axis.ndim != 1:
        raise ValueError("x_values and angles_deg must be one-dimensional")
    radius = np.zeros((x_axis.size, angle_axis.size), dtype=np.float64)
    within_x = (x_axis >= 0.0) & (x_axis <= stock.length)
    for x_index in np.flatnonzero(within_x):
        radius[x_index] = np.fromiter(
            (stock.radius_at(float(x_axis[x_index]), float(angle)) for angle in angle_axis),
            dtype=np.float64,
            count=angle_axis.size,
        )
    valid = np.broadcast_to(within_x[:, None], radius.shape).copy()
    return RotaryGrid(x_axis, angle_axis, radius, valid)
