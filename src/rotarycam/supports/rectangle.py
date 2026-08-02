"""Rectangular support masks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from rotarycam.supports._weights import smooth_footprint, surface_angular_distance
from rotarycam.supports.models import RectangularSupport

if TYPE_CHECKING:
    from rotarycam.geometry.models import RotaryGrid


def rectangular_support_mask(
    grid: RotaryGrid,
    support: RectangularSupport,
) -> npt.NDArray[np.float64]:
    """Build a periodic X/A weight mask for a rectangular support."""

    x_values = np.asarray(grid.x_values, dtype=np.float64)
    angles_deg = np.asarray(grid.angles_deg, dtype=np.float64)
    radius = np.asarray(grid.radius, dtype=np.float64)

    x_distance = np.abs(x_values[:, None] - support.x)
    tangent_distance = surface_angular_distance(
        angles_deg,
        support.angle_deg,
        radius,
    )
    x_weight = smooth_footprint(
        x_distance,
        support.length_x / 2.0,
        support.transition,
    )
    angle_weight = smooth_footprint(
        tangent_distance,
        support.width_surface / 2.0,
        support.transition,
    )
    return x_weight * angle_weight
