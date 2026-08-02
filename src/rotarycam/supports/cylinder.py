"""Cylindrical support masks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from rotarycam.supports._weights import smooth_footprint, surface_angular_distance
from rotarycam.supports.models import CylindricalSupport

if TYPE_CHECKING:
    from rotarycam.geometry.models import RotaryGrid


def cylindrical_support_mask(
    grid: RotaryGrid,
    support: CylindricalSupport,
) -> npt.NDArray[np.float64]:
    """Build a periodic circular X/A weight mask for a cylindrical support."""

    x_values = np.asarray(grid.x_values, dtype=np.float64)
    angles_deg = np.asarray(grid.angles_deg, dtype=np.float64)
    radius = np.asarray(grid.radius, dtype=np.float64)

    x_distance = x_values[:, None] - support.x
    tangent_distance = surface_angular_distance(
        angles_deg,
        support.angle_deg,
        radius,
    )
    surface_distance = np.hypot(x_distance, tangent_distance)
    return smooth_footprint(
        surface_distance,
        support.diameter / 2.0,
        support.transition,
    )
