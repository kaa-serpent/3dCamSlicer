from __future__ import annotations

import numpy as np
import pytest

from rotarycam.geometry import RotaryGrid, interpolate_radius


def test_interpolation_is_periodic_at_angular_seam() -> None:
    grid = RotaryGrid(
        np.array([0.0, 1.0]),
        np.array([0.0, 90.0, 180.0, 270.0]),
        np.array([[10.0, 20.0, 30.0, 20.0], [10.0, 20.0, 30.0, 20.0]]),
        np.ones((2, 4), dtype=bool),
    )
    negative_seam = interpolate_radius(grid, 0.5, -1.0)
    assert interpolate_radius(grid, 0.5, 359.0) == pytest.approx(negative_seam)
    assert interpolate_radius(grid, 0.5, 315.0) == pytest.approx(15.0)
