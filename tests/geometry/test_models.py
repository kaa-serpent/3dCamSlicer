from __future__ import annotations

import numpy as np
import pytest

from rotarycam.geometry.models import RotaryGrid, UndercutStatus


def test_rotary_grid_copies_and_freezes_arrays() -> None:
    radius = np.ones((2, 4))
    grid = RotaryGrid(np.array([0.0, 1.0]), np.array([0.0, 90.0, 180.0, 270.0]), radius, radius > 0)
    radius[0, 0] = 99.0

    assert grid.shape == (2, 4)
    assert grid.radius[0, 0] == 1.0
    with pytest.raises(ValueError):
        grid.radius[0, 0] = 2.0


@pytest.mark.parametrize(
    ("angles", "message"),
    [
        ([0.0, 90.0, 90.0], "strictly increasing"),
        ([0.0, 360.0], r"\[0, 360\)"),
    ],
)
def test_rotary_grid_rejects_invalid_angles(angles: list[float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        RotaryGrid(
            np.array([0.0, 1.0]),
            np.array(angles),
            np.zeros((2, len(angles))),
            np.ones((2, len(angles)), dtype=bool),
        )


def test_with_radius_preserves_metadata_without_mutating_source() -> None:
    grid = RotaryGrid(
        np.array([0.0, 1.0]),
        np.array([0.0, 180.0]),
        np.ones((2, 2)),
        np.ones((2, 2), dtype=bool),
        UndercutStatus.ABSENT,
    )
    changed = grid.with_radius(np.full((2, 2), 2.0))
    assert np.all(grid.radius == 1.0)
    assert np.all(changed.radius == 2.0)
    assert changed.undercut_status is UndercutStatus.ABSENT
