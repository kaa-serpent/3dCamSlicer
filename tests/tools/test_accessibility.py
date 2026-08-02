from __future__ import annotations

import numpy as np
import pytest

from rotarycam.geometry import RotaryGrid
from rotarycam.tools import Tool, ToolType, compute_accessibility_mask


def _tool(
    *,
    cutting_length: float = 4.0,
    diameter: float = 2.0,
    shank_diameter: float = 2.0,
) -> Tool:
    return Tool(
        number=1,
        name="access test",
        tool_type=ToolType.FLAT,
        diameter=diameter,
        cutting_length=cutting_length,
        flute_length=cutting_length,
        overall_length=20.0,
        shank_diameter=shank_diameter,
        max_stepdown=min(1.0, cutting_length),
        stepover=1.0,
        feed=300.0,
        plunge_feed=80.0,
        spindle_rpm=12_000,
    )


def _grid(radius: np.ndarray, *, valid: np.ndarray | None = None) -> RotaryGrid:
    return RotaryGrid(
        np.arange(radius.shape[0], dtype=np.float64),
        np.arange(radius.shape[1], dtype=np.float64) * (360.0 / radius.shape[1]),
        radius,
        np.ones(radius.shape, dtype=bool) if valid is None else valid,
    )


def test_accessibility_respects_cutting_length_and_validity() -> None:
    target = _grid(np.full((2, 4), 5.0))
    stock_radius = np.array([[7.0, 10.0, 7.0, 7.0], [7.0, 7.0, 7.0, 7.0]])
    stock_valid = np.ones((2, 4), dtype=bool)
    stock_valid[1, 3] = False
    stock_radius[1, 3] = 0.0
    current = _grid(stock_radius, valid=stock_valid)
    mask = compute_accessibility_mask(target, current, _tool(cutting_length=4.0))
    assert mask[0].tolist() == [True, False, True, True]
    assert not mask[1, 3]
    assert not mask.flags.writeable


def test_wide_shank_collision_is_rejected_beyond_flute() -> None:
    target = _grid(np.array([[1.0], [1.0], [8.0]]))
    current = _grid(target.radius.copy())
    mask = compute_accessibility_mask(
        target,
        current,
        _tool(cutting_length=3.0, diameter=2.0, shank_diameter=6.0),
    )
    assert not mask[0, 0]
    assert mask[2, 0]


def test_accessibility_rejects_layout_mismatch_and_target_outside_stock() -> None:
    target = _grid(np.full((2, 2), 5.0))
    different_layout = RotaryGrid(
        np.array([0.0, 2.0]),
        target.angles_deg,
        np.full((2, 2), 6.0),
        np.ones((2, 2), dtype=bool),
    )
    with pytest.raises(ValueError, match="same X/A layout"):
        compute_accessibility_mask(target, different_layout, _tool())
    with pytest.raises(ValueError, match="must not exceed"):
        compute_accessibility_mask(target, _grid(np.full((2, 2), 4.0)), _tool())
