from __future__ import annotations

import numpy as np
import pytest

from rotarycam.geometry import RotaryGrid
from rotarycam.strategies.rest_machining import (
    dilate_residual_mask,
    generate_rest_machining,
)
from rotarycam.strategies.settings import FinishingSettings
from rotarycam.tools import Tool, ToolType


def _tool() -> Tool:
    return Tool(
        number=2,
        name="Ball 1 mm",
        tool_type=ToolType.BALL,
        diameter=1.0,
        cutting_length=4.0,
        flute_length=4.0,
        overall_length=15.0,
        shank_diameter=1.0,
        max_stepdown=0.5,
        stepover=0.5,
        feed=200.0,
        plunge_feed=60.0,
        spindle_rpm=15_000,
    )


def _grid(radius: np.ndarray) -> RotaryGrid:
    return RotaryGrid(
        np.arange(radius.shape[0], dtype=np.float64),
        np.arange(radius.shape[1], dtype=np.float64) * (360.0 / radius.shape[1]),
        radius,
        np.ones(radius.shape, dtype=bool),
    )


def _settings() -> FinishingSettings:
    return FinishingSettings(stepover=2.0, safe_radius=8.0, tolerance=0.1)


def test_residual_dilation_wraps_angular_seam_but_not_x() -> None:
    grid = _grid(np.full((7, 12), 5.0))
    residual = np.zeros(grid.shape, dtype=bool)
    residual[3, 0] = True
    dilated = dilate_residual_mask(residual, grid, 0.5)
    assert dilated[3, -1]
    assert dilated[3, 1]
    assert not dilated[0, 0]


def test_rest_machining_is_limited_to_local_residual_component() -> None:
    target = _grid(np.full((10, 12), 5.0))
    stock_radius = target.radius.copy()
    stock_radius[4:6, 4:7] += 0.5
    current = _grid(stock_radius)
    residual = current.radius - target.radius > _settings().tolerance
    permitted = dilate_residual_mask(residual, target, _tool().diameter / 2.0)
    operation = generate_rest_machining(current, target, _tool(), _settings())
    cutting = [point for point in operation.toolpaths[0].points if not point.rapid]

    assert operation.strategy == "rest_machining"
    assert cutting
    assert len(cutting) < target.radius.size
    for point in cutting:
        x_index = int(np.argmin(np.abs(target.x_values - point.x)))
        angular_delta = np.abs(
            (target.angles_deg - (point.a % 360.0) + 180.0) % 360.0 - 180.0
        )
        angle_index = int(np.argmin(angular_delta))
        assert permitted[x_index, angle_index]


def test_rest_machining_honors_explicit_limit_mask() -> None:
    target = _grid(np.full((10, 12), 5.0))
    current = _grid(np.full((10, 12), 5.5))
    limit = np.zeros(target.shape, dtype=bool)
    limit[3:7, 3:9] = True
    operation = generate_rest_machining(current, target, _tool(), _settings(), limit)
    cutting = [point for point in operation.toolpaths[0].points if not point.rapid]
    assert cutting
    assert len(cutting) < target.radius.size


def test_rest_machining_refuses_empty_residual() -> None:
    target = _grid(np.full((4, 8), 5.0))
    with pytest.raises(ValueError, match="no residual"):
        generate_rest_machining(target, target, _tool(), _settings())
