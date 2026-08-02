from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from rotarycam.geometry import RotaryGrid
from rotarycam.strategies.indexed_roughing import (
    INDEXED_ORIENTATIONS_DEG,
    generate_indexed_roughing,
)
from rotarycam.strategies.rotary_roughing import generate_rotary_roughing
from rotarycam.strategies.roughing_settings import RoughingSettings
from rotarycam.tools import Tool, ToolType


def _tool() -> Tool:
    return Tool(
        number=1,
        name="Flat 2 mm",
        tool_type=ToolType.FLAT,
        diameter=2.0,
        cutting_length=8.0,
        flute_length=8.0,
        overall_length=20.0,
        shank_diameter=2.0,
        max_stepdown=2.0,
        stepover=1.0,
        feed=300.0,
        plunge_feed=80.0,
        spindle_rpm=12_000,
    )


def _grid(radius: float) -> RotaryGrid:
    values = np.full((3, 4), radius)
    return RotaryGrid(
        np.array([0.0, 5.0, 10.0]),
        np.array([0.0, 90.0, 180.0, 270.0]),
        values,
        np.ones(values.shape, dtype=bool),
    )


def _settings() -> RoughingSettings:
    return RoughingSettings(
        stepdown=2.0,
        stepover=1.0,
        allowance=0.0,
        safe_radius=12.0,
        tolerance=0.01,
    )


def test_rotary_roughing_uses_descending_levels_and_safe_links() -> None:
    operation = generate_rotary_roughing(_grid(10.0), _grid(5.0), _tool(), _settings())
    points = operation.toolpaths[0].points
    cutting = [point for point in points if not point.rapid]
    cutting_levels = sorted({point.z for point in cutting}, reverse=True)
    assert operation.strategy == "rotary_roughing"
    assert cutting_levels == pytest.approx([8.0, 6.0, 5.0])
    assert points[0].rapid and points[0].z == 12.0
    assert points[-1].rapid and points[-1].z == 12.0


def test_indexed_roughing_retracts_before_every_orientation_change() -> None:
    operation = generate_indexed_roughing(_grid(10.0), _grid(5.0), _tool(), _settings())
    points = operation.toolpaths[0].points
    cutting_angles = {point.a for point in points if not point.rapid}
    assert cutting_angles == set(INDEXED_ORIENTATIONS_DEG)
    for previous, current in pairwise(points):
        if previous.a != current.a:
            assert previous.rapid and current.rapid
            assert previous.z == current.z == _settings().safe_radius


def test_roughing_rejects_unsafe_parameters() -> None:
    with pytest.raises(ValueError, match="max_stepdown"):
        generate_rotary_roughing(
            _grid(10.0),
            _grid(5.0),
            _tool(),
            RoughingSettings(3.0, 1.0, 0.0, 12.0),
        )
    with pytest.raises(ValueError, match="safe_radius"):
        generate_indexed_roughing(
            _grid(10.0),
            _grid(5.0),
            _tool(),
            RoughingSettings(2.0, 1.0, 0.0, 10.0),
        )


def test_roughing_does_not_mutate_input_grids() -> None:
    stock = _grid(10.0)
    target = _grid(5.0)
    stock_before = stock.radius.copy()
    target_before = target.radius.copy()
    generate_rotary_roughing(stock, target, _tool(), _settings())
    assert np.array_equal(stock.radius, stock_before)
    assert np.array_equal(target.radius, target_before)
