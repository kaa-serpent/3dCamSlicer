import numpy as np
import pytest

from rotarycam.geometry.models import RotaryGrid
from rotarycam.simulation import (
    ball_cutter_surface,
    cutter_kernel,
    flat_cutter_surface,
    tapered_cutter_surface,
)
from rotarycam.toolpath.models import ToolpathPoint
from rotarycam.tools.models import Tool, ToolType


def tool(tool_type: ToolType) -> Tool:
    return Tool(
        number=1,
        name=f"{tool_type.value} 2 mm",
        tool_type=tool_type,
        diameter=2.0,
        cutting_length=5.0,
        flute_length=5.0,
        overall_length=20.0,
        shank_diameter=2.0,
        max_stepdown=1.0,
        stepover=1.0,
        feed=100.0,
        plunge_feed=50.0,
        spindle_rpm=10_000,
    )


def grid() -> RotaryGrid:
    x = np.asarray([-1.0, 0.0, 1.0])
    angles = np.asarray([0.0, 90.0, 180.0, 270.0])
    radius = np.full((3, 4), 10.0)
    return RotaryGrid(x, angles, radius, np.ones_like(radius, dtype=np.bool_))


def test_flat_kernel_has_constant_cut_radius_inside_footprint() -> None:
    result = flat_cutter_surface(6.0, np.asarray([0.0, 0.5, 1.0, 1.1]), 1.0)
    np.testing.assert_allclose(result[:3], 6.0)
    assert np.isinf(result[3])


def test_ball_kernel_rises_toward_cutter_edge() -> None:
    result = ball_cutter_surface(6.0, np.asarray([0.0, 0.5, 1.0]), 1.0)
    assert result[0] == pytest.approx(6.0)
    assert result[0] < result[1] < result[2]
    assert result[2] == pytest.approx(7.0)


def test_tapered_kernel_uses_narrow_tip_and_linear_widening() -> None:
    result = tapered_cutter_surface(
        6.0,
        np.asarray([0.0, 0.2, 0.85, 1.5, 1.6]),
        0.2,
        1.5,
        10.0,
    )
    assert result[:4] == pytest.approx([6.0, 6.0, 11.0, 16.0])
    assert np.isinf(result[-1])


def test_kernel_is_periodic_around_a_zero() -> None:
    candidate = cutter_kernel(
        grid(),
        ToolpathPoint(x=0.0, z=6.0, a=360.0),
        tool(ToolType.FLAT),
    )
    assert candidate[1, 0] == pytest.approx(6.0)
    assert np.isinf(candidate[1, 2])
