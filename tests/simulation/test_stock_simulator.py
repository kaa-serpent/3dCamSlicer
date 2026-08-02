import numpy as np
import pytest

from rotarycam.geometry.models import RotaryGrid
from rotarycam.simulation import removed_volume, simulate_toolpath
from rotarycam.toolpath.models import Toolpath, ToolpathPoint
from rotarycam.tools.models import Tool, ToolType


def make_tool(tool_type: ToolType = ToolType.FLAT) -> Tool:
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


def make_grid(radius: float) -> RotaryGrid:
    x = np.asarray([-1.0, 0.0, 1.0])
    angles = np.asarray([0.0, 90.0, 180.0, 270.0])
    values = np.full((3, 4), radius)
    return RotaryGrid(x, angles, values, np.ones_like(values, dtype=np.bool_))


def test_flat_simulation_is_monotonic_and_does_not_mutate_inputs() -> None:
    stock = make_grid(10.0)
    target = make_grid(5.0)
    original = stock.radius.copy()
    path = Toolpath(1, "test", [ToolpathPoint(x=0.0, z=6.0, a=0.0)])

    result = simulate_toolpath(stock, target, path, make_tool())

    assert result.stock.radius[1, 0] == pytest.approx(6.0)
    assert np.all(result.stock.radius <= stock.radius)
    assert np.all(result.stock.radius >= target.radius)
    np.testing.assert_array_equal(stock.radius, original)
    assert result.removed_volume > 0.0


def test_ball_simulation_cuts_less_at_footprint_edge() -> None:
    stock = make_grid(10.0)
    target = make_grid(5.0)
    path = Toolpath(1, "test", [ToolpathPoint(x=0.0, z=6.0, a=0.0)])

    result = simulate_toolpath(stock, target, path, make_tool(ToolType.BALL))

    assert result.stock.radius[1, 0] == pytest.approx(6.0)
    assert result.stock.radius[0, 0] == pytest.approx(7.0)
    assert result.stock.radius[0, 0] > result.stock.radius[1, 0]


def test_effective_target_preserves_support_material() -> None:
    stock = make_grid(10.0)
    target_values = np.full(stock.shape, 5.0)
    target_values[1, 0] = 8.0
    target = stock.with_radius(target_values)
    path = Toolpath(1, "test", [ToolpathPoint(x=0.0, z=4.0, a=0.0)])

    result = simulate_toolpath(stock, target, path, make_tool())

    assert result.stock.radius[1, 0] == pytest.approx(8.0)
    assert np.all(result.stock.radius >= target.radius)


def test_cutting_segment_is_sampled_between_endpoints() -> None:
    stock = make_grid(10.0)
    target = make_grid(5.0)
    path = Toolpath(
        1,
        "test",
        [
            ToolpathPoint(x=-1.0, z=6.0, a=0.0),
            ToolpathPoint(x=1.0, z=6.0, a=0.0),
        ],
    )

    result = simulate_toolpath(stock, target, path, make_tool())

    assert result.stock.radius[1, 0] == pytest.approx(6.0)


def test_rapid_move_removes_no_material() -> None:
    stock = make_grid(10.0)
    path = Toolpath(1, "test", [ToolpathPoint(x=0.0, z=2.0, a=0.0, rapid=True)])

    result = simulate_toolpath(stock, make_grid(5.0), path, make_tool())

    np.testing.assert_array_equal(result.stock.radius, stock.radius)
    assert result.removed_volume == pytest.approx(0.0)


def test_cylindrical_removed_volume_is_non_negative_and_scaled() -> None:
    old = make_grid(10.0)
    new = make_grid(8.0)

    volume = removed_volume(old, new)

    # Length 2 mm, full revolution, annulus area pi*(10²-8²).
    assert volume == pytest.approx(72.0 * np.pi)
    assert removed_volume(new, old) == pytest.approx(0.0)
