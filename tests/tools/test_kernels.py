from __future__ import annotations

import numpy as np
import pytest

from rotarycam.geometry import RotaryGrid
from rotarycam.tools import (
    BallCutterKernel,
    FlatCutterKernel,
    TaperedCutterKernel,
    Tool,
    ToolType,
    compute_compensated_tool_grid,
    discretize_kernel,
    kernel_for_tool,
)
from rotarycam.tools.compensation import conservative_lateral_distance


def _tool(tool_type: ToolType, *, diameter: float = 4.0) -> Tool:
    return Tool(
        number=1,
        name=f"{tool_type.value} cutter",
        tool_type=tool_type,
        diameter=diameter,
        cutting_length=8.0,
        flute_length=8.0,
        overall_length=20.0,
        shank_diameter=diameter,
        max_stepdown=2.0,
        stepover=1.0,
        feed=300.0,
        plunge_feed=80.0,
        spindle_rpm=12_000,
    )


def _target() -> RotaryGrid:
    radius = np.array(
        [
            [5.0, 5.0, 5.0, 8.0],
            [5.0, 5.0, 5.0, 8.0],
            [5.0, 5.0, 5.0, 8.0],
        ]
    )
    return RotaryGrid(
        np.array([0.0, 1.0, 2.0]),
        np.array([0.0, 90.0, 180.0, 270.0]),
        radius,
        np.ones(radius.shape, dtype=bool),
    )


def test_flat_and_ball_profiles_are_referenced_at_tip() -> None:
    flat = FlatCutterKernel(2.0)
    ball = BallCutterKernel(2.0)
    distances = np.array([0.0, 1.0, 2.0, 2.01])
    assert flat.height_at(distances)[:3] == pytest.approx([0.0, 0.0, 0.0])
    assert ball.height_at(distances)[:3] == pytest.approx([0.0, 2.0 - np.sqrt(3.0), 2.0])
    assert np.isinf(flat.height_at(distances)[-1])
    assert np.isinf(ball.height_at(distances)[-1])


def test_tapered_profile_widens_linearly_over_configured_distance() -> None:
    kernel = TaperedCutterKernel(0.2, 1.5, 10.0)
    distances = np.array([0.0, 0.2, 0.85, 1.5, 1.51])

    assert kernel.height_at(distances)[:4] == pytest.approx([0.0, 0.0, 5.0, 10.0])
    assert np.isinf(kernel.height_at(distances)[-1])

    tool = Tool(
        9,
        "Tapered",
        ToolType.TAPERED,
        3.0,
        10.0,
        10.0,
        40.0,
        3.175,
        1.0,
        0.4,
        300.0,
        80.0,
        12_000,
        0.4,
        10.0,
    )
    assert isinstance(kernel_for_tool(tool), TaperedCutterKernel)


def test_kernel_dispatch_and_discretization_are_deterministic() -> None:
    kernel = kernel_for_tool(_tool(ToolType.BALL))
    first = discretize_kernel(kernel, 0.7, 0.6)
    second = discretize_kernel(kernel, 0.7, 0.6)
    assert isinstance(kernel, BallCutterKernel)
    assert np.array_equal(first.dx, second.dx)
    assert np.array_equal(first.surface_offset, second.surface_offset)
    assert np.array_equal(first.height, second.height)
    assert np.any((first.dx == 0.0) & (first.surface_offset == 0.0))
    assert not first.height.flags.writeable


@pytest.mark.parametrize("tool_type", [ToolType.FLAT, ToolType.BALL, ToolType.TAPERED])
def test_compensated_envelope_does_not_penetrate_sampled_target(tool_type: ToolType) -> None:
    target = _target()
    tool = (
        Tool(
            1,
            "tapered cutter",
            ToolType.TAPERED,
            12.0,
            8.0,
            8.0,
            20.0,
            3.175,
            2.0,
            1.0,
            300.0,
            80.0,
            12_000,
            0.5,
            8.0,
        )
        if tool_type is ToolType.TAPERED
        else _tool(tool_type, diameter=12.0)
    )
    kernel = kernel_for_tool(tool)
    compensated = compute_compensated_tool_grid(target, tool)

    assert np.array_equal(target.radius, _target().radius)
    for x_index, angle_index in np.ndindex(target.shape):
        distance = conservative_lateral_distance(
            target.x_values,
            target.angles_deg,
            target.radius,
            center_x=float(target.x_values[x_index]),
            center_angle_deg=float(target.angles_deg[angle_index]),
            center_radius=float(target.radius[x_index, angle_index]),
        )
        covered = distance <= kernel.footprint_radius
        lower_surface = compensated.radius[x_index, angle_index] + kernel.height_at(distance)
        assert np.all(lower_surface[covered] + 1e-12 >= target.radius[covered])


def test_flat_compensation_observes_periodic_seam() -> None:
    target = _target()
    compensated = compute_compensated_tool_grid(target, _tool(ToolType.FLAT, diameter=16.0))
    assert np.all(compensated.radius[:, 0] >= target.radius[:, 3])
