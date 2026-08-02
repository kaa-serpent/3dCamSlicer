from __future__ import annotations

import dataclasses

import numpy as np

from rotarycam.accessibility import AccessibilitySettings, digest_accessibility_context
from rotarycam.config import AxisLimits, MachineDefinition, XYZAConfiguration
from rotarycam.tools import Tool, ToolAssembly, ToolHolder, ToolType
from rotarycam.volumetric import SolidVolume, VoxelLattice


def target(mask: np.ndarray | None = None) -> SolidVolume:
    lattice = VoxelLattice((0.0, 0.0, 0.0), (0.1, 0.1, 0.1), (2, 2, 2))
    values = np.ones(lattice.shape, dtype=np.bool_) if mask is None else mask
    return SolidVolume.from_dense(lattice, values, brick_size=1)


def tool(*, number: int = 1, diameter: float = 1.0) -> ToolAssembly:
    return ToolAssembly.from_tool(
        Tool(
            number,
            f"flat-{number}",
            ToolType.FLAT,
            diameter,
            2.0,
            2.0,
            5.0,
            1.0,
            min(0.5, diameter),
            min(0.5, diameter),
            100.0,
            50.0,
            10_000,
            stickout=3.0,
            holder=ToolHolder(2.0, 2.0),
        )
    )


def machine(*, x_max: float = 10.0) -> MachineDefinition:
    return MachineDefinition(
        x_limits=AxisLimits(minimum=-10.0, maximum=x_max),
        y_limits=AxisLimits(minimum=-10.0, maximum=10.0),
        z_limits=AxisLimits(minimum=-10.0, maximum=10.0),
        xyza_configuration=XYZAConfiguration(
            rotary_pivot_y=0.0,
            rotary_pivot_z=0.0,
            rotary_zero_deg=0.0,
            spindle_axis=(0.0, 0.0, -1.0),
            g54_origin=(0.0, 0.0, 0.0),
        ),
    )


def test_digest_is_deterministic_and_tool_order_is_irrelevant() -> None:
    settings = AccessibilitySettings((360.0, 180.0, 0.0), 0.1)

    first = digest_accessibility_context(target(), (tool(number=2), tool()), machine(), settings)
    second = digest_accessibility_context(target(), (tool(), tool(number=2)), machine(), settings)

    assert settings.angles_deg == (0.0, 180.0)
    assert first == second


def test_digest_changes_with_target_tool_machine_settings_or_plan() -> None:
    base_target = target()
    base_tool = tool()
    base_machine = machine()
    settings = AccessibilitySettings((0.0,), 0.1)
    base = digest_accessibility_context(
        base_target, (base_tool,), base_machine, settings, "plan-1"
    )
    changed_mask = base_target.to_dense().copy()
    changed_mask[0, 0, 0] = False

    assert digest_accessibility_context(
        target(changed_mask), (base_tool,), base_machine, settings, "plan-1"
    ) != base
    assert digest_accessibility_context(
        base_target, (tool(diameter=1.2),), base_machine, settings, "plan-1"
    ) != base
    assert digest_accessibility_context(
        base_target, (base_tool,), machine(x_max=11.0), settings, "plan-1"
    ) != base
    assert digest_accessibility_context(
        base_target,
        (base_tool,),
        base_machine,
        dataclasses.replace(settings, tolerance=0.02),
        "plan-1",
    ) != base
    assert digest_accessibility_context(
        base_target, (base_tool,), base_machine, settings, "plan-2"
    ) != base


def test_digest_rejects_duplicate_tool_numbers() -> None:
    with np.testing.assert_raises_regex(ValueError, "unique"):
        digest_accessibility_context(
            target(), (tool(), tool(diameter=1.2)), machine(), AccessibilitySettings((0.0,), 0.1)
        )
