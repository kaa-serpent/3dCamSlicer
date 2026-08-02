from __future__ import annotations

import numpy as np

from rotarycam.accessibility import (
    AccessibilitySettings,
    InaccessibilityReason,
    analyze_accessibility,
)
from rotarycam.collisions import (
    CollisionEvent,
    CollisionKind,
    CollisionReport,
    ToolComponent,
)
from rotarycam.config import AxisLimits, MachineDefinition, XYZAConfiguration
from rotarycam.tools import Tool, ToolAssembly, ToolHolder, ToolType
from rotarycam.volumetric import SolidVolume, VoxelLattice


def _target(*, origin_x: float = 0.0) -> SolidVolume:
    lattice = VoxelLattice((origin_x, 0.0, 0.0), (0.1, 0.1, 0.1), (1, 1, 2))
    return SolidVolume.from_dense(lattice, np.ones(lattice.shape, dtype=np.bool_), brick_size=1)


def _tool() -> ToolAssembly:
    return ToolAssembly.from_tool(
        Tool(
            1, "flat", ToolType.FLAT, 1.0, 1.0, 1.0, 3.0, 1.0,
            0.5, 0.5, 100.0, 50.0, 10_000,
            stickout=2.0, holder=ToolHolder(2.0, 1.0),
        )
    )


def _machine(*, x_max: float = 10.0) -> MachineDefinition:
    return MachineDefinition(
        x_limits=AxisLimits(minimum=-1.0, maximum=x_max),
        y_limits=AxisLimits(minimum=-1.0, maximum=1.0),
        z_limits=AxisLimits(minimum=-1.0, maximum=1.0),
        xyza_configuration=XYZAConfiguration(
            rotary_pivot_y=0.0,
            rotary_pivot_z=0.0,
            rotary_zero_deg=0.0,
            spindle_axis=(0.0, 0.0, -1.0),
            g54_origin=(0.0, 0.0, 0.0),
        ),
    )


def test_analyzer_is_deterministic_and_finds_angles_for_both_faces() -> None:
    settings = AccessibilitySettings((180.0, 0.0, 360.0), max_error=0.1)

    first = analyze_accessibility(_target(), (_tool(),), _machine(), settings)
    second = analyze_accessibility(_target(), (_tool(),), _machine(), settings)

    assert first.complete
    assert first.digest == second.digest
    assert first.candidates == second.candidates
    assert {candidate.pose.a for candidate in first.candidates} == {0.0, 180.0}


def test_travel_failures_form_one_six_connected_region() -> None:
    report = analyze_accessibility(
        _target(origin_x=5.0),
        (_tool(),),
        _machine(x_max=1.0),
        AccessibilitySettings((0.0, 180.0), max_error=0.1),
    )

    assert not report.complete
    assert len(report.regions) == 1
    assert report.regions[0].voxel_count == 2
    assert InaccessibilityReason.TRAVEL in report.regions[0].reasons


def test_collision_report_maps_machine_fixture_and_non_cutting_causes() -> None:
    def collision(candidate: object, tool: object) -> CollisionReport:
        del tool
        pose = candidate.pose  # type: ignore[attr-defined]
        return CollisionReport(
            (
                CollisionEvent(
                    CollisionKind.FIXTURE,
                    "fixture",
                    ToolComponent.HOLDER,
                    0.0,
                    pose,
                    -0.1,
                ),
                CollisionEvent(
                    CollisionKind.MACHINE,
                    "spindle",
                    ToolComponent.CUTTER,
                    0.0,
                    pose,
                    -0.1,
                ),
            )
        )

    report = analyze_accessibility(
        _target(),
        (_tool(),),
        _machine(),
        AccessibilitySettings((0.0, 180.0), max_error=0.1),
        collision_check=collision,
    )

    assert not report.complete
    assert set(report.regions[0].reasons) >= {
        InaccessibilityReason.FIXTURE_COLLISION,
        InaccessibilityReason.MACHINE_COLLISION,
        InaccessibilityReason.NON_CUTTING_COLLISION,
    }
