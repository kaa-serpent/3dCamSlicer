from __future__ import annotations

import numpy as np
import pytest

from rotarycam.collisions import (
    CollisionBudgetExceeded,
    CollisionKind,
    SweptCollisionChecker,
    SweptCollisionSettings,
    ToolComponent,
    primitive_obstacle,
    stock_obstacle,
)
from rotarycam.machine import AssemblyRole, Box, FrameKind
from rotarycam.motion import MachinePose, MotionBlock, MotionKind
from rotarycam.tools import Tool, ToolAssembly, ToolHolder, ToolType
from rotarycam.volumetric import SparseVolume, VoxelLattice


def _tool() -> ToolAssembly:
    return ToolAssembly.from_tool(
        Tool(
            number=1,
            name="short flat",
            tool_type=ToolType.FLAT,
            diameter=1.0,
            cutting_length=1.0,
            flute_length=1.0,
            overall_length=2.0,
            shank_diameter=1.0,
            max_stepdown=0.5,
            stepover=0.5,
            feed=100.0,
            plunge_feed=50.0,
            spindle_rpm=10_000,
            stickout=1.0,
            holder=ToolHolder(diameter=1.0, length=1.0),
        )
    )


def _block(x: float, kind: MotionKind) -> MotionBlock:
    return MotionBlock(MachinePose(x, 0.0, 0.0, 0.0), kind, duration_s=1.0)


def test_collision_present_only_in_middle_of_segment_is_detected() -> None:
    fixture = Box(
        role=AssemblyRole.FIXTURE,
        frame=FrameKind.FIXED,
        center=(0.0, 0.0, 0.0),
        size=(1.0, 1.0, 1.0),
    )
    checker = SweptCollisionChecker(
        _tool(),
        (primitive_obstacle(fixture, name="middle clamp"),),
        SweptCollisionSettings(tolerance=0.01),
    )

    report = checker.check_segment(MachinePose(-5.0, 0.0, 0.0, 0.0), _block(5.0, MotionKind.LINEAR))

    assert not report.collision_free
    assert report.first_event is not None
    assert report.first_event.kind is CollisionKind.FIXTURE
    assert 0.0 < report.first_event.fraction < 1.0


def test_rapid_crossing_stock_is_blocked_and_cutting_contact_is_allowed() -> None:
    lattice = VoxelLattice((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (1, 1, 1))
    source = np.ones((1, 1, 1), dtype=np.bool_)
    volume = SparseVolume.from_dense(lattice, source, brick_size=1)
    obstacle = stock_obstacle(volume, frame=FrameKind.FIXED)
    checker = SweptCollisionChecker(
        _tool(), (obstacle,), SweptCollisionSettings(tolerance=0.01)
    )

    rapid = checker.check(MachinePose(-5.0, 0.0, 0.0, 0.0), _block(5.0, MotionKind.RAPID))
    cutting = checker.check(MachinePose(-5.0, 0.0, 0.0, 0.0), _block(5.0, MotionKind.LINEAR))

    assert rapid.first_event is not None
    assert rapid.first_event.kind is CollisionKind.STOCK
    assert cutting.collision_free
    assert np.array_equal(volume.to_dense(), source)


def test_non_cutting_tool_component_crossing_stock_is_blocked_during_cut() -> None:
    lattice = VoxelLattice((0.0, 0.0, 1.5), (1.0, 1.0, 1.0), (1, 1, 1))
    volume = SparseVolume.from_dense(
        lattice, np.ones((1, 1, 1), dtype=np.bool_), brick_size=1
    )
    checker = SweptCollisionChecker(
        _tool(),
        (stock_obstacle(volume, frame=FrameKind.FIXED),),
        SweptCollisionSettings(tolerance=0.01),
    )

    report = checker.check(
        MachinePose(-5.0, 0.0, 0.0, 0.0), _block(5.0, MotionKind.LINEAR)
    )

    assert report.first_event is not None
    assert report.first_event.kind is CollisionKind.STOCK
    assert report.first_event.tool_component is ToolComponent.HOLDER


def test_rotary_collision_present_only_at_mid_angle_is_detected() -> None:
    rotary_fixture = Box(
        role=AssemblyRole.FIXTURE,
        frame=FrameKind.ROTARY,
        center=(0.0, 0.0, -2.0),
        size=(0.2, 0.2, 0.2),
    )
    checker = SweptCollisionChecker(
        _tool(),
        (primitive_obstacle(rotary_fixture, name="rotating clamp"),),
        SweptCollisionSettings(tolerance=0.01),
    )
    start = MachinePose(0.0, 2.0, -0.5, 0.0)
    block = MotionBlock(
        MachinePose(0.0, 2.0, -0.5, 180.0), MotionKind.LINEAR, duration_s=1.0
    )

    report = checker.check(start, block)

    assert report.first_event is not None
    assert 0.0 < report.first_event.fraction < 1.0
    assert 45.0 < report.first_event.pose.a < 135.0


def test_insufficient_subdivision_budget_refuses_to_claim_clearance() -> None:
    fixture = Box(
        role=AssemblyRole.FIXTURE,
        frame=FrameKind.FIXED,
        center=(0.0, 0.0, 0.0),
        size=(1.0, 1.0, 1.0),
    )
    checker = SweptCollisionChecker(
        _tool(),
        (primitive_obstacle(fixture),),
        SweptCollisionSettings(tolerance=0.001, max_subdivisions=0),
    )

    with pytest.raises(CollisionBudgetExceeded, match="cannot prove clearance"):
        checker.check(MachinePose(-5.0, 0.0, 0.0, 0.0), _block(5.0, MotionKind.LINEAR))
