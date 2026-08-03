"""Safety-focused tests for volumetric XYZA simulation."""

from __future__ import annotations

import numpy as np
import pytest

from rotarycam.collisions import (
    CollisionEvent,
    CollisionKind,
    CollisionReport,
    ToolComponent,
)
from rotarycam.config import XYZAConfiguration
from rotarycam.machine import XYZAKinematics
from rotarycam.motion import MachinePose, MotionBlock, MotionKind
from rotarycam.simulation.volumetric import (
    ConnectivityStatus,
    SimulationBudgetExceeded,
    SimulationIssueKind,
    VolumetricSimulationError,
    VolumetricSimulationSettings,
    simulate_volumetric_motion,
)
from rotarycam.tools import Tool, ToolAssembly, ToolHolder, ToolType
from rotarycam.volumetric import SolidVolume, StockVolume, VoxelLattice


def _tool() -> ToolAssembly:
    return ToolAssembly.from_tool(
        Tool(
            number=1,
            name="narrow flat cutter",
            tool_type=ToolType.FLAT,
            diameter=0.8,
            cutting_length=0.6,
            flute_length=1.0,
            overall_length=10.0,
            shank_diameter=0.8,
            max_stepdown=0.5,
            stepover=0.4,
            feed=100.0,
            plunge_feed=50.0,
            spindle_rpm=10_000,
            stickout=2.0,
            holder=ToolHolder(diameter=2.0, length=2.0),
        )
    )


def _kinematics() -> XYZAKinematics:
    return XYZAKinematics(
        XYZAConfiguration(
            rotary_pivot_y=0.0,
            rotary_pivot_z=0.0,
            rotary_zero_deg=0.0,
            spindle_axis=(0.0, 0.0, -1.0),
            g54_origin=(0.0, 0.0, 0.0),
        )
    )


def _volumes(
    initial_mask: np.ndarray, target_mask: np.ndarray
) -> tuple[StockVolume, SolidVolume]:
    lattice = VoxelLattice((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), initial_mask.shape)
    return (
        StockVolume.from_dense(lattice, initial_mask, brick_size=2),
        SolidVolume.from_dense(lattice, target_mask, brick_size=2),
    )


def _pose(x: float) -> MachinePose:
    return MachinePose(x=x, y=0.0, z=-0.5, a=0.0)


def _settings(max_samples: int = 100) -> VolumetricSimulationSettings:
    return VolumetricSimulationSettings(tolerance=0.25, max_samples=max_samples)


def test_volumetric_simulation_module_imports() -> None:
    assert VolumetricSimulationSettings(0.1, 10).max_samples == 10


def test_removes_only_cutter_solid_and_reports_residual() -> None:
    initial_mask = np.ones((3, 1, 1), dtype=np.bool_)
    target_mask = initial_mask.copy()
    target_mask[1, 0, 0] = False
    initial, target = _volumes(initial_mask, target_mask)

    report = simulate_volumetric_motion(
        initial,
        target,
        _tool(),
        _kinematics(),
        _pose(1.0),
        (
            MotionBlock(
                _pose(1.0), MotionKind.LINEAR, duration_s=1.0, feed=100.0
            ),
        ),
        settings=_settings(),
    )

    assert report.removed_voxels == 1
    assert report.residual_voxels == 0
    assert report.gouge_voxels == 0
    assert report.connectivity is ConnectivityStatus.UNKNOWN
    assert not report.blocking
    np.testing.assert_array_equal(report.final_stock.occupation, target_mask)


def test_gouge_is_not_restored_or_clamped_to_target() -> None:
    initial_mask = np.ones((3, 1, 1), dtype=np.bool_)
    initial, target = _volumes(initial_mask, initial_mask)

    report = simulate_volumetric_motion(
        initial,
        target,
        _tool(),
        _kinematics(),
        _pose(1.0),
        (
            MotionBlock(
                _pose(1.0), MotionKind.LINEAR, duration_s=1.0, feed=100.0
            ),
        ),
        settings=_settings(),
    )

    assert not report.final_stock.occupation[1, 0, 0]
    assert report.gouge[1, 0, 0]
    assert report.blocking
    assert [issue.kind for issue in report.issues] == [SimulationIssueKind.GOUGE]


def test_collision_report_is_propagated_as_blocking() -> None:
    initial_mask = np.zeros((1, 1, 1), dtype=np.bool_)
    initial, target = _volumes(initial_mask, initial_mask)
    start = _pose(5.0)
    block = MotionBlock(start, MotionKind.RAPID, duration_s=1.0)
    event = CollisionEvent(
        CollisionKind.FIXTURE,
        "clamp",
        ToolComponent.HOLDER,
        0.5,
        start,
        -0.1,
    )
    collision = CollisionReport((event,))

    report = simulate_volumetric_motion(
        initial,
        target,
        _tool(),
        _kinematics(),
        start,
        (block,),
        settings=_settings(),
        collision_reports=(collision,),
    )

    assert report.collision_reports == (collision,)
    assert report.blocking
    assert SimulationIssueKind.COLLISION in {issue.kind for issue in report.issues}


def test_six_neighbor_connectivity_detects_detached_target() -> None:
    initial_mask = np.ones((3, 1, 1), dtype=np.bool_)
    target_mask = np.zeros_like(initial_mask)
    target_mask[(0, 2), 0, 0] = True
    initial, target = _volumes(initial_mask, target_mask)
    anchor = np.zeros_like(initial_mask)
    anchor[0, 0, 0] = True

    report = simulate_volumetric_motion(
        initial,
        target,
        _tool(),
        _kinematics(),
        _pose(1.0),
        (
            MotionBlock(
                _pose(1.0), MotionKind.LINEAR, duration_s=1.0, feed=100.0
            ),
        ),
        settings=_settings(),
        anchor_mask=anchor,
    )

    assert report.connectivity is ConnectivityStatus.DETACHED
    assert report.blocking
    assert SimulationIssueKind.DETACHED in {issue.kind for issue in report.issues}
    assert report.gouge_voxels == 0


def test_inputs_and_report_masks_are_immutable() -> None:
    initial_mask = np.ones((2, 1, 1), dtype=np.bool_)
    target_mask = np.zeros_like(initial_mask)
    initial_copy = initial_mask.copy()
    target_copy = target_mask.copy()
    initial, target = _volumes(initial_mask, target_mask)

    report = simulate_volumetric_motion(
        initial,
        target,
        _tool(),
        _kinematics(),
        _pose(0.0),
        (
            MotionBlock(
                _pose(0.0), MotionKind.LINEAR, duration_s=1.0, feed=100.0
            ),
        ),
        settings=_settings(),
    )

    np.testing.assert_array_equal(initial_mask, initial_copy)
    np.testing.assert_array_equal(target_mask, target_copy)
    np.testing.assert_array_equal(initial.occupation, initial_copy)
    with pytest.raises(ValueError, match="read-only"):
        report.removed[0, 0, 0] = False


def test_refuses_to_undersample_when_budget_is_insufficient() -> None:
    initial_mask = np.ones((2, 1, 1), dtype=np.bool_)
    initial, target = _volumes(initial_mask, np.zeros_like(initial_mask))
    block = MotionBlock(_pose(10.0), MotionKind.LINEAR, duration_s=1.0, feed=100.0)

    with pytest.raises(SimulationBudgetExceeded, match="requires"):
        simulate_volumetric_motion(
            initial,
            target,
            _tool(),
            _kinematics(),
            _pose(0.0),
            (block,),
            settings=_settings(max_samples=1),
        )


def test_rapid_motion_never_removes_material() -> None:
    initial_mask = np.ones((3, 1, 1), dtype=np.bool_)
    initial, target = _volumes(initial_mask, np.zeros_like(initial_mask))
    start = _pose(0.0)
    block = MotionBlock(_pose(2.0), MotionKind.RAPID, duration_s=1.0)

    report = simulate_volumetric_motion(
        initial,
        target,
        _tool(),
        _kinematics(),
        start,
        (block,),
        settings=_settings(),
    )

    np.testing.assert_array_equal(report.final_stock.occupation, initial_mask)
    assert report.removed_voxels == 0
    assert report.sample_count == 0


def test_rejects_different_lattices() -> None:
    initial_mask = np.ones((1, 1, 1), dtype=np.bool_)
    initial, _ = _volumes(initial_mask, initial_mask)
    other_lattice = VoxelLattice((0.5, 0.0, 0.0), (1.0, 1.0, 1.0), (1, 1, 1))
    target = SolidVolume.from_dense(other_lattice, initial_mask)

    with pytest.raises(VolumetricSimulationError, match="identical lattices"):
        simulate_volumetric_motion(
            initial,
            target,
            _tool(),
            _kinematics(),
            _pose(0.0),
            (),
            settings=_settings(),
        )


def test_rejects_target_outside_initial_stock() -> None:
    initial_mask = np.zeros((1, 1, 1), dtype=np.bool_)
    target_mask = np.ones_like(initial_mask)
    initial, target = _volumes(initial_mask, target_mask)

    with pytest.raises(VolumetricSimulationError, match="contained"):
        simulate_volumetric_motion(
            initial,
            target,
            _tool(),
            _kinematics(),
            _pose(0.0),
            (),
            settings=_settings(),
        )
