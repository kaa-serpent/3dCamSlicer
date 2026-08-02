from __future__ import annotations

import numpy as np

from rotarycam.accessibility import AccessibilityReport, CandidatePose
from rotarycam.config import AxisLimits, MachineDefinition, XYZAConfiguration
from rotarycam.motion import MachinePose, MotionKind
from rotarycam.planning.freeform import (
    FreeformOperationKind,
    FreeformPlannerSettings,
    plan_freeform,
    unwrap_angle,
)
from rotarycam.tools import Tool, ToolAssembly, ToolHolder, ToolType
from rotarycam.volumetric import SolidVolume, StockVolume, VoxelLattice


def _tool(number: int, kind: ToolType, diameter: float) -> ToolAssembly:
    extra: dict[str, float] = {}
    if kind is ToolType.TAPERED:
        extra = {"tip_diameter": 0.5, "taper_length": 1.0}
    return ToolAssembly.from_tool(
        Tool(
            number=number,
            name=kind.value,
            tool_type=kind,
            diameter=diameter,
            cutting_length=2.0,
            flute_length=2.0,
            overall_length=10.0,
            shank_diameter=diameter,
            max_stepdown=1.0,
            stepover=min(1.0, diameter),
            feed=120.0,
            plunge_feed=60.0,
            spindle_rpm=10_000,
            stickout=3.0,
            holder=ToolHolder(6.0, 4.0),
            **extra,
        )
    )


def _machine() -> MachineDefinition:
    return MachineDefinition(
        x_limits=AxisLimits(minimum=-30.0, maximum=30.0),
        y_limits=AxisLimits(minimum=-30.0, maximum=30.0),
        z_limits=AxisLimits(minimum=-30.0, maximum=30.0),
        xyza_configuration=XYZAConfiguration(
            rotary_pivot_y=0.0,
            rotary_pivot_z=0.0,
            rotary_zero_deg=0.0,
            spindle_axis=(0.0, 0.0, -1.0),
            g54_origin=(0.0, 0.0, 0.0),
        ),
    )


def _volumes(
    *, residual: bool, shape: tuple[int, int, int] = (2, 1, 1)
) -> tuple[StockVolume, SolidVolume]:
    lattice = VoxelLattice((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), shape)
    stock = np.ones(shape, dtype=np.bool_)
    target = stock.copy()
    if residual:
        target[-1, 0, 0] = False
    return (
        StockVolume.from_dense(lattice, stock, brick_size=1),
        SolidVolume.from_dense(lattice, target, brick_size=1),
    )


def _candidate(
    index: tuple[int, int, int], tool_number: int, pose: MachinePose
) -> CandidatePose:
    return CandidatePose(
        index,
        (float(index[0]), 0.0, 0.0),
        (0.0, 0.0, 1.0),
        tool_number,
        pose,
        0.1,
    )


def _report(
    shape: tuple[int, int, int], candidates: tuple[CandidatePose, ...]
) -> AccessibilityReport:
    accessible = np.zeros(shape, dtype=np.bool_)
    for candidate in candidates:
        accessible[candidate.surface_index] = True
    return AccessibilityReport(
        candidates,
        accessible,
        np.zeros(shape, dtype=np.bool_),
        (),
        "a" * 64,
    )


def test_unwrap_prefers_continuous_winding() -> None:
    assert unwrap_angle(350.0, 10.0) == 370.0
    assert unwrap_angle(-350.0, 10.0) == -350.0


def test_plan_is_deterministic_and_contains_simultaneous_xyza_motion() -> None:
    initial, target = _volumes(residual=False)
    tools = (
        _tool(1, ToolType.FLAT, 6.0),
        _tool(2, ToolType.BALL, 4.0),
        _tool(3, ToolType.TAPERED, 2.0),
    )
    report = _report(
        target.lattice.shape,
        (
            _candidate((0, 0, 0), 1, MachinePose(0.0, 0.0, 0.0, 350.0)),
            _candidate((1, 0, 0), 1, MachinePose(1.0, 2.0, 3.0, 10.0)),
        ),
    )

    first = plan_freeform(initial, target, report, tools, _machine())
    second = plan_freeform(initial, target, report, tools, _machine())

    assert first == second
    linear = [block for block in first.blocks if block.kind is MotionKind.LINEAR]
    assert any(block.pose.a == 370.0 for block in linear)
    previous = first.passes[0].start_pose
    simultaneous = False
    for block in first.blocks:
        if block.kind is MotionKind.LINEAR:
            simultaneous |= (
                previous.x != block.pose.x
                and previous.y != block.pose.y
                and previous.z != block.pose.z
                and previous.a != block.pose.a
            )
        previous = block.pose
    assert simultaneous


def test_stock_aware_roughing_and_smaller_tool_rest_are_separate() -> None:
    initial, target = _volumes(residual=True)
    tools = (
        _tool(1, ToolType.FLAT, 6.0),
        _tool(2, ToolType.BALL, 4.0),
        _tool(3, ToolType.TAPERED, 2.0),
    )
    report = _report(
        target.lattice.shape,
        (
            _candidate((0, 0, 0), 1, MachinePose(0.0, 0.0, 0.0, 0.0)),
            _candidate((0, 0, 0), 2, MachinePose(0.0, 0.0, 0.0, 0.0)),
            _candidate((1, 0, 0), 3, MachinePose(1.0, 0.0, 0.0, 15.0)),
        ),
    )

    plan = plan_freeform(initial, target, report, tools, _machine())

    assert plan.operation_kinds == (
        FreeformOperationKind.ROUGHING,
        FreeformOperationKind.FINISHING,
        FreeformOperationKind.REST,
    )
    assert {planned_pass.tool_number for planned_pass in plan.passes} == {1, 3}
    assert plan.accessibility is report


def test_discontinuities_split_passes_and_link_at_safe_clearance() -> None:
    shape = (3, 1, 1)
    initial, target = _volumes(residual=False, shape=shape)
    tool = _tool(1, ToolType.BALL, 4.0)
    report = _report(
        shape,
        (
            _candidate((0, 0, 0), 1, MachinePose(0.0, 0.0, 0.0, 0.0)),
            _candidate((2, 0, 0), 1, MachinePose(20.0, 0.0, 0.0, 30.0)),
        ),
    )

    plan = plan_freeform(
        initial,
        target,
        report,
        (tool,),
        _machine(),
        FreeformPlannerSettings(max_link_distance=2.0),
    )

    assert len(plan.passes) == 2
    assert any(block.kind is MotionKind.RAPID for block in plan.blocks)
    for planned_pass in plan.passes:
        assert planned_pass.blocks[-1].kind is MotionKind.RAPID
        assert planned_pass.blocks[-1].pose.a == planned_pass.blocks[-2].pose.a
