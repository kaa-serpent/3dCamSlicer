from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import rotarycam.engine as engine_module
from rotarycam.accessibility import (
    AccessibilityReport,
    InaccessibilityReason,
    InaccessibleRegion,
)
from rotarycam.config import AxisLimits, MachineDefinition, XYZAConfiguration
from rotarycam.engine import RotaryCamEngine
from rotarycam.errors import ToolpathValidationError
from rotarycam.machine import (
    AssemblyRole,
    AxisDynamics,
    Box,
    FrameKind,
    MachineAssembly,
    MachineCapabilities,
)
from rotarycam.motion import MachinePose, MotionBlock, MotionKind
from rotarycam.planning import FreeformOperationKind, FreeformPlan, PlannedPass
from rotarycam.simulation import (
    ConnectivityStatus,
    SimulationIssue,
    SimulationIssueKind,
    VolumetricSimulationReport,
)
from rotarycam.tools import Tool, ToolAssembly, ToolHolder, ToolType
from rotarycam.volumetric import SolidVolume, StockVolume, VoxelLattice


def _machine() -> MachineDefinition:
    primitives = tuple(
        Box(
            role=role,
            frame=FrameKind.FIXED,
            center=(100.0 + index * 10.0, 100.0, 100.0),
            size=(1.0, 1.0, 1.0),
        )
        for index, role in enumerate(
            (
                AssemblyRole.CHUCK,
                AssemblyRole.JAWS,
                AssemblyRole.TAILSTOCK,
                AssemblyRole.PLATTER,
                AssemblyRole.SPINDLE,
                AssemblyRole.SUPPORT,
            )
        )
    )
    dynamics = {
        axis: AxisDynamics(max_velocity=6000.0, max_acceleration=1000.0)
        for axis in ("X", "Y", "Z", "A")
    }
    return MachineDefinition(
        profile_verified=True,
        x_limits=AxisLimits(minimum=-10.0, maximum=10.0),
        y_limits=AxisLimits(minimum=-10.0, maximum=10.0),
        z_limits=AxisLimits(minimum=-10.0, maximum=10.0),
        xyza_configuration=XYZAConfiguration(
            rotary_pivot_y=0.0,
            rotary_pivot_z=0.0,
            rotary_zero_deg=0.0,
            spindle_axis=(0.0, 0.0, 1.0),
            g54_origin=(0.0, 0.0, 0.0),
        ),
        machine_assembly_configured=True,
        dynamics=dynamics,
        capabilities=MachineCapabilities(
            simultaneous_xyza=True, inverse_time_feed_g93=True
        ),
        assembly=MachineAssembly(primitives=primitives),
        max_spindle_rpm=20_000,
    )


def _accessibility(*, incomplete: bool, digest: str = "a" * 64) -> AccessibilityReport:
    inaccessible = np.full((1, 1, 1), incomplete, dtype=np.bool_)
    accessible = ~inaccessible
    regions = (
        (
            InaccessibleRegion(
                ((0, 0, 0),),
                1,
                0.1,
                (InaccessibilityReason.OCCLUDED,),
            ),
        )
        if incomplete
        else ()
    )
    return AccessibilityReport((), accessible, inaccessible, regions, digest)


def _ready_engine(*, incomplete: bool = True) -> RotaryCamEngine:
    engine = RotaryCamEngine()
    engine.configure_machine(_machine())
    tool = Tool(
        1,
        "Flat",
        ToolType.FLAT,
        2.0,
        4.0,
        4.0,
        20.0,
        2.0,
        1.0,
        1.0,
        300.0,
        100.0,
        12_000,
        stickout=10.0,
        holder=ToolHolder(12.0, 20.0),
    )
    engine.set_tools([tool])
    accessibility = _accessibility(incomplete=incomplete)
    start = MachinePose(0.0, 0.0, 0.0, 0.0)
    block = MotionBlock(MachinePose(1.0, 1.0, 1.0, 10.0), MotionKind.LINEAR, 100.0, 300.0)
    planned_pass = PlannedPass(
        FreeformOperationKind.FINISHING,
        1,
        ((0, 0, 0),),
        start,
        (block,),
    )
    engine.accessibility_report = accessibility
    engine.freeform_plan = FreeformPlan((planned_pass,), accessibility)
    engine.timed_passes = (planned_pass,)
    lattice = VoxelLattice((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (1, 1, 1))
    stock = StockVolume.from_dense(lattice, np.ones((1, 1, 1), dtype=np.bool_))
    target = SolidVolume.from_dense(lattice, np.ones((1, 1, 1), dtype=np.bool_))
    engine.volumetric_stock = stock
    engine.volumetric_target = target
    current_digest = engine._current_accessibility_digest()
    assert current_digest is not None
    accessibility = replace(accessibility, digest=current_digest)
    engine.accessibility_report = accessibility
    engine.freeform_plan = FreeformPlan((planned_pass,), accessibility)
    engine.volumetric_simulations = (
        VolumetricSimulationReport(
            final_stock=stock,
            removed=np.zeros((1, 1, 1), dtype=np.bool_),
            residual=np.zeros((1, 1, 1), dtype=np.bool_),
            gouge=np.zeros((1, 1, 1), dtype=np.bool_),
            connectivity=ConnectivityStatus.CONNECTED,
        ),
    )
    return engine


def test_inaccessible_acknowledgement_must_match_and_is_invalidated() -> None:
    engine = _ready_engine()
    assert engine.accessibility_report is not None
    digest = engine.accessibility_report.digest

    assert "--ack-inaccessible" in "; ".join(engine.validate_xyza().errors)
    with pytest.raises(ToolpathValidationError, match="expected"):
        engine.acknowledge_inaccessible("b" * 64)

    engine.acknowledge_inaccessible(digest)
    assert engine.validate_xyza().valid

    engine.configure_machine(_machine().model_copy(update={"name": "changed"}))
    assert engine.inaccessible_ack_digest is None
    assert not engine.validate_xyza().valid


def test_acknowledgement_does_not_bypass_tool_or_profile_blockers() -> None:
    engine = _ready_engine()
    assert engine.accessibility_report is not None
    engine.acknowledge_inaccessible(engine.accessibility_report.digest)
    engine.tool_assemblies = (
        ToolAssembly.from_tool(
            engine.tool_assemblies[0].tool.__class__(
                2,
                "Incomplete",
                ToolType.FLAT,
                2.0,
                4.0,
                4.0,
                20.0,
                2.0,
                1.0,
                1.0,
                300.0,
                100.0,
                12_000,
            )
        ),
    )
    assert any("stickout" in item for item in engine.validate_xyza().errors)


def test_acknowledgement_is_bound_to_the_exact_timed_plan() -> None:
    engine = _ready_engine()
    assert engine.accessibility_report is not None
    engine.acknowledge_inaccessible(engine.accessibility_report.digest)

    planned_pass = engine.timed_passes[0]
    changed_block = replace(planned_pass.blocks[0], duration_s=101.0)
    engine.timed_passes = (replace(planned_pass, blocks=(changed_block,)),)

    assert any("stale" in item for item in engine.validate_xyza().errors)
    with pytest.raises(ToolpathValidationError, match="stale"):
        engine.acknowledge_inaccessible(engine.accessibility_report.digest)


def test_acknowledgement_never_bypasses_simulation_or_connectivity() -> None:
    engine = _ready_engine()
    assert engine.accessibility_report is not None
    engine.acknowledge_inaccessible(engine.accessibility_report.digest)
    simulation = engine.volumetric_simulations[0]
    engine.volumetric_simulations = (
        replace(
            simulation,
            issues=(SimulationIssue(SimulationIssueKind.GOUGE, "Target gouge detected."),),
        ),
    )

    assert "Target gouge detected." in engine.validate_xyza().errors

    engine.volumetric_simulations = (
        replace(simulation, connectivity=ConnectivityStatus.UNKNOWN),
    )
    assert any("connectivity is unknown" in item for item in engine.validate_xyza().errors)


def test_xyza_export_is_atomic_and_emits_simultaneous_axes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    engine = _ready_engine(incomplete=False)
    output = tmp_path / "part.nc"

    engine.export_xyza_gcode(output)

    text = output.read_text(encoding="ascii")
    assert "T1 M6\nS12000 M3" in text
    assert "G1 X1.000 Y1.000 Z1.000 A10.000" in text
    assert text.endswith("M5\nG94\n")

    output.unlink()

    def fail(*_args: object, **_kwargs: object) -> str:
        raise ToolpathValidationError("post failure")

    monkeypatch.setattr(engine_module, "generate_xyza_gcode", fail)
    with pytest.raises(ToolpathValidationError, match="post failure"):
        engine.export_xyza_gcode(output)
    assert not output.exists()
    assert not output.with_name(output.name + ".tmp").exists()


def test_xyza_export_schedules_tool_changes_only_after_safe_rapid(tmp_path: Path) -> None:
    engine = _ready_engine(incomplete=False)
    first_tool = engine.tool_assemblies[0].tool
    second_tool = replace(
        first_tool,
        number=2,
        name="Ball",
        tool_type=ToolType.BALL,
        spindle_rpm=10_000,
    )
    engine.tool_assemblies = (
        engine.tool_assemblies[0],
        ToolAssembly.from_tool(second_tool),
    )
    start = MachinePose(0.0, 0.0, 0.0, 0.0)
    first_cut = MotionBlock(
        MachinePose(1.0, 1.0, 1.0, 10.0),
        MotionKind.LINEAR,
        100.0,
        300.0,
    )
    retracted = MotionBlock(
        MachinePose(2.0, 1.0, 1.0, 10.0), MotionKind.RAPID, 100.0
    )
    second_cut = MotionBlock(
        MachinePose(3.0, 2.0, 2.0, 20.0),
        MotionKind.LINEAR,
        100.0,
        300.0,
    )
    passes = (
        PlannedPass(
            FreeformOperationKind.ROUGHING,
            1,
            ((0, 0, 0),),
            start,
            (first_cut, retracted),
        ),
        PlannedPass(
            FreeformOperationKind.FINISHING,
            2,
            ((0, 0, 0),),
            retracted.pose,
            (second_cut,),
        ),
    )
    assert engine.accessibility_report is not None
    engine.timed_passes = passes
    engine.freeform_plan = FreeformPlan(passes, engine.accessibility_report)
    current_digest = engine._current_accessibility_digest()
    assert current_digest is not None
    accessibility = replace(engine.accessibility_report, digest=current_digest)
    engine.accessibility_report = accessibility
    engine.freeform_plan = FreeformPlan(passes, accessibility)
    engine.volumetric_simulations = (
        engine.volumetric_simulations[0],
        engine.volumetric_simulations[0],
    )

    output = tmp_path / "multi-tool.nc"
    engine.export_xyza_gcode(output)

    text = output.read_text(encoding="ascii")
    assert "T1 M6\nS12000 M3" in text
    assert "G0 X2.000\nM5\nG94\nT2 M6\nS10000 M3" in text
