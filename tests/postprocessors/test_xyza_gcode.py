import pytest

from rotarycam.config import AxisLimits, MachineDefinition, XYZAConfiguration
from rotarycam.errors import ToolpathValidationError, UnverifiedMachineProfileError
from rotarycam.machine.assemblies import (
    AssemblyRole,
    AxisDynamics,
    Box,
    FrameKind,
    MachineAssembly,
    MachineCapabilities,
)
from rotarycam.motion import MachinePose, MotionBlock, MotionKind
from rotarycam.postprocessors import generate_xyza_gcode, generate_xyza_preview


def export_machine() -> MachineDefinition:
    assembly = MachineAssembly(
        primitives=tuple(
            Box(
                role=role,
                frame=(
                    FrameKind.SPINDLE
                    if role is AssemblyRole.SPINDLE
                    else FrameKind.FIXED
                ),
                center=(float(index) * 10.0, 0.0, 0.0),
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
    )
    return MachineDefinition(
        name="Verified XYZA test machine",
        profile_verified=True,
        x_limits=AxisLimits(minimum=-100.0, maximum=100.0),
        y_limits=AxisLimits(minimum=-100.0, maximum=100.0),
        z_limits=AxisLimits(minimum=-100.0, maximum=100.0),
        xyza_configuration=XYZAConfiguration(
            rotary_pivot_y=0.0,
            rotary_pivot_z=0.0,
            rotary_zero_deg=0.0,
            spindle_axis=(0.0, 0.0, -1.0),
            g54_origin=(0.0, 0.0, 0.0),
        ),
        assembly=assembly,
        machine_assembly_configured=True,
        dynamics={
            axis: AxisDynamics(max_velocity=100_000.0, max_acceleration=100_000.0)
            for axis in ("X", "Y", "Z", "A")
        },
        capabilities=MachineCapabilities(
            simultaneous_xyza=True,
            inverse_time_feed_g93=True,
        ),
        coordinate_precision=3,
        program_header=("G54",),
        program_footer=("M5",),
    )


def test_generate_xyza_gcode_golden_emits_changed_axes_and_restores_g94() -> None:
    start = MachinePose(0.0, 0.0, 0.0, 0.0)
    blocks = (
        MotionBlock(MachinePose(1.0, 0.0, 0.0, 0.0), MotionKind.RAPID, 1.0),
        MotionBlock(
            MachinePose(2.0, 3.0, 4.0, 90.0),
            MotionKind.LINEAR,
            duration_s=2.0,
            feed=300.0,
        ),
        MotionBlock(
            MachinePose(2.0, 3.0, 4.0, 450.0),
            MotionKind.LINEAR,
            duration_s=3.0,
            feed=300.0,
        ),
    )

    assert generate_xyza_gcode(start, blocks, export_machine()) == (
        "G21\n"
        "G90\n"
        "G54\n"
        "G0 X1.000\n"
        "G93\n"
        "G1 X2.000 Y3.000 Z4.000 A90.000 F30.000\n"
        "G1 A450.000 F20.000\n"
        "G94\n"
        "M5\n"
    )


def test_generate_xyza_preview_is_available_for_incomplete_profile() -> None:
    machine = export_machine().model_copy(
        update={"profile_verified": False, "capabilities": None, "dynamics": None}
    )
    block = MotionBlock(
        MachinePose(0.0, 1.0, 0.0, 0.0), MotionKind.LINEAR, 1.0, feed=100.0
    )

    preview = generate_xyza_preview(MachinePose(0.0, 0.0, 0.0, 0.0), (block,), machine)

    assert preview.startswith("(RotaryCAM XYZA PREVIEW - NOT FOR MACHINE EXECUTION)\n")
    assert "G1 Y1.000 F60.000" in preview
    assert preview.endswith("G94\nM5\n")


def test_generate_xyza_gcode_blocks_unverified_or_unconfirmed_controller() -> None:
    start = MachinePose(0.0, 0.0, 0.0, 0.0)
    block = MotionBlock(
        MachinePose(1.0, 0.0, 0.0, 0.0), MotionKind.LINEAR, 1.0, feed=100.0
    )
    verified = export_machine()

    with pytest.raises(UnverifiedMachineProfileError):
        generate_xyza_gcode(
            start, (block,), verified.model_copy(update={"profile_verified": False})
        )
    with pytest.raises(ToolpathValidationError, match="capabilities"):
        generate_xyza_gcode(
            start, (block,), verified.model_copy(update={"capabilities": None})
        )
    with pytest.raises(ToolpathValidationError, match="simultaneous"):
        generate_xyza_gcode(
            start,
            (block,),
            verified.model_copy(
                update={
                    "capabilities": MachineCapabilities(
                        simultaneous_xyza=False, inverse_time_feed_g93=True
                    )
                }
            ),
        )
    with pytest.raises(ToolpathValidationError, match="G93"):
        generate_xyza_gcode(
            start,
            (block,),
            verified.model_copy(
                update={
                    "capabilities": MachineCapabilities(
                        simultaneous_xyza=True, inverse_time_feed_g93=False
                    )
                }
            ),
        )


def test_generate_xyza_gcode_blocks_incomplete_dynamics_and_unsafe_duration() -> None:
    machine = export_machine()
    start = MachinePose(0.0, 0.0, 0.0, 0.0)
    block = MotionBlock(
        MachinePose(100.0, 0.0, 0.0, 0.0),
        MotionKind.LINEAR,
        duration_s=0.001,
        feed=10_000.0,
    )
    assert machine.dynamics is not None

    with pytest.raises(ToolpathValidationError, match="dynamics"):
        generate_xyza_gcode(
            start,
            (block,),
            machine.model_copy(update={"dynamics": {"X": machine.dynamics["X"]}}),
        )
    with pytest.raises(ToolpathValidationError, match="duration"):
        generate_xyza_gcode(start, (block,), machine)


def test_generate_xyza_gcode_blocks_incomplete_assembly_and_erased_motion() -> None:
    machine = export_machine()
    start = MachinePose(0.0, 0.0, 0.0, 0.0)
    block = MotionBlock(
        MachinePose(1.0, 0.0, 0.0, 0.0),
        MotionKind.LINEAR,
        duration_s=1.0,
        feed=100.0,
    )

    with pytest.raises(ToolpathValidationError, match="assembly"):
        generate_xyza_gcode(
            start, (block,), machine.model_copy(update={"assembly": None})
        )
    with pytest.raises(ToolpathValidationError, match="erase X"):
        generate_xyza_gcode(
            start,
            (
                MotionBlock(
                    MachinePose(0.0004, 0.0, 0.0, 0.0),
                    MotionKind.LINEAR,
                    duration_s=1.0,
                    feed=100.0,
                ),
            ),
            machine,
        )
