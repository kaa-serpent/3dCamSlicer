import pytest
from pydantic import ValidationError

from rotarycam.config import AxisLimits, MachineDefinition, XYZAConfiguration
from rotarycam.machine.assemblies import (
    AssemblyRole,
    AxisDynamics,
    Box,
    Cylinder,
    FrameKind,
    Frustum,
    MachineAssembly,
    MachineCapabilities,
)


def complete_assembly() -> MachineAssembly:
    return MachineAssembly(
        primitives=(
            Cylinder(
                role=AssemblyRole.CHUCK,
                frame=FrameKind.ROTARY,
                center=(0.0, 0.0, 0.0),
                radius=20.0,
                length=30.0,
            ),
            Box(
                role=AssemblyRole.JAWS,
                frame=FrameKind.ROTARY,
                center=(0.0, 0.0, 0.0),
                size=(20.0, 20.0, 20.0),
            ),
            Frustum(
                role=AssemblyRole.TAILSTOCK,
                frame=FrameKind.FIXED,
                center=(100.0, 0.0, 0.0),
                radius_start=8.0,
                radius_end=2.0,
                length=20.0,
            ),
            Cylinder(
                role=AssemblyRole.PLATTER,
                frame=FrameKind.ROTARY,
                center=(-5.0, 0.0, 0.0),
                radius=30.0,
                length=5.0,
            ),
            Box(
                role=AssemblyRole.SPINDLE,
                frame=FrameKind.SPINDLE,
                center=(0.0, 0.0, 50.0),
                size=(60.0, 60.0, 100.0),
            ),
            Box(
                role=AssemblyRole.SUPPORT,
                frame=FrameKind.FIXED,
                center=(50.0, 0.0, -20.0),
                size=(10.0, 20.0, 10.0),
            ),
        )
    )


def test_machine_assembly_reports_missing_required_roles() -> None:
    assembly = MachineAssembly(
        primitives=(
            Box(
                role="chuck",
                frame="rotary",
                center=(0.0, 0.0, 0.0),
                size=(1.0, 2.0, 3.0),
            ),
        )
    )

    assert assembly.is_complete is False
    assert AssemblyRole.CHUCK not in assembly.missing_roles
    assert AssemblyRole.SPINDLE in assembly.missing_roles
    assert AssemblyRole.SUPPORT in assembly.missing_roles
    assert complete_assembly().is_complete is True


def test_setup_transform_must_be_a_right_handed_rigid_transform() -> None:
    with pytest.raises(ValidationError, match="orthonormal"):
        XYZAConfiguration(
            rotary_pivot_y=0.0,
            rotary_pivot_z=0.0,
            rotary_zero_deg=0.0,
            spindle_axis=(0.0, 0.0, -1.0),
            g54_origin=(0.0, 0.0, 0.0),
            setup_transform=(
                (2.0, 0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            ),
        )


def test_measured_primitive_normalizes_axis_and_rejects_invalid_size() -> None:
    cylinder = Cylinder(
        role="chuck",
        frame="rotary",
        center=(0.0, 0.0, 0.0),
        radius=2.0,
        length=3.0,
        axis=(2.0, 0.0, 0.0),
    )
    assert cylinder.axis == pytest.approx((1.0, 0.0, 0.0))

    with pytest.raises(ValidationError):
        Box(
            role="jaws",
            frame="rotary",
            center=(0.0, 0.0, 0.0),
            size=(1.0, 0.0, 3.0),
        )


def test_machine_definition_v2_roundtrip_keeps_optional_xyza_contracts_unverified() -> None:
    machine = MachineDefinition(
        name="Measured test machine",
        profile_verified=False,
        x_limits=AxisLimits(minimum=0.0, maximum=200.0),
        y_limits=AxisLimits(minimum=-50.0, maximum=50.0),
        z_limits=AxisLimits(minimum=-20.0, maximum=100.0),
        xyza_configuration=XYZAConfiguration(
            rotary_pivot_y=1.0,
            rotary_pivot_z=2.0,
            rotary_zero_deg=3.0,
            spindle_axis=(0.0, 0.0, -1.0),
            g54_origin=(10.0, 20.0, 30.0),
        ),
        dynamics={
            "x": AxisDynamics(max_velocity=1_000.0, max_acceleration=100.0),
            "A": AxisDynamics(max_velocity=3_600.0, max_acceleration=500.0),
        },
        capabilities=MachineCapabilities(
            simultaneous_xyza=True,
            inverse_time_feed_g93=True,
        ),
        assembly=complete_assembly(),
    )

    restored = MachineDefinition.model_validate_json(machine.model_dump_json())

    assert restored == machine
    assert restored.profile_verified is False
    assert restored.dynamics is not None
    assert set(restored.dynamics) == {"X", "A"}
    assert restored.assembly is not None and restored.assembly.is_complete
