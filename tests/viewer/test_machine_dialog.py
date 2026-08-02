import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6.QtWidgets import QDialog

from rotarycam.config import (
    AxisLimits,
    MachineDefinition,
    MachineObservationMetadata,
    RotaryAxisConfig,
)
from rotarycam.machine import (
    AssemblyRole,
    Box,
    FrameKind,
    MachineAssembly,
    MachineCapabilities,
)
from rotarycam.viewer.machine_dialog import MachineDialog


def verified_machine() -> MachineDefinition:
    return MachineDefinition(
        name="Reviewed elsewhere",
        profile_verified=True,
        x_limits=AxisLimits(minimum=-10.0, maximum=210.0),
        z_limits=AxisLimits(minimum=2.0, maximum=120.0),
        rotary_axis=RotaryAxisConfig(
            axis_letter="B",
            direction=-1,
            degrees_per_revolution=720.0,
            allow_unbounded_angles=False,
            reset_between_operations=True,
            max_speed_deg_per_min=2_400.0,
            positioning_precision_deg=0.01,
            drive_system="gearbox",
            motor="servo",
        ),
        max_spindle_rpm=24_000,
        spindle_power_w=800.0,
        max_linear_speed_mm_min=3_000.0,
        safe_radius=55.0,
        max_rotary_stock_length=180.0,
        max_rotary_stock_radius=50.0,
        coordinate_precision=4,
        program_header=("G54", "G90"),
        program_footer=("M5", "M30"),
        observations=MachineObservationMetadata(
            controller_firmware="test-firmware",
            home_display_position_mm=(1.0, 2.0, 3.0),
        ),
    )


def test_editing_round_trips_all_fields_but_clears_verification(qtbot: object) -> None:
    original = verified_machine()
    dialog = MachineDialog(original)
    qtbot.addWidget(dialog)  # type: ignore[attr-defined]

    edited = dialog.machine()

    assert edited == original.model_copy(update={"profile_verified": False})
    assert edited.profile_verified is False
    assert edited.observations == original.observations
    assert "always marked unverified" in dialog.findChild(
        type(dialog.validation_label), "verificationWarning"
    ).text()


def test_new_machine_uses_unverified_safe_defaults(qtbot: object) -> None:
    dialog = MachineDialog()
    qtbot.addWidget(dialog)  # type: ignore[attr-defined]

    profile = dialog.machine()

    assert profile.profile_verified is False
    assert profile.x_limits == AxisLimits(minimum=0.0, maximum=200.0)
    assert profile.rotary_axis.axis_letter == "A"


def test_dialog_blocks_safe_radius_inside_declared_stock(qtbot: object) -> None:
    dialog = MachineDialog()
    qtbot.addWidget(dialog)  # type: ignore[attr-defined]
    dialog.safe_radius.setValue(40.0)
    dialog.max_stock_radius.setValue(40.0)

    dialog.accept()

    assert dialog.result() == QDialog.DialogCode.Rejected
    assert "greater than maximum stock radius" in dialog.validation_label.text()


def test_dialog_keeps_invalid_axis_range_open(qtbot: object) -> None:
    dialog = MachineDialog()
    qtbot.addWidget(dialog)  # type: ignore[attr-defined]
    dialog.x_maximum.setValue(dialog.x_minimum.value())

    dialog.accept()

    assert dialog.result() == QDialog.DialogCode.Rejected
    assert "axis maximum must be greater" in dialog.validation_label.text()


def test_dialog_collects_xyza_setup_dynamics_and_controller_capabilities(
    qtbot: object,
) -> None:
    dialog = MachineDialog()
    qtbot.addWidget(dialog)  # type: ignore[attr-defined]
    dialog.xyza_measured.setChecked(True)
    dialog.y_minimum.setValue(-40.0)
    dialog.y_maximum.setValue(60.0)
    dialog.rotary_pivot_y.setValue(12.0)
    dialog.rotary_pivot_z.setValue(34.0)
    dialog.rotary_zero.setValue(5.0)
    dialog.g54_x.setValue(1.0)
    dialog.g54_y.setValue(2.0)
    dialog.g54_z.setValue(3.0)
    dialog.dynamics_measured.setChecked(True)
    dialog.axis_dynamics["A"][0].setValue(3_600.0)
    dialog.axis_dynamics["A"][1].setValue(240.0)
    dialog.simultaneous_xyza.setChecked(True)
    dialog.inverse_time_g93.setChecked(True)

    profile = dialog.machine()

    assert profile.profile_verified is False
    assert profile.y_limits == AxisLimits(minimum=-40.0, maximum=60.0)
    assert profile.xyza_configuration is not None
    assert profile.xyza_configuration.rotary_pivot_y == pytest.approx(12.0)
    assert profile.xyza_configuration.g54_origin == pytest.approx((1.0, 2.0, 3.0))
    assert profile.dynamics is not None
    assert profile.dynamics["A"].max_velocity == pytest.approx(3_600.0)
    assert profile.dynamics["A"].max_acceleration == pytest.approx(240.0)
    assert profile.capabilities == MachineCapabilities(
        simultaneous_xyza=True,
        inverse_time_feed_g93=True,
    )


def test_unknown_xyza_measurements_remain_explicitly_unset(qtbot: object) -> None:
    dialog = MachineDialog()
    qtbot.addWidget(dialog)  # type: ignore[attr-defined]

    profile = dialog.machine()

    assert dialog.y_minimum.isEnabled() is False
    assert dialog.axis_dynamics["X"][0].isEnabled() is False
    assert profile.y_limits is None
    assert profile.xyza_configuration is None
    assert profile.dynamics is None
    assert profile.capabilities is None


def test_dialog_validates_and_persists_measured_assembly_json(qtbot: object) -> None:
    assembly = MachineAssembly(
        primitives=tuple(
            Box(
                role=role,
                frame=FrameKind.FIXED,
                center=(float(index), 0.0, 0.0),
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
    dialog = MachineDialog()
    qtbot.addWidget(dialog)  # type: ignore[attr-defined]
    dialog.assembly_json.setPlainText(assembly.model_dump_json())

    profile = dialog.machine()

    assert profile.profile_verified is False
    assert profile.machine_assembly_configured is True
    assert profile.assembly == assembly

    dialog.assembly_json.setPlainText("not json")
    with pytest.raises(ValueError, match="Invalid measured assembly JSON"):
        dialog.machine()
