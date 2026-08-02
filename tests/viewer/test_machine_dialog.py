import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6.QtWidgets import QDialog

from rotarycam.config import AxisLimits, MachineDefinition, RotaryAxisConfig
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
    )


def test_editing_round_trips_all_fields_but_clears_verification(qtbot: object) -> None:
    original = verified_machine()
    dialog = MachineDialog(original)
    qtbot.addWidget(dialog)  # type: ignore[attr-defined]

    edited = dialog.machine()

    assert edited == original.model_copy(update={"profile_verified": False})
    assert edited.profile_verified is False
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
