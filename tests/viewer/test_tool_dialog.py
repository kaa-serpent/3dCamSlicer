import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6.QtWidgets import QDialog

from rotarycam.tools.models import Tool, ToolHolder, ToolType
from rotarycam.viewer.tool_dialog import ONE_EIGHTH_INCH_MM, ToolDialog


def _tapered_tool() -> Tool:
    return Tool(
        number=12,
        name="Detailed taper",
        tool_type=ToolType.TAPERED,
        diameter=3.0,
        tip_diameter=0.3,
        taper_length=7.5,
        cutting_length=12.0,
        flute_length=15.0,
        overall_length=48.0,
        shank_diameter=3.175,
        max_stepdown=0.8,
        stepover=0.2,
        feed=420.0,
        plunge_feed=90.0,
        spindle_rpm=18_000,
    )


def test_dialog_defaults_to_personal_makera_z1_eighth_inch_bits(qtbot: object) -> None:
    dialog = ToolDialog(next_number=2)
    qtbot.addWidget(dialog)  # type: ignore[attr-defined]

    tool = dialog.tool()

    assert tool.name == "Makera Z1 Bit 2"
    assert tool.diameter == pytest.approx(ONE_EIGHTH_INCH_MM)
    assert tool.shank_diameter == pytest.approx(ONE_EIGHTH_INCH_MM)
    assert tool.max_stepdown == pytest.approx(1.0)
    assert tool.stepover == pytest.approx(1.0)
    assert dialog.tip_diameter.isEnabled() is False
    assert dialog.taper_length.isEnabled() is False


def test_dialog_builds_ball_bit_with_all_machining_fields(qtbot: object) -> None:
    dialog = ToolDialog(next_number=7)
    qtbot.addWidget(dialog)  # type: ignore[attr-defined]
    dialog.name.setText("Fine ball")
    dialog.tool_type.setCurrentIndex(1)
    dialog.diameter.setValue(2.0)
    dialog.shank_diameter.setValue(2.0)
    dialog.stepover.setValue(0.25)

    tool = dialog.tool()

    assert tool.number == 7
    assert tool.name == "Fine ball"
    assert tool.tool_type is ToolType.BALL
    assert tool.diameter == pytest.approx(2.0)
    assert tool.stepover == pytest.approx(0.25)


def test_dialog_builds_tapered_bit_with_independent_eighth_inch_shank(qtbot: object) -> None:
    dialog = ToolDialog(next_number=4)
    qtbot.addWidget(dialog)  # type: ignore[attr-defined]
    dialog.tool_type.setCurrentIndex(2)
    dialog.diameter.setValue(3.0)
    dialog.tip_diameter.setValue(0.4)
    dialog.taper_length.setValue(8.0)

    tool = dialog.tool()

    assert tool.tool_type is ToolType.TAPERED
    assert tool.tip_diameter == pytest.approx(0.4)
    assert tool.diameter == pytest.approx(3.0)
    assert tool.taper_length == pytest.approx(8.0)
    assert tool.shank_diameter == pytest.approx(ONE_EIGHTH_INCH_MM)
    assert dialog.tip_diameter.isEnabled() is True


def test_dialog_keeps_invalid_bit_open_and_shows_domain_error(qtbot: object) -> None:
    dialog = ToolDialog(next_number=1)
    qtbot.addWidget(dialog)  # type: ignore[attr-defined]
    dialog.diameter.setValue(1.0)
    dialog.stepover.setValue(2.0)

    dialog.accept()

    assert dialog.result() == QDialog.DialogCode.Rejected
    assert "stepover must not exceed diameter" in dialog.validation_label.text()


def test_dialog_prefills_every_field_when_editing(qtbot: object) -> None:
    existing = _tapered_tool()
    dialog = ToolDialog(tool=existing)
    qtbot.addWidget(dialog)  # type: ignore[attr-defined]

    rebuilt = dialog.tool()

    assert dialog.windowTitle() == "Edit cutting bit"
    assert rebuilt == existing
    assert dialog.number.value() == 12
    assert dialog.name.text() == "Detailed taper"
    assert dialog.tool_type.currentData() == ToolType.TAPERED
    assert dialog.tip_diameter.value() == pytest.approx(0.3)
    assert dialog.taper_length.value() == pytest.approx(7.5)


def test_dialog_collects_measured_stickout_and_holder_for_xyza_export(
    qtbot: object,
) -> None:
    dialog = ToolDialog(next_number=5)
    qtbot.addWidget(dialog)  # type: ignore[attr-defined]
    dialog.assembly_measured.setChecked(True)
    dialog.stickout.setValue(25.0)
    dialog.holder_diameter.setValue(18.0)
    dialog.holder_length.setValue(30.0)

    tool = dialog.tool()

    assert tool.stickout == pytest.approx(25.0)
    assert tool.holder == ToolHolder(diameter=18.0, length=30.0)


def test_legacy_tool_remains_explicitly_incomplete_for_export(qtbot: object) -> None:
    dialog = ToolDialog(tool=_tapered_tool())
    qtbot.addWidget(dialog)  # type: ignore[attr-defined]

    rebuilt = dialog.tool()

    assert dialog.assembly_measured.isChecked() is False
    assert rebuilt.stickout is None
    assert rebuilt.holder is None
