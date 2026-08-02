import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6.QtWidgets import QDialog

from rotarycam.config import AxisLimits, MachineDefinition
from rotarycam.viewer.machine_library_dialog import MachineLibraryDialog


def machine(name: str, *, verified: bool = False) -> MachineDefinition:
    return MachineDefinition(
        name=name,
        profile_verified=verified,
        x_limits=AxisLimits(minimum=0.0, maximum=200.0),
        z_limits=AxisLimits(minimum=0.0, maximum=100.0),
    )


def test_manager_selects_requested_profile_and_exposes_copy(qtbot: object) -> None:
    first = machine("First")
    second = machine("Second", verified=True)
    dialog = MachineLibraryDialog([first, second], selected_name="second")
    qtbot.addWidget(dialog)  # type: ignore[attr-defined]

    assert dialog.selected_profile() == second
    returned = dialog.profiles()
    returned.clear()
    assert dialog.profiles() == [first, second]
    assert "verified" in dialog.profile_list.currentItem().text()


def test_manager_add_edit_delete_and_select(qtbot: object) -> None:
    dialog = MachineLibraryDialog([machine("First")])
    qtbot.addWidget(dialog)  # type: ignore[attr-defined]

    dialog.add_profile(machine("Second"))
    assert dialog.selected_profile() == machine("Second")
    dialog.replace_selected_profile(machine("Renamed"))
    assert [profile.name for profile in dialog.profiles()] == ["First", "Renamed"]
    removed = dialog.delete_selected_profile()
    assert removed.name == "Renamed"
    assert dialog.selected_profile() == machine("First")

    dialog.select_button.click()
    assert dialog.result() == QDialog.DialogCode.Accepted


def test_manager_rejects_duplicate_names_case_insensitively(qtbot: object) -> None:
    dialog = MachineLibraryDialog([machine("Router")])
    qtbot.addWidget(dialog)  # type: ignore[attr-defined]

    with pytest.raises(ValueError, match="unique"):
        dialog.add_profile(machine("ROUTER"))


def test_manager_disables_selection_actions_when_empty(qtbot: object) -> None:
    dialog = MachineLibraryDialog()
    qtbot.addWidget(dialog)  # type: ignore[attr-defined]

    assert dialog.selected_profile() is None
    assert dialog.edit_button.isEnabled() is False
    assert dialog.delete_button.isEnabled() is False
    assert dialog.select_button.isEnabled() is False
