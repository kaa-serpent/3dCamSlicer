import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6.QtCore import Qt

from rotarycam.tools.models import Tool, ToolType
from rotarycam.viewer.tool_library_dialog import ToolLibraryDialog


def _tool(number: int, name: str) -> Tool:
    return Tool(
        number=number,
        name=name,
        tool_type=ToolType.FLAT,
        diameter=3.0,
        cutting_length=10.0,
        flute_length=12.0,
        overall_length=45.0,
        shank_diameter=3.175,
        max_stepdown=1.0,
        stepover=1.0,
        feed=500.0,
        plunge_feed=100.0,
        spindle_rpm=12_000,
    )


def test_library_starts_without_selection_and_returns_a_copy(qtbot: object) -> None:
    original = [_tool(1, "Rougher"), _tool(3, "Finisher")]
    dialog = ToolLibraryDialog(original)
    qtbot.addWidget(dialog)  # type: ignore[attr-defined]

    exported = dialog.tools()
    exported.clear()

    assert dialog.tool_list.count() == 2
    assert dialog.tool_list.currentRow() == -1
    assert dialog.edit_button.isEnabled() is False
    assert dialog.delete_button.isEnabled() is False
    assert dialog.tools() == original


def test_library_adds_edits_and_deletes_selected_tool_deterministically(
    qtbot: object,
) -> None:
    first = _tool(1, "Rougher")
    second = _tool(2, "Finisher")
    dialog = ToolLibraryDialog([first])
    qtbot.addWidget(dialog)  # type: ignore[attr-defined]

    dialog.add_tool(second)
    assert dialog.tools() == [first, second]
    assert dialog.tool_list.currentRow() == 1

    replacement = _tool(2, "Fine finisher")
    dialog.replace_selected_tool(replacement)
    assert dialog.tools() == [first, replacement]
    assert dialog.tool_list.currentRow() == 1

    dialog.tool_list.setCurrentRow(0)
    dialog.delete_selected_tool()
    assert dialog.tools() == [replacement]
    assert dialog.tool_list.currentRow() == 0
    assert dialog.edit_button.isEnabled() is True

    dialog.delete_selected_tool()
    assert dialog.tools() == []
    assert dialog.tool_list.currentRow() == -1
    assert dialog.edit_button.isEnabled() is False
    assert dialog.delete_button.isEnabled() is False


def test_library_rejects_duplicate_numbers_without_mutating_state(qtbot: object) -> None:
    first = _tool(1, "Rougher")
    second = _tool(2, "Finisher")
    dialog = ToolLibraryDialog([first, second])
    qtbot.addWidget(dialog)  # type: ignore[attr-defined]

    with pytest.raises(ValueError, match="duplicates: 1"):
        dialog.add_tool(_tool(1, "Duplicate"))
    assert dialog.tools() == [first, second]

    dialog.tool_list.setCurrentRow(1)
    with pytest.raises(ValueError, match="duplicates: 1"):
        dialog.replace_selected_tool(_tool(1, "Also duplicate"))
    assert dialog.tools() == [first, second]


def test_library_rejects_duplicate_initial_numbers() -> None:
    with pytest.raises(ValueError, match="duplicates: 4"):
        ToolLibraryDialog([_tool(4, "One"), _tool(4, "Two")])


def test_library_delete_without_selection_is_a_noop(qtbot: object) -> None:
    tool = _tool(1, "Rougher")
    dialog = ToolLibraryDialog([tool])
    qtbot.addWidget(dialog)  # type: ignore[attr-defined]

    dialog.delete_selected_tool()

    assert dialog.tools() == [tool]


def test_library_selection_is_independent_from_available_tools(qtbot: object) -> None:
    rougher = _tool(1, "Rougher")
    finisher = _tool(2, "Finisher")
    dialog = ToolLibraryDialog(
        [rougher, finisher],
        selected_numbers=[finisher.number],
    )
    qtbot.addWidget(dialog)  # type: ignore[attr-defined]

    assert dialog.tools() == [rougher, finisher]
    assert dialog.selected_tools() == [finisher]
    assert dialog.tool_list.item(0).checkState() == Qt.CheckState.Unchecked
    assert dialog.tool_list.item(1).checkState() == Qt.CheckState.Checked

    dialog.tool_list.item(0).setCheckState(Qt.CheckState.Checked)
    dialog.tool_list.item(1).setCheckState(Qt.CheckState.Unchecked)

    assert dialog.selected_tools() == [rougher]


def test_library_rejects_selected_number_that_is_not_available() -> None:
    with pytest.raises(ValueError, match="not in the library: 9"):
        ToolLibraryDialog([_tool(1, "Rougher")], selected_numbers=[9])
