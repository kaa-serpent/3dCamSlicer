"""Cutting-tool library management dialog."""

from __future__ import annotations

from collections.abc import Iterable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from rotarycam.tools.models import Tool, validate_unique_tool_numbers
from rotarycam.viewer.tool_dialog import ToolDialog


class ToolLibraryDialog(QDialog):
    """Edit a private, deterministic copy of a cutting-tool collection."""

    def __init__(
        self,
        tools: Iterable[Tool] = (),
        *,
        selected_numbers: Iterable[int] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        initial_tools = list(tools)
        validate_unique_tool_numbers(initial_tools)
        self._tools = initial_tools
        available_numbers = {tool.number for tool in initial_tools}
        self._selected_numbers = (
            set(available_numbers) if selected_numbers is None else set(selected_numbers)
        )
        unknown_numbers = self._selected_numbers - available_numbers
        if unknown_numbers:
            rendered = ", ".join(str(number) for number in sorted(unknown_numbers))
            raise ValueError(f"selected tool numbers are not in the library: {rendered}")

        self.setWindowTitle("Manage cutting bits")
        self.setModal(True)

        self.tool_list = QListWidget(self)
        self.tool_list.currentRowChanged.connect(self._on_selection_changed)
        self.tool_list.itemChanged.connect(self._on_item_checked)

        self.add_button = QPushButton("Add", self)
        self.add_button.clicked.connect(self.open_add_dialog)
        self.edit_button = QPushButton("Edit", self)
        self.edit_button.clicked.connect(self.open_edit_dialog)
        self.delete_button = QPushButton("Delete", self)
        self.delete_button.clicked.connect(self.delete_selected_tool)

        actions = QHBoxLayout()
        actions.addWidget(self.add_button)
        actions.addWidget(self.edit_button)
        actions.addWidget(self.delete_button)

        self.validation_label = QLabel(self)
        self.validation_label.setWordWrap(True)
        self.validation_label.setStyleSheet("color: #b00020;")

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(self.tool_list)
        layout.addLayout(actions)
        layout.addWidget(self.validation_label)
        layout.addWidget(buttons)

        self._refresh_list()

    def tools(self) -> list[Tool]:
        """Return a shallow copy; :class:`Tool` records are immutable."""

        return list(self._tools)

    def selected_tools(self) -> list[Tool]:
        """Return only the library tools selected for the current project."""

        return [tool for tool in self._tools if tool.number in self._selected_numbers]

    def _next_tool_number(self) -> int:
        used = {tool.number for tool in self._tools}
        candidate = 1
        while candidate in used:
            candidate += 1
        return candidate

    @staticmethod
    def _tool_label(tool: Tool) -> str:
        return f"T{tool.number} - {tool.name} ({tool.tool_type.value}, Ø{tool.diameter:g} mm)"

    def _refresh_list(self, *, selected_row: int = -1) -> None:
        previous = self.tool_list.blockSignals(True)
        self.tool_list.clear()
        for tool in self._tools:
            item = QListWidgetItem(self._tool_label(tool))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked
                if tool.number in self._selected_numbers
                else Qt.CheckState.Unchecked
            )
            self.tool_list.addItem(item)
        self.tool_list.setCurrentRow(selected_row)
        self.tool_list.blockSignals(previous)
        self._on_selection_changed(selected_row)

    def _on_selection_changed(self, row: int) -> None:
        selected = 0 <= row < len(self._tools)
        self.edit_button.setEnabled(selected)
        self.delete_button.setEnabled(selected)

    def _on_item_checked(self, item: QListWidgetItem) -> None:
        row = self.tool_list.row(item)
        if not 0 <= row < len(self._tools):
            return
        tool_number = self._tools[row].number
        if item.checkState() is Qt.CheckState.Checked:
            self._selected_numbers.add(tool_number)
        else:
            self._selected_numbers.discard(tool_number)

    def add_tool(self, tool: Tool, *, selected: bool = True) -> None:
        """Append a validated tool while preserving collection uniqueness."""

        candidate = [*self._tools, tool]
        validate_unique_tool_numbers(candidate)
        self._tools = candidate
        if selected:
            self._selected_numbers.add(tool.number)
        self.validation_label.clear()
        self._refresh_list(selected_row=len(self._tools) - 1)

    def replace_selected_tool(self, tool: Tool) -> None:
        """Replace the selected tool without changing library order."""

        row = self.tool_list.currentRow()
        if not 0 <= row < len(self._tools):
            raise ValueError("Select a cutting bit to edit.")
        candidate = list(self._tools)
        previous_number = candidate[row].number
        was_selected = previous_number in self._selected_numbers
        candidate[row] = tool
        validate_unique_tool_numbers(candidate)
        self._tools = candidate
        self._selected_numbers.discard(previous_number)
        if was_selected:
            self._selected_numbers.add(tool.number)
        self.validation_label.clear()
        self._refresh_list(selected_row=row)

    def delete_selected_tool(self) -> None:
        """Delete only the current selection and select its nearest neighbour."""

        row = self.tool_list.currentRow()
        if not 0 <= row < len(self._tools):
            return
        removed = self._tools.pop(row)
        self._selected_numbers.discard(removed.number)
        next_row = min(row, len(self._tools) - 1)
        self.validation_label.clear()
        self._refresh_list(selected_row=next_row)

    def open_add_dialog(self) -> None:
        """Open the editor with the first available positive tool number."""

        dialog = ToolDialog(next_number=self._next_tool_number(), parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            self.add_tool(dialog.tool())
        except (TypeError, ValueError) as error:
            self.validation_label.setText(str(error))

    def open_edit_dialog(self) -> None:
        """Open the editor prefilled with the selected tool."""

        row = self.tool_list.currentRow()
        if not 0 <= row < len(self._tools):
            return
        dialog = ToolDialog(tool=self._tools[row], parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            self.replace_selected_tool(dialog.tool())
        except (TypeError, ValueError) as error:
            self.validation_label.setText(str(error))
