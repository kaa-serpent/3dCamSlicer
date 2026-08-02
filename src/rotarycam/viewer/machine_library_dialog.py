"""Add, edit, delete and select machine profiles in one desktop dialog."""

from __future__ import annotations

from collections.abc import Iterable

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from rotarycam.config import MachineDefinition
from rotarycam.machine.library import validate_unique_machine_names
from rotarycam.viewer.machine_dialog import MachineDialog


class MachineLibraryDialog(QDialog):
    """Manage a private list of machine profiles and choose the active one."""

    def __init__(
        self,
        profiles: Iterable[MachineDefinition] = (),
        *,
        selected_name: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Machine profiles")
        self.setModal(True)
        self.resize(640, 420)
        self._profiles = list(profiles)
        validate_unique_machine_names(self._profiles)

        self.profile_list = QListWidget(self)
        self.profile_list.setObjectName("machineProfileList")
        self.profile_list.currentRowChanged.connect(self._refresh_buttons)
        self.profile_list.itemDoubleClicked.connect(self._select_current)

        self.add_button = QPushButton("Add", self)
        self.edit_button = QPushButton("Edit", self)
        self.delete_button = QPushButton("Delete", self)
        self.add_button.clicked.connect(self._add_from_dialog)
        self.edit_button.clicked.connect(self._edit_from_dialog)
        self.delete_button.clicked.connect(self._delete_with_confirmation)

        actions = QHBoxLayout()
        actions.addWidget(self.add_button)
        actions.addWidget(self.edit_button)
        actions.addWidget(self.delete_button)
        actions.addStretch()

        self.safety_label = QLabel(
            "Editing a profile clears its verified status. Verify the controller, simulate, "
            "and dry-run before enabling CNC export.",
            self,
        )
        self.safety_label.setWordWrap(True)
        self.safety_label.setStyleSheet("color: #9a6700;")

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        self.select_button = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.select_button.setText("Select")
        self.buttons.accepted.connect(self._select_current)
        self.buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Available machine profiles", self))
        layout.addWidget(self.profile_list)
        layout.addLayout(actions)
        layout.addWidget(self.safety_label)
        layout.addWidget(self.buttons)

        self._rebuild_list(selected_name)

    def _rebuild_list(self, selected_name: str | None = None) -> None:
        self.profile_list.clear()
        for profile in self._profiles:
            status = "verified" if profile.profile_verified else "unverified"
            self.profile_list.addItem(f"{profile.name}  —  {status}")
        target = -1
        if selected_name is not None:
            normalized = selected_name.strip().casefold()
            target = next(
                (
                    index
                    for index, profile in enumerate(self._profiles)
                    if profile.name.casefold() == normalized
                ),
                -1,
            )
        if target < 0 and self._profiles:
            target = 0
        self.profile_list.setCurrentRow(target)
        self._refresh_buttons()

    def _refresh_buttons(self, _row: int = -1) -> None:
        has_selection = 0 <= self.profile_list.currentRow() < len(self._profiles)
        self.edit_button.setEnabled(has_selection)
        self.delete_button.setEnabled(has_selection)
        self.select_button.setEnabled(has_selection)

    def profiles(self) -> list[MachineDefinition]:
        """Return the manager's current validated profiles in display order."""

        return list(self._profiles)

    def selected_profile(self) -> MachineDefinition | None:
        """Return the currently highlighted profile, if any."""

        row = self.profile_list.currentRow()
        return self._profiles[row] if 0 <= row < len(self._profiles) else None

    def add_profile(self, profile: MachineDefinition) -> None:
        """Add a profile, rejecting ambiguous duplicate names."""

        validate_unique_machine_names([*self._profiles, profile])
        self._profiles.append(profile)
        self._rebuild_list(profile.name)

    def replace_selected_profile(self, profile: MachineDefinition) -> None:
        """Replace the selected profile while preserving its display position."""

        row = self.profile_list.currentRow()
        if not 0 <= row < len(self._profiles):
            raise ValueError("Select a machine profile to edit.")
        replacement = list(self._profiles)
        replacement[row] = profile
        validate_unique_machine_names(replacement)
        self._profiles = replacement
        self._rebuild_list(profile.name)

    def delete_selected_profile(self) -> MachineDefinition:
        """Delete and return the selected profile without showing a confirmation box."""

        row = self.profile_list.currentRow()
        if not 0 <= row < len(self._profiles):
            raise ValueError("Select a machine profile to delete.")
        removed = self._profiles.pop(row)
        self._rebuild_list()
        if self._profiles:
            self.profile_list.setCurrentRow(min(row, len(self._profiles) - 1))
        return removed

    def _show_duplicate_error(self, error: ValueError) -> None:
        QMessageBox.warning(self, "Machine profile", str(error))

    def _add_from_dialog(self) -> None:
        dialog = MachineDialog(parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            self.add_profile(dialog.machine())
        except ValueError as error:
            self._show_duplicate_error(error)

    def _edit_from_dialog(self) -> None:
        current = self.selected_profile()
        if current is None:
            return
        dialog = MachineDialog(current, parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            self.replace_selected_profile(dialog.machine())
        except ValueError as error:
            self._show_duplicate_error(error)

    def _delete_with_confirmation(self) -> None:
        current = self.selected_profile()
        if current is None:
            return
        answer = QMessageBox.question(
            self,
            "Delete machine profile",
            f'Delete "{current.name}"? This cannot be undone.',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.delete_selected_profile()

    def _select_current(self, _item: object | None = None) -> None:
        if self.selected_profile() is not None:
            self.accept()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Keep the platform-standard delete shortcut scoped to the selected row."""

        if event.key() == Qt.Key.Key_Delete:
            self._delete_with_confirmation()
            return
        super().keyPressEvent(event)


__all__ = ["MachineLibraryDialog"]
