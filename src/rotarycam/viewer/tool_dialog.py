"""Validated cutting-bit editor for the desktop UI."""

from __future__ import annotations

from collections.abc import Iterable

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from rotarycam.tools.models import Tool, ToolType

ONE_EIGHTH_INCH_MM = 3.175


class ToolDialog(QDialog):
    """Collect every field required to create or edit a validated cutting bit."""

    def __init__(
        self,
        *,
        next_number: int = 1,
        tool: Tool | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Edit cutting bit" if tool is not None else "Add cutting bit")
        self.setModal(True)

        self.number = QSpinBox(self)
        self.number.setRange(1, 9999)
        self.number.setValue(next_number)
        self.name = QLineEdit(f"Makera Z1 Bit {next_number}", self)
        self.tool_type = QComboBox(self)
        self.tool_type.addItem("Flat end mill", ToolType.FLAT)
        self.tool_type.addItem("Ball end mill", ToolType.BALL)
        self.tool_type.addItem("Tapered / V-bit", ToolType.TAPERED)

        self.diameter = self._dimension(ONE_EIGHTH_INCH_MM)
        self.tip_diameter = self._dimension(0.5)
        self.taper_length = self._dimension(10.0)
        self.cutting_length = self._dimension(18.0)
        self.flute_length = self._dimension(18.0)
        self.overall_length = self._dimension(50.0)
        self.shank_diameter = self._dimension(ONE_EIGHTH_INCH_MM)
        self.shank_diameter.setToolTip(
            "Your Makera Z1 bits use a 1/8 inch (3.175 mm) shank; change only if needed."
        )
        self.max_stepdown = self._dimension(1.0)
        self.stepover = self._dimension(1.0)
        self.feed = self._dimension(500.0, maximum=100_000.0)
        self.plunge_feed = self._dimension(120.0, maximum=100_000.0)
        self.spindle_rpm = QSpinBox(self)
        self.spindle_rpm.setRange(1, 100_000)
        self.spindle_rpm.setValue(12_000)
        self.tool_type.currentIndexChanged.connect(self._update_taper_fields)
        if tool is not None:
            self._populate(tool)
        self._update_taper_fields()

        form = QFormLayout()
        form.addRow("Tool number", self.number)
        form.addRow("Name", self.name)
        form.addRow("Bit type", self.tool_type)
        for label, widget in self._dimension_rows():
            form.addRow(label, widget)
        form.addRow("Spindle (rpm)", self.spindle_rpm)

        self.validation_label = QLabel(self)
        self.validation_label.setWordWrap(True)
        self.validation_label.setStyleSheet("color: #b00020;")
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        personal_default = QLabel(
            "Makera Z1 default: 1/8 in (3.175 mm) shank. "
            "The cutting tip diameter may be smaller.",
            self,
        )
        personal_default.setWordWrap(True)
        layout.addWidget(personal_default)
        layout.addLayout(form)
        layout.addWidget(self.validation_label)
        layout.addWidget(buttons)

    def _populate(self, tool: Tool) -> None:
        """Populate every persisted field from an existing immutable tool."""

        self.number.setValue(tool.number)
        self.name.setText(tool.name)
        tool_type_index = self.tool_type.findData(tool.tool_type)
        if tool_type_index < 0:
            raise ValueError(f"Unsupported bit type: {tool.tool_type}")
        self.tool_type.setCurrentIndex(tool_type_index)
        self.diameter.setValue(tool.diameter)
        self.cutting_length.setValue(tool.cutting_length)
        self.flute_length.setValue(tool.flute_length)
        self.overall_length.setValue(tool.overall_length)
        self.shank_diameter.setValue(tool.shank_diameter)
        self.max_stepdown.setValue(tool.max_stepdown)
        self.stepover.setValue(tool.stepover)
        self.feed.setValue(tool.feed)
        self.plunge_feed.setValue(tool.plunge_feed)
        self.spindle_rpm.setValue(tool.spindle_rpm)
        if tool.tip_diameter is not None:
            self.tip_diameter.setValue(tool.tip_diameter)
        if tool.taper_length is not None:
            self.taper_length.setValue(tool.taper_length)

    def _dimension(self, value: float, *, maximum: float = 10_000.0) -> QDoubleSpinBox:
        editor = QDoubleSpinBox(self)
        editor.setDecimals(3)
        editor.setRange(0.001, maximum)
        editor.setSingleStep(0.1)
        editor.setValue(value)
        return editor

    def _dimension_rows(self) -> Iterable[tuple[str, QDoubleSpinBox]]:
        return (
            ("Maximum cutting diameter (mm)", self.diameter),
            ("Tip diameter (mm; tapered bits)", self.tip_diameter),
            ("Distance to maximum diameter (mm)", self.taper_length),
            ("Cutting length (mm)", self.cutting_length),
            ("Flute length (mm)", self.flute_length),
            ("Overall length (mm)", self.overall_length),
            ("Shank diameter (mm; 1/8 in = 3.175 mm)", self.shank_diameter),
            ("Maximum stepdown (mm)", self.max_stepdown),
            ("Stepover (mm)", self.stepover),
            ("Cutting feed (mm/min)", self.feed),
            ("Plunge feed (mm/min)", self.plunge_feed),
        )

    def _update_taper_fields(self, _index: int = 0) -> None:
        tapered = self.tool_type.currentData() == ToolType.TAPERED
        self.tip_diameter.setEnabled(tapered)
        self.taper_length.setEnabled(tapered)

    def tool(self) -> Tool:
        """Build the immutable domain model and run all physical validation."""

        try:
            selected_type = ToolType(str(self.tool_type.currentData()))
        except ValueError as error:
            raise ValueError("Select a supported bit type.") from error
        return Tool(
            number=self.number.value(),
            name=self.name.text().strip(),
            tool_type=selected_type,
            diameter=self.diameter.value(),
            cutting_length=self.cutting_length.value(),
            flute_length=self.flute_length.value(),
            overall_length=self.overall_length.value(),
            shank_diameter=self.shank_diameter.value(),
            max_stepdown=self.max_stepdown.value(),
            stepover=self.stepover.value(),
            feed=self.feed.value(),
            plunge_feed=self.plunge_feed.value(),
            spindle_rpm=self.spindle_rpm.value(),
            tip_diameter=(self.tip_diameter.value() if selected_type is ToolType.TAPERED else None),
            taper_length=(self.taper_length.value() if selected_type is ToolType.TAPERED else None),
        )

    def accept(self) -> None:
        """Keep the dialog open and show domain validation failures inline."""

        try:
            self.tool()
        except (TypeError, ValueError) as error:
            self.validation_label.setText(str(error))
            return
        self.validation_label.clear()
        super().accept()
