"""Complete, safety-conscious machine-profile editor for the desktop UI."""

from __future__ import annotations

from collections.abc import Iterable

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from rotarycam.config import AxisLimits, MachineDefinition, RotaryAxisConfig


class MachineDialog(QDialog):
    """Create or edit every persisted machine field except verification status.

    A profile assembled here is always unverified.  Verification represents an
    out-of-band controller review and must never be granted by an ordinary edit.
    """

    def __init__(
        self,
        machine: MachineDefinition | None = None,
        *,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add machine profile" if machine is None else "Edit machine profile")
        self.setModal(True)

        self.name = QLineEdit(machine.name if machine is not None else "New machine", self)
        self.x_minimum = self._number(-100_000.0, 100_000.0, 0.0)
        self.x_maximum = self._number(-100_000.0, 100_000.0, 200.0)
        self.z_minimum = self._number(-100_000.0, 100_000.0, 0.0)
        self.z_maximum = self._number(-100_000.0, 100_000.0, 100.0)

        self.axis_letter = QLineEdit("A", self)
        self.axis_letter.setMaxLength(1)
        self.rotary_direction = QComboBox(self)
        self.rotary_direction.addItem("Positive", 1)
        self.rotary_direction.addItem("Negative", -1)
        self.degrees_per_revolution = self._positive_number(360.0, maximum=100_000.0)
        self.allow_unbounded_angles = QCheckBox("Allow continuous/unbounded A angles", self)
        self.allow_unbounded_angles.setChecked(True)
        self.reset_between_operations = QCheckBox("Reset A only between operations", self)
        self.max_rotary_speed = self._optional_number(maximum=10_000_000.0)
        self.positioning_precision = self._optional_number(maximum=360.0, decimals=6)
        self.drive_system = QLineEdit(self)
        self.motor = QLineEdit(self)

        self.max_spindle_rpm = self._optional_integer(maximum=1_000_000)
        self.spindle_power = self._optional_number(maximum=10_000_000.0)
        self.max_linear_speed = self._optional_number(maximum=10_000_000.0)
        self.safe_radius = self._optional_number(maximum=100_000.0)
        self.max_stock_length = self._optional_number(maximum=100_000.0)
        self.max_stock_radius = self._optional_number(maximum=100_000.0)
        self.coordinate_precision = QSpinBox(self)
        self.coordinate_precision.setRange(0, 6)
        self.coordinate_precision.setValue(3)
        self.program_header = QPlainTextEdit(self)
        self.program_header.setPlaceholderText("One controller command per line")
        self.program_header.setMaximumHeight(72)
        self.program_footer = QPlainTextEdit(self)
        self.program_footer.setPlaceholderText("One controller command per line")
        self.program_footer.setMaximumHeight(72)

        if machine is not None:
            self._load_machine(machine)

        form = QFormLayout()
        for label, widget in self._rows():
            form.addRow(label, widget)

        warning = QLabel(
            "Safety: profiles created or edited here are always marked unverified. "
            "Controller documentation, simulation and a dry-run must be checked before export.",
            self,
        )
        warning.setObjectName("verificationWarning")
        warning.setWordWrap(True)
        warning.setStyleSheet("color: #9a6700; font-weight: 600;")
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
        layout.addWidget(warning)
        layout.addLayout(form)
        layout.addWidget(self.validation_label)
        layout.addWidget(buttons)

    def _rows(self) -> Iterable[tuple[str, QWidget]]:
        return (
            ("Profile name", self.name),
            ("X minimum (mm)", self.x_minimum),
            ("X maximum (mm)", self.x_maximum),
            ("Radial Z minimum (mm)", self.z_minimum),
            ("Radial Z maximum (mm)", self.z_maximum),
            ("Rotary axis letter", self.axis_letter),
            ("Rotary direction", self.rotary_direction),
            ("Degrees per revolution", self.degrees_per_revolution),
            ("Rotary winding", self.allow_unbounded_angles),
            ("Rotary reset policy", self.reset_between_operations),
            ("Maximum rotary speed (deg/min)", self.max_rotary_speed),
            ("A positioning precision (deg)", self.positioning_precision),
            ("Rotary drive system", self.drive_system),
            ("Rotary motor", self.motor),
            ("Maximum spindle speed (rpm)", self.max_spindle_rpm),
            ("Spindle power (W)", self.spindle_power),
            ("Maximum linear speed (mm/min)", self.max_linear_speed),
            ("Safe radial TCP (mm)", self.safe_radius),
            ("Maximum rotary stock length (mm)", self.max_stock_length),
            ("Maximum rotary stock radius (mm)", self.max_stock_radius),
            ("G-code decimal places", self.coordinate_precision),
            ("Program header", self.program_header),
            ("Program footer", self.program_footer),
        )

    def _number(
        self,
        minimum: float,
        maximum: float,
        value: float,
        *,
        decimals: int = 3,
    ) -> QDoubleSpinBox:
        editor = QDoubleSpinBox(self)
        editor.setDecimals(decimals)
        editor.setRange(minimum, maximum)
        editor.setValue(value)
        return editor

    def _positive_number(
        self,
        value: float,
        *,
        maximum: float,
        decimals: int = 3,
    ) -> QDoubleSpinBox:
        return self._number(0.001, maximum, value, decimals=decimals)

    def _optional_number(
        self,
        *,
        maximum: float,
        decimals: int = 3,
    ) -> QDoubleSpinBox:
        editor = self._number(0.0, maximum, 0.0, decimals=decimals)
        editor.setSpecialValueText("Not set")
        return editor

    def _optional_integer(self, *, maximum: int) -> QSpinBox:
        editor = QSpinBox(self)
        editor.setRange(0, maximum)
        editor.setSpecialValueText("Not set")
        return editor

    @staticmethod
    def _set_optional_float(editor: QDoubleSpinBox, value: float | None) -> None:
        editor.setValue(0 if value is None else value)

    @staticmethod
    def _set_optional_int(editor: QSpinBox, value: int | None) -> None:
        editor.setValue(0 if value is None else value)

    def _load_machine(self, machine: MachineDefinition) -> None:
        self.x_minimum.setValue(machine.x_limits.minimum)
        self.x_maximum.setValue(machine.x_limits.maximum)
        self.z_minimum.setValue(machine.z_limits.minimum)
        self.z_maximum.setValue(machine.z_limits.maximum)
        rotary = machine.rotary_axis
        self.axis_letter.setText(rotary.axis_letter)
        self.rotary_direction.setCurrentIndex(0 if rotary.direction == 1 else 1)
        self.degrees_per_revolution.setValue(rotary.degrees_per_revolution)
        self.allow_unbounded_angles.setChecked(rotary.allow_unbounded_angles)
        self.reset_between_operations.setChecked(rotary.reset_between_operations)
        self._set_optional_float(self.max_rotary_speed, rotary.max_speed_deg_per_min)
        self._set_optional_float(self.positioning_precision, rotary.positioning_precision_deg)
        self.drive_system.setText(rotary.drive_system or "")
        self.motor.setText(rotary.motor or "")
        self._set_optional_int(self.max_spindle_rpm, machine.max_spindle_rpm)
        self._set_optional_float(self.spindle_power, machine.spindle_power_w)
        self._set_optional_float(self.max_linear_speed, machine.max_linear_speed_mm_min)
        self._set_optional_float(self.safe_radius, machine.safe_radius)
        self._set_optional_float(self.max_stock_length, machine.max_rotary_stock_length)
        self._set_optional_float(self.max_stock_radius, machine.max_rotary_stock_radius)
        self.coordinate_precision.setValue(machine.coordinate_precision)
        self.program_header.setPlainText("\n".join(machine.program_header))
        self.program_footer.setPlainText("\n".join(machine.program_footer))

    @staticmethod
    def _optional_float(editor: QDoubleSpinBox) -> float | None:
        return None if editor.value() == editor.minimum() else editor.value()

    @staticmethod
    def _optional_int(editor: QSpinBox) -> int | None:
        return None if editor.value() == editor.minimum() else editor.value()

    @staticmethod
    def _optional_text(editor: QLineEdit) -> str | None:
        value = editor.text().strip()
        return value or None

    @staticmethod
    def _program_lines(editor: QPlainTextEdit) -> tuple[str, ...]:
        return tuple(line.strip() for line in editor.toPlainText().splitlines() if line.strip())

    def machine(self) -> MachineDefinition:
        """Build a fully validated, explicitly unverified machine definition."""

        safe_radius = self._optional_float(self.safe_radius)
        maximum_stock_radius = self._optional_float(self.max_stock_radius)
        if (
            safe_radius is not None
            and maximum_stock_radius is not None
            and safe_radius <= maximum_stock_radius
        ):
            raise ValueError("safe radius must be greater than maximum stock radius")
        direction = self.rotary_direction.currentData()
        if direction not in (-1, 1):
            raise ValueError("Select a supported rotary direction.")
        return MachineDefinition(
            name=self.name.text().strip(),
            profile_verified=False,
            x_limits=AxisLimits(
                minimum=self.x_minimum.value(),
                maximum=self.x_maximum.value(),
            ),
            z_limits=AxisLimits(
                minimum=self.z_minimum.value(),
                maximum=self.z_maximum.value(),
            ),
            rotary_axis=RotaryAxisConfig(
                axis_letter=self.axis_letter.text(),
                direction=direction,
                degrees_per_revolution=self.degrees_per_revolution.value(),
                allow_unbounded_angles=self.allow_unbounded_angles.isChecked(),
                reset_between_operations=self.reset_between_operations.isChecked(),
                max_speed_deg_per_min=self._optional_float(self.max_rotary_speed),
                positioning_precision_deg=self._optional_float(self.positioning_precision),
                drive_system=self._optional_text(self.drive_system),
                motor=self._optional_text(self.motor),
            ),
            max_spindle_rpm=self._optional_int(self.max_spindle_rpm),
            spindle_power_w=self._optional_float(self.spindle_power),
            max_linear_speed_mm_min=self._optional_float(self.max_linear_speed),
            safe_radius=safe_radius,
            max_rotary_stock_length=self._optional_float(self.max_stock_length),
            max_rotary_stock_radius=maximum_stock_radius,
            coordinate_precision=self.coordinate_precision.value(),
            program_header=self._program_lines(self.program_header),
            program_footer=self._program_lines(self.program_footer),
        )

    def accept(self) -> None:
        """Keep the dialog open and show validation failures inline."""

        try:
            self.machine()
        except (TypeError, ValueError) as error:
            self.validation_label.setText(str(error))
            return
        self.validation_label.clear()
        super().accept()


__all__ = ["MachineDialog"]
