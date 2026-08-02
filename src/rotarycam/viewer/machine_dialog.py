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

from rotarycam.config import (
    AxisLimits,
    MachineDefinition,
    RotaryAxisConfig,
    XYZAConfiguration,
    identity_transform,
)
from rotarycam.machine import AxisDynamics, MachineAssembly, MachineCapabilities


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
        self._source_machine = machine

        self.name = QLineEdit(machine.name if machine is not None else "New machine", self)
        self.x_minimum = self._number(-100_000.0, 100_000.0, 0.0)
        self.x_maximum = self._number(-100_000.0, 100_000.0, 200.0)
        self.z_minimum = self._number(-100_000.0, 100_000.0, 0.0)
        self.z_maximum = self._number(-100_000.0, 100_000.0, 100.0)
        self.xyza_measured = QCheckBox("Y travel and XYZA setup were measured", self)
        self.y_minimum = self._number(-100_000.0, 100_000.0, -100.0)
        self.y_maximum = self._number(-100_000.0, 100_000.0, 100.0)
        self.rotary_pivot_y = self._number(-100_000.0, 100_000.0, 0.0)
        self.rotary_pivot_z = self._number(-100_000.0, 100_000.0, 0.0)
        self.rotary_zero = self._number(-1_000_000.0, 1_000_000.0, 0.0)
        self.spindle_axis_x = self._number(-1.0, 1.0, 0.0, decimals=6)
        self.spindle_axis_y = self._number(-1.0, 1.0, 0.0, decimals=6)
        self.spindle_axis_z = self._number(-1.0, 1.0, -1.0, decimals=6)
        self.g54_x = self._number(-100_000.0, 100_000.0, 0.0)
        self.g54_y = self._number(-100_000.0, 100_000.0, 0.0)
        self.g54_z = self._number(-100_000.0, 100_000.0, 0.0)
        self.xyza_measured.toggled.connect(self._update_xyza_fields)

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

        self.dynamics_measured = QCheckBox("X/Y/Z/A dynamics were measured", self)
        self.axis_dynamics: dict[str, tuple[QDoubleSpinBox, QDoubleSpinBox]] = {}
        for axis in "XYZA":
            velocity = self._positive_number(1_200.0, maximum=10_000_000.0)
            acceleration = self._positive_number(100.0, maximum=10_000_000.0)
            self.axis_dynamics[axis] = (velocity, acceleration)
        self.dynamics_measured.toggled.connect(self._update_dynamics_fields)
        self.simultaneous_xyza = QCheckBox(
            "Controller support for simultaneous X/Y/Z/A is verified", self
        )
        self.inverse_time_g93 = QCheckBox(
            "Controller support for inverse-time G93 is verified", self
        )
        self.assembly_json = QPlainTextEdit(self)
        self.assembly_json.setPlaceholderText(
            '{"primitives":[{"primitive_type":"box","role":"chuck",'
            '"frame":"rotary","center":[0,0,0],"size":[1,1,1]}]}'
        )
        self.assembly_json.setMaximumHeight(150)

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
        self._update_xyza_fields()
        self._update_dynamics_fields()

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
            ("Measured XYZA setup", self.xyza_measured),
            ("Y minimum (mm)", self.y_minimum),
            ("Y maximum (mm)", self.y_maximum),
            ("Rotary pivot Y in G54 (mm)", self.rotary_pivot_y),
            ("Rotary pivot Z in G54 (mm)", self.rotary_pivot_z),
            ("Mechanical A zero (deg)", self.rotary_zero),
            ("Spindle axis X", self.spindle_axis_x),
            ("Spindle axis Y", self.spindle_axis_y),
            ("Spindle axis Z", self.spindle_axis_z),
            ("G54 origin X (mm)", self.g54_x),
            ("G54 origin Y (mm)", self.g54_y),
            ("G54 origin Z (mm)", self.g54_z),
            ("Rotary axis letter", self.axis_letter),
            ("Rotary direction", self.rotary_direction),
            ("Degrees per revolution", self.degrees_per_revolution),
            ("Rotary winding", self.allow_unbounded_angles),
            ("Rotary reset policy", self.reset_between_operations),
            ("Maximum rotary speed (deg/min)", self.max_rotary_speed),
            ("A positioning precision (deg)", self.positioning_precision),
            ("Rotary drive system", self.drive_system),
            ("Rotary motor", self.motor),
            ("Measured axis dynamics", self.dynamics_measured),
            ("X maximum velocity (mm/min)", self.axis_dynamics["X"][0]),
            ("X maximum acceleration (mm/s2)", self.axis_dynamics["X"][1]),
            ("Y maximum velocity (mm/min)", self.axis_dynamics["Y"][0]),
            ("Y maximum acceleration (mm/s2)", self.axis_dynamics["Y"][1]),
            ("Z maximum velocity (mm/min)", self.axis_dynamics["Z"][0]),
            ("Z maximum acceleration (mm/s2)", self.axis_dynamics["Z"][1]),
            ("A maximum velocity (deg/min)", self.axis_dynamics["A"][0]),
            ("A maximum acceleration (deg/s2)", self.axis_dynamics["A"][1]),
            ("Simultaneous XYZA capability", self.simultaneous_xyza),
            ("Inverse-time G93 capability", self.inverse_time_g93),
            ("Measured assembly primitives (JSON)", self.assembly_json),
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
        if machine.y_limits is not None and machine.xyza_configuration is not None:
            self.xyza_measured.setChecked(True)
            self.y_minimum.setValue(machine.y_limits.minimum)
            self.y_maximum.setValue(machine.y_limits.maximum)
            configuration = machine.xyza_configuration
            self.rotary_pivot_y.setValue(configuration.rotary_pivot_y)
            self.rotary_pivot_z.setValue(configuration.rotary_pivot_z)
            self.rotary_zero.setValue(configuration.rotary_zero_deg)
            self.spindle_axis_x.setValue(configuration.spindle_axis[0])
            self.spindle_axis_y.setValue(configuration.spindle_axis[1])
            self.spindle_axis_z.setValue(configuration.spindle_axis[2])
            self.g54_x.setValue(configuration.g54_origin[0])
            self.g54_y.setValue(configuration.g54_origin[1])
            self.g54_z.setValue(configuration.g54_origin[2])
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
        if machine.dynamics is not None and all(axis in machine.dynamics for axis in "XYZA"):
            self.dynamics_measured.setChecked(True)
            for axis, (velocity, acceleration) in self.axis_dynamics.items():
                velocity.setValue(machine.dynamics[axis].max_velocity)
                acceleration.setValue(machine.dynamics[axis].max_acceleration)
        if machine.capabilities is not None:
            self.simultaneous_xyza.setChecked(machine.capabilities.simultaneous_xyza)
            self.inverse_time_g93.setChecked(
                machine.capabilities.inverse_time_feed_g93
            )
        if machine.assembly is not None:
            self.assembly_json.setPlainText(machine.assembly.model_dump_json(indent=2))
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

    def _update_xyza_fields(self, _checked: bool = False) -> None:
        enabled = self.xyza_measured.isChecked()
        for editor in (
            self.y_minimum,
            self.y_maximum,
            self.rotary_pivot_y,
            self.rotary_pivot_z,
            self.rotary_zero,
            self.spindle_axis_x,
            self.spindle_axis_y,
            self.spindle_axis_z,
            self.g54_x,
            self.g54_y,
            self.g54_z,
        ):
            editor.setEnabled(enabled)

    def _update_dynamics_fields(self, _checked: bool = False) -> None:
        enabled = self.dynamics_measured.isChecked()
        for velocity, acceleration in self.axis_dynamics.values():
            velocity.setEnabled(enabled)
            acceleration.setEnabled(enabled)

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
        source_configuration = (
            None if self._source_machine is None else self._source_machine.xyza_configuration
        )
        xyza_configuration = None
        y_limits = None
        if self.xyza_measured.isChecked():
            y_limits = AxisLimits(
                minimum=self.y_minimum.value(), maximum=self.y_maximum.value()
            )
            xyza_configuration = XYZAConfiguration(
                rotary_pivot_y=self.rotary_pivot_y.value(),
                rotary_pivot_z=self.rotary_pivot_z.value(),
                rotary_zero_deg=self.rotary_zero.value(),
                spindle_axis=(
                    self.spindle_axis_x.value(),
                    self.spindle_axis_y.value(),
                    self.spindle_axis_z.value(),
                ),
                g54_origin=(
                    self.g54_x.value(), self.g54_y.value(), self.g54_z.value()
                ),
                setup_transform=(
                    source_configuration.setup_transform
                    if source_configuration is not None
                    else identity_transform()
                ),
            )
        dynamics = None
        if self.dynamics_measured.isChecked():
            dynamics = {
                axis: AxisDynamics(
                    max_velocity=velocity.value(),
                    max_acceleration=acceleration.value(),
                )
                for axis, (velocity, acceleration) in self.axis_dynamics.items()
            }
        capabilities = None
        if (
            (self._source_machine is not None and self._source_machine.capabilities is not None)
            or self.simultaneous_xyza.isChecked()
            or self.inverse_time_g93.isChecked()
        ):
            capabilities = MachineCapabilities(
                simultaneous_xyza=self.simultaneous_xyza.isChecked(),
                inverse_time_feed_g93=self.inverse_time_g93.isChecked(),
            )
        assembly_text = self.assembly_json.toPlainText().strip()
        try:
            assembly = (
                MachineAssembly.model_validate_json(assembly_text)
                if assembly_text
                else None
            )
        except ValueError as error:
            raise ValueError(f"Invalid measured assembly JSON: {error}") from error
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
            y_limits=y_limits,
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
            xyza_configuration=xyza_configuration,
            machine_assembly_configured=assembly is not None,
            dynamics=dynamics,
            capabilities=capabilities,
            assembly=assembly,
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
