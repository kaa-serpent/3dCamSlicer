"""Application-wide, serializable configuration models.

The CAM algorithms consume these models but do not contain machine-specific
defaults.  All distances are millimetres and all angles are degrees.
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from rotarycam.machine.assemblies import (
    AxisDynamics,
    MachineAssembly,
    MachineCapabilities,
)

PositiveFloat = Annotated[float, Field(gt=0.0, allow_inf_nan=False)]
NonNegativeFloat = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]
type MatrixRow = tuple[float, float, float, float]
type Matrix4x4 = tuple[MatrixRow, MatrixRow, MatrixRow, MatrixRow]


def identity_transform() -> Matrix4x4:
    """Return an immutable affine identity transform."""

    return (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


class RadialSamplingMode(StrEnum):
    """Policy used when converting a mesh to the single-radius CAM grid."""

    STRICT = "strict"
    OUTER_ENVELOPE = "outer_envelope"


class FinishingStrategy(StrEnum):
    """Direction used for the primary surface-finishing operation."""

    HELICAL = "helical"
    LONGITUDINAL = "longitudinal"


class StrictConfigModel(BaseModel):
    """Base class shared by persisted configuration records."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class AxisLimits(StrictConfigModel):
    """Inclusive travel limits for one linear machine axis."""

    minimum: float
    maximum: float

    @model_validator(mode="after")
    def validate_range(self) -> AxisLimits:
        if not math.isfinite(self.minimum) or not math.isfinite(self.maximum):
            raise ValueError("axis limits must be finite")
        if self.maximum <= self.minimum:
            raise ValueError("axis maximum must be greater than minimum")
        return self


class RotaryAxisConfig(StrictConfigModel):
    """Mapping, winding policy, and optional physical rotary-axis limits."""

    axis_letter: str = "A"
    direction: Literal[-1, 1] = 1
    degrees_per_revolution: PositiveFloat = 360.0
    allow_unbounded_angles: bool = True
    reset_between_operations: bool = False
    max_speed_deg_per_min: PositiveFloat | None = None
    positioning_precision_deg: PositiveFloat | None = None
    drive_system: str | None = None
    motor: str | None = None

    @field_validator("axis_letter")
    @classmethod
    def validate_axis_letter(cls, value: str) -> str:
        normalized = value.strip().upper()
        if len(normalized) != 1 or not normalized.isascii() or not normalized.isalpha():
            raise ValueError("axis_letter must be one ASCII letter")
        return normalized

    @field_validator("drive_system", "motor")
    @classmethod
    def validate_optional_description(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("rotary hardware descriptions must not be empty")
        return normalized


class XYZAConfiguration(StrictConfigModel):
    """Measured work-coordinate inputs needed by the future XYZA mapper.

    The record is optional on ``MachineDefinition`` so migration can represent
    unknown measurements without manufacturing defaults.
    """

    rotary_pivot_y: float
    rotary_pivot_z: float
    rotary_zero_deg: float
    spindle_axis: tuple[float, float, float]
    g54_origin: tuple[float, float, float]
    setup_transform: Matrix4x4 = Field(default_factory=identity_transform)

    @field_validator(
        "rotary_pivot_y",
        "rotary_pivot_z",
        "rotary_zero_deg",
    )
    @classmethod
    def validate_finite_scalar(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("XYZA coordinates and angles must be finite")
        return value

    @field_validator("spindle_axis", "g54_origin")
    @classmethod
    def validate_vector(cls, value: tuple[float, float, float]) -> tuple[float, float, float]:
        if not all(math.isfinite(component) for component in value):
            raise ValueError("XYZA vectors must be finite")
        return value

    @field_validator("setup_transform")
    @classmethod
    def validate_setup_transform(cls, value: Matrix4x4) -> Matrix4x4:
        if not all(math.isfinite(component) for row in value for component in row):
            raise ValueError("setup_transform values must be finite")
        if value[3] != (0.0, 0.0, 0.0, 1.0):
            raise ValueError("setup_transform must be an affine 4x4 matrix")
        rotation = tuple(
            tuple(value[row][column] for row in range(3)) for column in range(3)
        )
        for index, column in enumerate(rotation):
            magnitude = math.sqrt(sum(component * component for component in column))
            if not math.isclose(magnitude, 1.0, rel_tol=0.0, abs_tol=1e-9):
                raise ValueError("setup_transform rotation must be orthonormal")
            for other in rotation[index + 1 :]:
                dot_product = sum(
                    left * right for left, right in zip(column, other, strict=True)
                )
                if not math.isclose(dot_product, 0.0, rel_tol=0.0, abs_tol=1e-9):
                    raise ValueError("setup_transform rotation must be orthonormal")
        determinant = (
            value[0][0] * (value[1][1] * value[2][2] - value[1][2] * value[2][1])
            - value[0][1]
            * (value[1][0] * value[2][2] - value[1][2] * value[2][0])
            + value[0][2]
            * (value[1][0] * value[2][1] - value[1][1] * value[2][0])
        )
        if not math.isclose(determinant, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("setup_transform rotation must be right-handed")
        return value

    @model_validator(mode="after")
    def validate_spindle_axis(self) -> XYZAConfiguration:
        if math.sqrt(sum(component * component for component in self.spindle_axis)) <= 1e-12:
            raise ValueError("spindle_axis must be non-zero")
        return self


class MachineObservationMetadata(StrictConfigModel):
    """Informational observations that are not machine-coordinate contracts.

    These fields preserve controller and display readings without treating them
    as travel limits, G54 coordinates, rotary-pivot measurements, positioning
    accuracy, or controller-capability evidence.  Planning and export validation
    must use the dedicated :class:`MachineDefinition` fields instead.
    """

    controller_firmware: str | None = None
    home_display_position_mm: tuple[float, float, float] | None = None
    rotary_mount_display_xy_mm: tuple[float, float] | None = None
    coordinate_display_decimals: int | None = Field(default=None, ge=0, le=9)
    unresolved_rotary_direction_report: str | None = None

    @field_validator("controller_firmware", "unresolved_rotary_direction_report")
    @classmethod
    def validate_observation_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("machine observation text must not be empty")
        if "\n" in normalized or "\r" in normalized:
            raise ValueError("machine observation text must be one line")
        return normalized

    @field_validator("home_display_position_mm")
    @classmethod
    def validate_home_display_position(
        cls, value: tuple[float, float, float] | None
    ) -> tuple[float, float, float] | None:
        if value is not None and not all(math.isfinite(component) for component in value):
            raise ValueError("observed Home display coordinates must be finite")
        return value

    @field_validator("rotary_mount_display_xy_mm")
    @classmethod
    def validate_rotary_mount_display_position(
        cls, value: tuple[float, float] | None
    ) -> tuple[float, float] | None:
        if value is not None and not all(math.isfinite(component) for component in value):
            raise ValueError("observed rotary mount display coordinates must be finite")
        return value


class MachineDefinition(StrictConfigModel):
    """Minimum machine profile required by planning and validation."""

    name: str = "Makera rotary"
    profile_verified: bool = False
    x_limits: AxisLimits
    y_limits: AxisLimits | None = None
    z_limits: AxisLimits
    rotary_axis: RotaryAxisConfig = Field(default_factory=RotaryAxisConfig)
    xyza_configuration: XYZAConfiguration | None = None
    machine_assembly_configured: bool = False
    dynamics: dict[str, AxisDynamics] | None = None
    capabilities: MachineCapabilities | None = None
    assembly: MachineAssembly | None = None
    observations: MachineObservationMetadata | None = None
    max_spindle_rpm: int | None = Field(default=None, gt=0)
    spindle_power_w: PositiveFloat | None = None
    max_linear_speed_mm_min: PositiveFloat | None = None
    safe_radius: float | None = Field(default=None, gt=0.0)
    max_rotary_stock_length: PositiveFloat | None = None
    max_rotary_stock_radius: PositiveFloat | None = None
    coordinate_precision: int = Field(default=3, ge=0, le=6)
    program_header: tuple[str, ...] = ()
    program_footer: tuple[str, ...] = ()

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("machine name must not be empty")
        return value.strip()

    @field_validator("safe_radius")
    @classmethod
    def validate_safe_radius(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("safe_radius must be finite")
        return value

    @field_validator("program_header", "program_footer")
    @classmethod
    def validate_program_lines(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any("\n" in line or "\r" in line for line in value):
            raise ValueError("program header/footer entries must each be one line")
        return value

    @field_validator("dynamics")
    @classmethod
    def validate_dynamics(
        cls, value: dict[str, AxisDynamics] | None
    ) -> dict[str, AxisDynamics] | None:
        if value is None:
            return None
        normalized = {axis.strip().upper(): limits for axis, limits in value.items()}
        if any(axis not in {"X", "Y", "Z", "A"} for axis in normalized):
            raise ValueError("dynamics axes must be X, Y, Z, or A")
        if len(normalized) != len(value):
            raise ValueError("dynamics axes must be unique ignoring case")
        return normalized


class MachiningSettings(StrictConfigModel):
    """Project-level sampling and safety settings."""

    x_step: PositiveFloat = 0.25
    angle_step_deg: PositiveFloat = 1.0
    radial_sampling_mode: RadialSamplingMode = RadialSamplingMode.STRICT
    finishing_strategy: FinishingStrategy = FinishingStrategy.HELICAL
    roughing_allowance: NonNegativeFloat = 0.5
    final_tolerance: PositiveFloat = 0.05
    safe_clearance: PositiveFloat = 5.0

    @field_validator("angle_step_deg")
    @classmethod
    def validate_angle_step(cls, value: float) -> float:
        if value > 360.0:
            raise ValueError("angle_step_deg must not exceed 360 degrees")
        return value
