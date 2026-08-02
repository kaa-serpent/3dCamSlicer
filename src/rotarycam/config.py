"""Application-wide, serializable configuration models.

The CAM algorithms consume these models but do not contain machine-specific
defaults.  All distances are millimetres and all angles are degrees.
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PositiveFloat = Annotated[float, Field(gt=0.0, allow_inf_nan=False)]
NonNegativeFloat = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]


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


class MachineDefinition(StrictConfigModel):
    """Minimum machine profile required by planning and validation."""

    name: str = "Makera rotary"
    profile_verified: bool = False
    x_limits: AxisLimits
    z_limits: AxisLimits
    rotary_axis: RotaryAxisConfig = Field(default_factory=RotaryAxisConfig)
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
