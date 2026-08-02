"""Material-retaining support definitions."""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Annotated, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SupportType(StrEnum):
    """Supported keep-out footprint families."""

    RECTANGLE = "rectangle"
    CYLINDER = "cylinder"


class _SupportBase(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
    )

    id: UUID = Field(default_factory=uuid4)
    x: float
    angle_deg: float
    thickness: float = Field(gt=0.0)
    transition: float = Field(ge=0.0)
    enabled: bool = True

    @field_validator("x", "angle_deg", "thickness", "transition")
    @classmethod
    def validate_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("support dimensions and coordinates must be finite")
        return value

    @field_validator("angle_deg")
    @classmethod
    def normalize_angle(cls, value: float) -> float:
        return value % 360.0


class RectangularSupport(_SupportBase):
    """Rectangular keep-out expressed on the unwrapped X/A surface."""

    support_type: Literal[SupportType.RECTANGLE] = Field(
        default=SupportType.RECTANGLE,
        alias="type",
    )
    length_x: float = Field(gt=0.0)
    width_surface: float = Field(gt=0.0)

    @field_validator("length_x", "width_surface")
    @classmethod
    def validate_dimensions(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("support dimensions must be finite")
        return value


class CylindricalSupport(_SupportBase):
    """Circular keep-out expressed on the unwrapped X/A surface."""

    support_type: Literal[SupportType.CYLINDER] = Field(
        default=SupportType.CYLINDER,
        alias="type",
    )
    diameter: float = Field(gt=0.0)

    @field_validator("diameter")
    @classmethod
    def validate_diameter(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("support diameter must be finite")
        return value


type Support = Annotated[
    RectangularSupport | CylindricalSupport,
    Field(discriminator="support_type"),
]
