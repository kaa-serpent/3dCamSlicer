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


class RetentionType(StrEnum):
    """Volumetric-engine retention definitions migrated from radial supports."""

    SURFACE_RECTANGLE = "surface_rectangle"
    SURFACE_CYLINDER = "surface_cylinder"


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


class _SurfaceRetentionBase(BaseModel):
    """A target-relative retained volume anchored on the stock surface.

    This representation preserves v1 support intent without inventing a stock
    radius or a Cartesian fixture measurement during migration.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    id: UUID = Field(default_factory=uuid4)
    x: float
    angle_deg: float
    radial_thickness: float = Field(gt=0.0)
    transition: float = Field(ge=0.0)
    enabled: bool = True

    @field_validator("x", "angle_deg", "radial_thickness", "transition")
    @classmethod
    def validate_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("retention dimensions and coordinates must be finite")
        return value

    @field_validator("angle_deg")
    @classmethod
    def normalize_angle(cls, value: float) -> float:
        return value % 360.0


class RectangularSurfaceRetention(_SurfaceRetentionBase):
    """Target-relative retained prism with a rectangular surface footprint."""

    retention_type: Literal[RetentionType.SURFACE_RECTANGLE] = Field(
        default=RetentionType.SURFACE_RECTANGLE,
        alias="type",
    )
    length_x: float = Field(gt=0.0)
    width_surface: float = Field(gt=0.0)

    @field_validator("length_x", "width_surface")
    @classmethod
    def validate_dimensions(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("retention dimensions must be finite")
        return value


class CylindricalSurfaceRetention(_SurfaceRetentionBase):
    """Target-relative retained cylinder with a circular surface footprint."""

    retention_type: Literal[RetentionType.SURFACE_CYLINDER] = Field(
        default=RetentionType.SURFACE_CYLINDER,
        alias="type",
    )
    diameter: float = Field(gt=0.0)

    @field_validator("diameter")
    @classmethod
    def validate_diameter(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("retention diameter must be finite")
        return value


type RetentionVolume = Annotated[
    RectangularSurfaceRetention | CylindricalSurfaceRetention,
    Field(discriminator="retention_type"),
]


def migrate_support_to_retention(support: Support) -> RetentionVolume:
    """Convert one v1 radial support without changing its protected footprint."""

    if isinstance(support, RectangularSupport):
        return RectangularSurfaceRetention(
            id=support.id,
            x=support.x,
            angle_deg=support.angle_deg,
            radial_thickness=support.thickness,
            transition=support.transition,
            enabled=support.enabled,
            length_x=support.length_x,
            width_surface=support.width_surface,
        )
    if isinstance(support, CylindricalSupport):
        return CylindricalSurfaceRetention(
            id=support.id,
            x=support.x,
            angle_deg=support.angle_deg,
            radial_thickness=support.thickness,
            transition=support.transition,
            enabled=support.enabled,
            diameter=support.diameter,
        )
    raise TypeError(f"unsupported support type: {type(support).__name__}")
