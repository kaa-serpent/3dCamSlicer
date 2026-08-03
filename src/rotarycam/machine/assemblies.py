"""Measured machine envelopes and axis capabilities.

These records are persistence-safe descriptions, not collision algorithms.
Every primitive is attached to a frame so later validators can place fixed,
rotary and spindle-carried geometry with the appropriate transform.
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PositiveFloat = Annotated[float, Field(gt=0.0, allow_inf_nan=False)]
Vector3 = tuple[float, float, float]


class _AssemblyRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FrameKind(StrEnum):
    """Machine frame carrying a measured envelope."""

    FIXED = "fixed"
    ROTARY = "rotary"
    SPINDLE = "spindle"


class AssemblyRole(StrEnum):
    """Safety-significant roles required for a complete machine assembly."""

    CHUCK = "chuck"
    JAWS = "jaws"
    TAILSTOCK = "tailstock"
    PLATTER = "platter"
    SPINDLE = "spindle"
    FIXTURE = "fixture"
    SUPPORT = "support"


REQUIRED_MACHINE_ROLES = frozenset(
    {
        AssemblyRole.CHUCK,
        AssemblyRole.JAWS,
        AssemblyRole.TAILSTOCK,
        AssemblyRole.PLATTER,
        AssemblyRole.SPINDLE,
        AssemblyRole.SUPPORT,
    }
)


class AxisDynamics(_AssemblyRecord):
    """Measured velocity and acceleration bounds for one machine axis.

    Values use mm/min and mm/s² for linear axes, degrees/min and degrees/s²
    for A.  The axis name provides the unit context at the configuration edge.
    """

    max_velocity: PositiveFloat
    max_acceleration: PositiveFloat


class MachineCapabilities(_AssemblyRecord):
    """Controller features that require explicit verification."""

    simultaneous_xyza: bool = False
    inverse_time_feed_g93: bool = False


def _finite_vector(value: Vector3, *, name: str) -> Vector3:
    if not all(math.isfinite(component) for component in value):
        raise ValueError(f"{name} must contain only finite values")
    return value


def _unit_vector(value: Vector3) -> Vector3:
    _finite_vector(value, name="axis")
    magnitude = math.sqrt(sum(component * component for component in value))
    if magnitude <= 1e-12:
        raise ValueError("axis must be non-zero")
    return tuple(component / magnitude for component in value)  # type: ignore[return-value]


class _MeasuredPrimitive(_AssemblyRecord):
    role: AssemblyRole
    frame: FrameKind
    center: Vector3

    @field_validator("center")
    @classmethod
    def validate_center(cls, value: Vector3) -> Vector3:
        return _finite_vector(value, name="center")


class Box(_MeasuredPrimitive):
    """Axis-aligned measured box in its carrying frame."""

    primitive_type: Literal["box"] = "box"
    size: Vector3

    @field_validator("size")
    @classmethod
    def validate_size(cls, value: Vector3) -> Vector3:
        _finite_vector(value, name="size")
        if any(component <= 0.0 for component in value):
            raise ValueError("box size components must be greater than zero")
        return value


class Cylinder(_MeasuredPrimitive):
    """Measured finite cylinder centred on ``center``."""

    primitive_type: Literal["cylinder"] = "cylinder"
    radius: PositiveFloat
    length: PositiveFloat
    axis: Vector3 = (1.0, 0.0, 0.0)

    @field_validator("axis")
    @classmethod
    def normalize_axis(cls, value: Vector3) -> Vector3:
        return _unit_vector(value)


class Frustum(_MeasuredPrimitive):
    """Measured conical frustum centred along ``axis``."""

    primitive_type: Literal["frustum"] = "frustum"
    radius_start: PositiveFloat
    radius_end: PositiveFloat
    length: PositiveFloat
    axis: Vector3 = (1.0, 0.0, 0.0)

    @field_validator("axis")
    @classmethod
    def normalize_axis(cls, value: Vector3) -> Vector3:
        return _unit_vector(value)


type MeasuredPrimitive = Annotated[
    Box | Cylinder | Frustum,
    Field(discriminator="primitive_type"),
]


class MachineAssembly(_AssemblyRecord):
    """Collection of measured collision envelopes for a machine setup."""

    primitives: tuple[MeasuredPrimitive, ...] = ()

    @model_validator(mode="after")
    def require_unique_spindle(self) -> MachineAssembly:
        spindle_count = sum(item.role is AssemblyRole.SPINDLE for item in self.primitives)
        if spindle_count > 1:
            raise ValueError("machine assembly must contain at most one spindle envelope")
        return self

    @property
    def represented_roles(self) -> frozenset[AssemblyRole]:
        return frozenset(primitive.role for primitive in self.primitives)

    @property
    def missing_roles(self) -> frozenset[AssemblyRole]:
        """Return every safety role absent from the measured assembly."""

        return REQUIRED_MACHINE_ROLES - self.represented_roles

    @property
    def is_complete(self) -> bool:
        """Whether all mandatory machine envelopes have been measured."""

        return not self.missing_roles


__all__ = [
    "REQUIRED_MACHINE_ROLES",
    "AssemblyRole",
    "AxisDynamics",
    "Box",
    "Cylinder",
    "FrameKind",
    "Frustum",
    "MachineAssembly",
    "MachineCapabilities",
    "MeasuredPrimitive",
]
