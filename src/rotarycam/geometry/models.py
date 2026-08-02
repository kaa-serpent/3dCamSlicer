"""Validated numerical models shared by the geometry engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


class UndercutStatus(StrEnum):
    """State of radial-undercut evaluation for a mesh or grid."""

    NOT_EVALUATED = "not_evaluated"
    ABSENT = "absent"
    PRESENT = "present"
    INDETERMINATE = "indeterminate"


@dataclass(frozen=True, slots=True)
class RotaryGrid:
    """A cylindrical radius field indexed as ``(X, A)``.

    Angles are expressed in degrees in the canonical half-open interval
    ``[0, 360)``. Invalid cells have a radius of zero.
    """

    x_values: FloatArray
    angles_deg: FloatArray
    radius: FloatArray
    valid: BoolArray
    undercut_status: UndercutStatus = UndercutStatus.NOT_EVALUATED
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        x_values = np.array(self.x_values, dtype=np.float64, copy=True)
        angles_deg = np.array(self.angles_deg, dtype=np.float64, copy=True)
        radius = np.array(self.radius, dtype=np.float64, copy=True)
        valid = np.array(self.valid, dtype=np.bool_, copy=True)

        if x_values.ndim != 1 or x_values.size == 0:
            raise ValueError("x_values must be a non-empty one-dimensional array")
        if angles_deg.ndim != 1 or angles_deg.size == 0:
            raise ValueError("angles_deg must be a non-empty one-dimensional array")
        if not np.all(np.isfinite(x_values)) or not np.all(np.diff(x_values) > 0.0):
            raise ValueError("x_values must be finite and strictly increasing")
        if not np.all(np.isfinite(angles_deg)):
            raise ValueError("angles_deg must be finite")
        if np.any(angles_deg < 0.0) or np.any(angles_deg >= 360.0):
            raise ValueError("angles_deg must lie in [0, 360)")
        if not np.all(np.diff(angles_deg) > 0.0):
            raise ValueError("angles_deg must be unique and strictly increasing")

        expected_shape = (x_values.size, angles_deg.size)
        if radius.shape != expected_shape or valid.shape != expected_shape:
            raise ValueError(f"radius and valid must have shape {expected_shape}")
        if np.any(~np.isfinite(radius)) or np.any(radius < 0.0):
            raise ValueError("radius values must be finite and non-negative")
        if np.any(radius[~valid] != 0.0):
            raise ValueError("invalid cells must have a zero radius")

        for array in (x_values, angles_deg, radius, valid):
            array.setflags(write=False)
        object.__setattr__(self, "x_values", x_values)
        object.__setattr__(self, "angles_deg", angles_deg)
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "warnings", tuple(self.warnings))

    @property
    def shape(self) -> tuple[int, int]:
        """Return the grid shape in canonical ``(X, A)`` order."""

        return self.radius.shape

    def with_radius(
        self,
        radius: npt.ArrayLike,
        *,
        valid: npt.ArrayLike | None = None,
    ) -> RotaryGrid:
        """Return an independent grid carrying replacement radius values."""

        return RotaryGrid(
            self.x_values,
            self.angles_deg,
            np.asarray(radius, dtype=np.float64),
            self.valid if valid is None else np.asarray(valid, dtype=np.bool_),
            self.undercut_status,
            self.warnings,
        )
