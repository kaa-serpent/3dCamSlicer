"""Deterministic XYZA kinematics around the machine X rotary axis.

The mapper is deliberately independent from planning and controller commands.
Points enter in the part frame and leave in G54 coordinates.  The configured
setup transform locates the part at A0, then the commanded (possibly unwrapped)
angle is applied around the measured rotary pivot.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians, sin

import numpy as np
from numpy.typing import NDArray

from rotarycam.config import MachineDefinition, XYZAConfiguration
from rotarycam.motion.models import MachinePose

FloatArray = NDArray[np.float64]


def _validated_points(points: FloatArray) -> FloatArray:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim < 1 or array.shape[-1] != 3:
        raise ValueError("points must have shape (..., 3)")
    if not np.all(np.isfinite(array)):
        raise ValueError("points must contain only finite values")
    return np.array(array, dtype=np.float64, copy=True)


def _rotation_x(angle_deg: float) -> FloatArray:
    angle = radians(angle_deg)
    cosine = cos(angle)
    sine = sin(angle)
    return np.asarray(
        (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, cosine, -sine, 0.0),
            (0.0, sine, cosine, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )


def _translation(x: float, y: float, z: float) -> FloatArray:
    result = np.eye(4, dtype=np.float64)
    result[:3, 3] = (x, y, z)
    return result


@dataclass(frozen=True, slots=True)
class XYZAKinematics:
    """Forward and inverse part/G54 transformations for one machine profile.

    ``A`` remains unwrapped: 0, 360 and 720 degrees are distinct motion
    positions even though their Cartesian transforms are equivalent.
    """

    configuration: XYZAConfiguration
    rotary_direction: int = 1

    def __post_init__(self) -> None:
        if self.rotary_direction not in (-1, 1):
            raise ValueError("rotary_direction must be -1 or 1")

    @classmethod
    def from_machine(cls, machine: MachineDefinition) -> XYZAKinematics:
        """Build the mapper from the direction and measurements in a profile."""

        if machine.xyza_configuration is None:
            raise ValueError("machine profile has no XYZA configuration")
        return cls(machine.xyza_configuration, machine.rotary_axis.direction)

    def matrix_at(self, a_deg: float) -> FloatArray:
        """Return ``T_rotary_zero * Rx(direction*A) * T_setup``.

        The rotary-zero term is expressed as a rotation about the measured
        ``(pivot_y, pivot_z)`` in G54.  ``setup_transform`` is a full affine
        part-to-G54 transform at the mechanical zero.
        """

        if not np.isfinite(a_deg):
            raise ValueError("a_deg must be finite")
        pivot = _translation(
            0.0,
            self.configuration.rotary_pivot_y,
            self.configuration.rotary_pivot_z,
        )
        unpivot = _translation(
            0.0,
            -self.configuration.rotary_pivot_y,
            -self.configuration.rotary_pivot_z,
        )
        angle = self.configuration.rotary_zero_deg + self.rotary_direction * a_deg
        setup = np.asarray(self.configuration.setup_transform, dtype=np.float64)
        return pivot @ _rotation_x(angle) @ unpivot @ setup

    def inverse_matrix_at(self, a_deg: float) -> FloatArray:
        """Return the exact inverse of :meth:`matrix_at`."""

        return np.asarray(np.linalg.inv(self.matrix_at(a_deg)), dtype=np.float64)

    def part_to_g54(self, points: FloatArray, a_deg: float) -> FloatArray:
        """Transform one point or an array of part-frame points into G54."""

        return _apply_transform(_validated_points(points), self.matrix_at(a_deg))

    def g54_to_part(self, points: FloatArray, a_deg: float) -> FloatArray:
        """Transform one point or an array of G54 points into the part frame."""

        return _apply_transform(_validated_points(points), self.inverse_matrix_at(a_deg))

    def machine_pose_for_part_point(
        self, point: FloatArray, a_deg: float
    ) -> MachinePose:
        """Map one part-frame point to a distinctly typed G54 machine pose."""

        validated = _validated_points(point)
        if validated.shape != (3,):
            raise ValueError("point must have shape (3,)")
        transformed = self.part_to_g54(validated, a_deg)
        return MachinePose(
            x=float(transformed[0]),
            y=float(transformed[1]),
            z=float(transformed[2]),
            a=a_deg,
        )

    def part_point_for_machine_pose(self, pose: MachinePose) -> FloatArray:
        """Recover a part-frame point without labelling it as a machine pose."""

        return self.g54_to_part(np.asarray((pose.x, pose.y, pose.z)), pose.a)


def _apply_transform(points: FloatArray, transform: FloatArray) -> FloatArray:
    flat = points.reshape((-1, 3))
    homogeneous = np.concatenate(
        (flat, np.ones((flat.shape[0], 1), dtype=np.float64)), axis=1
    )
    result = (transform @ homogeneous.T).T[:, :3]
    return result.reshape(points.shape)


def transform_points(
    points: FloatArray,
    a_deg: float,
    configuration: XYZAConfiguration,
    *,
    rotary_direction: int = 1,
) -> FloatArray:
    """Functional forward-transform convenience wrapper."""

    return XYZAKinematics(configuration, rotary_direction).part_to_g54(points, a_deg)


def inverse_transform_points(
    points: FloatArray,
    a_deg: float,
    configuration: XYZAConfiguration,
    *,
    rotary_direction: int = 1,
) -> FloatArray:
    """Functional inverse-transform convenience wrapper."""

    return XYZAKinematics(configuration, rotary_direction).g54_to_part(points, a_deg)


__all__ = ["XYZAKinematics", "inverse_transform_points", "transform_points"]
