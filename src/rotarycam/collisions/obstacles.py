"""Signed-distance adapters for measured and voxel collision obstacles."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

from rotarycam.collisions.models import CollisionKind
from rotarycam.machine import Box, Cylinder, FrameKind
from rotarycam.machine.assemblies import MeasuredPrimitive
from rotarycam.motion import MachinePose
from rotarycam.volumetric import SparseVolume

FloatArray = NDArray[np.float64]


class SignedDistanceGeometry(Protocol):
    """Geometry queried in its carrying frame."""

    @property
    def envelope_radius(self) -> float: ...

    def signed_distance(self, points: FloatArray) -> FloatArray: ...


def _points(points: FloatArray) -> FloatArray:
    value = np.asarray(points, dtype=np.float64)
    if value.ndim < 1 or value.shape[-1] != 3:
        raise ValueError("points must have shape (..., 3)")
    if not np.all(np.isfinite(value)):
        raise ValueError("points must contain only finite values")
    return np.array(value, copy=True)


def _box_distance(points: FloatArray, center: FloatArray, half_size: FloatArray) -> FloatArray:
    offset = np.abs(points - center) - half_size
    outside = np.linalg.norm(np.maximum(offset, 0.0), axis=-1)
    inside = np.minimum(np.max(offset, axis=-1), 0.0)
    return np.asarray(outside + inside, dtype=np.float64)


def _basis(axis: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    direction = axis / np.linalg.norm(axis)
    seed = np.asarray((0.0, 0.0, 1.0))
    if abs(float(np.dot(direction, seed))) > 0.9:
        seed = np.asarray((0.0, 1.0, 0.0))
    first = np.cross(direction, seed)
    first /= np.linalg.norm(first)
    second = np.cross(direction, first)
    return direction, first, second


@dataclass(frozen=True, slots=True)
class PrimitiveSDF:
    """Exact SDF adapter for a measured box, cylinder, or frustum."""

    primitive: MeasuredPrimitive

    @property
    def envelope_radius(self) -> float:
        center_radius = float(np.linalg.norm(np.asarray(self.primitive.center)))
        if isinstance(self.primitive, Box):
            local = 0.5 * float(np.linalg.norm(np.asarray(self.primitive.size)))
        elif isinstance(self.primitive, Cylinder):
            local = math.hypot(0.5 * self.primitive.length, self.primitive.radius)
        else:
            local = math.hypot(
                0.5 * self.primitive.length,
                max(self.primitive.radius_start, self.primitive.radius_end),
            )
        return center_radius + local

    def signed_distance(self, points: FloatArray) -> FloatArray:
        value = _points(points)
        primitive = self.primitive
        center = np.asarray(primitive.center, dtype=np.float64)
        if isinstance(primitive, Box):
            return _box_distance(value, center, 0.5 * np.asarray(primitive.size))

        axis, _, _ = _basis(np.asarray(primitive.axis, dtype=np.float64))
        relative = value - center
        axial = np.sum(relative * axis, axis=-1)
        radial_vector = relative - axial[..., None] * axis
        radial = np.linalg.norm(radial_vector, axis=-1)
        half_length = 0.5 * primitive.length
        if isinstance(primitive, Cylinder):
            delta = np.stack((radial - primitive.radius, np.abs(axial) - half_length), axis=-1)
            outside = np.linalg.norm(np.maximum(delta, 0.0), axis=-1)
            inside = np.minimum(np.max(delta, axis=-1), 0.0)
            return np.asarray(outside + inside, dtype=np.float64)

        z = axial + half_length
        slope = (primitive.radius_end - primitive.radius_start) / primitive.length
        expected_radius = primitive.radius_start + slope * z
        side_inside = (expected_radius - radial) / math.sqrt(1.0 + slope * slope)
        lower_inside = z
        upper_inside = primitive.length - z
        depth = np.minimum(np.minimum(side_inside, lower_inside), upper_inside)

        radius_delta = primitive.radius_end - primitive.radius_start
        denominator = radius_delta * radius_delta + primitive.length * primitive.length
        side_t = np.clip(
            ((radial - primitive.radius_start) * radius_delta + z * primitive.length)
            / denominator,
            0.0,
            1.0,
        )
        side_r = primitive.radius_start + side_t * radius_delta
        side_z = side_t * primitive.length
        side_distance = np.hypot(radial - side_r, z - side_z)
        lower_distance = np.hypot(np.maximum(radial - primitive.radius_start, 0.0), z)
        upper_distance = np.hypot(
            np.maximum(radial - primitive.radius_end, 0.0), z - primitive.length
        )
        outside = np.minimum(np.minimum(side_distance, lower_distance), upper_distance)
        return np.asarray(np.where(depth >= 0.0, -depth, outside), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class SparseVolumeSDF:
    """SDF of the union of occupied voxel cells, without mutating the volume."""

    volume: SparseVolume

    @property
    def envelope_radius(self) -> float:
        lower, upper = self.volume.lattice.bounds
        return max(
            float(np.linalg.norm(corner))
            for corner in (
                np.asarray((x, y, z), dtype=np.float64)
                for x in (lower[0], upper[0])
                for y in (lower[1], upper[1])
                for z in (lower[2], upper[2])
            )
        )

    def signed_distance(self, points: FloatArray) -> FloatArray:
        value = _points(points)
        occupied = np.argwhere(self.volume.to_dense())
        if occupied.size == 0:
            return np.full(value.shape[:-1], np.inf, dtype=np.float64)
        lattice = self.volume.lattice
        origins = np.asarray(lattice.origin)
        spacing = np.asarray(lattice.spacing)
        centers = origins + occupied * spacing
        half_size = 0.5 * spacing
        flat = value.reshape((-1, 3))
        result = np.full(flat.shape[0], np.inf, dtype=np.float64)
        for center in centers:
            result = np.minimum(result, _box_distance(flat, center, half_size))
        return result.reshape(value.shape[:-1])


@dataclass(frozen=True, slots=True)
class CollisionObstacle:
    """Typed obstacle plus its machine carrying frame."""

    name: str
    kind: CollisionKind
    geometry: SignedDistanceGeometry
    frame: FrameKind = FrameKind.FIXED
    rotary_pivot_yz: tuple[float, float] = (0.0, 0.0)
    rotary_direction: int = 1

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("obstacle name must not be empty")
        if not isinstance(self.kind, CollisionKind):
            raise TypeError("kind must be a CollisionKind")
        if not isinstance(self.frame, FrameKind):
            raise TypeError("frame must be a FrameKind")
        if (
            len(self.rotary_pivot_yz) != 2
            or not all(math.isfinite(item) for item in self.rotary_pivot_yz)
        ):
            raise ValueError("rotary_pivot_yz must contain two finite values")
        if self.rotary_direction not in (-1, 1):
            raise ValueError("rotary_direction must be -1 or 1")
        if not hasattr(self.geometry, "signed_distance"):
            raise TypeError("geometry must provide signed_distance")

    @property
    def envelope_radius(self) -> float:
        radius = float(self.geometry.envelope_radius)
        if not math.isfinite(radius) or radius < 0.0:
            raise ValueError("geometry envelope_radius must be finite and non-negative")
        return radius

    def _to_local(self, points: FloatArray, pose: MachinePose) -> FloatArray:
        value = _points(points)
        if self.frame is FrameKind.FIXED:
            return value
        if self.frame is FrameKind.SPINDLE:
            return np.asarray(
                value - np.asarray((pose.x, pose.y, pose.z), dtype=np.float64),
                dtype=np.float64,
            )
        pivot_y, pivot_z = self.rotary_pivot_yz
        angle = math.radians(-self.rotary_direction * pose.a)
        cosine = math.cos(angle)
        sine = math.sin(angle)
        result = value.copy()
        y = value[..., 1] - pivot_y
        z = value[..., 2] - pivot_z
        result[..., 1] = pivot_y + cosine * y - sine * z
        result[..., 2] = pivot_z + sine * y + cosine * z
        return result

    def signed_distance(self, points: FloatArray, pose: MachinePose) -> FloatArray:
        return np.asarray(
            self.geometry.signed_distance(self._to_local(points, pose)), dtype=np.float64
        )


def primitive_obstacle(
    primitive: MeasuredPrimitive,
    *,
    name: str | None = None,
    kind: CollisionKind | None = None,
    rotary_pivot_yz: tuple[float, float] = (0.0, 0.0),
    rotary_direction: int = 1,
) -> CollisionObstacle:
    """Adapt one measured machine/fixture primitive."""

    inferred_kind = (
        CollisionKind.FIXTURE
        if primitive.role.value in {"fixture", "support"}
        else CollisionKind.MACHINE
    )
    return CollisionObstacle(
        name or primitive.role.value,
        kind or inferred_kind,
        PrimitiveSDF(primitive),
        primitive.frame,
        rotary_pivot_yz,
        rotary_direction,
    )


def stock_obstacle(
    volume: SparseVolume,
    *,
    name: str = "stock",
    frame: FrameKind = FrameKind.ROTARY,
    rotary_pivot_yz: tuple[float, float] = (0.0, 0.0),
    rotary_direction: int = 1,
) -> CollisionObstacle:
    """Adapt occupied stock cells as a rotary-frame obstacle."""

    return CollisionObstacle(
        name,
        CollisionKind.STOCK,
        cast(SignedDistanceGeometry, SparseVolumeSDF(volume)),
        frame,
        rotary_pivot_yz,
        rotary_direction,
    )


__all__ = [
    "CollisionObstacle",
    "PrimitiveSDF",
    "SignedDistanceGeometry",
    "SparseVolumeSDF",
    "primitive_obstacle",
    "stock_obstacle",
]
