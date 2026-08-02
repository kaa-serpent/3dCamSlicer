"""Analytical cutter, flute, shank, and holder envelopes.

The tool TCP is at ``(0, 0, 0)`` and the tool extends along positive Z toward
the spindle.  Only ``cutter`` removes material; every solid participates in
collision checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from rotarycam.tools.models import Tool, ToolType

FloatArray = NDArray[np.float64]


def _points(points: FloatArray) -> FloatArray:
    result = np.asarray(points, dtype=np.float64)
    if result.ndim < 1 or result.shape[-1] != 3:
        raise ValueError("points must have shape (..., 3)")
    if not np.all(np.isfinite(result)):
        raise ValueError("points must contain only finite values")
    return np.array(result, copy=True)


class AnalyticalSolid(Protocol):
    def signed_distance(self, points: FloatArray) -> FloatArray: ...

    def contains_points(
        self, points: FloatArray, *, tolerance: float = 0.0
    ) -> NDArray[np.bool_]: ...


@dataclass(frozen=True, slots=True)
class AxialCylinder:
    radius: float
    z_min: float
    z_max: float

    def __post_init__(self) -> None:
        if not np.isfinite((self.radius, self.z_min, self.z_max)).all():
            raise ValueError("cylinder dimensions must be finite")
        if self.radius <= 0.0 or self.z_max <= self.z_min:
            raise ValueError("cylinder must have positive radius and length")

    def signed_distance(self, points: FloatArray) -> FloatArray:
        value = _points(points)
        radial = np.hypot(value[..., 0], value[..., 1])
        half_length = 0.5 * (self.z_max - self.z_min)
        center = 0.5 * (self.z_max + self.z_min)
        radial_delta = radial - self.radius
        axial_delta = np.abs(value[..., 2] - center) - half_length
        outside = np.hypot(np.maximum(radial_delta, 0.0), np.maximum(axial_delta, 0.0))
        inside = np.minimum(np.maximum(radial_delta, axial_delta), 0.0)
        return np.asarray(outside + inside, dtype=np.float64)

    def contains_points(
        self, points: FloatArray, *, tolerance: float = 0.0
    ) -> NDArray[np.bool_]:
        return np.asarray(self.signed_distance(points) <= tolerance, dtype=np.bool_)


@dataclass(frozen=True, slots=True)
class AxialFrustum:
    radius_start: float
    radius_end: float
    z_min: float
    z_max: float

    def __post_init__(self) -> None:
        values = (self.radius_start, self.radius_end, self.z_min, self.z_max)
        if not np.isfinite(values).all():
            raise ValueError("frustum dimensions must be finite")
        if min(self.radius_start, self.radius_end) <= 0.0 or self.z_max <= self.z_min:
            raise ValueError("frustum must have positive radii and length")

    def _radius_at(self, z: FloatArray) -> FloatArray:
        fraction = (z - self.z_min) / (self.z_max - self.z_min)
        return np.asarray(
            self.radius_start + fraction * (self.radius_end - self.radius_start),
            dtype=np.float64,
        )

    def signed_distance(self, points: FloatArray) -> FloatArray:
        value = _points(points)
        radial = np.hypot(value[..., 0], value[..., 1])
        z = value[..., 2]
        length = self.z_max - self.z_min
        slope = (self.radius_end - self.radius_start) / length
        side_inside = (self._radius_at(z) - radial) / np.sqrt(1.0 + slope * slope)
        lower_inside = z - self.z_min
        upper_inside = self.z_max - z
        inside_depth = np.minimum(np.minimum(side_inside, lower_inside), upper_inside)

        side_t = np.clip(
            ((radial - self.radius_start) * (self.radius_end - self.radius_start)
             + (z - self.z_min) * length)
            / ((self.radius_end - self.radius_start) ** 2 + length**2),
            0.0,
            1.0,
        )
        side_r = self.radius_start + side_t * (self.radius_end - self.radius_start)
        side_z = self.z_min + side_t * length
        side_distance = np.hypot(radial - side_r, z - side_z)
        lower_distance = np.hypot(
            np.maximum(radial - self.radius_start, 0.0), z - self.z_min
        )
        upper_distance = np.hypot(
            np.maximum(radial - self.radius_end, 0.0), z - self.z_max
        )
        outside_distance = np.minimum(np.minimum(side_distance, lower_distance), upper_distance)
        inside = inside_depth >= 0.0
        return np.asarray(np.where(inside, -inside_depth, outside_distance), dtype=np.float64)

    def contains_points(
        self, points: FloatArray, *, tolerance: float = 0.0
    ) -> NDArray[np.bool_]:
        return np.asarray(self.signed_distance(points) <= tolerance, dtype=np.bool_)


@dataclass(frozen=True, slots=True)
class BallEndSolid:
    radius: float
    cutting_length: float

    def __post_init__(self) -> None:
        if not np.isfinite((self.radius, self.cutting_length)).all():
            raise ValueError("ball cutter dimensions must be finite")
        if self.radius <= 0.0 or self.cutting_length <= 0.0:
            raise ValueError("ball cutter dimensions must be greater than zero")

    def signed_distance(self, points: FloatArray) -> FloatArray:
        value = _points(points)
        sphere = np.linalg.norm(value - np.asarray((0.0, 0.0, self.radius)), axis=-1) - self.radius
        lower_half = np.maximum(sphere, value[..., 2] - self.radius)
        if self.cutting_length <= self.radius:
            return np.asarray(np.maximum(lower_half, value[..., 2] - self.cutting_length))
        cylinder = AxialCylinder(self.radius, self.radius, self.cutting_length)
        return np.asarray(np.minimum(lower_half, cylinder.signed_distance(value)))

    def contains_points(
        self, points: FloatArray, *, tolerance: float = 0.0
    ) -> NDArray[np.bool_]:
        return np.asarray(self.signed_distance(points) <= tolerance, dtype=np.bool_)


@dataclass(frozen=True, slots=True)
class CompositeSolid:
    solids: tuple[AnalyticalSolid, ...]

    def signed_distance(self, points: FloatArray) -> FloatArray:
        value = _points(points)
        if not self.solids:
            return np.full(value.shape[:-1], np.inf, dtype=np.float64)
        distances = tuple(solid.signed_distance(value) for solid in self.solids)
        return np.asarray(np.minimum.reduce(distances), dtype=np.float64)

    def contains_points(
        self, points: FloatArray, *, tolerance: float = 0.0
    ) -> NDArray[np.bool_]:
        return np.asarray(self.signed_distance(points) <= tolerance, dtype=np.bool_)


@dataclass(frozen=True, slots=True)
class ToolAssembly:
    """Complete analytical envelope built from a v2 :class:`Tool`."""

    tool: Tool
    cutter: AnalyticalSolid
    flute: AnalyticalSolid | None
    shank: AnalyticalSolid | None
    holder: AnalyticalSolid | None

    @classmethod
    def from_tool(cls, tool: Tool) -> ToolAssembly:
        radius = 0.5 * tool.diameter
        if tool.tool_type is ToolType.FLAT:
            cutter: AnalyticalSolid = AxialCylinder(radius, 0.0, tool.cutting_length)
        elif tool.tool_type is ToolType.BALL:
            cutter = BallEndSolid(radius, tool.cutting_length)
        else:
            if tool.tip_diameter is None or tool.taper_length is None:
                raise ValueError("tapered tool requires tip_diameter and taper_length")
            tapered = AxialFrustum(
                0.5 * tool.tip_diameter,
                radius,
                0.0,
                tool.taper_length,
            )
            cutter_sections: list[AnalyticalSolid] = [tapered]
            if tool.taper_length < tool.cutting_length:
                cutter_sections.append(
                    AxialCylinder(radius, tool.taper_length, tool.cutting_length)
                )
            cutter = CompositeSolid(tuple(cutter_sections))

        flute = (
            AxialCylinder(radius, tool.cutting_length, tool.flute_length)
            if tool.flute_length > tool.cutting_length
            else None
        )
        exposed_length = tool.stickout if tool.stickout is not None else tool.overall_length
        shank_start = tool.flute_length
        shank = (
            AxialCylinder(0.5 * tool.shank_diameter, shank_start, exposed_length)
            if exposed_length > shank_start
            else None
        )
        holder = (
            AxialCylinder(
                0.5 * tool.holder.diameter,
                exposed_length,
                exposed_length + tool.holder.length,
            )
            if tool.holder is not None and tool.stickout is not None
            else None
        )
        return cls(tool=tool, cutter=cutter, flute=flute, shank=shank, holder=holder)

    @property
    def is_complete(self) -> bool:
        """Whether stickout and holder were explicitly measured."""

        return self.tool.stickout is not None and self.tool.holder is not None

    @property
    def solids(self) -> tuple[AnalyticalSolid, ...]:
        return (
            self.cutter,
            *(solid for solid in (self.flute, self.shank, self.holder) if solid is not None),
        )

    def signed_distance(self, points: FloatArray) -> FloatArray:
        """Signed distance to the union of cutter, flute, shank and holder."""

        return CompositeSolid(self.solids).signed_distance(points)

    def contains_points(
        self, points: FloatArray, *, tolerance: float = 0.0
    ) -> NDArray[np.bool_]:
        return CompositeSolid(self.solids).contains_points(points, tolerance=tolerance)

    def cutter_signed_distance(self, points: FloatArray) -> FloatArray:
        """Signed distance to the material-removing solid only."""

        return self.cutter.signed_distance(points)

    def cutter_contains_points(
        self, points: FloatArray, *, tolerance: float = 0.0
    ) -> NDArray[np.bool_]:
        return self.cutter.contains_points(points, tolerance=tolerance)


__all__ = [
    "AnalyticalSolid",
    "AxialCylinder",
    "AxialFrustum",
    "BallEndSolid",
    "CompositeSolid",
    "ToolAssembly",
]
