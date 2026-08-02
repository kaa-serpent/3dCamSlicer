"""Conservative lower-surface kernels for radial cutter compensation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

from rotarycam.tools.models import Tool, ToolType

FloatArray = npt.NDArray[np.float64]


class CutterKernel(Protocol):
    """Axisymmetric cutter surface measured outward from its TCP tip."""

    @property
    def footprint_radius(self) -> float: ...

    def height_at(self, lateral_distance: npt.ArrayLike) -> FloatArray:
        """Return lower-surface height, or infinity outside the footprint."""

        ...


@dataclass(frozen=True, slots=True)
class FlatCutterKernel:
    """Flat end mill: a zero-height disk at the commanded TCP radius."""

    radius: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.radius) or self.radius <= 0.0:
            raise ValueError("kernel radius must be finite and positive")

    @property
    def footprint_radius(self) -> float:
        return self.radius

    def height_at(self, lateral_distance: npt.ArrayLike) -> FloatArray:
        distance = np.asarray(lateral_distance, dtype=np.float64)
        if np.any(~np.isfinite(distance)) or np.any(distance < 0.0):
            raise ValueError("lateral distances must be finite and non-negative")
        return np.where(distance <= self.radius, 0.0, np.inf).astype(np.float64)


@dataclass(frozen=True, slots=True)
class BallCutterKernel:
    """Spherical lower half of a ball end mill, referenced at its tip."""

    radius: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.radius) or self.radius <= 0.0:
            raise ValueError("kernel radius must be finite and positive")

    @property
    def footprint_radius(self) -> float:
        return self.radius

    def height_at(self, lateral_distance: npt.ArrayLike) -> FloatArray:
        distance = np.asarray(lateral_distance, dtype=np.float64)
        if np.any(~np.isfinite(distance)) or np.any(distance < 0.0):
            raise ValueError("lateral distances must be finite and non-negative")
        inside = distance <= self.radius
        height = np.full(distance.shape, np.inf, dtype=np.float64)
        squared = np.maximum(0.0, self.radius * self.radius - distance[inside] ** 2)
        height[inside] = self.radius - np.sqrt(squared)
        return height


@dataclass(frozen=True, slots=True)
class TaperedCutterKernel:
    """Conical frustum expanding from a flat narrow tip over a fixed length."""

    tip_radius: float
    maximum_radius: float
    taper_length: float

    def __post_init__(self) -> None:
        values = np.asarray((self.tip_radius, self.maximum_radius, self.taper_length))
        if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("tapered kernel dimensions must be finite and positive")
        if self.tip_radius >= self.maximum_radius:
            raise ValueError("tip radius must be smaller than maximum radius")

    @property
    def footprint_radius(self) -> float:
        return self.maximum_radius

    def height_at(self, lateral_distance: npt.ArrayLike) -> FloatArray:
        distance = np.asarray(lateral_distance, dtype=np.float64)
        if np.any(~np.isfinite(distance)) or np.any(distance < 0.0):
            raise ValueError("lateral distances must be finite and non-negative")
        inside = distance <= self.maximum_radius
        height = np.full(distance.shape, np.inf, dtype=np.float64)
        rise = self.taper_length * np.maximum(0.0, distance - self.tip_radius)
        rise /= self.maximum_radius - self.tip_radius
        height[inside] = rise[inside]
        return height


@dataclass(frozen=True, slots=True)
class CutterKernelSamples:
    """Immutable Cartesian sampling of a cutter's circular footprint."""

    dx: FloatArray
    surface_offset: FloatArray
    height: FloatArray

    def __post_init__(self) -> None:
        dx = np.array(self.dx, dtype=np.float64, copy=True)
        surface_offset = np.array(self.surface_offset, dtype=np.float64, copy=True)
        height = np.array(self.height, dtype=np.float64, copy=True)
        if dx.ndim != 1 or dx.shape != surface_offset.shape or dx.shape != height.shape:
            raise ValueError("kernel sample arrays must be one-dimensional and equally shaped")
        if dx.size == 0 or np.any(~np.isfinite(dx)) or np.any(~np.isfinite(surface_offset)):
            raise ValueError("kernel offsets must be non-empty and finite")
        if np.any(~np.isfinite(height)) or np.any(height < 0.0):
            raise ValueError("kernel heights must be finite and non-negative")
        for values in (dx, surface_offset, height):
            values.setflags(write=False)
        object.__setattr__(self, "dx", dx)
        object.__setattr__(self, "surface_offset", surface_offset)
        object.__setattr__(self, "height", height)


def kernel_for_tool(tool: Tool) -> CutterKernel:
    """Build the lower-surface kernel selected by a validated tool definition."""

    radius = tool.diameter / 2.0
    if tool.tool_type is ToolType.FLAT:
        return FlatCutterKernel(radius)
    if tool.tool_type is ToolType.BALL:
        return BallCutterKernel(radius)
    if tool.tool_type is ToolType.TAPERED:
        assert tool.tip_diameter is not None and tool.taper_length is not None
        return TaperedCutterKernel(tool.tip_diameter / 2.0, radius, tool.taper_length)
    raise ValueError(f"Unsupported cutter type: {tool.tool_type}")


def discretize_kernel(
    kernel: CutterKernel,
    x_spacing: float,
    surface_spacing: float,
) -> CutterKernelSamples:
    """Sample a kernel footprint, always including its center and boundary.

    Cell-center heights are biased downward by half a cell diagonal before use.
    This makes the sampled lower envelope conservative: compensation can only move
    the TCP farther away from the target, never closer.
    """

    spacings = np.asarray((x_spacing, surface_spacing), dtype=np.float64)
    if np.any(~np.isfinite(spacings)) or np.any(spacings <= 0.0):
        raise ValueError("kernel spacings must be finite and positive")
    radius = kernel.footprint_radius
    x_axis = np.unique(
        np.append(
            np.arange(-radius, radius + x_spacing, x_spacing),
            (0.0, radius, -radius),
        )
    )
    s_axis = np.unique(
        np.append(
            np.arange(-radius, radius + surface_spacing, surface_spacing),
            (0.0, radius, -radius),
        )
    )
    dx_grid, surface_grid = np.meshgrid(x_axis, s_axis, indexing="ij")
    distance = np.hypot(dx_grid, surface_grid)
    inside = distance <= radius + 1e-12
    dx = dx_grid[inside]
    surface = surface_grid[inside]
    conservative_distance = np.maximum(
        0.0,
        distance[inside] - 0.5 * float(np.hypot(x_spacing, surface_spacing)),
    )
    height = kernel.height_at(conservative_distance)
    return CutterKernelSamples(dx, surface, height)
