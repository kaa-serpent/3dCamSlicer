"""Conservative discrete stock simulation on a cylindrical radius grid."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from math import ceil

import numpy as np

from rotarycam.geometry.models import RotaryGrid
from rotarycam.simulation.cutter_kernel import cutter_kernel
from rotarycam.simulation.models import SimulationResult
from rotarycam.simulation.residual import compute_residual
from rotarycam.toolpath.models import Toolpath, ToolpathPoint
from rotarycam.tools.models import Tool


def _minimum_positive_spacing(values: np.ndarray) -> float | None:
    if values.size < 2:
        return None
    spacing = np.diff(values)
    positive = spacing[spacing > 0.0]
    return float(np.min(positive)) if positive.size else None


def _sampling_step(stock: RotaryGrid, tool: Tool) -> float:
    candidates = [tool.diameter / 4.0]
    x_spacing = _minimum_positive_spacing(np.asarray(stock.x_values))
    if x_spacing is not None:
        candidates.append(x_spacing / 2.0)
    if stock.angles_deg.size > 1:
        cyclic = np.diff(np.r_[stock.angles_deg, stock.angles_deg[0] + 360.0])
        angle_spacing = float(np.min(cyclic[cyclic > 0.0]))
        max_radius = float(np.max(stock.radius))
        if max_radius > 0.0:
            candidates.append(np.deg2rad(angle_spacing) * max_radius / 2.0)
    return max(min(candidates), 1e-6)


def _cutting_samples(
    points: Sequence[ToolpathPoint],
    stock: RotaryGrid,
    tool: Tool,
) -> Iterator[ToolpathPoint]:
    if not points:
        return

    step = _sampling_step(stock, tool)
    previous: ToolpathPoint | None = None
    reference_radius = float(np.max(stock.radius))
    for point in points:
        if point.rapid:
            previous = point
            continue
        if previous is None:
            yield point
            previous = point
            continue

        dx = point.x - previous.x
        dz = point.z - previous.z
        da_arc = np.deg2rad(point.a - previous.a) * reference_radius
        distance = float(np.sqrt(dx * dx + dz * dz + da_arc * da_arc))
        sample_count = max(1, ceil(distance / step))
        for index in range(1, sample_count + 1):
            fraction = index / sample_count
            yield ToolpathPoint(
                x=previous.x + dx * fraction,
                z=previous.z + dz * fraction,
                a=previous.a + (point.a - previous.a) * fraction,
                feed=point.feed,
                rapid=False,
            )
        previous = point


def _cell_widths(values: np.ndarray) -> np.ndarray:
    if values.size == 1:
        return np.zeros(1, dtype=np.float64)
    boundaries = np.empty(values.size + 1, dtype=np.float64)
    boundaries[1:-1] = (values[:-1] + values[1:]) / 2.0
    boundaries[0] = values[0]
    boundaries[-1] = values[-1]
    return np.diff(boundaries)


def _angular_widths_rad(angles_deg: np.ndarray) -> np.ndarray:
    if angles_deg.size == 1:
        return np.asarray([2.0 * np.pi], dtype=np.float64)
    previous_gap = (angles_deg - np.roll(angles_deg, 1)) % 360.0
    next_gap = (np.roll(angles_deg, -1) - angles_deg) % 360.0
    return np.asarray(
        np.deg2rad((previous_gap + next_gap) / 2.0),
        dtype=np.float64,
    )


def removed_volume(old_stock: RotaryGrid, new_stock: RotaryGrid) -> float:
    """Approximate removed annular-sector volume in cubic millimetres."""

    if old_stock.shape != new_stock.shape:
        raise ValueError("old and new stock grids must have the same shape")
    radial_area = np.maximum(old_stock.radius**2 - new_stock.radius**2, 0.0) / 2.0
    x_width = _cell_widths(np.asarray(old_stock.x_values))[:, None]
    angle_width = _angular_widths_rad(np.asarray(old_stock.angles_deg))[None, :]
    valid = np.asarray(old_stock.valid) & np.asarray(new_stock.valid)
    return float(np.sum(radial_area * x_width * angle_width, where=valid))


def simulate_toolpath(
    stock: RotaryGrid,
    target: RotaryGrid,
    toolpath: Toolpath,
    tool: Tool,
    *,
    tolerance: float = 0.0,
) -> SimulationResult:
    """Apply a discrete cutter sweep without crossing the effective target."""

    baseline = compute_residual(stock, target, tolerance=tolerance)
    if np.any(np.asarray(target.radius) > np.asarray(stock.radius) + 1e-9):
        raise ValueError("effective target must not exceed current stock")
    if toolpath.tool_number != tool.number:
        raise ValueError("toolpath tool number does not match the cutter")

    usable = np.asarray(stock.valid) & np.asarray(target.valid)
    new_radius = np.asarray(stock.radius).copy()
    kernel_grid = stock
    for point in _cutting_samples(toolpath.points, stock, tool):
        swept_radius = cutter_kernel(kernel_grid, point, tool)
        cut_radius = np.maximum(swept_radius, np.asarray(target.radius))
        new_radius[usable] = np.minimum(new_radius[usable], cut_radius[usable])
        kernel_grid = stock.with_radius(new_radius)

    new_stock = stock.with_radius(new_radius)
    residual = compute_residual(new_stock, target, tolerance=tolerance)
    # This assertion documents and defends the simulator's central invariant.
    if np.any(residual.error > baseline.error + 1e-9):
        raise RuntimeError("simulation increased residual stock unexpectedly")
    return SimulationResult(
        stock=new_stock,
        removed_volume=removed_volume(stock, new_stock),
        max_remaining_error=residual.max_error,
        mean_remaining_error=residual.mean_error,
        remaining_mask=residual.mask,
    )
