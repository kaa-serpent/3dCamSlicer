"""Rotary roughing by descending radial levels."""

from __future__ import annotations

import numpy as np

from rotarycam.geometry.models import RotaryGrid
from rotarycam.planning.operation import MachiningOperation
from rotarycam.strategies._common import build_safe_toolpath, center_radius, contiguous_segments
from rotarycam.strategies.roughing_settings import RoughingSettings
from rotarycam.toolpath.models import ToolpathPoint
from rotarycam.tools.compensation import compute_compensated_tool_grid
from rotarycam.tools.models import Tool


def _validate_roughing_inputs(
    stock: RotaryGrid,
    target: RotaryGrid,
    tool: Tool,
    settings: RoughingSettings,
) -> None:
    if not np.array_equal(stock.x_values, target.x_values) or not np.array_equal(
        stock.angles_deg, target.angles_deg
    ):
        raise ValueError("stock and target must use the same X/A layout")
    common = stock.valid & target.valid
    if not np.any(common):
        raise ValueError("stock and target share no valid cells")
    if np.any(target.radius[common] > stock.radius[common] + settings.tolerance):
        raise ValueError("target radius must not exceed stock radius")
    if settings.stepdown > tool.max_stepdown:
        raise ValueError("roughing stepdown must not exceed tool max_stepdown")
    if settings.stepover > tool.diameter:
        raise ValueError("roughing stepover must not exceed tool diameter")
    if settings.safe_radius <= float(np.max(stock.radius[stock.valid])):
        raise ValueError("safe_radius must be greater than the initial stock radius")


def _radial_levels(maximum: float, minimum: float, stepdown: float) -> list[float]:
    levels: list[float] = []
    current = maximum
    while current - stepdown > minimum:
        current -= stepdown
        levels.append(current)
    if not levels or levels[-1] > minimum:
        levels.append(minimum)
    return levels


def generate_rotary_roughing(
    stock: RotaryGrid,
    target: RotaryGrid,
    tool: Tool,
    settings: RoughingSettings,
) -> MachiningOperation:
    """Generate fixed-X rotary passes at monotonically descending radial levels."""

    _validate_roughing_inputs(stock, target, tool, settings)
    compensated = center_radius(
        target,
        tool,
        settings.allowance,
        compute_compensated_tool_grid,
    )
    if float(np.max(compensated[target.valid])) >= settings.safe_radius:
        raise ValueError("safe_radius must exceed every compensated cutting radius")
    common = stock.valid & target.valid
    maximum = float(np.max(stock.radius[common]))
    minimum = float(np.min(compensated[common]))
    levels = _radial_levels(maximum, minimum, settings.stepdown)
    reached = np.array(stock.radius, copy=True)
    segments: list[list[ToolpathPoint]] = []
    angle_indices = list(range(len(target.angles_deg)))
    if not settings.climb_milling:
        angle_indices.reverse()

    for level in levels:
        for x_index, x_value in enumerate(target.x_values):
            candidates: list[ToolpathPoint | None] = []
            for angle_index in angle_indices:
                cut_radius = max(level, float(compensated[x_index, angle_index]))
                active = (
                    common[x_index, angle_index]
                    and reached[x_index, angle_index] > cut_radius + settings.tolerance
                )
                candidates.append(
                    ToolpathPoint(
                        float(x_value),
                        cut_radius,
                        float(target.angles_deg[angle_index]),
                        tool.feed,
                    )
                    if active
                    else None
                )
                if active:
                    reached[x_index, angle_index] = cut_radius
            segments.extend(contiguous_segments(candidates))
    path = build_safe_toolpath(
        strategy="rotary_roughing",
        tool=tool,
        segments=segments,
        safe_radius=settings.safe_radius,
    )
    return MachiningOperation(
        "Rotary roughing",
        tool,
        "rotary_roughing",
        settings.allowance,
        settings.tolerance,
        [path],
    )
