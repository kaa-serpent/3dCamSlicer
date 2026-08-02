"""Safe indexed roughing for an initial rectangular stock."""

from __future__ import annotations

import numpy as np

from rotarycam.geometry.models import RotaryGrid
from rotarycam.planning.operation import MachiningOperation
from rotarycam.strategies._common import build_safe_toolpath, center_radius, contiguous_segments
from rotarycam.strategies.rotary_roughing import _radial_levels, _validate_roughing_inputs
from rotarycam.strategies.roughing_settings import RoughingSettings
from rotarycam.toolpath.models import ToolpathPoint
from rotarycam.tools.compensation import compute_compensated_tool_grid
from rotarycam.tools.models import Tool

INDEXED_ORIENTATIONS_DEG = (0.0, 90.0, 180.0, 270.0)


def _nearest_angle_index(grid: RotaryGrid, angle_deg: float) -> int:
    distance = np.abs((grid.angles_deg - angle_deg + 180.0) % 360.0 - 180.0)
    return int(np.argmin(distance))


def generate_indexed_roughing(
    stock: RotaryGrid,
    target: RotaryGrid,
    tool: Tool,
    settings: RoughingSettings,
) -> MachiningOperation:
    """Generate four fixed-A roughing groups, retracting before every rotation."""

    _validate_roughing_inputs(stock, target, tool, settings)
    compensated = center_radius(
        target,
        tool,
        settings.allowance,
        compute_compensated_tool_grid,
    )
    if float(np.max(compensated[target.valid])) >= settings.safe_radius:
        raise ValueError("safe_radius must exceed every compensated cutting radius")
    reached = np.array(stock.radius, copy=True)
    segments: list[list[ToolpathPoint]] = []
    x_indices = list(range(len(target.x_values)))

    for orientation_index, orientation in enumerate(INDEXED_ORIENTATIONS_DEG):
        angle_index = _nearest_angle_index(target, orientation)
        common = stock.valid[:, angle_index] & target.valid[:, angle_index]
        if not np.any(common):
            continue
        maximum = float(np.max(stock.radius[common, angle_index]))
        minimum = float(np.min(compensated[common, angle_index]))
        levels = _radial_levels(maximum, minimum, settings.stepdown)
        ordered_x = x_indices if orientation_index % 2 == 0 else list(reversed(x_indices))
        for level in levels:
            candidates: list[ToolpathPoint | None] = []
            for x_index in ordered_x:
                cut_radius = max(level, float(compensated[x_index, angle_index]))
                active = (
                    common[x_index]
                    and reached[x_index, angle_index] > cut_radius + settings.tolerance
                )
                candidates.append(
                    ToolpathPoint(
                        float(target.x_values[x_index]),
                        cut_radius,
                        orientation,
                        tool.feed,
                    )
                    if active
                    else None
                )
                if active:
                    reached[x_index, angle_index] = cut_radius
            segments.extend(contiguous_segments(candidates))

    path = build_safe_toolpath(
        strategy="indexed_roughing",
        tool=tool,
        segments=segments,
        safe_radius=settings.safe_radius,
    )
    return MachiningOperation(
        "Indexed rectangular-stock roughing",
        tool,
        "indexed_roughing",
        settings.allowance,
        settings.tolerance,
        [path],
    )
