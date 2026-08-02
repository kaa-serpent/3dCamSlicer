"""Circular finishing with continuous seam angles."""

from __future__ import annotations

import numpy.typing as npt

from rotarycam.geometry.models import RotaryGrid
from rotarycam.planning.operation import MachiningOperation
from rotarycam.strategies._common import (
    Compensator,
    build_safe_toolpath,
    center_radius,
    contiguous_segments,
    effective_mask,
)
from rotarycam.strategies.settings import FinishingSettings
from rotarycam.toolpath.models import ToolpathPoint
from rotarycam.tools.models import Tool


def generate_circular_finishing(
    target: RotaryGrid,
    tool: Tool,
    settings: FinishingSettings,
    mask: npt.ArrayLike | None = None,
    *,
    compensator: Compensator | None = None,
) -> MachiningOperation:
    """Generate fixed-X full-circle passes without a 359° to 0° reversal."""
    radii = center_radius(target, tool, settings.allowance, compensator)
    active = effective_mask(target, mask)
    segments: list[list[ToolpathPoint]] = []
    angle_count = len(target.angles_deg)
    for x_index, x_value in enumerate(target.x_values):
        reverse = settings.bidirectional and x_index % 2 == 1
        indices = list(range(angle_count - 1, -1, -1) if reverse else range(angle_count))
        candidates: list[ToolpathPoint | None] = []
        if reverse and bool(active[x_index].all()):
            candidates.append(
                ToolpathPoint(
                    float(x_value),
                    float(radii[x_index, 0]),
                    float(target.angles_deg[0] + 360.0),
                    tool.feed,
                )
            )
        for angle_index in indices:
            angle = float(target.angles_deg[angle_index])
            candidates.append(
                ToolpathPoint(
                    float(x_value),
                    float(radii[x_index, angle_index]),
                    angle,
                    tool.feed,
                )
                if active[x_index, angle_index]
                else None
            )
        if bool(active[x_index].all()) and not reverse:
            first_index = indices[0]
            candidates.append(
                ToolpathPoint(
                    float(x_value),
                    float(radii[x_index, first_index]),
                    float(target.angles_deg[first_index] + 360.0),
                    tool.feed,
                )
            )
        segments.extend(contiguous_segments(candidates))
    path = build_safe_toolpath(
        strategy="circular",
        tool=tool,
        segments=segments,
        safe_radius=settings.safe_radius,
    )
    return MachiningOperation(
        "Circular finishing",
        tool,
        "circular",
        settings.allowance,
        settings.tolerance,
        [path],
    )
