"""Indexed longitudinal finishing with safe links."""

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


def generate_longitudinal_finishing(
    target: RotaryGrid,
    tool: Tool,
    settings: FinishingSettings,
    mask: npt.ArrayLike | None = None,
    *,
    compensator: Compensator | None = None,
) -> MachiningOperation:
    """Generate fixed-A passes along X, alternating direction when requested."""
    radii = center_radius(target, tool, settings.allowance, compensator)
    active = effective_mask(target, mask)
    segments: list[list[ToolpathPoint]] = []
    for angle_index, angle in enumerate(target.angles_deg):
        indices = list(range(len(target.x_values)))
        if settings.bidirectional and angle_index % 2:
            indices.reverse()
        candidates = [
            ToolpathPoint(
                float(target.x_values[index]),
                float(radii[index, angle_index]),
                float(angle),
                tool.feed,
            )
            if active[index, angle_index]
            else None
            for index in indices
        ]
        segments.extend(contiguous_segments(candidates))
    path = build_safe_toolpath(
        strategy="longitudinal",
        tool=tool,
        segments=segments,
        safe_radius=settings.safe_radius,
    )
    return MachiningOperation(
        "Longitudinal finishing",
        tool,
        "longitudinal",
        settings.allowance,
        settings.tolerance,
        [path],
    )
