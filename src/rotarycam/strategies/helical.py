"""Continuous helical finishing over a cylindrical radius field."""

from __future__ import annotations

from math import ceil

import numpy as np
import numpy.typing as npt

from rotarycam.geometry.interpolation import interpolate_radius
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


def _nearest_mask_value(grid: RotaryGrid, mask: npt.NDArray[np.bool_], x: float, a: float) -> bool:
    x_index = int(np.argmin(np.abs(grid.x_values - x)))
    angular_delta = np.abs((grid.angles_deg - (a % 360.0) + 180.0) % 360.0 - 180.0)
    angle_index = int(np.argmin(angular_delta))
    return bool(mask[x_index, angle_index])


def generate_helical_finishing(
    target: RotaryGrid,
    tool: Tool,
    settings: FinishingSettings,
    mask: npt.ArrayLike | None = None,
    *,
    compensator: Compensator | None = None,
) -> MachiningOperation:
    """Generate a monotonic X/A helix and retract across masked interruptions."""
    compensated = target.with_radius(
        center_radius(target, tool, settings.allowance, compensator)
    )
    active = effective_mask(target, mask)
    length = float(target.x_values[-1] - target.x_values[0])
    revolutions = max(1, ceil(length / settings.stepover))
    angular_step = (
        float(np.min(np.diff(target.angles_deg))) if len(target.angles_deg) > 1 else 360.0
    )
    sample_count = max(2, revolutions * ceil(360.0 / angular_step) + 1)
    angles = np.linspace(0.0, revolutions * 360.0, sample_count)
    x_values = np.linspace(target.x_values[0], target.x_values[-1], sample_count)
    candidates: list[ToolpathPoint | None] = []
    for x_value, angle in zip(x_values, angles, strict=True):
        x = float(x_value)
        a = float(angle)
        if not _nearest_mask_value(target, active, x, a):
            candidates.append(None)
            continue
        try:
            radius = interpolate_radius(compensated, x, a)
        except ValueError:
            candidates.append(None)
            continue
        candidates.append(ToolpathPoint(x, radius, a, tool.feed))
    path = build_safe_toolpath(
        strategy="helical",
        tool=tool,
        segments=contiguous_segments(candidates),
        safe_radius=settings.safe_radius,
    )
    return MachiningOperation(
        "Helical finishing",
        tool,
        "helical",
        settings.allowance,
        settings.tolerance,
        [path],
    )
