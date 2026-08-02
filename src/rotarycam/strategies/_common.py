"""Internal shared helpers for finishing strategies."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt

from rotarycam.geometry.models import RotaryGrid
from rotarycam.toolpath.models import Toolpath, ToolpathPoint
from rotarycam.toolpath.safety_moves import connect_cutting_segments
from rotarycam.tools.models import Tool

LOGGER = logging.getLogger(__name__)
Compensator = Callable[[RotaryGrid, Tool], RotaryGrid | npt.ArrayLike]


def effective_mask(grid: RotaryGrid, mask: npt.ArrayLike | None) -> npt.NDArray[np.bool_]:
    if mask is None:
        return np.array(grid.valid, copy=True)
    candidate = np.asarray(mask, dtype=np.bool_)
    if candidate.shape != grid.shape:
        raise ValueError(f"mask must have shape {grid.shape}")
    return candidate & grid.valid


def center_radius(
    grid: RotaryGrid,
    tool: Tool,
    allowance: float,
    compensator: Compensator | None,
) -> npt.NDArray[np.float64]:
    """Resolve tool-center radii through an injected kernel or explicit fallback.

    The fallback adds the cutter radius in the machine radial direction. It is a
    deterministic preview contract, exact only where the surface normal is radial.
    Production callers should inject the cutter-kernel compensator.
    """
    if compensator is None:
        try:
            from rotarycam.tools import compute_compensated_tool_grid
        except ImportError:
            LOGGER.warning(
                "No cutter compensator available; using radial preview compensation for tool %s",
                tool.number,
            )
            resolved: RotaryGrid | npt.ArrayLike = grid.radius + tool.diameter / 2.0
        else:
            resolved = compute_compensated_tool_grid(grid, tool)
    else:
        resolved = compensator(grid, tool)
    result = resolved.radius if isinstance(resolved, RotaryGrid) else np.asarray(resolved)
    result = np.asarray(result, dtype=np.float64) + allowance
    if result.shape != grid.shape:
        raise ValueError(f"compensated radius must have shape {grid.shape}")
    if not np.all(np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError("compensated radius must be finite and non-negative")
    return np.where(grid.valid, result, 0.0)


def contiguous_segments(points: Sequence[ToolpathPoint | None]) -> list[list[ToolpathPoint]]:
    segments: list[list[ToolpathPoint]] = []
    current: list[ToolpathPoint] = []
    for point in points:
        if point is None:
            if current:
                segments.append(current)
                current = []
        else:
            current.append(point)
    if current:
        segments.append(current)
    return segments


def build_safe_toolpath(
    *,
    strategy: str,
    tool: Tool,
    segments: Sequence[Sequence[ToolpathPoint]],
    safe_radius: float,
) -> Toolpath:
    points = connect_cutting_segments(
        segments,
        safe_radius=safe_radius,
        plunge_feed=tool.plunge_feed,
    )
    if not points:
        raise ValueError("mask leaves no machinable finishing segment")
    return Toolpath(tool.number, strategy, points)
