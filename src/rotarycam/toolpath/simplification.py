"""Tolerance-bound simplification for dense toolpaths."""

from __future__ import annotations

from math import isfinite

import numpy as np

from rotarycam.toolpath.models import Toolpath, ToolpathPoint


def _can_remove(
    previous: ToolpathPoint,
    current: ToolpathPoint,
    following: ToolpathPoint,
    linear_tolerance: float,
    angular_tolerance_deg: float,
) -> bool:
    if previous.rapid != current.rapid or current.rapid != following.rapid:
        return False
    if current.feed != following.feed:
        return False
    span = np.array([following.x - previous.x, following.z - previous.z], dtype=np.float64)
    offset = np.array([current.x - previous.x, current.z - previous.z], dtype=np.float64)
    denominator = float(np.dot(span, span))
    fraction = (
        0.0
        if denominator == 0.0
        else float(np.clip(np.dot(offset, span) / denominator, 0, 1))
    )
    projected = np.array([previous.x, previous.z]) + fraction * span
    linear_error = float(np.linalg.norm(np.array([current.x, current.z]) - projected))
    expected_a = previous.a + fraction * (following.a - previous.a)
    return linear_error <= linear_tolerance and abs(current.a - expected_a) <= angular_tolerance_deg


def simplify_toolpath(
    toolpath: Toolpath,
    *,
    linear_tolerance: float,
    angular_tolerance_deg: float,
) -> Toolpath:
    """Remove redundant interior points without crossing motion-mode boundaries."""
    if not isfinite(linear_tolerance) or linear_tolerance < 0.0:
        raise ValueError("linear_tolerance must be finite and non-negative")
    if not isfinite(angular_tolerance_deg) or angular_tolerance_deg < 0.0:
        raise ValueError("angular_tolerance_deg must be finite and non-negative")
    if len(toolpath.points) <= 2:
        return Toolpath(toolpath.tool_number, toolpath.strategy, toolpath.points)
    kept = [toolpath.points[0]]
    for index in range(1, len(toolpath.points) - 1):
        if not _can_remove(
            kept[-1],
            toolpath.points[index],
            toolpath.points[index + 1],
            linear_tolerance,
            angular_tolerance_deg,
        ):
            kept.append(toolpath.points[index])
    kept.append(toolpath.points[-1])
    return Toolpath(toolpath.tool_number, toolpath.strategy, kept)
