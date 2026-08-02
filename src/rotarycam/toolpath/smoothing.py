"""Conservative smoothing of cutting-only radial coordinates."""

from __future__ import annotations

from rotarycam.toolpath.models import Toolpath, ToolpathPoint


def smooth_toolpath(toolpath: Toolpath, *, window_size: int = 3) -> Toolpath:
    """Smooth Z within cutting runs while preserving endpoints and rapid moves."""
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("window_size must be an odd integer of at least three")
    points = list(toolpath.points)
    half = window_size // 2
    output = list(points)
    for index, point in enumerate(points):
        if point.rapid or index < half or index + half >= len(points):
            continue
        neighborhood = points[index - half : index + half + 1]
        if any(candidate.rapid for candidate in neighborhood):
            continue
        smoothed_z = sum(candidate.z for candidate in neighborhood) / window_size
        output[index] = ToolpathPoint(point.x, smoothed_z, point.a, point.feed, point.rapid)
    return Toolpath(toolpath.tool_number, toolpath.strategy, output)
