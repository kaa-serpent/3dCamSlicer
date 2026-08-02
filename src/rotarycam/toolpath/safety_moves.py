"""Construction of safe rapid links between cutting segments."""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite

from rotarycam.toolpath.models import ToolpathPoint


def connect_cutting_segments(
    segments: Sequence[Sequence[ToolpathPoint]],
    *,
    safe_radius: float,
    plunge_feed: float,
) -> list[ToolpathPoint]:
    """Wrap each cutting segment in retract, traverse, plunge and retract moves."""
    if not isfinite(safe_radius) or safe_radius <= 0.0:
        raise ValueError("safe_radius must be finite and positive")
    if not isfinite(plunge_feed) or plunge_feed <= 0.0:
        raise ValueError("plunge_feed must be finite and positive")

    result: list[ToolpathPoint] = []
    for segment in segments:
        if not segment:
            continue
        if any(point.rapid for point in segment):
            raise ValueError("input cutting segments must not contain rapid points")
        if any(point.z >= safe_radius for point in segment):
            raise ValueError("safe_radius must be greater than every cutting radius")
        first = segment[0]
        last = segment[-1]
        result.append(ToolpathPoint(first.x, safe_radius, first.a, rapid=True))
        result.append(ToolpathPoint(first.x, first.z, first.a, feed=plunge_feed))
        result.extend(segment[1:])
        result.append(ToolpathPoint(last.x, safe_radius, last.a, rapid=True))
    return result
