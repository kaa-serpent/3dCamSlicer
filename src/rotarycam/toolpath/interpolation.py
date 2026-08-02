"""Interpolation helpers for continuous X/Z/A machine motion."""

from __future__ import annotations

from math import ceil, isfinite

import numpy as np
import numpy.typing as npt

from rotarycam.toolpath.models import ToolpathPoint


def unwrap_angles(angles_deg: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Return the closest continuous representation of wrapped input angles."""
    angles = np.asarray(angles_deg, dtype=np.float64)
    if angles.ndim != 1:
        raise ValueError("angles_deg must be one-dimensional")
    if not np.all(np.isfinite(angles)):
        raise ValueError("angles_deg must be finite")
    if angles.size == 0:
        return angles.copy()
    return np.degrees(np.unwrap(np.radians(angles), period=2.0 * np.pi))


def interpolate_segment(
    start: ToolpathPoint,
    end: ToolpathPoint,
    *,
    max_linear_step: float,
    max_angle_step_deg: float,
) -> list[ToolpathPoint]:
    """Densify one homogeneous move, including both endpoints."""
    if not isfinite(max_linear_step) or max_linear_step <= 0.0:
        raise ValueError("max_linear_step must be finite and positive")
    if not isfinite(max_angle_step_deg) or max_angle_step_deg <= 0.0:
        raise ValueError("max_angle_step_deg must be finite and positive")
    if start.rapid != end.rapid:
        raise ValueError("cannot interpolate across a rapid/cutting mode transition")
    linear_distance = float(np.hypot(end.x - start.x, end.z - start.z))
    angular_distance = abs(end.a - start.a)
    count = max(
        1,
        ceil(linear_distance / max_linear_step),
        ceil(angular_distance / max_angle_step_deg),
    )
    points: list[ToolpathPoint] = []
    for index in range(count + 1):
        fraction = index / count
        feed = end.feed if index else start.feed
        points.append(
            ToolpathPoint(
                x=start.x + (end.x - start.x) * fraction,
                z=start.z + (end.z - start.z) * fraction,
                a=start.a + (end.a - start.a) * fraction,
                feed=feed,
                rapid=start.rapid,
            )
        )
    return points
