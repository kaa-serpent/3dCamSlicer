"""Rectangular stock represented as an analytical radial profile."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def rectangular_radius_at(angle_rad: float, half_width: float, half_height: float) -> float:
    """Intersect a center-origin ray with a Y/Z-aligned rectangle."""

    values = np.asarray((angle_rad, half_width, half_height), dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("angle and half dimensions must be finite")
    if half_width <= 0.0 or half_height <= 0.0:
        raise ValueError("half dimensions must be positive")
    cosine = abs(float(np.cos(angle_rad)))
    sine = abs(float(np.sin(angle_rad)))
    y_limit = np.inf if cosine < 1e-15 else half_width / cosine
    z_limit = np.inf if sine < 1e-15 else half_height / sine
    return float(min(y_limit, z_limit))


@dataclass(frozen=True, slots=True)
class RectangularStock:
    """A Y/Z-centered rectangular prism spanning X from zero to length."""

    length: float
    width: float
    height: float

    def __post_init__(self) -> None:
        dimensions = np.asarray((self.length, self.width, self.height), dtype=np.float64)
        if not np.all(np.isfinite(dimensions)) or np.any(dimensions <= 0.0):
            raise ValueError("length, width, and height must be finite and positive")

    def radius_at(self, x: float, angle_deg: float) -> float:
        """Return the rectangle boundary radius at an X/A location."""

        if not np.isfinite(x) or not np.isfinite(angle_deg):
            raise ValueError("x and angle_deg must be finite")
        if x < 0.0 or x > self.length:
            return 0.0
        return rectangular_radius_at(np.radians(angle_deg), self.width / 2.0, self.height / 2.0)
