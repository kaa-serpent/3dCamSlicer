"""Cylindrical stock definition."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class CylindricalStock:
    """A cylindrical stock centered on the X rotary axis."""

    length: float
    diameter: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.length) or self.length <= 0.0:
            raise ValueError("length must be finite and positive")
        if not np.isfinite(self.diameter) or self.diameter <= 0.0:
            raise ValueError("diameter must be finite and positive")

    def radius_at(self, x: float, angle_deg: float) -> float:
        """Return the constant cylinder radius within its X extent."""

        if not np.isfinite(x) or not np.isfinite(angle_deg):
            raise ValueError("x and angle_deg must be finite")
        if x < 0.0 or x > self.length:
            return 0.0
        return self.diameter / 2.0
