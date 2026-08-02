"""Validated settings shared by indexed and rotary roughing."""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True, slots=True)
class RoughingSettings:
    """Radial levels and safe-link parameters for roughing operations."""

    stepdown: float
    stepover: float
    allowance: float
    safe_radius: float
    climb_milling: bool = True
    tolerance: float = 0.05

    def __post_init__(self) -> None:
        for field_name in ("stepdown", "stepover", "safe_radius", "tolerance"):
            value = getattr(self, field_name)
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{field_name} must be finite and positive")
        if not isfinite(self.allowance) or self.allowance < 0.0:
            raise ValueError("allowance must be finite and non-negative")
