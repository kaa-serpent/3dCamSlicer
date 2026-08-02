"""Shared finishing strategy settings."""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True, slots=True)
class FinishingSettings:
    """Sampling, allowance and safe-link settings for finishing."""

    stepover: float
    safe_radius: float
    allowance: float = 0.0
    tolerance: float = 0.05
    bidirectional: bool = True

    def __post_init__(self) -> None:
        for field_name in ("stepover", "safe_radius", "tolerance"):
            value = getattr(self, field_name)
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{field_name} must be finite and positive")
        if not isfinite(self.allowance) or self.allowance < 0.0:
            raise ValueError("allowance must be finite and non-negative")
