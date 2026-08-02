"""Simulation result models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from rotarycam.geometry.models import RotaryGrid

BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Immutable result of applying one toolpath to a stock grid."""

    stock: RotaryGrid
    removed_volume: float
    max_remaining_error: float
    mean_remaining_error: float
    remaining_mask: BoolArray

    def __post_init__(self) -> None:
        if not np.isfinite(self.removed_volume) or self.removed_volume < 0.0:
            raise ValueError("removed_volume must be finite and non-negative")
        if not np.isfinite(self.max_remaining_error) or self.max_remaining_error < 0.0:
            raise ValueError("max_remaining_error must be finite and non-negative")
        if not np.isfinite(self.mean_remaining_error) or self.mean_remaining_error < 0.0:
            raise ValueError("mean_remaining_error must be finite and non-negative")
        mask = np.array(self.remaining_mask, dtype=np.bool_, copy=True)
        if mask.shape != self.stock.shape:
            raise ValueError("remaining_mask must match the stock grid shape")
        mask.setflags(write=False)
        object.__setattr__(self, "remaining_mask", mask)
