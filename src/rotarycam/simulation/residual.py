"""Residual-stock measurements shared by simulation and rest machining."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from rotarycam.geometry.models import RotaryGrid

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class ResidualResult:
    """Error field and aggregate values above an effective target."""

    error: FloatArray
    max_error: float
    mean_error: float
    mask: BoolArray

    def __post_init__(self) -> None:
        error = np.array(self.error, dtype=np.float64, copy=True)
        mask = np.array(self.mask, dtype=np.bool_, copy=True)
        if error.shape != mask.shape:
            raise ValueError("residual error and mask shapes must match")
        error.setflags(write=False)
        mask.setflags(write=False)
        object.__setattr__(self, "error", error)
        object.__setattr__(self, "mask", mask)


def compute_residual(
    stock: RotaryGrid,
    target: RotaryGrid,
    *,
    tolerance: float = 0.0,
) -> ResidualResult:
    """Measure remaining stock while ignoring invalid cells."""

    if tolerance < 0.0 or not np.isfinite(tolerance):
        raise ValueError("tolerance must be finite and non-negative")
    if stock.shape != target.shape:
        raise ValueError("stock and target grids must have the same shape")
    if not np.array_equal(stock.x_values, target.x_values) or not np.array_equal(
        stock.angles_deg,
        target.angles_deg,
    ):
        raise ValueError("stock and target coordinates must match")

    valid = np.asarray(stock.valid) & np.asarray(target.valid)
    error = np.zeros(stock.shape, dtype=np.float64)
    error[valid] = np.maximum(
        np.asarray(stock.radius)[valid] - np.asarray(target.radius)[valid],
        0.0,
    )
    mask = valid & (error > tolerance)
    values = error[valid]
    max_error = float(np.max(values)) if values.size else 0.0
    mean_error = float(np.mean(values)) if values.size else 0.0
    return ResidualResult(error, max_error, mean_error, mask)
