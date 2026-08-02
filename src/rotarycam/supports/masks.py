"""Support mask dispatch and protected-target construction."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from rotarycam.geometry.models import RotaryGrid
from rotarycam.supports.cylinder import cylindrical_support_mask
from rotarycam.supports.models import CylindricalSupport, RectangularSupport, Support
from rotarycam.supports.rectangle import rectangular_support_mask


def support_mask(grid: RotaryGrid, support: Support) -> npt.NDArray[np.float64]:
    """Return the weight mask for one support definition."""

    if isinstance(support, RectangularSupport):
        return rectangular_support_mask(grid, support)
    if isinstance(support, CylindricalSupport):
        return cylindrical_support_mask(grid, support)
    raise TypeError(f"unsupported support type: {type(support).__name__}")


def _validate_compatible_grids(target: RotaryGrid, stock: RotaryGrid) -> None:
    target_radius = np.asarray(target.radius)
    stock_radius = np.asarray(stock.radius)
    target_valid = np.asarray(target.valid, dtype=np.bool_)
    stock_valid = np.asarray(stock.valid, dtype=np.bool_)

    if target_radius.shape != stock_radius.shape:
        raise ValueError("target and stock radius arrays must have the same shape")
    if target_valid.shape != target_radius.shape or stock_valid.shape != stock_radius.shape:
        raise ValueError("grid validity masks must match their radius arrays")
    if not np.array_equal(np.asarray(target.x_values), np.asarray(stock.x_values)):
        raise ValueError("target and stock X coordinates must match")
    if not np.array_equal(np.asarray(target.angles_deg), np.asarray(stock.angles_deg)):
        raise ValueError("target and stock angle coordinates must match")

    usable = target_valid & stock_valid
    if not np.all(np.isfinite(target_radius[usable])):
        raise ValueError("target contains non-finite radii in valid cells")
    if not np.all(np.isfinite(stock_radius[usable])):
        raise ValueError("stock contains non-finite radii in valid cells")
    if np.any(target_radius[usable] > stock_radius[usable] + 1e-9):
        raise ValueError("target radius must not exceed stock radius")


def apply_supports(
    target: RotaryGrid,
    stock: RotaryGrid,
    supports: Sequence[Support],
) -> RotaryGrid:
    """Return an effective target that preserves every enabled support.

    Overlapping supports use the largest requested protected radius rather than
    summing thicknesses.  Invalid target cells are left unchanged.
    """

    _validate_compatible_grids(target, stock)
    target_radius = np.asarray(target.radius, dtype=np.float64)
    stock_radius = np.asarray(stock.radius, dtype=np.float64)
    target_valid = np.asarray(target.valid, dtype=np.bool_)
    stock_valid = np.asarray(stock.valid, dtype=np.bool_)
    usable = target_valid & stock_valid

    protected = target_radius.copy()
    for support in supports:
        if not support.enabled:
            continue
        weight = support_mask(target, support)
        candidate = target_radius + support.thickness * weight
        protected[usable] = np.maximum(protected[usable], candidate[usable])

    effective = target_radius.copy()
    effective[usable] = np.minimum(stock_radius[usable], protected[usable])
    return target.with_radius(effective)
