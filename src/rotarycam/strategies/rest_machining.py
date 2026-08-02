"""Residual-mask construction and spatially limited rest machining."""

from __future__ import annotations

from math import ceil

import numpy as np
import numpy.typing as npt
from scipy import ndimage  # type: ignore[import-untyped]

from rotarycam.geometry.models import RotaryGrid
from rotarycam.planning.operation import MachiningOperation
from rotarycam.strategies.helical import generate_helical_finishing
from rotarycam.strategies.settings import FinishingSettings
from rotarycam.toolpath.models import Toolpath
from rotarycam.tools.compensation import compute_compensated_tool_grid
from rotarycam.tools.models import Tool

BoolArray = npt.NDArray[np.bool_]


def _minimum_grid_spacing(grid: RotaryGrid) -> float:
    x_spacing = float(np.min(np.diff(grid.x_values))) if len(grid.x_values) > 1 else np.inf
    angle_spacing = (
        float(np.min(np.diff(grid.angles_deg))) if len(grid.angles_deg) > 1 else 360.0
    )
    positive_radius = grid.radius[grid.valid & (grid.radius > 0.0)]
    angular_spacing = (
        float(np.min(positive_radius)) * np.radians(angle_spacing)
        if positive_radius.size
        else np.inf
    )
    spacing = min(x_spacing, angular_spacing)
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("grid must provide a positive linear sampling spacing")
    return spacing


def dilate_residual_mask(
    mask: npt.ArrayLike,
    grid: RotaryGrid,
    tool_radius: float,
) -> BoolArray:
    """Dilate residual components while wrapping only the angular dimension."""

    source = np.asarray(mask, dtype=np.bool_)
    if source.shape != grid.shape:
        raise ValueError(f"mask must have shape {grid.shape}")
    if not np.isfinite(tool_radius) or tool_radius <= 0.0:
        raise ValueError("tool_radius must be finite and positive")
    iterations = max(1, ceil(tool_radius / _minimum_grid_spacing(grid)))
    tiled = np.concatenate((source, source, source), axis=1)
    dilated = ndimage.binary_dilation(
        tiled,
        structure=np.ones((3, 3), dtype=np.bool_),
        iterations=iterations,
    )
    angle_count = source.shape[1]
    return np.asarray(dilated[:, angle_count : 2 * angle_count] & grid.valid, dtype=np.bool_)


def generate_rest_machining(
    current_stock: RotaryGrid,
    target: RotaryGrid,
    tool: Tool,
    settings: FinishingSettings,
    mask: npt.ArrayLike | None = None,
) -> MachiningOperation:
    """Generate only helical segments near residual material above tolerance."""

    if not np.array_equal(current_stock.x_values, target.x_values) or not np.array_equal(
        current_stock.angles_deg, target.angles_deg
    ):
        raise ValueError("current_stock and target must use the same X/A layout")
    common = current_stock.valid & target.valid
    if np.any(target.radius[common] > current_stock.radius[common] + settings.tolerance):
        raise ValueError("target radius must not exceed current stock radius")
    residual = common & (current_stock.radius - target.radius > settings.tolerance)
    limit_mask: BoolArray | None = None
    if mask is not None:
        limit_mask = np.asarray(mask, dtype=np.bool_)
        if limit_mask.shape != target.shape:
            raise ValueError(f"mask must have shape {target.shape}")
        residual &= limit_mask
    if not np.any(residual):
        raise ValueError("no residual material exceeds tolerance")
    mask = dilate_residual_mask(residual, target, tool.diameter / 2.0)
    if limit_mask is not None:
        mask &= limit_mask
    finishing = generate_helical_finishing(
        target,
        tool,
        settings,
        mask,
        compensator=compute_compensated_tool_grid,
    )
    paths = [Toolpath(tool.number, "rest_machining", path.points) for path in finishing.toolpaths]
    return MachiningOperation(
        "Rest machining",
        tool,
        "rest_machining",
        settings.allowance,
        settings.tolerance,
        paths,
    )
