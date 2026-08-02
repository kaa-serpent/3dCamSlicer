"""Flat-end cutter compensation entry points."""

from rotarycam.geometry.models import RotaryGrid
from rotarycam.tools.compensation import compute_compensated_tool_grid
from rotarycam.tools.cutter_kernel import FlatCutterKernel
from rotarycam.tools.models import Tool, ToolType


def compensate_flat_cutter(target: RotaryGrid, tool: Tool) -> RotaryGrid:
    """Return a conservative TCP envelope for a flat end mill."""

    if tool.tool_type is not ToolType.FLAT:
        raise ValueError("compensate_flat_cutter requires a flat tool")
    return compute_compensated_tool_grid(target, tool)


__all__ = ["FlatCutterKernel", "compensate_flat_cutter"]
