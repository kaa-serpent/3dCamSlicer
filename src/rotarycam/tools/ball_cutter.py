"""Ball-end cutter compensation entry points."""

from rotarycam.geometry.models import RotaryGrid
from rotarycam.tools.compensation import compute_compensated_tool_grid
from rotarycam.tools.cutter_kernel import BallCutterKernel
from rotarycam.tools.models import Tool, ToolType


def compensate_ball_cutter(target: RotaryGrid, tool: Tool) -> RotaryGrid:
    """Return a conservative TCP envelope for a ball end mill."""

    if tool.tool_type is not ToolType.BALL:
        raise ValueError("compensate_ball_cutter requires a ball tool")
    return compute_compensated_tool_grid(target, tool)


__all__ = ["BallCutterKernel", "compensate_ball_cutter"]
