import pytest

from rotarycam.planning.tool_selector import ToolSelectionSettings, sorted_tool_candidates
from rotarycam.tools.models import Tool, ToolType


def make_tool(number: int, diameter: float) -> Tool:
    return Tool(
        number,
        f"Tool {number}",
        ToolType.BALL,
        diameter,
        10.0,
        10.0,
        40.0,
        diameter,
        1.0,
        diameter / 4.0,
        300.0,
        80.0,
        12_000,
    )


def test_candidates_are_sorted_by_decreasing_diameter() -> None:
    tools = [make_tool(1, 2.0), make_tool(2, 6.0), make_tool(3, 4.0)]

    assert [tool.number for tool in sorted_tool_candidates(tools)] == [2, 3, 1]


def test_duplicate_tool_numbers_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicates: 1"):
        sorted_tool_candidates([make_tool(1, 2.0), make_tool(1, 4.0)])


def test_selection_settings_reject_invalid_ratios() -> None:
    with pytest.raises(ValueError, match="must not exceed one"):
        ToolSelectionSettings(minimum_gain_ratio=1.1)
