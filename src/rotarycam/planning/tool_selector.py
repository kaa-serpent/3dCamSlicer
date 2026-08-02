"""Tool collection validation and deterministic candidate ordering."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from rotarycam.tools.models import Tool, validate_unique_tool_numbers


@dataclass(frozen=True, slots=True)
class ToolSelectionSettings:
    """Minimum useful gain required before accepting a tool change."""

    minimum_removed_volume: float = 0.0
    minimum_machined_area: float = 0.0
    minimum_gain_ratio: float = 0.0
    final_tolerance: float = 0.05
    tool_change_penalty: float = 0.0

    def __post_init__(self) -> None:
        for field_name in (
            "minimum_removed_volume",
            "minimum_machined_area",
            "minimum_gain_ratio",
            "tool_change_penalty",
        ):
            value = getattr(self, field_name)
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{field_name} must be finite and non-negative")
        if not isfinite(self.final_tolerance) or self.final_tolerance <= 0.0:
            raise ValueError("final_tolerance must be finite and positive")
        if self.minimum_gain_ratio > 1.0 or self.tool_change_penalty > 1.0:
            raise ValueError("gain ratio and tool change penalty must not exceed one")


def sorted_tool_candidates(tools: list[Tool]) -> list[Tool]:
    """Validate and sort cutters by decreasing diameter, then stable tool number."""
    if not tools:
        raise ValueError("at least one tool is required")
    validate_unique_tool_numbers(tools)
    return sorted(tools, key=lambda tool: (-tool.diameter, tool.number))
