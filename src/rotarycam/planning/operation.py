"""A planned machining operation and its generated paths."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from rotarycam.toolpath.models import Toolpath
from rotarycam.tools.models import Tool


@dataclass(slots=True)
class MachiningOperation:
    """One strategy executed with one cutter."""

    name: str
    tool: Tool
    strategy: str
    allowance: float
    tolerance: float
    toolpaths: list[Toolpath]
    estimated_removed_volume: float = 0.0

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.strategy.strip():
            raise ValueError("operation name and strategy must not be empty")
        for field_name, value in (
            ("allowance", self.allowance),
            ("tolerance", self.tolerance),
            ("estimated_removed_volume", self.estimated_removed_volume),
        ):
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{field_name} must be finite and non-negative")
        if not self.toolpaths:
            raise ValueError("an operation must contain at least one toolpath")
        if any(path.tool_number != self.tool.number for path in self.toolpaths):
            raise ValueError("all toolpaths must use the operation tool number")
        self.toolpaths = list(self.toolpaths)
