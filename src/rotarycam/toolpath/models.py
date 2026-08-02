"""Toolpath point definitions."""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True, slots=True)
class ToolpathPoint:
    """A simultaneous machine position in X, radial Z and continuous A.

    ``x`` and ``z`` are millimetres, ``a`` is an unwrapped angle in degrees, and
    ``feed`` is millimetres per minute when specified.
    """

    x: float
    z: float
    a: float
    feed: float | None = None
    rapid: bool = False

    def __post_init__(self) -> None:
        """Reject non-finite positions and invalid feeds."""
        for field_name, value in (("x", self.x), ("z", self.z), ("a", self.a)):
            if not isfinite(value):
                raise ValueError(f"{field_name} must be finite")
        if self.feed is not None and (not isfinite(self.feed) or self.feed <= 0):
            raise ValueError("feed must be finite and greater than zero when specified")


@dataclass(slots=True)
class Toolpath:
    """Ordered machine moves for one tool and one strategy."""

    tool_number: int
    strategy: str
    points: list[ToolpathPoint]

    def __post_init__(self) -> None:
        if self.tool_number <= 0:
            raise ValueError("tool_number must be greater than zero")
        if not self.strategy.strip():
            raise ValueError("strategy must not be empty")
        if not self.points:
            raise ValueError("a toolpath must contain at least one point")
        self.points = list(self.points)
