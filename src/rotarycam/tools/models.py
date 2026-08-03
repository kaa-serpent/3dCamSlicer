"""Cutting tool definitions and validation."""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite


class ToolType(StrEnum):
    """Supported cutter geometries."""

    FLAT = "flat"
    BALL = "ball"
    TAPERED = "tapered"


@dataclass(frozen=True, slots=True)
class ToolHolder:
    """Conservative cylindrical holder envelope in millimetres."""

    diameter: float
    length: float

    def __post_init__(self) -> None:
        for field_name, value in (("diameter", self.diameter), ("length", self.length)):
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"holder {field_name} must be finite and greater than zero")


@dataclass(frozen=True, slots=True)
class Tool:
    """Physical cutter and conservative machining parameters.

    Linear dimensions and feeds are expressed in millimetres and millimetres per
    minute. ``spindle_rpm`` is expressed in revolutions per minute.
    """

    number: int
    name: str
    tool_type: ToolType
    diameter: float
    cutting_length: float
    flute_length: float
    overall_length: float
    shank_diameter: float
    max_stepdown: float
    stepover: float
    feed: float
    plunge_feed: float
    spindle_rpm: int
    tip_diameter: float | None = None
    taper_length: float | None = None
    stickout: float | None = None
    holder: ToolHolder | None = None

    def __post_init__(self) -> None:
        """Reject physically inconsistent tool definitions."""
        if self.number <= 0:
            raise ValueError("tool number must be greater than zero")
        if not self.name.strip():
            raise ValueError("tool name must not be empty")
        if not isinstance(self.tool_type, ToolType):
            raise TypeError("tool_type must be a ToolType")

        positive_values = {
            "diameter": self.diameter,
            "cutting_length": self.cutting_length,
            "flute_length": self.flute_length,
            "overall_length": self.overall_length,
            "shank_diameter": self.shank_diameter,
            "max_stepdown": self.max_stepdown,
            "stepover": self.stepover,
            "feed": self.feed,
            "plunge_feed": self.plunge_feed,
        }
        for field_name, value in positive_values.items():
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{field_name} must be finite and greater than zero")

        if self.spindle_rpm <= 0:
            raise ValueError("spindle_rpm must be greater than zero")
        if self.max_stepdown > self.cutting_length:
            raise ValueError("max_stepdown must not exceed cutting_length")
        if self.stepover > self.diameter:
            raise ValueError("stepover must not exceed diameter")
        if self.cutting_length > self.flute_length:
            raise ValueError("cutting_length must not exceed flute_length")
        if self.flute_length > self.overall_length:
            raise ValueError("flute_length must not exceed overall_length")
        if self.tool_type is ToolType.TAPERED:
            if self.tip_diameter is None:
                raise ValueError("tip_diameter is required for a tapered bit")
            if not isfinite(self.tip_diameter) or self.tip_diameter <= 0.0:
                raise ValueError("tip_diameter must be finite and greater than zero")
            if self.tip_diameter >= self.diameter:
                raise ValueError("tip_diameter must be smaller than diameter")
            if self.taper_length is None:
                raise ValueError("taper_length is required for a tapered bit")
            if not isfinite(self.taper_length) or self.taper_length <= 0.0:
                raise ValueError("taper_length must be finite and greater than zero")
            if self.taper_length > self.cutting_length:
                raise ValueError("taper_length must not exceed cutting_length")
        elif self.tip_diameter is not None or self.taper_length is not None:
            raise ValueError("tip_diameter and taper_length are only valid for tapered bits")
        if self.stickout is not None:
            if not isfinite(self.stickout) or self.stickout <= 0.0:
                raise ValueError("stickout must be finite and greater than zero")
            if self.stickout < self.flute_length:
                raise ValueError("stickout must not be shorter than flute_length")
            if self.stickout > self.overall_length:
                raise ValueError("stickout must not exceed overall_length")
        if self.holder is not None and not isinstance(self.holder, ToolHolder):
            raise TypeError("holder must be a ToolHolder")


def validate_unique_tool_numbers(tools: Sequence[Tool]) -> None:
    """Validate the collection-level uniqueness invariant for tool numbers."""
    seen: set[int] = set()
    duplicates: set[int] = set()
    for tool in tools:
        if tool.number in seen:
            duplicates.add(tool.number)
        seen.add(tool.number)
    if duplicates:
        duplicate_list = ", ".join(str(number) for number in sorted(duplicates))
        raise ValueError(f"tool numbers must be unique; duplicates: {duplicate_list}")
