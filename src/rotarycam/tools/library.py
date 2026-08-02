"""Versioned JSON persistence for a personal cutting-bit library."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rotarycam.tools.models import Tool, ToolType, validate_unique_tool_numbers

TOOL_LIBRARY_SCHEMA_VERSION: Literal[1] = 1


class ToolRecord(BaseModel):
    """JSON-safe representation of one validated cutting bit."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    number: int = Field(gt=0)
    name: str
    tool_type: ToolType = Field(alias="type")
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

    @classmethod
    def from_tool(cls, tool: Tool) -> ToolRecord:
        return cls(
            number=tool.number,
            name=tool.name,
            type=tool.tool_type,
            diameter=tool.diameter,
            cutting_length=tool.cutting_length,
            flute_length=tool.flute_length,
            overall_length=tool.overall_length,
            shank_diameter=tool.shank_diameter,
            max_stepdown=tool.max_stepdown,
            stepover=tool.stepover,
            feed=tool.feed,
            plunge_feed=tool.plunge_feed,
            spindle_rpm=tool.spindle_rpm,
            tip_diameter=tool.tip_diameter,
            taper_length=tool.taper_length,
        )

    def to_tool(self) -> Tool:
        return Tool(
            self.number,
            self.name,
            self.tool_type,
            self.diameter,
            self.cutting_length,
            self.flute_length,
            self.overall_length,
            self.shank_diameter,
            self.max_stepdown,
            self.stepover,
            self.feed,
            self.plunge_feed,
            self.spindle_rpm,
            self.tip_diameter,
            self.taper_length,
        )


class ToolLibraryDocument(BaseModel):
    """Portable schema envelope supporting future migrations."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = TOOL_LIBRARY_SCHEMA_VERSION
    tools: tuple[ToolRecord, ...] = ()

    @model_validator(mode="after")
    def validate_unique_numbers(self) -> ToolLibraryDocument:
        validate_unique_tool_numbers([record.to_tool() for record in self.tools])
        return self


def default_tool_library_path() -> Path:
    """Return the current user's personal RotaryCAM bit-library path."""

    app_data = os.environ.get("APPDATA")
    base = Path(app_data) if app_data else Path.home() / ".config"
    return base / "RotaryCAM" / "tools.json"


def save_tool_library(tools: list[Tool], path: Path) -> None:
    """Atomically save a validated library as UTF-8 JSON."""

    validate_unique_tool_numbers(tools)
    document = ToolLibraryDocument(tools=tuple(ToolRecord.from_tool(tool) for tool in tools))
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        document.model_dump_json(by_alias=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def load_tool_library(path: Path) -> list[Tool]:
    """Load and validate all cutting bits from a versioned JSON document."""

    document = ToolLibraryDocument.model_validate_json(path.read_text(encoding="utf-8"))
    return [record.to_tool() for record in document.tools]
