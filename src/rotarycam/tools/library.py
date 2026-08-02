"""Versioned JSON persistence for a personal cutting-bit library."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from rotarycam.persistence import atomic_write_text, preserve_v1_source
from rotarycam.tools.models import Tool, ToolHolder, ToolType, validate_unique_tool_numbers

TOOL_LIBRARY_SCHEMA_VERSION: Literal[2] = 2


class ToolHolderRecord(BaseModel):
    """JSON-safe cylindrical holder envelope."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    diameter: float = Field(gt=0.0)
    length: float = Field(gt=0.0)

    @field_validator("diameter", "length")
    @classmethod
    def validate_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("holder dimensions must be finite")
        return value

    @classmethod
    def from_holder(cls, holder: ToolHolder) -> ToolHolderRecord:
        return cls(diameter=holder.diameter, length=holder.length)

    def to_holder(self) -> ToolHolder:
        return ToolHolder(diameter=self.diameter, length=self.length)


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
    stickout: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)
    holder: ToolHolderRecord | None = None

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
            stickout=tool.stickout,
            holder=None if tool.holder is None else ToolHolderRecord.from_holder(tool.holder),
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
            self.stickout,
            None if self.holder is None else self.holder.to_holder(),
        )


class ToolLibraryDocument(BaseModel):
    """Portable schema envelope supporting future migrations."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[2] = TOOL_LIBRARY_SCHEMA_VERSION
    tools: tuple[ToolRecord, ...] = ()

    @model_validator(mode="after")
    def validate_unique_numbers(self) -> ToolLibraryDocument:
        validate_unique_tool_numbers([record.to_tool() for record in self.tools])
        return self


class ToolRecordV1(BaseModel):
    """Read-only compatibility record for one schema-v1 tool."""

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


class ToolLibraryDocumentV1(BaseModel):
    """Read-only compatibility DTO for tool-library schema v1."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    tools: tuple[ToolRecordV1, ...] = ()

    @model_validator(mode="after")
    def validate_unique_numbers(self) -> ToolLibraryDocumentV1:
        validate_unique_tool_numbers([record.to_tool() for record in self.tools])
        return self


class _LoadedTools(list[Tool]):
    """List-compatible result carrying non-persisted migration provenance."""

    def __init__(self, tools: Sequence[Tool], migrated_v1_source: Path | None = None) -> None:
        super().__init__(tools)
        self.migrated_v1_source = migrated_v1_source


def migrate_tool_library_v1(document: ToolLibraryDocumentV1) -> tuple[Tool, ...]:
    """Convert v1 tools while leaving unknown stickout and holder unset."""

    return tuple(record.to_tool() for record in document.tools)


def default_tool_library_path() -> Path:
    """Return the current user's personal RotaryCAM bit-library path."""

    app_data = os.environ.get("APPDATA")
    base = Path(app_data) if app_data else Path.home() / ".config"
    return base / "RotaryCAM" / "tools.json"


def save_tool_library(tools: Sequence[Tool], path: Path) -> None:
    """Atomically save a validated library as UTF-8 JSON."""

    validate_unique_tool_numbers(tools)
    document = ToolLibraryDocument(tools=tuple(ToolRecord.from_tool(tool) for tool in tools))
    destination = path.resolve()
    if isinstance(tools, _LoadedTools) and tools.migrated_v1_source is not None:
        preserve_v1_source(tools.migrated_v1_source)
    atomic_write_text(
        destination,
        document.model_dump_json(by_alias=True, indent=2) + "\n",
    )
    if isinstance(tools, _LoadedTools):
        tools.migrated_v1_source = None


def load_tool_library(path: Path) -> list[Tool]:
    """Load and validate all cutting bits from a versioned JSON document."""

    source = path.resolve()
    raw: Any = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("tool library must be a JSON object")
    version = raw.get("schema_version")
    if version == 1:
        migrated = migrate_tool_library_v1(ToolLibraryDocumentV1.model_validate(raw))
        return _LoadedTools(migrated, source)
    if version == TOOL_LIBRARY_SCHEMA_VERSION:
        document = ToolLibraryDocument.model_validate(raw)
        return _LoadedTools([record.to_tool() for record in document.tools])
    raise ValueError(f"unsupported tool-library schema_version: {version!r}")
