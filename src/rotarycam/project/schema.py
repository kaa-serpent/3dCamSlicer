"""Versioned, JSON-safe project schema.

This module contains persistence DTOs.  They deliberately do not leak into the
geometry engine, whose stock and tool models remain independent domain types.
"""

from __future__ import annotations

import math
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator

from rotarycam.config import MachineDefinition, MachiningSettings
from rotarycam.supports.models import (
    RetentionVolume,
    Support,
    migrate_support_to_retention,
)

PROJECT_SCHEMA_VERSION: Literal[2] = 2

type MatrixRow = tuple[float, float, float, float]
type Matrix4x4 = tuple[MatrixRow, MatrixRow, MatrixRow, MatrixRow]


def identity_transform() -> Matrix4x4:
    """Return a fresh identity transform in JSON-friendly tuple form."""

    return (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


class _ProjectRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class StockType(StrEnum):
    RECTANGLE = "rectangle"
    CYLINDER = "cylinder"


class RectangularStockConfig(_ProjectRecord):
    """Persisted rectangular stock definition in millimetres."""

    stock_type: Literal[StockType.RECTANGLE] = Field(
        default=StockType.RECTANGLE,
        alias="type",
    )
    length: float = Field(gt=0.0)
    width: float = Field(gt=0.0)
    height: float = Field(gt=0.0)

    @field_validator("length", "width", "height")
    @classmethod
    def finite_dimensions(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("stock dimensions must be finite")
        return value


class CylindricalStockConfig(_ProjectRecord):
    """Persisted cylindrical stock definition in millimetres."""

    stock_type: Literal[StockType.CYLINDER] = Field(
        default=StockType.CYLINDER,
        alias="type",
    )
    length: float = Field(gt=0.0)
    diameter: float = Field(gt=0.0)

    @field_validator("length", "diameter")
    @classmethod
    def finite_dimensions(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("stock dimensions must be finite")
        return value


type StockDefinition = Annotated[
    RectangularStockConfig | CylindricalStockConfig,
    Field(discriminator="stock_type"),
]


class ToolType(StrEnum):
    FLAT = "flat"
    BALL = "ball"
    TAPERED = "tapered"


class PipelineMode(StrEnum):
    """Persisted CAM pipeline selection."""

    XYZA = "xyza"


class ToolHolderConfig(_ProjectRecord):
    """Conservative cylindrical holder envelope in millimetres."""

    diameter: float = Field(gt=0.0)
    length: float = Field(gt=0.0)

    @field_validator("diameter", "length")
    @classmethod
    def validate_dimensions(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("holder dimensions must be finite")
        return value


class ToolConfig(_ProjectRecord):
    """JSON representation of a supported cutter geometry."""

    number: int = Field(gt=0)
    name: str
    tool_type: ToolType = Field(alias="type")
    diameter: float = Field(gt=0.0)
    cutting_length: float = Field(gt=0.0)
    flute_length: float = Field(gt=0.0)
    overall_length: float = Field(gt=0.0)
    shank_diameter: float = Field(gt=0.0)
    max_stepdown: float = Field(gt=0.0)
    stepover: float = Field(gt=0.0)
    feed: float = Field(gt=0.0)
    plunge_feed: float = Field(gt=0.0)
    spindle_rpm: int = Field(gt=0)
    tip_diameter: float | None = Field(default=None, gt=0.0)
    taper_length: float | None = Field(default=None, gt=0.0)
    stickout: float | None = Field(default=None, gt=0.0)
    holder: ToolHolderConfig | None = None

    @model_validator(mode="after")
    def validate_geometry(self) -> ToolConfig:
        numeric_values = (
            self.diameter,
            self.cutting_length,
            self.flute_length,
            self.overall_length,
            self.shank_diameter,
            self.max_stepdown,
            self.stepover,
            self.feed,
            self.plunge_feed,
        )
        if not all(math.isfinite(value) for value in numeric_values):
            raise ValueError("tool dimensions and feeds must be finite")
        if not self.name.strip():
            raise ValueError("tool name must not be empty")
        if self.max_stepdown > self.cutting_length:
            raise ValueError("max_stepdown must not exceed cutting_length")
        if self.stepover > self.diameter:
            raise ValueError("stepover must not exceed diameter")
        if self.cutting_length > self.flute_length:
            raise ValueError("cutting_length must not exceed flute_length")
        if self.flute_length > self.overall_length:
            raise ValueError("flute_length must not exceed overall_length")
        if self.tool_type is ToolType.TAPERED:
            if self.tip_diameter is None or self.taper_length is None:
                raise ValueError("tapered bits require tip_diameter and taper_length")
            if self.tip_diameter >= self.diameter:
                raise ValueError("tip_diameter must be smaller than diameter")
            if self.taper_length > self.cutting_length:
                raise ValueError("taper_length must not exceed cutting_length")
        elif self.tip_diameter is not None or self.taper_length is not None:
            raise ValueError("taper dimensions are only valid for tapered bits")
        if self.stickout is not None:
            if not math.isfinite(self.stickout):
                raise ValueError("stickout must be finite")
            if self.stickout > self.overall_length:
                raise ValueError("stickout must not exceed overall_length")
        return self


class RotaryCamProject(_ProjectRecord):
    """Complete persisted project envelope for schema version 2."""

    schema_version: Literal[2] = PROJECT_SCHEMA_VERSION
    pipeline: Literal[PipelineMode.XYZA] = PipelineMode.XYZA
    mesh_path: Path
    mesh_scale: float = Field(default=1.0, gt=0.0)
    mesh_transform: Matrix4x4 = Field(default_factory=identity_transform)
    stock: StockDefinition
    tools: list[ToolConfig] = Field(default_factory=list)
    supports: list[Support] = Field(default_factory=list)
    retention_volumes: list[RetentionVolume] = Field(default_factory=list)
    machining_settings: MachiningSettings = Field(default_factory=MachiningSettings)
    machine: MachineDefinition
    generated_operations: list[dict[str, Any]] = Field(default_factory=list)

    _migrated_v1_source: Path | None = PrivateAttr(default=None)

    @field_validator("mesh_path")
    @classmethod
    def validate_mesh_path(cls, value: Path) -> Path:
        if not str(value).strip():
            raise ValueError("mesh_path must not be empty")
        if value.suffix.lower() not in {".stl", ".obj"}:
            raise ValueError("mesh_path must reference an STL or OBJ file")
        return value

    @field_validator("mesh_scale")
    @classmethod
    def validate_scale(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("mesh_scale must be finite")
        return value

    @field_validator("mesh_transform")
    @classmethod
    def validate_transform(cls, value: Matrix4x4) -> Matrix4x4:
        if not all(math.isfinite(component) for row in value for component in row):
            raise ValueError("mesh_transform values must be finite")
        if value[3] != (0.0, 0.0, 0.0, 1.0):
            raise ValueError("mesh_transform must be an affine 4x4 matrix")
        return value

    @model_validator(mode="after")
    def validate_tool_numbers(self) -> RotaryCamProject:
        numbers = [tool.number for tool in self.tools]
        if len(numbers) != len(set(numbers)):
            raise ValueError("tool numbers must be unique")
        return self


class RotaryCamProjectV1(_ProjectRecord):
    """Read-only compatibility DTO for the original radial project schema."""

    schema_version: Literal[1] = 1
    mesh_path: Path
    mesh_scale: float = Field(default=1.0, gt=0.0)
    mesh_transform: Matrix4x4 = Field(default_factory=identity_transform)
    stock: StockDefinition
    tools: list[ToolConfig] = Field(default_factory=list)
    supports: list[Support] = Field(default_factory=list)
    machining_settings: MachiningSettings = Field(default_factory=MachiningSettings)
    machine: MachineDefinition
    generated_operations: list[dict[str, Any]] = Field(default_factory=list)


def migrate_project_v1(project: RotaryCamProjectV1) -> RotaryCamProject:
    """Create a v2 XYZA project while invalidating every unsafe derived value."""

    machine = project.machine.model_copy(
        update={
            "profile_verified": False,
            "y_limits": None,
            "xyza_configuration": None,
            "machine_assembly_configured": False,
        }
    )
    return RotaryCamProject(
        mesh_path=project.mesh_path,
        mesh_scale=project.mesh_scale,
        mesh_transform=project.mesh_transform,
        stock=project.stock,
        tools=[
            tool.model_copy(update={"stickout": None, "holder": None})
            for tool in project.tools
        ],
        supports=[],
        retention_volumes=[migrate_support_to_retention(item) for item in project.supports],
        machining_settings=project.machining_settings,
        machine=machine,
        generated_operations=[],
    )
