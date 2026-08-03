"""Typed persistence and browser-preview contracts for the local web UI."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from rotarycam.config import MachineDefinition, MachiningSettings
from rotarycam.project import RotaryCamProject, StockDefinition, ToolConfig
from rotarycam.supports import Support

WEB_WORKSPACE_SCHEMA_VERSION: Literal[1] = 1


def utc_now() -> datetime:
    """Return a timezone-aware timestamp for persisted workspace metadata."""

    return datetime.now(UTC)


class _WebRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class WebWorkspaceDocument(_WebRecord):
    """Autosaved, possibly incomplete project inputs for the local web UI."""

    schema_version: Literal[1] = WEB_WORKSPACE_SCHEMA_VERSION
    project_id: UUID
    name: str = "Untitled project"
    revision: int = Field(default=0, ge=0)
    source_asset: str | None = None
    mesh_asset: str | None = None
    stock: StockDefinition | None = None
    tools: tuple[ToolConfig, ...] = ()
    supports: tuple[Support, ...] = ()
    settings: MachiningSettings = Field(default_factory=MachiningSettings)
    machine: MachineDefinition
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("project name must not be empty")
        return normalized

    @field_validator("source_asset", "mesh_asset")
    @classmethod
    def validate_asset_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        candidate = Path(value)
        if (
            not value.strip()
            or candidate.is_absolute()
            or candidate.name != value
            or value in {".", ".."}
        ):
            raise ValueError("workspace assets must be plain file names")
        return value

    @model_validator(mode="after")
    def validate_unique_tool_numbers(self) -> WebWorkspaceDocument:
        numbers = [tool.number for tool in self.tools]
        if len(numbers) != len(set(numbers)):
            raise ValueError("tool numbers must be unique")
        return self

    @property
    def is_complete(self) -> bool:
        """Return whether the draft can materialize the canonical project schema."""

        return self.mesh_asset is not None and self.stock is not None and bool(self.tools)

    def to_project(self, project_directory: Path) -> RotaryCamProject:
        """Build a canonical project without weakening its completeness requirements."""

        if not self.is_complete or self.mesh_asset is None or self.stock is None:
            raise ValueError("mesh, stock, and at least one tool are required")
        return RotaryCamProject(
            mesh_path=(project_directory / "assets" / self.mesh_asset).resolve(),
            stock=self.stock,
            tools=list(self.tools),
            supports=list(self.supports),
            machining_settings=self.settings,
            machine=self.machine,
        )


class JobState(StrEnum):
    """Lifecycle of the single in-process CAM job for a project."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class GenerationJobSnapshot(_WebRecord):
    """Small immutable status record rendered by the HTMX polling fragment."""

    job_id: UUID
    project_id: UUID
    revision: int = Field(ge=0)
    state: JobState = JobState.QUEUED
    step: int = Field(default=0, ge=0)
    total: int = Field(default=5, gt=0)
    message: str = "Waiting to start"
    error: str | None = None

    @model_validator(mode="after")
    def validate_progress(self) -> GenerationJobSnapshot:
        if self.step > self.total:
            raise ValueError("job step must not exceed its total")
        if not self.message.strip():
            raise ValueError("job message must not be empty")
        if self.state is JobState.FAILED and not self.error:
            raise ValueError("failed jobs require an error message")
        return self


class StockSceneRecord(_WebRecord):
    """Browser-friendly analytical stock primitive."""

    kind: Literal["cylinder", "rectangle"]
    length: float = Field(gt=0.0)
    diameter: float | None = Field(default=None, gt=0.0)
    width: float | None = Field(default=None, gt=0.0)
    height: float | None = Field(default=None, gt=0.0)

    @model_validator(mode="after")
    def validate_dimensions(self) -> StockSceneRecord:
        if self.kind == "cylinder" and self.diameter is None:
            raise ValueError("cylindrical stock requires a diameter")
        if self.kind == "rectangle" and (self.width is None or self.height is None):
            raise ValueError("rectangular stock requires width and height")
        return self


class ToolpathSceneRecord(_WebRecord):
    """Metadata for one raw Float32 XYZ line-segment buffer."""

    operation_index: int = Field(ge=0)
    operation_name: str
    tool_number: int = Field(gt=0)
    tool_name: str
    strategy: str
    color: str
    buffer_url: str
    segment_count: int = Field(ge=0)
    path_count: int = Field(ge=0)
    point_count: int = Field(ge=0)


class SceneManifest(_WebRecord):
    """Revisioned description of every independently visible browser scene layer."""

    revision: int = Field(ge=0)
    target_url: str | None = None
    residual_url: str | None = None
    stock: StockSceneRecord | None = None
    supports: tuple[Support, ...] = ()
    toolpaths: tuple[ToolpathSceneRecord, ...] = ()


__all__ = [
    "WEB_WORKSPACE_SCHEMA_VERSION",
    "GenerationJobSnapshot",
    "JobState",
    "SceneManifest",
    "StockSceneRecord",
    "ToolpathSceneRecord",
    "WebWorkspaceDocument",
]
