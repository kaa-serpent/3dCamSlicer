"""Versioned JSON persistence for personal machine profiles."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator

from rotarycam.config import MachineDefinition
from rotarycam.persistence import atomic_write_text, preserve_v1_source

MACHINE_LIBRARY_SCHEMA_VERSION: Literal[2] = 2


def validate_unique_machine_names(profiles: Sequence[MachineDefinition]) -> None:
    """Reject ambiguous profile names, ignoring case and surrounding whitespace."""

    normalized_names = [profile.name.strip().casefold() for profile in profiles]
    if len(normalized_names) != len(set(normalized_names)):
        raise ValueError("machine profile names must be unique (ignoring case)")


class MachineLibraryDocument(BaseModel):
    """Portable schema envelope supporting future migrations."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[2] = MACHINE_LIBRARY_SCHEMA_VERSION
    profiles: tuple[MachineDefinition, ...] = ()

    @model_validator(mode="after")
    def validate_unique_names(self) -> MachineLibraryDocument:
        validate_unique_machine_names(self.profiles)
        return self


class MachineLibraryDocumentV1(BaseModel):
    """Read-only compatibility DTO for machine-library schema v1."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    profiles: tuple[MachineDefinition, ...] = ()

    @model_validator(mode="after")
    def validate_unique_names(self) -> MachineLibraryDocumentV1:
        validate_unique_machine_names(self.profiles)
        return self


class _LoadedMachineProfiles(list[MachineDefinition]):
    """List-compatible result carrying non-persisted migration provenance."""

    def __init__(
        self,
        profiles: Sequence[MachineDefinition],
        migrated_v1_source: Path | None = None,
    ) -> None:
        super().__init__(profiles)
        self.migrated_v1_source = migrated_v1_source


def migrate_machine_library_v1(
    document: MachineLibraryDocumentV1,
) -> tuple[MachineDefinition, ...]:
    """Invalidate v1 verification and expose all unknown XYZA inputs as missing."""

    return tuple(
        profile.model_copy(
            update={
                "profile_verified": False,
                "y_limits": None,
                "xyza_configuration": None,
                "machine_assembly_configured": False,
            }
        )
        for profile in document.profiles
    )


def default_machine_library_path() -> Path:
    """Return the current user's personal RotaryCAM machine-library path."""

    app_data = os.environ.get("APPDATA")
    base = Path(app_data) if app_data else Path.home() / ".config"
    return base / "RotaryCAM" / "machines.json"


def save_machine_library(profiles: Sequence[MachineDefinition], path: Path) -> None:
    """Atomically save validated machine profiles as UTF-8 JSON."""

    validate_unique_machine_names(profiles)
    document = MachineLibraryDocument(profiles=tuple(profiles))
    destination = path.resolve()
    if isinstance(profiles, _LoadedMachineProfiles) and profiles.migrated_v1_source is not None:
        preserve_v1_source(profiles.migrated_v1_source)
    atomic_write_text(destination, document.model_dump_json(indent=2) + "\n")
    if isinstance(profiles, _LoadedMachineProfiles):
        profiles.migrated_v1_source = None


def load_machine_library(path: Path) -> list[MachineDefinition]:
    """Load and validate all profiles from a versioned JSON document."""

    source = path.resolve()
    raw: Any = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("machine library must be a JSON object")
    version = raw.get("schema_version")
    if version == 1:
        migrated = migrate_machine_library_v1(MachineLibraryDocumentV1.model_validate(raw))
        return _LoadedMachineProfiles(migrated, source)
    if version == MACHINE_LIBRARY_SCHEMA_VERSION:
        document = MachineLibraryDocument.model_validate(raw)
        return _LoadedMachineProfiles(document.profiles)
    raise ValueError(f"unsupported machine-library schema_version: {version!r}")


__all__ = [
    "MACHINE_LIBRARY_SCHEMA_VERSION",
    "MachineLibraryDocument",
    "MachineLibraryDocumentV1",
    "default_machine_library_path",
    "load_machine_library",
    "migrate_machine_library_v1",
    "save_machine_library",
    "validate_unique_machine_names",
]
