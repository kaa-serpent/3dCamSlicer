"""Versioned JSON persistence for personal machine profiles."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from rotarycam.config import MachineDefinition

MACHINE_LIBRARY_SCHEMA_VERSION: Literal[1] = 1


def validate_unique_machine_names(profiles: Sequence[MachineDefinition]) -> None:
    """Reject ambiguous profile names, ignoring case and surrounding whitespace."""

    normalized_names = [profile.name.strip().casefold() for profile in profiles]
    if len(normalized_names) != len(set(normalized_names)):
        raise ValueError("machine profile names must be unique (ignoring case)")


class MachineLibraryDocument(BaseModel):
    """Portable schema envelope supporting future migrations."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = MACHINE_LIBRARY_SCHEMA_VERSION
    profiles: tuple[MachineDefinition, ...] = ()

    @model_validator(mode="after")
    def validate_unique_names(self) -> MachineLibraryDocument:
        validate_unique_machine_names(self.profiles)
        return self


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
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        temporary.write_text(document.model_dump_json(indent=2) + "\n", encoding="utf-8")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_machine_library(path: Path) -> list[MachineDefinition]:
    """Load and validate all profiles from a versioned JSON document."""

    document = MachineLibraryDocument.model_validate_json(path.read_text(encoding="utf-8"))
    return list(document.profiles)


__all__ = [
    "MACHINE_LIBRARY_SCHEMA_VERSION",
    "MachineLibraryDocument",
    "default_machine_library_path",
    "load_machine_library",
    "save_machine_library",
    "validate_unique_machine_names",
]
