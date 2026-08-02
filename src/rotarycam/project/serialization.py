"""Load and save versioned RotaryCAM project documents."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from rotarycam.persistence import atomic_write_text, preserve_v1_source
from rotarycam.project.schema import (
    PROJECT_SCHEMA_VERSION,
    RotaryCamProject,
    RotaryCamProjectV1,
    migrate_project_v1,
)


def _relative_mesh_path(mesh_path: Path, project_directory: Path) -> Path:
    if not mesh_path.is_absolute():
        return mesh_path
    try:
        return Path(os.path.relpath(mesh_path, start=project_directory))
    except ValueError:
        # Windows cannot create a relative path across drive letters.
        return mesh_path


def save_project(
    project: RotaryCamProject,
    path: Path,
    *,
    relative_mesh_path: bool = True,
) -> None:
    """Serialize a project as UTF-8 JSON, optionally using a portable mesh path."""

    destination = path.resolve()
    stored_mesh_path = project.mesh_path
    if relative_mesh_path:
        stored_mesh_path = _relative_mesh_path(
            project.mesh_path.resolve(),
            destination.parent,
        )
    persisted = project.model_copy(update={"mesh_path": stored_mesh_path})
    if project._migrated_v1_source is not None:
        preserve_v1_source(project._migrated_v1_source)
    atomic_write_text(
        destination,
        persisted.model_dump_json(by_alias=True, indent=2) + "\n",
    )
    project._migrated_v1_source = None


def load_project(path: Path) -> RotaryCamProject:
    """Parse a project and resolve a relative mesh path against its directory."""

    source = path.resolve()
    raw: Any = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("project document must be a JSON object")
    version = raw.get("schema_version")
    if version == 1:
        project = migrate_project_v1(RotaryCamProjectV1.model_validate(raw))
        project._migrated_v1_source = source
    elif version == PROJECT_SCHEMA_VERSION:
        project = RotaryCamProject.model_validate(raw)
    else:
        raise ValueError(f"unsupported project schema_version: {version!r}")
    if project.mesh_path.is_absolute():
        return project
    resolved = project.model_copy(
        update={"mesh_path": (source.parent / project.mesh_path).resolve()},
    )
    resolved._migrated_v1_source = project._migrated_v1_source
    return resolved
