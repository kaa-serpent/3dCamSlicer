"""Load and save versioned RotaryCAM project documents."""

from __future__ import annotations

import os
from pathlib import Path

from rotarycam.project.schema import RotaryCamProject


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
    destination.write_text(
        persisted.model_dump_json(by_alias=True, indent=2) + "\n",
        encoding="utf-8",
    )


def load_project(path: Path) -> RotaryCamProject:
    """Parse a project and resolve a relative mesh path against its directory."""

    source = path.resolve()
    project = RotaryCamProject.model_validate_json(source.read_text(encoding="utf-8"))
    if project.mesh_path.is_absolute():
        return project
    return project.model_copy(
        update={"mesh_path": (source.parent / project.mesh_path).resolve()},
    )
