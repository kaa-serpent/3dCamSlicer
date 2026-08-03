"""Confined, atomic persistence for local web workspaces."""

from __future__ import annotations

import os
import shutil
import threading
from pathlib import Path
from typing import BinaryIO
from uuid import UUID, uuid4

from pydantic import ValidationError

from rotarycam.config import MachineDefinition
from rotarycam.errors import RotaryCamError
from rotarycam.geometry import align_longest_axis_to_x, load_mesh
from rotarycam.web.models import WebWorkspaceDocument

MAX_MESH_UPLOAD_BYTES = 100 * 1024 * 1024


class WebStoreError(RotaryCamError):
    """Base class for expected web workspace storage failures."""


class ProjectNotFoundError(WebStoreError):
    """Raised when a requested workspace does not exist."""


class InvalidProjectIdError(WebStoreError):
    """Raised when an identifier cannot safely address a workspace directory."""


class MeshUploadError(WebStoreError):
    """Raised when an uploaded mesh is unsupported, invalid, or too large."""


def default_project_root() -> Path:
    """Return the per-user location used by the local web application."""

    app_data = os.environ.get("APPDATA")
    base = Path(app_data) if app_data else Path.home() / ".config"
    return base / "RotaryCAM" / "projects"


def _atomic_write_text(destination: Path, content: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8", newline="\n")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


class ProjectStore:
    """Persist versioned workspaces below one injectable, confined root."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    @staticmethod
    def parse_project_id(project_id: UUID | str) -> UUID:
        """Return a canonical UUID or reject path-like identifiers."""

        if isinstance(project_id, UUID):
            return project_id
        try:
            parsed = UUID(project_id)
        except (AttributeError, TypeError, ValueError) as exc:
            raise InvalidProjectIdError("Project ID must be a valid UUID.") from exc
        if str(parsed) != project_id:
            raise InvalidProjectIdError("Project ID must use canonical UUID syntax.")
        return parsed

    def project_directory(self, project_id: UUID | str, *, require: bool = True) -> Path:
        """Resolve a UUID workspace while rejecting symlink/path escapes."""

        parsed = self.parse_project_id(project_id)
        lexical = self.root / str(parsed)
        resolved = lexical.resolve()
        if resolved.parent != self.root or resolved != lexical:
            raise InvalidProjectIdError("Project directory escapes the configured root.")
        if require and not (resolved / "workspace.json").is_file():
            raise ProjectNotFoundError(f"Project does not exist: {parsed}")
        return resolved

    def create(self, name: str, machine: MachineDefinition) -> WebWorkspaceDocument:
        """Create and atomically persist a new incomplete workspace."""

        with self._lock:
            project_id = uuid4()
            directory = self.project_directory(project_id, require=False)
            directory.mkdir(parents=False, exist_ok=False)
            (directory / "assets").mkdir()
            workspace = WebWorkspaceDocument(
                project_id=project_id,
                name=name,
                machine=machine,
            )
            self.save(workspace)
            return workspace

    def load(self, project_id: UUID | str) -> WebWorkspaceDocument:
        """Load and validate one workspace, including its directory identity."""

        directory = self.project_directory(project_id)
        try:
            workspace = WebWorkspaceDocument.model_validate_json(
                (directory / "workspace.json").read_text(encoding="utf-8")
            )
        except (OSError, ValidationError) as exc:
            raise WebStoreError(f"Unable to load workspace: {exc}") from exc
        if directory.name != str(workspace.project_id):
            raise WebStoreError("Workspace project ID does not match its directory.")
        return workspace

    def list(self) -> list[WebWorkspaceDocument]:
        """Return valid projects newest-first, ignoring unrelated root entries."""

        workspaces: list[WebWorkspaceDocument] = []
        for candidate in self.root.iterdir():
            if not candidate.is_dir():
                continue
            try:
                workspaces.append(self.load(candidate.name))
            except WebStoreError:
                continue
        return sorted(workspaces, key=lambda item: item.updated_at, reverse=True)

    def save(self, workspace: WebWorkspaceDocument) -> None:
        """Atomically save a draft and materialize its canonical project if complete."""

        with self._lock:
            directory = self.project_directory(workspace.project_id, require=False)
            if not directory.is_dir():
                raise ProjectNotFoundError(f"Project does not exist: {workspace.project_id}")
            _atomic_write_text(
                directory / "workspace.json",
                workspace.model_dump_json(by_alias=True, indent=2) + "\n",
            )
            if not workspace.is_complete:
                return
            project = workspace.to_project(directory)
            portable = project.model_copy(
                update={"mesh_path": Path("assets") / str(workspace.mesh_asset)}
            )
            _atomic_write_text(
                directory / "project.json",
                portable.model_dump_json(by_alias=True, indent=2) + "\n",
            )

    def save_mesh(
        self,
        project_id: UUID | str,
        filename: str,
        source: BinaryIO,
        *,
        maximum_bytes: int = MAX_MESH_UPLOAD_BYTES,
    ) -> tuple[str, str]:
        """Validate, copy, align, and normalize an STL/OBJ upload atomically."""

        directory = self.project_directory(project_id)
        suffix = Path(filename).suffix.lower()
        if suffix not in {".stl", ".obj"}:
            raise MeshUploadError("Only STL and OBJ mesh files are supported.")
        if maximum_bytes <= 0:
            raise ValueError("maximum_bytes must be positive")
        assets = directory / "assets"
        upload = assets / f".upload-{uuid4().hex}{suffix}"
        source_name = f"source{suffix}"
        source_destination = assets / source_name
        target_destination = assets / "target.stl"
        target_temporary = assets / f".target-{uuid4().hex}.stl"
        size = 0
        try:
            with upload.open("wb") as destination:
                while chunk := source.read(1024 * 1024):
                    size += len(chunk)
                    if size > maximum_bytes:
                        raise MeshUploadError(
                            f"Mesh upload exceeds the {maximum_bytes // (1024 * 1024)} MiB limit."
                        )
                    destination.write(chunk)
            if size == 0:
                raise MeshUploadError("Uploaded mesh is empty.")
            try:
                aligned = align_longest_axis_to_x(load_mesh(upload))
                aligned.export(target_temporary, file_type="stl")
            except Exception as exc:
                if isinstance(exc, MeshUploadError):
                    raise
                raise MeshUploadError(f"Unable to import mesh: {exc}") from exc
            with self._lock:
                upload.replace(source_destination)
                target_temporary.replace(target_destination)
            return source_name, target_destination.name
        finally:
            upload.unlink(missing_ok=True)
            target_temporary.unlink(missing_ok=True)

    def asset_path(
        self,
        project_id: UUID | str,
        *parts: str,
        require: bool = True,
    ) -> Path:
        """Resolve an asset below its workspace and reject traversal or symlinks."""

        directory = self.project_directory(project_id)
        assets = (directory / "assets").resolve()
        if not parts or any(not part or Path(part).name != part for part in parts):
            raise WebStoreError("Asset path components must be plain file names.")
        candidate = assets.joinpath(*parts).resolve()
        if assets not in candidate.parents:
            raise WebStoreError("Asset path escapes the project asset directory.")
        if require and not candidate.is_file():
            raise ProjectNotFoundError("Project asset does not exist.")
        return candidate

    def copy_asset(self, project_id: UUID | str, source: Path, filename: str) -> Path:
        """Copy a trusted generated asset into the confined project asset directory."""

        destination = self.asset_path(project_id, filename, require=False)
        temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
        try:
            shutil.copyfile(source, temporary)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination


__all__ = [
    "MAX_MESH_UPLOAD_BYTES",
    "InvalidProjectIdError",
    "MeshUploadError",
    "ProjectNotFoundError",
    "ProjectStore",
    "WebStoreError",
    "default_project_root",
]
