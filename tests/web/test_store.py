from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import pytest
import trimesh

from rotarycam.machine.profiles import makera_z1_community_profile
from rotarycam.project import CylindricalStockConfig, ToolConfig
from rotarycam.web.models import WebWorkspaceDocument
from rotarycam.web.store import InvalidProjectIdError, MeshUploadError, ProjectStore


def _tool() -> ToolConfig:
    return ToolConfig(
        number=1,
        name="Flat 2",
        type="flat",
        diameter=2,
        cutting_length=5,
        flute_length=5,
        overall_length=20,
        shank_diameter=2,
        max_stepdown=1,
        stepover=1,
        feed=200,
        plunge_feed=80,
        spindle_rpm=10_000,
    )


def _mesh_bytes() -> bytes:
    mesh = trimesh.creation.box(extents=(3.0, 9.0, 4.0))
    exported = mesh.export(file_type="stl")
    assert isinstance(exported, bytes)
    return exported


def test_store_confines_uuid_paths_and_round_trips_workspace(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "projects")
    created = store.create("  Spindle test  ", makera_z1_community_profile())

    assert created.name == "Spindle test"
    assert store.load(created.project_id) == created
    assert store.list() == [created]
    with pytest.raises(InvalidProjectIdError):
        store.project_directory("../outside")
    with pytest.raises(InvalidProjectIdError):
        store.project_directory(str(created.project_id).upper())


def test_mesh_upload_copies_source_and_writes_aligned_target(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "projects")
    workspace = store.create("Mesh", makera_z1_community_profile())

    source_name, target_name = store.save_mesh(
        workspace.project_id,
        "unsafe-name.STL",
        BytesIO(_mesh_bytes()),
    )

    assert source_name == "source.stl"
    assert target_name == "target.stl"
    target = trimesh.load_mesh(store.asset_path(workspace.project_id, target_name))
    assert target.extents[0] == pytest.approx(9.0)
    assert target.bounds[0, 0] == pytest.approx(0.0)
    assert target.centroid[1:] == pytest.approx([0.0, 0.0])


def test_mesh_upload_rejects_format_empty_and_size_limit(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "projects")
    workspace = store.create("Mesh", makera_z1_community_profile())

    with pytest.raises(MeshUploadError, match="Only STL and OBJ"):
        store.save_mesh(workspace.project_id, "mesh.3mf", BytesIO(b"data"))
    with pytest.raises(MeshUploadError, match="empty"):
        store.save_mesh(workspace.project_id, "mesh.stl", BytesIO())
    with pytest.raises(MeshUploadError, match="exceeds"):
        store.save_mesh(
            workspace.project_id,
            "mesh.stl",
            BytesIO(b"too large"),
            maximum_bytes=2,
        )


def test_complete_workspace_atomically_materializes_canonical_project(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "projects")
    created = store.create("Complete", makera_z1_community_profile())
    _source, mesh_asset = store.save_mesh(
        created.project_id,
        "mesh.stl",
        BytesIO(_mesh_bytes()),
    )
    complete = WebWorkspaceDocument.model_validate(
        created.model_copy(
            update={
                "mesh_asset": mesh_asset,
                "stock": CylindricalStockConfig(length=9, diameter=12),
                "tools": (_tool(),),
            }
        ).model_dump(by_alias=True)
    )

    store.save(complete)

    directory = store.project_directory(created.project_id)
    project_json = json.loads((directory / "project.json").read_text(encoding="utf-8"))
    assert project_json["mesh_path"].replace("\\", "/") == "assets/target.stl"
    assert not list(directory.rglob("*.tmp"))
