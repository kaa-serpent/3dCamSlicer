from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import trimesh

from rotarycam.geometry import (
    MeshLoadError,
    UndercutStatus,
    load_mesh,
    normalize_mesh,
    validate_mesh,
)


def test_load_binary_stl(tmp_path: Path) -> None:
    source = trimesh.creation.box(extents=(3.0, 4.0, 5.0))
    path = tmp_path / "box.stl"
    path.write_bytes(source.export(file_type="stl"))
    loaded = load_mesh(path)
    assert loaded.extents == pytest.approx([3.0, 4.0, 5.0])
    assert loaded.is_watertight


def test_load_ascii_stl(tmp_path: Path) -> None:
    source = trimesh.creation.icosphere(subdivisions=1, radius=2.0)
    path = tmp_path / "sphere.stl"
    exported = trimesh.exchange.stl.export_stl_ascii(source)
    path.write_text(exported, encoding="ascii")
    assert load_mesh(path).is_watertight


def test_load_obj_and_merge_scene_parts(tmp_path: Path) -> None:
    scene = trimesh.Scene()
    scene.add_geometry(trimesh.creation.box(extents=(2.0, 2.0, 2.0)), geom_name="left")
    scene.add_geometry(
        trimesh.creation.box(extents=(2.0, 2.0, 2.0)),
        geom_name="right",
        transform=trimesh.transformations.translation_matrix((4.0, 0.0, 0.0)),
    )
    path = tmp_path / "parts.obj"
    exported = scene.export(file_type="obj")
    path.write_text(exported, encoding="utf-8")
    loaded = load_mesh(path)
    assert len(loaded.faces) == 24
    assert loaded.extents[0] == pytest.approx(6.0)


def test_normalize_mesh_is_non_mutating_and_removes_degenerate_face() -> None:
    mesh = trimesh.creation.box()
    faces = np.vstack((mesh.faces, [0, 0, 0]))
    dirty = trimesh.Trimesh(vertices=mesh.vertices.copy(), faces=faces, process=False)
    before = dirty.faces.copy()
    normalized = normalize_mesh(dirty)
    assert np.array_equal(dirty.faces, before)
    assert len(normalized.faces) == len(mesh.faces)


def test_validation_does_not_claim_undercut_evaluation() -> None:
    report = validate_mesh(trimesh.creation.box())
    assert report.radial_undercuts is UndercutStatus.NOT_EVALUATED
    assert not report.cam_compatible


def test_load_mesh_rejects_unknown_extension(tmp_path: Path) -> None:
    path = tmp_path / "mesh.ply"
    path.write_text("ply", encoding="ascii")
    with pytest.raises(MeshLoadError, match="Only STL and OBJ"):
        load_mesh(path)
