"""STL/OBJ loading with deterministic, non-mutating cleanup."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from rotarycam.geometry.errors import MeshLoadError

SUPPORTED_MESH_EXTENSIONS = frozenset({".stl", ".obj"})


def _scene_to_mesh(scene: trimesh.Scene) -> trimesh.Trimesh:
    dumped = scene.to_geometry()
    if isinstance(dumped, trimesh.Trimesh):
        return dumped
    raise MeshLoadError("The scene contains no triangle geometry.")


def normalize_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Return a cleaned mesh copy without changing its units or placement."""

    if not isinstance(mesh, trimesh.Trimesh):
        raise MeshLoadError("Loaded geometry is not a triangle mesh.")
    result = mesh.copy()
    if result.vertices.size == 0 or result.faces.size == 0:
        raise MeshLoadError("The mesh is empty.")

    vertices = np.asarray(result.vertices, dtype=np.float64)
    faces = np.asarray(result.faces)
    if not np.all(np.isfinite(vertices)):
        raise MeshLoadError("The mesh contains non-finite vertices.")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise MeshLoadError("The mesh must contain triangular faces.")

    result.process(validate=True)
    result.remove_unreferenced_vertices()
    if result.vertices.size == 0 or result.faces.size == 0:
        raise MeshLoadError("Mesh cleanup removed all geometry.")
    try:
        result.fix_normals(multibody=True)
    except TypeError:  # pragma: no cover - compatibility with older Trimesh
        result.fix_normals()

    extents = np.asarray(result.extents, dtype=np.float64)
    if extents.shape != (3,) or not np.all(np.isfinite(extents)) or np.any(extents <= 0.0):
        raise MeshLoadError("The mesh must have finite, positive dimensions on X, Y, and Z.")
    return result


def load_mesh(path: Path) -> trimesh.Trimesh:
    """Load and normalize a millimetre-based STL or OBJ mesh.

    Scene graph transformations are baked into OBJ geometry before all parts are
    concatenated. The source file and any object returned by Trimesh are never
    mutated by callers of this function.
    """

    mesh_path = Path(path)
    if mesh_path.suffix.lower() not in SUPPORTED_MESH_EXTENSIONS:
        raise MeshLoadError("Only STL and OBJ mesh files are supported.")
    if not mesh_path.is_file():
        raise MeshLoadError(f"Mesh file does not exist: {mesh_path}")
    try:
        loaded = trimesh.load(mesh_path, process=False, force="scene")
    except Exception as exc:  # Trimesh uses several format-specific exception types
        raise MeshLoadError(f"Unable to load mesh '{mesh_path}': {exc}") from exc

    if isinstance(loaded, trimesh.Scene):
        mesh = _scene_to_mesh(loaded)
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        raise MeshLoadError("The file contains no supported triangle geometry.")
    return normalize_mesh(mesh)
