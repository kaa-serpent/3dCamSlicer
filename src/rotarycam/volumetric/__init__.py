"""Public API for conservative volumetric geometry."""

from rotarycam.volumetric.builders import (
    MeshPointClassifier,
    build_cylindrical_stock,
    build_mesh_volume,
    build_rectangular_stock,
    classify_mesh_points,
    lattice_from_bounds,
    rasterize_retention_volumes,
    ray_parity_classifier,
    trimesh_point_classifier,
)
from rotarycam.volumetric.errors import (
    InvalidSolidMeshError,
    InvalidVolumeError,
    MemoryBudgetExceeded,
    VolumetricGeometryError,
)
from rotarycam.volumetric.models import (
    BrickClassification,
    SolidVolume,
    SparseBrickStore,
    SparseVolume,
    StockVolume,
    VolumetricSettings,
    VoxelBrick,
    VoxelLattice,
)

__all__ = [
    "BrickClassification",
    "InvalidSolidMeshError",
    "InvalidVolumeError",
    "MemoryBudgetExceeded",
    "MeshPointClassifier",
    "SolidVolume",
    "SparseBrickStore",
    "SparseVolume",
    "StockVolume",
    "VolumetricGeometryError",
    "VolumetricSettings",
    "VoxelBrick",
    "VoxelLattice",
    "build_cylindrical_stock",
    "build_mesh_volume",
    "build_rectangular_stock",
    "classify_mesh_points",
    "lattice_from_bounds",
    "rasterize_retention_volumes",
    "ray_parity_classifier",
    "trimesh_point_classifier",
]
