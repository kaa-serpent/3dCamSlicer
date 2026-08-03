"""Conservative analytical and triangle-mesh volume builders."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt
import trimesh

from rotarycam.stock.cylindrical import CylindricalStock
from rotarycam.stock.rectangular import RectangularStock
from rotarycam.supports.models import (
    CylindricalSurfaceRetention,
    RectangularSurfaceRetention,
    RetentionVolume,
)
from rotarycam.volumetric.errors import InvalidSolidMeshError, InvalidVolumeError
from rotarycam.volumetric.models import (
    BoolArray,
    Point3,
    SolidVolume,
    SparseBrickStore,
    SparseVolume,
    StockVolume,
    VolumetricSettings,
    VoxelLattice,
)

FloatArray = npt.NDArray[np.float64]
MeshPointClassifier = Callable[[trimesh.Trimesh, FloatArray], npt.ArrayLike]


def lattice_from_bounds(
    lower: Point3,
    upper: Point3,
    settings: VolumetricSettings,
    *,
    working_bytes_per_voxel: int = 1,
) -> VoxelLattice:
    """Cover finite bounds at the requested tolerance or reject the memory cost."""

    low = np.asarray(lower, dtype=np.float64)
    high = np.asarray(upper, dtype=np.float64)
    if low.shape != (3,) or high.shape != (3,):
        raise InvalidVolumeError("volume bounds must contain three coordinates")
    if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)):
        raise InvalidVolumeError("volume bounds must be finite")
    extents = high - low
    if np.any(extents <= 0.0):
        raise InvalidVolumeError("upper volume bounds must exceed lower bounds")
    spacing = settings.tolerance
    shape_array = np.ceil(extents / spacing).astype(np.int64)
    shape = tuple(int(value) for value in shape_array)
    origin = tuple(float(value) for value in low + spacing * 0.5)
    lattice = VoxelLattice(origin, (spacing, spacing, spacing), shape)  # type: ignore[arg-type]
    lattice.require_memory_budget(
        settings.memory_budget_bytes,
        bytes_per_voxel=working_bytes_per_voxel,
    )
    return lattice


def _centres(lattice: VoxelLattice) -> FloatArray:
    axes = tuple(lattice.axis_centres(axis) for axis in range(3))
    grid = np.meshgrid(*axes, indexing="ij")
    return np.column_stack(tuple(values.ravel() for values in grid))


def build_rectangular_stock(
    stock: RectangularStock,
    settings: VolumetricSettings,
) -> StockVolume:
    """Rasterize an X-aligned rectangular stock with conservative boundary cells."""

    lattice = lattice_from_bounds(
        (0.0, -stock.width / 2.0, -stock.height / 2.0),
        (stock.length, stock.width / 2.0, stock.height / 2.0),
        settings,
    )
    occupation = np.ones(lattice.shape, dtype=np.bool_)
    return StockVolume.from_dense(lattice, occupation, brick_size=settings.brick_size)


def build_cylindrical_stock(
    stock: CylindricalStock,
    settings: VolumetricSettings,
) -> StockVolume:
    """Rasterize an X-aligned cylinder, retaining every boundary-intersecting cell."""

    radius = stock.diameter / 2.0
    lattice = lattice_from_bounds(
        (0.0, -radius, -radius),
        (stock.length, radius, radius),
        settings,
    )
    y = lattice.axis_centres(1)[:, None]
    z = lattice.axis_centres(2)[None, :]
    half_y = lattice.spacing[1] * 0.5
    half_z = lattice.spacing[2] * 0.5
    closest_y = np.maximum(np.abs(y) - half_y, 0.0)
    closest_z = np.maximum(np.abs(z) - half_z, 0.0)
    radial_intersection = closest_y * closest_y + closest_z * closest_z <= radius * radius
    occupation = np.broadcast_to(radial_intersection, lattice.shape).copy()
    return StockVolume.from_dense(lattice, occupation, brick_size=settings.brick_size)


def ray_parity_classifier(mesh: trimesh.Trimesh, points: FloatArray) -> BoolArray:
    """Classify points by deterministic ray parity without optional spatial indexes."""

    query = np.asarray(points, dtype=np.float64)
    if query.ndim != 2 or query.shape[1] != 3 or not np.all(np.isfinite(query)):
        raise InvalidVolumeError("classifier points must have finite shape (N, 3)")
    triangles = np.asarray(mesh.triangles, dtype=np.float64)
    direction = np.asarray((1.0, 0.3713906763541037, 0.52999894000318), dtype=np.float64)
    direction /= np.linalg.norm(direction)
    counts = np.zeros(len(query), dtype=np.uint32)
    epsilon = np.finfo(np.float64).eps * 64.0
    for triangle in triangles:
        edge_1 = triangle[1] - triangle[0]
        edge_2 = triangle[2] - triangle[0]
        h = np.cross(direction, edge_2)
        determinant = float(np.dot(edge_1, h))
        if abs(determinant) <= epsilon:
            continue
        inverse = 1.0 / determinant
        offset = query - triangle[0]
        u = inverse * (offset @ h)
        cross = np.cross(offset, edge_1)
        v = inverse * (cross @ direction)
        distance = inverse * (cross @ edge_2)
        hit = (
            (u >= -epsilon)
            & (v >= -epsilon)
            & (u + v <= 1.0 + epsilon)
            & (distance > epsilon)
        )
        counts += hit
    result = np.asarray(counts % 2 == 1, dtype=np.bool_)
    result.setflags(write=False)
    return result


def trimesh_point_classifier(mesh: trimesh.Trimesh, points: FloatArray) -> BoolArray:
    """Use Trimesh containment when available and fall back to pure ray parity."""

    try:
        result = np.asarray(mesh.contains(points), dtype=np.bool_)
    except (ImportError, ModuleNotFoundError):
        return ray_parity_classifier(mesh, points)
    if result.shape != (len(points),):
        raise InvalidVolumeError("mesh classifier returned an invalid result shape")
    result = result.copy()
    result.setflags(write=False)
    return result


def classify_mesh_points(
    components: Sequence[trimesh.Trimesh],
    points: npt.ArrayLike,
    *,
    classifier: MeshPointClassifier = trimesh_point_classifier,
) -> BoolArray:
    """Combine disjoint shells by union and nested shells by even/odd parity."""

    query = np.array(points, dtype=np.float64, copy=True)
    if query.ndim != 2 or query.shape[1] != 3 or not np.all(np.isfinite(query)):
        raise InvalidVolumeError("classifier points must have finite shape (N, 3)")
    classified_components: list[BoolArray] = []
    depths: list[int] = []
    for index, component in enumerate(components):
        classified = np.asarray(classifier(component, query), dtype=np.bool_)
        if classified.shape != (len(query),):
            raise InvalidVolumeError("mesh classifier returned an invalid result shape")
        classified_components.append(classified)

        child_bounds = np.asarray(component.bounds, dtype=np.float64)
        child_vertices = np.asarray(component.vertices, dtype=np.float64)
        depth = 0
        for parent_index, parent in enumerate(components):
            if parent_index == index:
                continue
            parent_bounds = np.asarray(parent.bounds, dtype=np.float64)
            if not (
                np.all(child_bounds[0] > parent_bounds[0])
                and np.all(child_bounds[1] < parent_bounds[1])
            ):
                continue
            contained_vertices = np.asarray(classifier(parent, child_vertices), dtype=np.bool_)
            if contained_vertices.shape == (len(child_vertices),) and bool(
                np.all(contained_vertices)
            ):
                depth += 1
        depths.append(depth)

    deepest = np.full(len(query), -1, dtype=np.int64)
    for classified, depth in zip(classified_components, depths, strict=True):
        deepest[classified] = np.maximum(deepest[classified], depth)
    inside = np.asarray((deepest >= 0) & (deepest % 2 == 0), dtype=np.bool_)
    inside.setflags(write=False)
    return inside


def _validate_solid_mesh(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, ...]:
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise InvalidSolidMeshError("a non-empty triangle mesh is required")
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces)
    if not np.all(np.isfinite(vertices)) or faces.ndim != 2 or faces.shape[1] != 3:
        raise InvalidSolidMeshError("mesh vertices and triangular faces must be finite")
    if not bool(mesh.is_watertight):
        raise InvalidSolidMeshError("mesh must be watertight before volumetric planning")
    if not bool(mesh.is_winding_consistent):
        raise InvalidSolidMeshError("mesh winding must be consistent before volumetric planning")
    try:
        components = tuple(mesh.split(only_watertight=False))
    except Exception as exc:
        raise InvalidSolidMeshError("mesh components could not be separated") from exc
    if not components:
        raise InvalidSolidMeshError("mesh contains no solid components")
    return components


def _triangle_surface_guard(mesh: trimesh.Trimesh, lattice: VoxelLattice) -> BoolArray:
    """Mark every cell whose AABB overlaps a triangle AABB (conservative superset)."""

    guard = np.zeros(lattice.shape, dtype=np.bool_)
    lower, _ = lattice.bounds
    low = np.asarray(lower, dtype=np.float64)
    spacing = np.asarray(lattice.spacing, dtype=np.float64)
    maximum = np.asarray(lattice.shape, dtype=np.int64) - 1
    triangles = np.asarray(mesh.triangles, dtype=np.float64)
    for triangle in triangles:
        first = np.floor((np.min(triangle, axis=0) - low) / spacing).astype(np.int64)
        last = np.floor((np.max(triangle, axis=0) - low) / spacing).astype(np.int64)
        first = np.clip(first, 0, maximum)
        last = np.clip(last, 0, maximum)
        guard[
            first[0] : last[0] + 1,
            first[1] : last[1] + 1,
            first[2] : last[2] + 1,
        ] = True
    guard.setflags(write=False)
    return guard


def build_mesh_volume(
    mesh: trimesh.Trimesh,
    settings: VolumetricSettings,
    *,
    classifier: MeshPointClassifier = trimesh_point_classifier,
) -> SolidVolume:
    """Rasterize closed shells using component parity and a conservative surface guard."""

    components = _validate_solid_mesh(mesh)
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    lattice = lattice_from_bounds(
        tuple(float(value) for value in bounds[0]),  # type: ignore[arg-type]
        tuple(float(value) for value in bounds[1]),  # type: ignore[arg-type]
        settings,
        working_bytes_per_voxel=128 + len(components),
    )
    points = _centres(lattice)
    inside = classify_mesh_points(components, points, classifier=classifier).reshape(lattice.shape)
    guard = _triangle_surface_guard(mesh, lattice)
    occupation = np.asarray(inside | guard, dtype=np.bool_)
    store = SparseBrickStore.from_dense(
        lattice,
        occupation,
        brick_shape=settings.brick_shape,
    )
    return SolidVolume(lattice, store, guard)


def build_mesh_volume_on_lattice(
    mesh: trimesh.Trimesh,
    lattice: VoxelLattice,
    *,
    brick_size: int = 8,
    memory_budget_bytes: int | None = None,
    classifier: MeshPointClassifier = trimesh_point_classifier,
) -> SolidVolume:
    """Rasterize a closed mesh on an existing stock lattice.

    Sharing the exact lattice is required by volumetric planning and simulation;
    geometry outside the lattice is rejected by the engine's containment checks.
    The optional budget covers the dense classifier working set and is never
    traded for a coarser tolerance.
    """

    components = _validate_solid_mesh(mesh)
    if memory_budget_bytes is not None:
        lattice.require_memory_budget(
            memory_budget_bytes,
            bytes_per_voxel=128 + len(components),
        )
    points = _centres(lattice)
    inside = classify_mesh_points(components, points, classifier=classifier).reshape(
        lattice.shape
    )
    guard = _triangle_surface_guard(mesh, lattice)
    occupation = np.asarray(inside | guard, dtype=np.bool_)
    store = SparseBrickStore.from_dense(
        lattice,
        occupation,
        brick_shape=(brick_size, brick_size, brick_size),
    )
    return SolidVolume(lattice, store, guard)


def rasterize_retention_volumes(
    target: SparseVolume,
    retentions: Sequence[RetentionVolume],
    *,
    brick_size: int | None = None,
) -> SparseVolume:
    """Rasterize enabled v2 target-relative retention definitions on target rays."""

    lattice = target.lattice
    centres = _centres(lattice)
    occupied_centres = centres[target.to_dense().ravel()]
    retained = np.zeros(lattice.shape, dtype=np.bool_)
    if len(occupied_centres) == 0:
        return SparseVolume.from_dense(
            lattice,
            retained,
            brick_size=brick_size or target.brick_shape[0],
        )
    half_diagonal = math.sqrt(sum(value * value for value in lattice.spacing)) * 0.5
    x_values = centres[:, 0]
    yz_values = centres[:, 1:]
    for retention in retentions:
        if not retention.enabled:
            continue
        angle = math.radians(retention.angle_deg)
        radial = np.asarray((math.cos(angle), math.sin(angle)), dtype=np.float64)
        tangent = np.asarray((-math.sin(angle), math.cos(angle)), dtype=np.float64)
        occupied_projection = occupied_centres[:, 1:] @ radial
        x_distance = np.abs(occupied_centres[:, 0] - retention.x)
        local = x_distance <= max(lattice.spacing[0], half_diagonal)
        if np.any(local):
            anchor = float(np.max(occupied_projection[local]))
        else:
            nearest = np.min(x_distance)
            anchor = float(np.max(occupied_projection[x_distance <= nearest + 1e-12]))
        anchor += 0.5 * (
            abs(radial[0]) * lattice.spacing[1] + abs(radial[1]) * lattice.spacing[2]
        )
        longitudinal = x_values - retention.x
        tangential = yz_values @ tangent
        outward = yz_values @ radial - anchor
        radial_hit = (
            (outward >= -half_diagonal)
            & (outward <= retention.radial_thickness + half_diagonal)
        )
        if isinstance(retention, RectangularSurfaceRetention):
            footprint = (
                np.abs(longitudinal)
                <= retention.length_x / 2.0 + retention.transition + half_diagonal
            ) & (
                np.abs(tangential)
                <= retention.width_surface / 2.0 + retention.transition + half_diagonal
            )
        elif isinstance(retention, CylindricalSurfaceRetention):
            footprint_radius = retention.diameter / 2.0 + retention.transition + half_diagonal
            footprint = longitudinal * longitudinal + tangential * tangential <= (
                footprint_radius * footprint_radius
            )
        else:  # pragma: no cover - closed Pydantic union, defensive at domain boundary
            raise InvalidVolumeError(f"unsupported retention type: {type(retention).__name__}")
        retained.ravel()[footprint & radial_hit] = True
    return SparseVolume.from_dense(
        lattice,
        retained,
        brick_size=brick_size or target.brick_shape[0],
    )
