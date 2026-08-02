"""Batched ray/triangle sampling and strict radial-solid certification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import trimesh

from rotarycam.geometry.errors import RadialCompatibilityError
from rotarycam.geometry.mesh_validation import MeshValidationReport, validate_mesh
from rotarycam.geometry.models import RotaryGrid, UndercutStatus


@dataclass(frozen=True, slots=True)
class RadialSamplingResult:
    """A sampled radius grid paired with its topology-aware certification."""

    grid: RotaryGrid
    report: MeshValidationReport
    intersection_counts: npt.NDArray[np.int32]


def _sample_axes(
    mesh: trimesh.Trimesh,
    x_step: float,
    angle_step_deg: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    if not np.isfinite(x_step) or x_step <= 0.0:
        raise ValueError("x_step must be finite and positive")
    if not np.isfinite(angle_step_deg) or angle_step_deg <= 0.0 or angle_step_deg > 360.0:
        raise ValueError("angle_step_deg must lie in (0, 360]")
    x_min, x_max = (float(value) for value in np.asarray(mesh.bounds)[:, 0])
    x_values = np.arange(x_min, x_max, x_step, dtype=np.float64)
    if x_values.size == 0 or not np.isclose(x_values[-1], x_max):
        x_values = np.append(x_values, x_max)
    else:
        x_values[-1] = x_max
    angles = np.arange(0.0, 360.0, angle_step_deg, dtype=np.float64)
    return x_values, angles


def _deduplicated_positive(values: list[float], tolerance: float) -> npt.NDArray[np.float64]:
    if not values:
        return np.empty(0, dtype=np.float64)
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    ordered = ordered[ordered > tolerance]
    if ordered.size < 2:
        return ordered
    keep = np.concatenate(([True], np.diff(ordered) > tolerance))
    return ordered[keep]


def _batched_intersections(
    triangles: npt.NDArray[np.float64],
    origins: npt.NDArray[np.float64],
    directions: npt.NDArray[np.float64],
    *,
    ray_batch_size: int,
    triangle_batch_size: int,
    tolerance: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int32]]:
    """Return outer radii and unique positive boundary counts per ray."""

    ray_count = len(origins)
    outer = np.zeros(ray_count, dtype=np.float64)
    counts = np.zeros(ray_count, dtype=np.int32)
    edge_1_all = triangles[:, 1] - triangles[:, 0]
    edge_2_all = triangles[:, 2] - triangles[:, 0]

    for ray_start in range(0, ray_count, ray_batch_size):
        ray_stop = min(ray_start + ray_batch_size, ray_count)
        batch_origins = origins[ray_start:ray_stop]
        batch_directions = directions[ray_start:ray_stop]
        hits: list[list[float]] = [[] for _ in range(ray_stop - ray_start)]

        for triangle_start in range(0, len(triangles), triangle_batch_size):
            triangle_stop = min(triangle_start + triangle_batch_size, len(triangles))
            vertices = triangles[triangle_start:triangle_stop, 0]
            edge_1 = edge_1_all[triangle_start:triangle_stop]
            edge_2 = edge_2_all[triangle_start:triangle_stop]

            direction_grid = batch_directions[:, None, :]
            h = np.cross(direction_grid, edge_2[None, :, :])
            determinant = np.einsum("tj,rtj->rt", edge_1, h)
            non_parallel = np.abs(determinant) > tolerance
            inverse = np.zeros_like(determinant)
            inverse[non_parallel] = 1.0 / determinant[non_parallel]
            s = batch_origins[:, None, :] - vertices[None, :, :]
            u = inverse * np.einsum("rtj,rtj->rt", s, h)
            q = np.cross(s, edge_1[None, :, :])
            v = inverse * np.einsum("rj,rtj->rt", batch_directions, q)
            distance = inverse * np.einsum("tj,rtj->rt", edge_2, q)
            hit_mask = (
                non_parallel
                & (u >= -tolerance)
                & (v >= -tolerance)
                & (u + v <= 1.0 + tolerance)
                & (distance > tolerance)
            )
            for hit_ray_index in np.flatnonzero(np.any(hit_mask, axis=1)):
                hits[int(hit_ray_index)].extend(
                    distance[hit_ray_index, hit_mask[hit_ray_index]].tolist()
                )

        for local_index, ray_hits in enumerate(hits):
            unique = _deduplicated_positive(ray_hits, tolerance * 8.0)
            index = ray_start + local_index
            counts[index] = len(unique)
            if unique.size:
                outer[index] = float(unique[-1])
    return outer, counts


def sample_mesh_radially_with_report(
    mesh: trimesh.Trimesh,
    x_step: float,
    angle_step_deg: float,
    *,
    strict: bool = True,
    ray_batch_size: int = 128,
    triangle_batch_size: int = 2048,
) -> RadialSamplingResult:
    """Sample a mesh and certify one material boundary from the rotary axis.

    In strict mode, open meshes, inconsistent winding, multiple components,
    missing intersections and rays with more than one boundary all block CAM.
    ``strict=False`` retains the outermost intersection for inspection while the
    structured report explains why it is not safe for toolpath generation.
    """

    if ray_batch_size <= 0 or triangle_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    base_report = validate_mesh(mesh)
    if base_report.errors:
        raise RadialCompatibilityError("; ".join(base_report.errors))

    x_values, angles = _sample_axes(mesh, x_step, angle_step_deg)
    x_grid, angle_grid = np.meshgrid(x_values, np.radians(angles), indexing="ij")
    origins = np.column_stack((x_grid.ravel(), np.zeros(x_grid.size), np.zeros(x_grid.size)))
    directions = np.column_stack(
        (np.zeros(angle_grid.size), np.cos(angle_grid).ravel(), np.sin(angle_grid).ravel())
    )
    triangles = np.asarray(mesh.triangles, dtype=np.float64)
    scale = max(float(np.max(np.asarray(mesh.extents, dtype=np.float64))), 1.0)
    tolerance = max(np.finfo(np.float64).eps * scale * 128.0, 1e-10)
    outer, counts_flat = _batched_intersections(
        triangles,
        origins,
        directions,
        ray_batch_size=ray_batch_size,
        triangle_batch_size=triangle_batch_size,
        tolerance=tolerance,
    )
    shape = (len(x_values), len(angles))
    counts = counts_flat.reshape(shape)
    radius = outer.reshape(shape)
    valid = counts > 0
    has_multiple = bool(np.any(counts > 1))
    has_missing = bool(np.any(counts == 0))

    topology_uncertain = (
        not base_report.is_watertight
        or not base_report.is_winding_consistent
        or base_report.has_multiple_components
    )
    if has_multiple:
        status = UndercutStatus.PRESENT
    elif topology_uncertain or has_missing:
        status = UndercutStatus.INDETERMINATE
    else:
        status = UndercutStatus.ABSENT

    warnings: list[str] = []
    if has_missing:
        missing_count = int(np.count_nonzero(counts == 0))
        warnings.append(f"{missing_count} radial cells have no mesh intersection.")
    if has_multiple:
        warnings.append(
            f"{int(np.count_nonzero(counts > 1))} radial cells cross multiple material boundaries."
        )
    grid = RotaryGrid(x_values, angles, radius, valid, status, tuple(warnings))
    report = validate_mesh(mesh, radial_undercuts=status)
    result = RadialSamplingResult(grid, report, counts)
    if strict and not report.cam_compatible:
        details = report.errors + report.warnings + grid.warnings
        raise RadialCompatibilityError("Mesh is not a certified radial solid: " + " ".join(details))
    return result


def sample_mesh_radially(
    mesh: trimesh.Trimesh,
    x_step: float,
    angle_step_deg: float,
    *,
    strict: bool = True,
    ray_batch_size: int = 128,
    triangle_batch_size: int = 2048,
) -> RotaryGrid:
    """Return the radial grid; use the report variant for inspection details."""

    return sample_mesh_radially_with_report(
        mesh,
        x_step,
        angle_step_deg,
        strict=strict,
        ray_batch_size=ray_batch_size,
        triangle_batch_size=triangle_batch_size,
    ).grid
