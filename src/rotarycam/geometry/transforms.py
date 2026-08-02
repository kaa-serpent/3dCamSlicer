"""Coordinate and immutable mesh transformations for an X rotary axis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt
import trimesh


class _StockLike(Protocol):
    @property
    def length(self) -> float: ...

    def radius_at(self, x: float, angle_deg: float) -> float: ...


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Result of a non-mutating geometric containment check."""

    valid: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


def xyz_to_xar(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Convert XYZ to longitudinal X, rotary angle A (degrees), and radius R."""

    values = np.asarray((x, y, z), dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("XYZ coordinates must be finite")
    angle_deg = float(np.degrees(np.arctan2(z, y)) % 360.0)
    radius = float(np.hypot(y, z))
    return float(x), angle_deg, radius


def xyz_array_to_xar(points: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Vectorized :func:`xyz_to_xar` for an array with final dimension three."""

    xyz = np.asarray(points, dtype=np.float64)
    if xyz.ndim < 1 or xyz.shape[-1] != 3:
        raise ValueError("points must have a final dimension of size 3")
    if not np.all(np.isfinite(xyz)):
        raise ValueError("XYZ coordinates must be finite")
    result = np.empty_like(xyz, dtype=np.float64)
    result[..., 0] = xyz[..., 0]
    result[..., 1] = np.degrees(np.arctan2(xyz[..., 2], xyz[..., 1])) % 360.0
    result[..., 2] = np.hypot(xyz[..., 1], xyz[..., 2])
    return result


def xar_to_xyz(x: float, angle_deg: float, radius: float) -> tuple[float, float, float]:
    """Convert X/A/R to XYZ using A0=+Y and increasing A toward +Z."""

    values = np.asarray((x, angle_deg, radius), dtype=np.float64)
    if not np.all(np.isfinite(values)) or radius < 0.0:
        raise ValueError("X/A/R coordinates must be finite and radius non-negative")
    angle_rad = np.radians(angle_deg)
    return float(x), float(radius * np.cos(angle_rad)), float(radius * np.sin(angle_rad))


def center_mesh_on_rotary_axis(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Return a copy centered in Y/Z with its minimum X positioned at zero."""

    result = mesh.copy()
    bounds = np.asarray(result.bounds, dtype=np.float64)
    translation = np.array(
        (-bounds[0, 0], -0.5 * (bounds[0, 1] + bounds[1, 1]), -0.5 * (bounds[0, 2] + bounds[1, 2]))
    )
    result.apply_translation(translation)
    return result


def align_longest_axis_to_x(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Return a centered copy with its longest bounding-box axis mapped to X.

    Raw STL and OBJ files do not encode a rotary-axis convention.  This
    deterministic heuristic keeps an already X-long mesh unchanged, maps Y to
    X with a -90-degree Z rotation, and maps Z to X with a +90-degree Y
    rotation.  Persisted projects continue to use their explicit transform.
    """

    extents = np.asarray(mesh.extents, dtype=np.float64)
    if extents.shape != (3,) or not np.all(np.isfinite(extents)) or np.any(extents <= 0.0):
        raise ValueError("mesh extents must be finite and positive")
    longest_axis = int(np.argmax(extents))
    transform = np.eye(4, dtype=np.float64)
    if longest_axis == 1:
        transform[:3, :3] = np.array(
            (
                (0.0, 1.0, 0.0),
                (-1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0),
            ),
            dtype=np.float64,
        )
    elif longest_axis == 2:
        transform[:3, :3] = np.array(
            (
                (0.0, 0.0, 1.0),
                (0.0, 1.0, 0.0),
                (-1.0, 0.0, 0.0),
            ),
            dtype=np.float64,
        )
    result = mesh.copy()
    result.apply_transform(transform)
    return center_mesh_on_rotary_axis(result)


def apply_translation(mesh: trimesh.Trimesh, offset: npt.ArrayLike) -> trimesh.Trimesh:
    """Return a translated copy of ``mesh``."""

    translation = np.asarray(offset, dtype=np.float64)
    if translation.shape != (3,) or not np.all(np.isfinite(translation)):
        raise ValueError("offset must contain three finite coordinates")
    result = mesh.copy()
    result.apply_translation(translation)
    return result


def apply_uniform_scale(mesh: trimesh.Trimesh, scale: float) -> trimesh.Trimesh:
    """Return a uniformly scaled copy of ``mesh``."""

    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be finite and positive")
    result = mesh.copy()
    result.apply_scale(scale)  # type: ignore[no-untyped-call]
    return result


def apply_rotation(
    mesh: trimesh.Trimesh,
    angle_deg: float,
    axis: npt.ArrayLike,
    *,
    point: npt.ArrayLike | None = None,
) -> trimesh.Trimesh:
    """Return a copy rotated around an arbitrary axis and optional pivot."""

    direction = np.asarray(axis, dtype=np.float64)
    invalid_direction = (
        direction.shape != (3,)
        or not np.all(np.isfinite(direction))
        or np.linalg.norm(direction) == 0.0
    )
    if invalid_direction:
        raise ValueError("axis must be a finite, non-zero three-vector")
    pivot = np.zeros(3) if point is None else np.asarray(point, dtype=np.float64)
    if pivot.shape != (3,) or not np.all(np.isfinite(pivot)) or not np.isfinite(angle_deg):
        raise ValueError("rotation angle and point must be finite")
    matrix = trimesh.transformations.rotation_matrix(  # type: ignore[no-untyped-call]
        np.radians(angle_deg), direction, pivot
    )
    result = mesh.copy()
    result.apply_transform(matrix)
    return result


def validate_mesh_inside_stock(
    mesh: trimesh.Trimesh,
    stock: _StockLike,
    *,
    tolerance: float = 1e-6,
) -> ValidationResult:
    """Check every mesh vertex against a radial stock definition."""

    if tolerance < 0.0 or not np.isfinite(tolerance):
        raise ValueError("tolerance must be finite and non-negative")
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if vertices.size == 0:
        return ValidationResult(False, ("Mesh is empty.",))
    xar = xyz_array_to_xar(vertices)
    outside_x = (xar[:, 0] < -tolerance) | (xar[:, 0] > stock.length + tolerance)
    allowed = np.fromiter(
        (stock.radius_at(float(x), float(a)) for x, a in xar[:, :2]),
        dtype=np.float64,
        count=len(xar),
    )
    outside_radial = xar[:, 2] > allowed + tolerance
    errors: list[str] = []
    if np.any(outside_x):
        outside_x_count = int(np.count_nonzero(outside_x))
        errors.append(f"{outside_x_count} mesh vertices lie outside the stock X range.")
    if np.any(outside_radial):
        outside_radial_count = int(np.count_nonzero(outside_radial))
        errors.append(f"{outside_radial_count} mesh vertices lie outside the stock profile.")
    return ValidationResult(not errors, tuple(errors))


def unwrap_angles(angles_deg: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Unwrap a degree sequence so adjacent samples take the shortest turn."""

    angles = np.asarray(angles_deg, dtype=np.float64)
    if angles.ndim != 1 or not np.all(np.isfinite(angles)):
        raise ValueError("angles_deg must be a finite one-dimensional array")
    return np.degrees(np.unwrap(np.radians(angles), period=2.0 * np.pi))
