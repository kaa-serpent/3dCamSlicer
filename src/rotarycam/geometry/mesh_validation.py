"""Mesh inspection and CAM eligibility reporting."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import trimesh

from rotarycam.geometry.models import UndercutStatus


class RadialState(StrEnum):
    """Compact inspection state retained for CLI boundary compatibility."""

    UNKNOWN = "unknown"
    REPRESENTABLE = "representable"
    UNDERCUT = "undercut"


@dataclass(frozen=True, slots=True)
class MeshValidationReport:
    """Structured validation result suitable for CLI and JSON boundaries."""

    dimensions_mm: tuple[float, float, float]
    component_count: int
    is_watertight: bool
    is_winding_consistent: bool
    has_degenerate_faces: bool
    has_multiple_components: bool
    radial_undercuts: UndercutStatus
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    @property
    def radial_state(self) -> RadialState:
        """Map the canonical four-state result to the compact inspection label."""

        if self.radial_undercuts is UndercutStatus.ABSENT:
            return RadialState.REPRESENTABLE
        if self.radial_undercuts is UndercutStatus.PRESENT:
            return RadialState.UNDERCUT
        return RadialState.UNKNOWN

    @property
    def cam_compatible(self) -> bool:
        """Whether strict radial CAM may proceed from this report."""

        return (
            not self.errors
            and self.is_watertight
            and self.is_winding_consistent
            and not self.has_multiple_components
            and self.radial_undercuts is UndercutStatus.ABSENT
        )


def _component_count(mesh: trimesh.Trimesh) -> int:
    if len(mesh.faces) == 0:
        return 0
    try:
        return len(mesh.split(only_watertight=False))
    except Exception:
        return int(getattr(mesh, "body_count", 1))


def _has_degenerate_faces(mesh: trimesh.Trimesh) -> bool:
    if len(mesh.faces) == 0:
        return False
    triangles = np.asarray(mesh.triangles, dtype=np.float64)
    doubled_areas = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    scale = max(float(np.max(np.asarray(mesh.extents, dtype=np.float64))), 1.0)
    return bool(np.any(doubled_areas <= np.finfo(np.float64).eps * scale * scale * 16.0))


def validate_mesh(
    mesh: trimesh.Trimesh,
    *,
    radial_undercuts: UndercutStatus = UndercutStatus.NOT_EVALUATED,
) -> MeshValidationReport:
    """Inspect topology without mutating or implicitly sampling the mesh."""

    warnings: list[str] = []
    errors: list[str] = []
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        return MeshValidationReport(
            (0.0, 0.0, 0.0),
            0,
            False,
            False,
            False,
            False,
            UndercutStatus.INDETERMINATE,
            (),
            ("Mesh is empty.",),
        )

    extents = np.asarray(mesh.extents, dtype=np.float64)
    dimensions = (float(extents[0]), float(extents[1]), float(extents[2]))
    if not np.all(np.isfinite(dimensions)) or any(value <= 0.0 for value in dimensions):
        errors.append("Mesh dimensions must be finite and positive.")
    is_watertight = bool(mesh.is_watertight)
    is_winding_consistent = bool(mesh.is_winding_consistent)
    component_count = _component_count(mesh)
    has_multiple_components = component_count > 1
    degenerate = _has_degenerate_faces(mesh)
    if not is_watertight:
        warnings.append("Mesh is open; inspection is available but CAM generation is blocked.")
    if not is_winding_consistent:
        warnings.append("Mesh winding is inconsistent; radial certification is unreliable.")
    if has_multiple_components:
        warnings.append("Mesh has multiple connected components; strict radial CAM is blocked.")
    if degenerate:
        warnings.append("Mesh contains degenerate faces.")
    if radial_undercuts is UndercutStatus.PRESENT:
        warnings.append("Radial undercuts or multiple material intervals were detected.")
    elif radial_undercuts is UndercutStatus.INDETERMINATE:
        warnings.append("Radial undercuts could not be certified.")

    return MeshValidationReport(
        dimensions,
        component_count,
        is_watertight,
        is_winding_consistent,
        degenerate,
        has_multiple_components,
        radial_undercuts,
        tuple(warnings),
        tuple(errors),
    )
