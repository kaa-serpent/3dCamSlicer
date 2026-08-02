from __future__ import annotations

import numpy as np
import pytest
import trimesh

from rotarycam.geometry import (
    RadialCompatibilityError,
    UndercutStatus,
    center_mesh_on_rotary_axis,
    sample_mesh_radially,
    sample_mesh_radially_with_report,
)


def _x_aligned_cylinder(radius: float = 5.0, length: float = 10.0) -> trimesh.Trimesh:
    mesh = trimesh.creation.cylinder(radius=radius, height=length, sections=128)
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2.0, (0.0, 1.0, 0.0)))
    return center_mesh_on_rotary_axis(mesh)


def test_sample_cylinder_has_constant_radius() -> None:
    grid = sample_mesh_radially(_x_aligned_cylinder(), 2.5, 30.0)
    assert grid.undercut_status is UndercutStatus.ABSENT
    assert np.all(grid.valid)
    assert grid.radius == pytest.approx(5.0, abs=0.01)


def test_sample_cube_matches_analytical_profile() -> None:
    mesh = trimesh.creation.box(extents=(10.0, 8.0, 6.0))
    mesh.apply_translation((5.0, 0.0, 0.0))
    grid = sample_mesh_radially(mesh, 5.0, 45.0)
    middle = grid.radius[1]
    assert middle[[0, 4]] == pytest.approx([4.0, 4.0])
    assert middle[[2, 6]] == pytest.approx([3.0, 3.0])
    assert middle[[1, 3, 5, 7]] == pytest.approx(np.full(4, 3.0 * np.sqrt(2.0)))


def test_off_axis_solid_is_detected_as_multiple_radial_boundaries() -> None:
    mesh = _x_aligned_cylinder(radius=1.0, length=6.0)
    mesh.apply_translation((0.0, 4.0, 0.0))
    result = sample_mesh_radially_with_report(mesh, 2.0, 15.0, strict=False)
    assert result.grid.undercut_status is UndercutStatus.PRESENT
    assert np.any(result.intersection_counts > 1)
    assert not result.report.cam_compatible
    with pytest.raises(RadialCompatibilityError):
        sample_mesh_radially(mesh, 2.0, 15.0)


def test_open_mesh_is_inspectable_but_not_strictly_certified() -> None:
    mesh = _x_aligned_cylinder()
    mesh.update_faces(np.arange(len(mesh.faces) - 1))
    inspected = sample_mesh_radially_with_report(mesh, 5.0, 90.0, strict=False)
    assert inspected.grid.undercut_status in {UndercutStatus.PRESENT, UndercutStatus.INDETERMINATE}
    assert not inspected.report.cam_compatible
    with pytest.raises(RadialCompatibilityError):
        sample_mesh_radially(mesh, 5.0, 90.0)
