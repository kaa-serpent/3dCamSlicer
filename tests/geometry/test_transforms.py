from __future__ import annotations

import numpy as np
import pytest
import trimesh

from rotarycam.geometry.transforms import (
    align_longest_axis_to_x,
    apply_uniform_scale,
    center_mesh_on_rotary_axis,
    unwrap_angles,
    validate_mesh_inside_stock,
    xar_to_xyz,
    xyz_array_to_xar,
    xyz_to_xar,
)
from rotarycam.stock import CylindricalStock, RectangularStock


@pytest.mark.parametrize(
    ("xyz", "expected_angle"),
    [
        ((2.0, 4.0, 0.0), 0.0),
        ((2.0, 0.0, 4.0), 90.0),
        ((2.0, -4.0, 0.0), 180.0),
        ((2.0, 0.0, -4.0), 270.0),
    ],
)
def test_xyz_to_xar_cardinal_angles(xyz: tuple[float, float, float], expected_angle: float) -> None:
    x, angle, radius = xyz_to_xar(*xyz)
    assert x == 2.0
    assert angle == pytest.approx(expected_angle)
    assert radius == pytest.approx(4.0)
    assert xar_to_xyz(x, angle, radius) == pytest.approx(xyz)


def test_xyz_array_to_xar_is_vectorized() -> None:
    result = xyz_array_to_xar([[0.0, 1.0, 0.0], [3.0, 0.0, -2.0]])
    assert np.allclose(result, [[0.0, 0.0, 1.0], [3.0, 270.0, 2.0]])


def test_unwrap_angles_maintains_continuity_at_seam() -> None:
    assert unwrap_angles([358.0, 359.0, 0.0, 1.0]) == pytest.approx([358.0, 359.0, 360.0, 361.0])


def test_center_and_scale_do_not_mutate_mesh() -> None:
    mesh = trimesh.creation.box(extents=(2.0, 4.0, 6.0))
    original = mesh.vertices.copy()
    centered = center_mesh_on_rotary_axis(mesh)
    scaled = apply_uniform_scale(centered, 2.0)
    assert np.array_equal(mesh.vertices, original)
    assert centered.bounds[0, 0] == pytest.approx(0.0)
    assert centered.centroid[1:] == pytest.approx([0.0, 0.0])
    assert scaled.extents == pytest.approx([4.0, 8.0, 12.0])


@pytest.mark.parametrize(
    "extents",
    [
        (9.0, 4.0, 2.0),
        (2.0, 9.0, 4.0),
        (4.0, 2.0, 9.0),
    ],
)
def test_align_longest_axis_to_x_is_deterministic_and_non_mutating(
    extents: tuple[float, float, float],
) -> None:
    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_translation((3.0, -7.0, 11.0))
    original = mesh.vertices.copy()

    aligned = align_longest_axis_to_x(mesh)

    assert np.array_equal(mesh.vertices, original)
    assert aligned.extents[0] == pytest.approx(9.0)
    assert aligned.bounds[0, 0] == pytest.approx(0.0)
    assert aligned.centroid[1:] == pytest.approx([0.0, 0.0])


def test_mesh_inside_stock_checks_x_and_radial_profile() -> None:
    mesh = trimesh.creation.box(extents=(10.0, 4.0, 4.0))
    mesh.apply_translation((5.0, 0.0, 0.0))
    assert validate_mesh_inside_stock(mesh, RectangularStock(10.0, 4.0, 4.0)).valid
    result = validate_mesh_inside_stock(mesh, CylindricalStock(10.0, 4.0))
    assert not result.valid
    assert "stock profile" in result.errors[0]
