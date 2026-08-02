"""Conservative analytical, mesh, and retention rasterization tests."""

import numpy as np
import pytest
import trimesh

from rotarycam.stock import CylindricalStock, RectangularStock
from rotarycam.supports.models import (
    CylindricalSurfaceRetention,
    RectangularSurfaceRetention,
)
from rotarycam.volumetric import (
    InvalidSolidMeshError,
    MemoryBudgetExceeded,
    SolidVolume,
    SparseVolume,
    VolumetricSettings,
    VoxelLattice,
    build_cylindrical_stock,
    build_mesh_volume,
    build_mesh_volume_on_lattice,
    build_rectangular_stock,
    classify_mesh_points,
    rasterize_retention_volumes,
    ray_parity_classifier,
)


def _nearest_index(volume: SparseVolume, point: tuple[float, float, float]) -> tuple[int, int, int]:
    return tuple(
        int(np.argmin(np.abs(volume.lattice.axis_centres(axis) - point[axis])))
        for axis in range(3)
    )


def test_analytical_stock_builders_are_xyz_and_conservative() -> None:
    settings = VolumetricSettings(0.5, brick_size=2)
    rectangle = build_rectangular_stock(RectangularStock(4.0, 4.0, 2.0), settings)
    assert rectangle.lattice.shape == (8, 8, 4)
    assert np.all(rectangle.to_dense())

    cylinder = build_cylindrical_stock(CylindricalStock(4.0, 4.0), settings)
    dense = cylinder.to_dense()
    assert dense[_nearest_index(cylinder, (2.0, 0.0, 0.0))]
    assert not dense[0, 0, 0]


def test_mesh_builder_rejects_open_or_inconsistent_meshes() -> None:
    open_mesh = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    open_mesh.update_faces(np.arange(len(open_mesh.faces) - 1))
    open_mesh.remove_unreferenced_vertices()
    with pytest.raises(InvalidSolidMeshError, match="watertight"):
        build_mesh_volume(open_mesh, VolumetricSettings(0.5))

    inconsistent = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    inconsistent.faces[0] = inconsistent.faces[0][::-1]
    with pytest.raises(InvalidSolidMeshError, match="winding"):
        build_mesh_volume(inconsistent, VolumetricSettings(0.5))


def test_mesh_builder_uses_existing_lattice_and_refuses_insufficient_budget() -> None:
    mesh = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    lattice = VoxelLattice(
        (-2.25, -2.25, -2.25),
        (0.5, 0.5, 0.5),
        (10, 10, 10),
    )

    volume = build_mesh_volume_on_lattice(
        mesh,
        lattice,
        brick_size=2,
        memory_budget_bytes=1_000_000,
        classifier=ray_parity_classifier,
    )

    assert volume.lattice == lattice
    assert volume.to_dense()[4, 4, 4]
    assert not volume.to_dense()[0, 0, 0]
    with pytest.raises(MemoryBudgetExceeded):
        build_mesh_volume_on_lattice(mesh, lattice, memory_budget_bytes=1)


def test_parity_supports_disjoint_components_and_nested_cavity() -> None:
    left = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    left.apply_translation((-2.0, 0.0, 0.0))
    right = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    right.apply_translation((2.0, 0.0, 0.0))
    union = trimesh.util.concatenate((left, right))
    union_volume = build_mesh_volume(
        union,
        VolumetricSettings(0.5, brick_size=2),
        classifier=ray_parity_classifier,
    )
    union_dense = union_volume.to_dense()
    assert union_dense[_nearest_index(union_volume, (-2.0, 0.0, 0.0))]
    assert union_dense[_nearest_index(union_volume, (2.0, 0.0, 0.0))]
    assert not union_dense[_nearest_index(union_volume, (0.0, 0.0, 0.0))]

    outer = trimesh.creation.box(extents=(6.0, 6.0, 6.0))
    inner = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    shell = trimesh.util.concatenate((outer, inner))
    points = np.asarray(((0.0, 0.0, 0.0), (2.0, 0.0, 0.0)), dtype=np.float64)
    classified = classify_mesh_points(
        tuple(shell.split(only_watertight=False)),
        points,
        classifier=ray_parity_classifier,
    )
    assert classified.tolist() == [False, True]


def test_partially_overlapping_components_are_combined_as_a_union() -> None:
    left = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    right = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    right.apply_translation((1.0, 0.0, 0.0))
    components = (left, right)
    points = np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)), dtype=np.float64)

    classified = classify_mesh_points(
        components,
        points,
        classifier=ray_parity_classifier,
    )

    assert classified.tolist() == [True, True]


def test_torus_keeps_hole_and_surface_guard_is_occupied() -> None:
    torus = trimesh.creation.torus(major_radius=3.0, minor_radius=1.0)
    volume = build_mesh_volume(
        torus,
        VolumetricSettings(0.5, brick_size=4),
        classifier=ray_parity_classifier,
    )
    dense = volume.to_dense()
    assert not dense[_nearest_index(volume, (0.0, 0.0, 0.0))]
    assert dense[_nearest_index(volume, (3.0, 0.0, 0.0))]
    assert volume.conservative_guard is not None
    assert np.all(dense[volume.conservative_guard])


def test_v2_retention_volumes_are_rasterized_and_disabled_items_are_ignored() -> None:
    lattice = VoxelLattice((0.5, -2.5, -2.5), (1.0, 1.0, 1.0), (4, 6, 6))
    centres = np.meshgrid(
        *(lattice.axis_centres(axis) for axis in range(3)),
        indexing="ij",
    )
    target_mask = centres[1] * centres[1] + centres[2] * centres[2] <= 1.6**2
    target = SolidVolume.from_dense(lattice, target_mask, brick_size=2)
    enabled = RectangularSurfaceRetention(
        x=2.0,
        angle_deg=0.0,
        radial_thickness=1.0,
        transition=0.0,
        length_x=2.0,
        width_surface=2.0,
    )
    disabled = CylindricalSurfaceRetention(
        x=2.0,
        angle_deg=180.0,
        radial_thickness=1.0,
        transition=0.0,
        diameter=2.0,
        enabled=False,
    )
    retained = rasterize_retention_volumes(target, (enabled, disabled))
    dense = retained.to_dense()
    assert np.any(dense)
    assert np.any(dense[:, -1, :])
    assert not np.any(dense[:, 0, :])
