"""Immutable sparse-volume model tests."""

import numpy as np
import pytest

from rotarycam.volumetric import (
    BrickClassification,
    InvalidVolumeError,
    MemoryBudgetExceeded,
    SparseVolume,
    VolumetricSettings,
    VoxelLattice,
    lattice_from_bounds,
)


def test_settings_and_lattice_validate_finite_positive_values() -> None:
    settings = VolumetricSettings(0.25, memory_budget_mib=2, brick_size=4)
    assert settings.brick_shape == (4, 4, 4)
    assert settings.memory_budget_bytes == 2 * 1024 * 1024

    for tolerance in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(InvalidVolumeError):
            VolumetricSettings(tolerance)
    with pytest.raises(InvalidVolumeError):
        VolumetricSettings(1.0, memory_budget_mib=0)
    with pytest.raises(InvalidVolumeError):
        VoxelLattice((0.0, 0.0, 0.0), (1.0, float("nan"), 1.0), (1, 1, 1))


def test_lattice_memory_budget_and_axes_are_explicit() -> None:
    lattice = VoxelLattice((0.5, 0.5, 0.5), (1.0, 1.0, 1.0), (10, 20, 30))
    assert lattice.voxel_count == 6000
    assert lattice.estimate_memory_bytes(bytes_per_voxel=4, simultaneous_arrays=2) == 48000
    axis = lattice.axis_centres(0)
    assert not axis.flags.writeable
    with pytest.raises(ValueError):
        axis[0] = 99.0
    with pytest.raises(MemoryBudgetExceeded) as error:
        lattice.require_memory_budget(5999)
    assert error.value.required_bytes == 6000
    with pytest.raises(MemoryBudgetExceeded):
        lattice_from_bounds(
            (0.0, 0.0, 0.0),
            (101.0, 101.0, 101.0),
            VolumetricSettings(0.5, memory_budget_mib=1),
        )


def test_sparse_volume_copies_masks_and_sorts_bricks() -> None:
    lattice = VoxelLattice((0.5, 0.5, 0.5), (1.0, 1.0, 1.0), (4, 4, 4))
    occupation = np.zeros(lattice.shape, dtype=np.bool_)
    occupation[0:2, 0:2, 0:2] = True
    occupation[3, 3, 3] = True

    volume = SparseVolume.from_dense(lattice, occupation, brick_size=2)
    occupation[:] = False

    assert [brick.index for brick in volume.bricks] == [(0, 0, 0), (1, 1, 1)]
    assert [brick.classification for brick in volume.bricks] == [
        BrickClassification.FULL,
        BrickClassification.MIXED,
    ]
    dense = volume.to_dense()
    assert dense[0, 0, 0]
    assert dense[3, 3, 3]
    assert not dense.flags.writeable
    classes = [brick.classification for brick in volume.iter_bricks(include_empty=True)]
    assert classes.count(BrickClassification.EMPTY) == 6
