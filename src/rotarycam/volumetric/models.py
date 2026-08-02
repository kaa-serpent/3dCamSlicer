"""Immutable XYZ lattice and deterministic sparse brick records."""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Self

import numpy as np
import numpy.typing as npt

from rotarycam.volumetric.errors import InvalidVolumeError, MemoryBudgetExceeded

BoolArray = npt.NDArray[np.bool_]
Point3 = tuple[float, float, float]
Index3 = tuple[int, int, int]


def _finite_point(name: str, value: object) -> Point3:
    if not isinstance(value, Iterable):
        raise InvalidVolumeError(f"{name} must contain exactly three finite numbers")
    try:
        result = tuple(float(component) for component in value)
    except (TypeError, ValueError) as exc:
        raise InvalidVolumeError(f"{name} must contain exactly three finite numbers") from exc
    if len(result) != 3 or not all(math.isfinite(component) for component in result):
        raise InvalidVolumeError(f"{name} must contain exactly three finite numbers")
    return (result[0], result[1], result[2])


def _positive_index(name: str, value: object) -> Index3:
    if not isinstance(value, Iterable):
        raise InvalidVolumeError(f"{name} must contain exactly three positive integers")
    try:
        items: tuple[object, ...] = tuple(value)
    except TypeError as exc:
        raise InvalidVolumeError(f"{name} must contain exactly three positive integers") from exc
    if len(items) != 3:
        raise InvalidVolumeError(f"{name} must contain exactly three positive integers")
    normalized: list[int] = []
    for item in items:
        if isinstance(item, bool) or not isinstance(item, (int, np.integer)):
            raise InvalidVolumeError(f"{name} must contain exactly three positive integers")
        normalized.append(int(item))
    if any(item <= 0 for item in normalized):
        raise InvalidVolumeError(f"{name} must contain exactly three positive integers")
    return (normalized[0], normalized[1], normalized[2])


@dataclass(frozen=True, slots=True)
class VolumetricSettings:
    """Accuracy and explicit resource ceiling for volume construction."""

    tolerance: float
    memory_budget_mib: int = 1024
    brick_size: int = 32

    def __post_init__(self) -> None:
        tolerance = float(self.tolerance)
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise InvalidVolumeError("tolerance must be finite and greater than zero")
        if (
            isinstance(self.memory_budget_mib, bool)
            or not isinstance(self.memory_budget_mib, (int, np.integer))
            or int(self.memory_budget_mib) <= 0
        ):
            raise InvalidVolumeError("memory_budget_mib must be a positive integer")
        object.__setattr__(self, "tolerance", tolerance)
        object.__setattr__(self, "memory_budget_mib", int(self.memory_budget_mib))
        if (
            isinstance(self.brick_size, bool)
            or not isinstance(self.brick_size, (int, np.integer))
            or int(self.brick_size) <= 0
        ):
            raise InvalidVolumeError("brick_size must be a positive integer")
        object.__setattr__(self, "brick_size", int(self.brick_size))

    @property
    def memory_budget_bytes(self) -> int:
        """Return the exact binary MiB budget in bytes."""

        return self.memory_budget_mib * 1024 * 1024

    @property
    def brick_shape(self) -> Index3:
        """Return the cubic sparse-brick shape used by volume builders."""

        return (self.brick_size, self.brick_size, self.brick_size)


@dataclass(frozen=True, slots=True)
class VoxelLattice:
    """Regular voxel-centre lattice in canonical ``(X, Y, Z)`` order."""

    origin: Point3
    spacing: Point3
    shape: Index3

    def __post_init__(self) -> None:
        origin = _finite_point("origin", self.origin)
        spacing = _finite_point("spacing", self.spacing)
        if any(component <= 0.0 for component in spacing):
            raise InvalidVolumeError("spacing must be finite and strictly positive")
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "spacing", spacing)
        object.__setattr__(self, "shape", _positive_index("shape", self.shape))

    @property
    def voxel_count(self) -> int:
        """Return the exact number of XYZ cells."""

        return math.prod(self.shape)

    @property
    def bounds(self) -> tuple[Point3, Point3]:
        """Return outer cell-face bounds, not just centre coordinates."""

        lower = tuple(self.origin[axis] - 0.5 * self.spacing[axis] for axis in range(3))
        upper = tuple(
            self.origin[axis] + (self.shape[axis] - 0.5) * self.spacing[axis]
            for axis in range(3)
        )
        return lower, upper  # type: ignore[return-value]

    def estimate_memory_bytes(
        self,
        *,
        bytes_per_voxel: int = 1,
        simultaneous_arrays: int = 1,
    ) -> int:
        """Estimate dense working storage without platform-dependent overhead."""

        if (
            isinstance(bytes_per_voxel, bool)
            or isinstance(simultaneous_arrays, bool)
            or not isinstance(bytes_per_voxel, int)
            or not isinstance(simultaneous_arrays, int)
            or bytes_per_voxel <= 0
            or simultaneous_arrays <= 0
        ):
            raise InvalidVolumeError("memory estimation factors must be positive")
        return self.voxel_count * bytes_per_voxel * simultaneous_arrays

    def require_memory_budget(
        self,
        budget_bytes: int,
        *,
        bytes_per_voxel: int = 1,
        simultaneous_arrays: int = 1,
    ) -> None:
        """Reject a lattice which cannot be represented at the requested tolerance."""

        required = self.estimate_memory_bytes(
            bytes_per_voxel=bytes_per_voxel,
            simultaneous_arrays=simultaneous_arrays,
        )
        if required > budget_bytes:
            raise MemoryBudgetExceeded(required, budget_bytes)

    def axis_centres(self, axis: int) -> npt.NDArray[np.float64]:
        """Return an independent, read-only centre coordinate axis."""

        if axis not in (0, 1, 2):
            raise InvalidVolumeError("axis must be 0 (X), 1 (Y), or 2 (Z)")
        values = (
            self.origin[axis]
            + np.arange(self.shape[axis], dtype=np.float64) * self.spacing[axis]
        )
        values.setflags(write=False)
        return values


class BrickClassification(StrEnum):
    """Compact occupation state of one lattice brick."""

    EMPTY = "empty"
    FULL = "full"
    MIXED = "mixed"


@dataclass(frozen=True, slots=True)
class VoxelBrick:
    """One non-empty sparse brick; mixed values are copied and read-only."""

    index: Index3
    shape: Index3
    classification: BrickClassification
    values: BoolArray | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        index = tuple(self.index)
        if (
            len(index) != 3
            or any(
                isinstance(item, bool) or not isinstance(item, (int, np.integer))
                for item in index
            )
            or any(int(item) < 0 for item in index)
        ):
            raise InvalidVolumeError("brick index must contain three non-negative integers")
        shape = _positive_index("brick shape", self.shape)
        classification = BrickClassification(self.classification)
        values: BoolArray | None = None
        if classification is BrickClassification.MIXED:
            if self.values is None:
                raise InvalidVolumeError("mixed bricks require an occupation array")
            values = np.array(self.values, dtype=np.bool_, copy=True)
            if values.shape != shape:
                raise InvalidVolumeError(f"mixed brick values must have shape {shape}")
            if bool(np.all(values)) or not bool(np.any(values)):
                raise InvalidVolumeError("mixed brick values must contain empty and occupied cells")
            values.setflags(write=False)
        elif self.values is not None:
            dense = np.asarray(self.values, dtype=np.bool_)
            expected = classification is BrickClassification.FULL
            if dense.shape != shape or not bool(np.all(dense == expected)):
                raise InvalidVolumeError("uniform brick values disagree with classification")
        object.__setattr__(self, "index", tuple(int(item) for item in index))
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "classification", classification)
        object.__setattr__(self, "values", values)

    def dense(self) -> BoolArray:
        """Return a read-only dense copy of this brick."""

        if self.classification is BrickClassification.MIXED:
            assert self.values is not None
            result = self.values.copy()
        else:
            result = np.full(
                self.shape,
                self.classification is BrickClassification.FULL,
                dtype=np.bool_,
            )
        result.setflags(write=False)
        return result


@dataclass(frozen=True, slots=True)
class SparseBrickStore:
    """Sparse occupation bricks stored and traversed lexicographically."""

    lattice: VoxelLattice
    brick_shape: Index3
    bricks: tuple[VoxelBrick, ...] = ()

    def __post_init__(self) -> None:
        brick_shape = _positive_index("brick_shape", self.brick_shape)
        grid_shape = tuple(
            math.ceil(self.lattice.shape[axis] / brick_shape[axis]) for axis in range(3)
        )
        unique: dict[Index3, VoxelBrick] = {}
        for brick in self.bricks:
            if brick.classification is BrickClassification.EMPTY:
                continue
            if any(brick.index[axis] >= grid_shape[axis] for axis in range(3)):
                raise InvalidVolumeError("brick index lies outside the lattice")
            expected_shape = tuple(
                min(
                    brick_shape[axis],
                    self.lattice.shape[axis] - brick.index[axis] * brick_shape[axis],
                )
                for axis in range(3)
            )
            if brick.shape != expected_shape:
                raise InvalidVolumeError(f"brick {brick.index} must have shape {expected_shape}")
            if brick.index in unique:
                raise InvalidVolumeError(f"duplicate brick index {brick.index}")
            unique[brick.index] = brick
        object.__setattr__(self, "brick_shape", brick_shape)
        object.__setattr__(self, "bricks", tuple(unique[index] for index in sorted(unique)))

    @classmethod
    def from_dense(
        cls,
        lattice: VoxelLattice,
        occupation: npt.ArrayLike,
        *,
        brick_shape: Index3 = (32, 32, 32),
    ) -> SparseBrickStore:
        """Copy a dense XYZ mask into deterministic sparse bricks."""

        dense = np.array(occupation, dtype=np.bool_, copy=True)
        if dense.shape != lattice.shape:
            raise InvalidVolumeError(f"occupation must have XYZ shape {lattice.shape}")
        normalized_shape = _positive_index("brick_shape", brick_shape)
        bricks: list[VoxelBrick] = []
        for ix in range(0, lattice.shape[0], normalized_shape[0]):
            for iy in range(0, lattice.shape[1], normalized_shape[1]):
                for iz in range(0, lattice.shape[2], normalized_shape[2]):
                    block = dense[
                        ix : ix + normalized_shape[0],
                        iy : iy + normalized_shape[1],
                        iz : iz + normalized_shape[2],
                    ]
                    if not bool(np.any(block)):
                        continue
                    index = (
                        ix // normalized_shape[0],
                        iy // normalized_shape[1],
                        iz // normalized_shape[2],
                    )
                    shape = block.shape
                    if bool(np.all(block)):
                        bricks.append(VoxelBrick(index, shape, BrickClassification.FULL))
                    else:
                        bricks.append(VoxelBrick(index, shape, BrickClassification.MIXED, block))
        return cls(lattice, normalized_shape, tuple(bricks))

    def iter_bricks(self, *, include_empty: bool = False) -> Iterator[VoxelBrick]:
        """Yield bricks in canonical lexicographic ``(X, Y, Z)`` order."""

        if not include_empty:
            yield from self.bricks
            return
        occupied = {brick.index: brick for brick in self.bricks}
        counts = tuple(
            math.ceil(self.lattice.shape[axis] / self.brick_shape[axis]) for axis in range(3)
        )
        for bx in range(counts[0]):
            for by in range(counts[1]):
                for bz in range(counts[2]):
                    index = (bx, by, bz)
                    brick = occupied.get(index)
                    if brick is not None:
                        yield brick
                        continue
                    shape: Index3 = (
                        min(
                            self.brick_shape[0],
                            self.lattice.shape[0] - index[0] * self.brick_shape[0],
                        ),
                        min(
                            self.brick_shape[1],
                            self.lattice.shape[1] - index[1] * self.brick_shape[1],
                        ),
                        min(
                            self.brick_shape[2],
                            self.lattice.shape[2] - index[2] * self.brick_shape[2],
                        ),
                    )
                    yield VoxelBrick(index, shape, BrickClassification.EMPTY)

    def to_dense(self) -> BoolArray:
        """Materialize a read-only XYZ occupation mask."""

        result = np.zeros(self.lattice.shape, dtype=np.bool_)
        for brick in self.bricks:
            slices = tuple(
                slice(
                    brick.index[axis] * self.brick_shape[axis],
                    brick.index[axis] * self.brick_shape[axis] + brick.shape[axis],
                )
                for axis in range(3)
            )
            result[slices] = brick.dense()
        result.setflags(write=False)
        return result


@dataclass(frozen=True, slots=True)
class SparseVolume:
    """Immutable occupied volume backed by deterministic sparse bricks."""

    lattice: VoxelLattice
    store: SparseBrickStore

    def __post_init__(self) -> None:
        if self.store.lattice != self.lattice:
            raise InvalidVolumeError("volume and brick store must use the same lattice")

    @property
    def occupation(self) -> BoolArray:
        """Return a detached, read-only dense occupation mask."""

        return self.store.to_dense()

    @property
    def brick_shape(self) -> Index3:
        """Return the uniform brick allocation shape."""

        return self.store.brick_shape

    @property
    def bricks(self) -> tuple[VoxelBrick, ...]:
        """Return non-empty bricks in deterministic lexicographic order."""

        return self.store.bricks

    @classmethod
    def from_dense(
        cls,
        lattice: VoxelLattice,
        occupation: npt.ArrayLike,
        *,
        brick_size: int = 32,
    ) -> Self:
        """Build a typed immutable volume from an input mask copy."""

        if isinstance(brick_size, bool) or not isinstance(brick_size, int) or brick_size <= 0:
            raise InvalidVolumeError("brick_size must be a positive integer")
        shape = (brick_size, brick_size, brick_size)
        return cls(lattice, SparseBrickStore.from_dense(lattice, occupation, brick_shape=shape))

    def iter_bricks(self, *, include_empty: bool = False) -> Iterator[VoxelBrick]:
        """Yield bricks in canonical X/Y/Z order."""

        return self.store.iter_bricks(include_empty=include_empty)

    def to_dense(self) -> BoolArray:
        """Materialize an independent, read-only occupation array."""

        return self.store.to_dense()


@dataclass(frozen=True, slots=True)
class SolidVolume(SparseVolume):
    """Target solid occupation, including any parity-classified cavities."""

    conservative_guard: BoolArray | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        super(SolidVolume, self).__post_init__()
        if self.conservative_guard is None:
            return
        guard = np.array(self.conservative_guard, dtype=np.bool_, copy=True)
        if guard.shape != self.lattice.shape:
            raise InvalidVolumeError(
                f"conservative_guard must have XYZ shape {self.lattice.shape}"
            )
        if np.any(guard & ~self.store.to_dense()):
            raise InvalidVolumeError("conservative guard cells must be occupied")
        guard.setflags(write=False)
        object.__setattr__(self, "conservative_guard", guard)


@dataclass(frozen=True, slots=True)
class StockVolume(SparseVolume):
    """Initial or simulated stock occupation."""
