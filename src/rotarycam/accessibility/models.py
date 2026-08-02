"""Typed immutable results for volumetric surface accessibility."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
import numpy.typing as npt

from rotarycam.motion import MachinePose

BoolArray = npt.NDArray[np.bool_]
Index3 = tuple[int, int, int]
Point3 = tuple[float, float, float]


class InaccessibilityReason(StrEnum):
    """Stable reason codes used by reports and export acknowledgements."""

    TRAVEL = "travel"
    CUTTER_REACH = "cutter_reach"
    OCCLUDED = "occluded"
    NON_CUTTING_COLLISION = "non_cutting_collision"
    MACHINE_COLLISION = "machine_collision"
    FIXTURE_COLLISION = "fixture_collision"
    TOPOLOGY = "topology"


@dataclass(frozen=True, slots=True)
class AccessibilitySettings:
    """Orientation sampling and geometric acceptance tolerances."""

    angles_deg: tuple[float, ...]
    max_error: float
    tolerance: float = 0.05

    def __post_init__(self) -> None:
        if not self.angles_deg:
            raise ValueError("angles_deg must contain at least one angle")
        if not all(math.isfinite(angle) for angle in self.angles_deg):
            raise ValueError("angles_deg must contain only finite angles")
        normalized = tuple(sorted({float(angle) % 360.0 for angle in self.angles_deg}))
        object.__setattr__(self, "angles_deg", normalized)
        if not math.isfinite(self.max_error) or self.max_error < 0.0:
            raise ValueError("max_error must be finite and non-negative")
        if not math.isfinite(self.tolerance) or self.tolerance <= 0.0:
            raise ValueError("tolerance must be finite and greater than zero")


@dataclass(frozen=True, slots=True)
class CandidatePose:
    """One valid tool pose for one target surface voxel."""

    surface_index: Index3
    part_point: Point3
    normal: Point3
    tool_number: int
    pose: MachinePose
    error: float

    def __post_init__(self) -> None:
        if (
            len(self.surface_index) != 3
            or any(
                isinstance(index, bool) or not isinstance(index, int) or index < 0
                for index in self.surface_index
            )
        ):
            raise ValueError("surface_index must contain three non-negative indices")
        if len(self.part_point) != 3 or len(self.normal) != 3:
            raise ValueError("part_point and normal must contain three values")
        if not all(math.isfinite(value) for value in (*self.part_point, *self.normal)):
            raise ValueError("part_point and normal must contain only finite values")
        magnitude = math.sqrt(sum(value * value for value in self.normal))
        if not math.isclose(magnitude, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("normal must be a unit vector")
        if self.tool_number <= 0:
            raise ValueError("tool_number must be greater than zero")
        if not isinstance(self.pose, MachinePose):
            raise TypeError("pose must be a MachinePose")
        if not math.isfinite(self.error) or self.error < 0.0:
            raise ValueError("error must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class InaccessibleRegion:
    """One 6-connected component of inaccessible surface voxels."""

    indices: tuple[Index3, ...]
    voxel_count: int
    max_error: float
    reasons: tuple[InaccessibilityReason, ...]

    def __post_init__(self) -> None:
        normalized = tuple(sorted(set(self.indices)))
        if not normalized:
            raise ValueError("indices must contain at least one voxel")
        if any(
            len(index) != 3 or any(component < 0 for component in index)
            for index in normalized
        ):
            raise ValueError("indices must contain non-negative XYZ indices")
        if self.voxel_count != len(normalized):
            raise ValueError("voxel_count must equal the number of unique indices")
        if not math.isfinite(self.max_error) or self.max_error < 0.0:
            raise ValueError("max_error must be finite and non-negative")
        reasons = tuple(sorted(set(self.reasons), key=lambda reason: reason.value))
        if not reasons:
            raise ValueError("reasons must not be empty")
        object.__setattr__(self, "indices", normalized)
        object.__setattr__(self, "reasons", reasons)


@dataclass(frozen=True, slots=True)
class AccessibilityReport:
    """Deterministic candidates and the accessible/inaccessible surface split."""

    candidates: tuple[CandidatePose, ...]
    accessible_mask: BoolArray = field(repr=False, compare=False)
    inaccessible_mask: BoolArray = field(repr=False, compare=False)
    regions: tuple[InaccessibleRegion, ...]
    digest: str

    def __post_init__(self) -> None:
        accessible = np.array(self.accessible_mask, dtype=np.bool_, copy=True)
        inaccessible = np.array(self.inaccessible_mask, dtype=np.bool_, copy=True)
        if accessible.ndim != 3 or inaccessible.shape != accessible.shape:
            raise ValueError("accessibility masks must have the same three-dimensional shape")
        if np.any(accessible & inaccessible):
            raise ValueError("accessible and inaccessible masks must be disjoint")
        accessible.setflags(write=False)
        inaccessible.setflags(write=False)
        object.__setattr__(self, "accessible_mask", accessible)
        object.__setattr__(self, "inaccessible_mask", inaccessible)
        candidates = tuple(
            sorted(
                self.candidates,
                key=lambda item: (item.surface_index, item.tool_number, item.pose.a),
            )
        )
        regions = tuple(sorted(self.regions, key=lambda region: region.indices[0]))
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "regions", regions)
        if len(self.digest) != 64 or any(char not in "0123456789abcdef" for char in self.digest):
            raise ValueError("digest must be a lowercase SHA-256 digest")

    @property
    def complete(self) -> bool:
        """Whether every target surface voxel has at least one valid pose."""

        return not bool(np.any(self.inaccessible_mask))

    @property
    def residual_voxels(self) -> int:
        """Return the inaccessible surface-voxel count."""

        return int(np.count_nonzero(self.inaccessible_mask))

    @property
    def max_error(self) -> float:
        """Return the largest geometric error represented by any residue region."""

        return max((region.max_error for region in self.regions), default=0.0)

    @property
    def reasons(self) -> tuple[InaccessibilityReason, ...]:
        """Return the deterministic union of all inaccessible-region causes."""

        return tuple(
            sorted(
                {reason for region in self.regions for reason in region.reasons},
                key=lambda reason: reason.value,
            )
        )


__all__ = [
    "AccessibilityReport",
    "AccessibilitySettings",
    "CandidatePose",
    "InaccessibilityReason",
    "InaccessibleRegion",
]
