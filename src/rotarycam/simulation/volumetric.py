"""Deterministic volumetric material-removal simulation for XYZA motion."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
import numpy.typing as npt

from rotarycam.collisions import CollisionReport
from rotarycam.errors import RotaryCamError
from rotarycam.machine import XYZAKinematics
from rotarycam.motion import MachinePose, MotionBlock, MotionKind
from rotarycam.tools import ToolAssembly
from rotarycam.volumetric import SolidVolume, StockVolume, VoxelLattice

BoolArray = npt.NDArray[np.bool_]
FloatArray = npt.NDArray[np.float64]


class VolumetricSimulationError(RotaryCamError):
    """Base class for expected volumetric-simulation failures."""


class SimulationBudgetExceeded(VolumetricSimulationError):
    """Raised rather than silently undersampling a requested motion."""


@dataclass(frozen=True, slots=True)
class VolumetricSimulationSettings:
    """Spatial sampling tolerance and explicit hard sample budget."""

    tolerance: float
    max_samples: int = 1_000_000

    def __post_init__(self) -> None:
        if not math.isfinite(self.tolerance) or self.tolerance <= 0.0:
            raise ValueError("tolerance must be finite and greater than zero")
        if (
            isinstance(self.max_samples, bool)
            or not isinstance(self.max_samples, int)
            or self.max_samples <= 0
        ):
            raise ValueError("max_samples must be a positive integer")


class ConnectivityStatus(StrEnum):
    """Whether retained target material is still joined to a known anchor."""

    CONNECTED = "connected"
    DETACHED = "detached"
    UNKNOWN = "unknown"


class SimulationIssueKind(StrEnum):
    """Blocking safety conditions produced or propagated by simulation."""

    GOUGE = "gouge"
    COLLISION = "collision"
    DETACHED = "detached"


@dataclass(frozen=True, slots=True)
class SimulationIssue:
    """One typed blocking condition with a deterministic human-readable detail."""

    kind: SimulationIssueKind
    message: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, SimulationIssueKind):
            raise TypeError("kind must be a SimulationIssueKind")
        if not self.message:
            raise ValueError("message must not be empty")


def _immutable_mask(value: npt.ArrayLike, shape: tuple[int, int, int]) -> BoolArray:
    result = np.array(value, dtype=np.bool_, copy=True)
    if result.shape != shape:
        raise ValueError(f"simulation mask must have XYZ shape {shape}")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class VolumetricSimulationReport:
    """Immutable final stock and safety evidence for one simulated plan."""

    final_stock: StockVolume
    removed: BoolArray = field(repr=False, compare=False)
    residual: BoolArray = field(repr=False, compare=False)
    gouge: BoolArray = field(repr=False, compare=False)
    connectivity: ConnectivityStatus
    issues: tuple[SimulationIssue, ...] = ()
    collision_reports: tuple[CollisionReport, ...] = field(
        default=(), repr=False, compare=False
    )
    sample_count: int = 0

    def __post_init__(self) -> None:
        shape = self.final_stock.lattice.shape
        object.__setattr__(self, "removed", _immutable_mask(self.removed, shape))
        object.__setattr__(self, "residual", _immutable_mask(self.residual, shape))
        object.__setattr__(self, "gouge", _immutable_mask(self.gouge, shape))
        if not isinstance(self.connectivity, ConnectivityStatus):
            raise TypeError("connectivity must be a ConnectivityStatus")
        if (
            isinstance(self.sample_count, bool)
            or not isinstance(self.sample_count, int)
            or self.sample_count < 0
        ):
            raise ValueError("sample_count must be a non-negative integer")

    @property
    def blocking(self) -> bool:
        """Return whether export must be blocked by simulation evidence."""

        return bool(self.issues)

    @property
    def removed_voxels(self) -> int:
        return int(np.count_nonzero(self.removed))

    @property
    def residual_voxels(self) -> int:
        return int(np.count_nonzero(self.residual))

    @property
    def gouge_voxels(self) -> int:
        return int(np.count_nonzero(self.gouge))


def _same_lattice(left: VoxelLattice, right: VoxelLattice) -> bool:
    return (
        left.shape == right.shape
        and left.origin == right.origin
        and left.spacing == right.spacing
    )


def _voxel_centres(lattice: VoxelLattice, mask: BoolArray) -> FloatArray:
    indices = np.argwhere(mask).astype(np.float64, copy=False)
    return np.asarray(
        np.asarray(lattice.origin, dtype=np.float64)
        + indices * np.asarray(lattice.spacing, dtype=np.float64),
        dtype=np.float64,
    )


def _tool_coordinates(
    points_g54: FloatArray, pose: MachinePose, spindle_axis: tuple[float, float, float]
) -> FloatArray:
    """Express G54 points in the tool's axially symmetric TCP frame."""

    offset = points_g54 - np.asarray((pose.x, pose.y, pose.z), dtype=np.float64)
    cutting_direction = np.asarray(spindle_axis, dtype=np.float64)
    cutting_direction /= np.linalg.norm(cutting_direction)
    toward_spindle = -cutting_direction
    axial = offset @ toward_spindle
    squared_radius = np.maximum(
        np.einsum("ij,ij->i", offset, offset) - axial * axial, 0.0
    )
    radial = np.sqrt(squared_radius)
    return np.column_stack((radial, np.zeros_like(radial), axial))


def _interpolate(start: MachinePose, end: MachinePose, fraction: float) -> MachinePose:
    return MachinePose(
        x=start.x + fraction * (end.x - start.x),
        y=start.y + fraction * (end.y - start.y),
        z=start.z + fraction * (end.z - start.z),
        a=start.a + fraction * (end.a - start.a),
    )


def _rotary_radius(lattice: VoxelLattice, kinematics: XYZAKinematics) -> float:
    axes = tuple(lattice.axis_centres(axis) for axis in range(3))
    corners = np.asarray(
        [
            (x, y, z)
            for x in (axes[0][0], axes[0][-1])
            for y in (axes[1][0], axes[1][-1])
            for z in (axes[2][0], axes[2][-1])
        ],
        dtype=np.float64,
    )
    at_zero = kinematics.part_to_g54(corners, 0.0)
    pivot_y = kinematics.configuration.rotary_pivot_y
    pivot_z = kinematics.configuration.rotary_pivot_z
    return float(
        np.max(np.hypot(at_zero[:, 1] - pivot_y, at_zero[:, 2] - pivot_z))
    )


def _sample_count(
    start: MachinePose, end: MachinePose, rotary_radius: float, step: float
) -> int:
    translation = math.dist((start.x, start.y, start.z), (end.x, end.y, end.z))
    rotary_arc = abs(math.radians(end.a - start.a)) * rotary_radius
    return max(1, math.ceil((translation + rotary_arc) / step))


def _anchor_connectivity(
    stock: BoolArray, target: BoolArray, anchor_mask: BoolArray | None
) -> ConnectivityStatus:
    if anchor_mask is None:
        return ConnectivityStatus.UNKNOWN
    reachable = np.zeros(stock.shape, dtype=np.bool_)
    seeds = np.argwhere(stock & anchor_mask)
    queue: deque[tuple[int, int, int]] = deque()
    for raw in seeds:
        index = (int(raw[0]), int(raw[1]), int(raw[2]))
        reachable[index] = True
        queue.append(index)
    shape = stock.shape
    while queue:
        x, y, z = queue.popleft()
        for neighbor in (
            (x - 1, y, z),
            (x + 1, y, z),
            (x, y - 1, z),
            (x, y + 1, z),
            (x, y, z - 1),
            (x, y, z + 1),
        ):
            nx, ny, nz = neighbor
            if (
                0 <= nx < shape[0]
                and 0 <= ny < shape[1]
                and 0 <= nz < shape[2]
                and stock[neighbor]
                and not reachable[neighbor]
            ):
                reachable[neighbor] = True
                queue.append(neighbor)
    return (
        ConnectivityStatus.CONNECTED
        if not np.any((stock & target) & ~reachable)
        else ConnectivityStatus.DETACHED
    )


def simulate_volumetric_motion(
    initial: StockVolume,
    target: SolidVolume,
    tool: ToolAssembly,
    kinematics: XYZAKinematics,
    start: MachinePose,
    blocks: Sequence[MotionBlock],
    *,
    settings: VolumetricSimulationSettings,
    anchor_mask: npt.ArrayLike | None = None,
    collision_reports: Sequence[CollisionReport] | None = None,
) -> VolumetricSimulationReport:
    """Subtract only the cutter swept by a conservatively sampled XYZA plan."""

    if not _same_lattice(initial.lattice, target.lattice):
        raise VolumetricSimulationError("initial and target must use identical lattices")
    motions = tuple(blocks)
    if not all(isinstance(block, MotionBlock) for block in motions):
        raise TypeError("blocks must contain MotionBlock values")
    reports = tuple(collision_reports or ())
    if collision_reports is not None and len(reports) != len(motions):
        raise ValueError("collision_reports must contain one report per motion block")
    if not all(isinstance(report, CollisionReport) for report in reports):
        raise TypeError("collision_reports must contain CollisionReport values")

    anchor = (
        None
        if anchor_mask is None
        else _immutable_mask(anchor_mask, initial.lattice.shape)
    )
    occupied = np.array(initial.occupation, dtype=np.bool_, copy=True)
    old = occupied.copy()
    target_mask = np.asarray(target.occupation, dtype=np.bool_)
    if np.any(target_mask & ~old):
        raise VolumetricSimulationError("target must be contained in the initial stock")
    voxel_half_diagonal = 0.5 * math.sqrt(
        sum(spacing * spacing for spacing in initial.lattice.spacing)
    )
    step = min(settings.tolerance, voxel_half_diagonal)
    rotary_radius = _rotary_radius(initial.lattice, kinematics)

    segment_starts = (
        (start, *(block.pose for block in motions[:-1])) if motions else ()
    )
    intervals = tuple(
        _sample_count(segment_start, block.pose, rotary_radius, step)
        for segment_start, block in zip(
            segment_starts, motions, strict=True
        )
    )
    required_samples = sum(
        interval_count + 1
        for block, interval_count in zip(motions, intervals, strict=True)
        if block.kind is MotionKind.LINEAR
    )
    if required_samples > settings.max_samples:
        raise SimulationBudgetExceeded(
            f"simulation requires {required_samples} samples, budget is "
            f"{settings.max_samples}"
        )

    def remove_at(pose: MachinePose) -> None:
        part_points = _voxel_centres(initial.lattice, occupied)
        if part_points.size == 0:
            return
        points_g54 = kinematics.part_to_g54(part_points, pose.a)
        local = _tool_coordinates(
            points_g54, pose, kinematics.configuration.spindle_axis
        )
        remove = tool.cutter_contains_points(local)
        occupied[tuple(np.argwhere(occupied)[remove].T)] = False

    segment_start = start
    for block, interval_count in zip(motions, intervals, strict=True):
        if block.kind is MotionKind.LINEAR:
            remove_at(segment_start)
            for index in range(1, interval_count + 1):
                remove_at(_interpolate(segment_start, block.pose, index / interval_count))
        segment_start = block.pose

    removed = old & ~occupied
    residual = occupied & ~target_mask
    gouge = target_mask & ~occupied
    connectivity = _anchor_connectivity(occupied, target_mask, anchor)
    issues: list[SimulationIssue] = []
    gouge_count = int(np.count_nonzero(gouge))
    if gouge_count:
        issues.append(
            SimulationIssue(
                SimulationIssueKind.GOUGE,
                f"cutter removed {gouge_count} target voxels",
            )
        )
    collision_count = sum(len(report.events) for report in reports)
    if collision_count:
        issues.append(
            SimulationIssue(
                SimulationIssueKind.COLLISION,
                f"collision validation reported {collision_count} events",
            )
        )
    if connectivity is ConnectivityStatus.DETACHED:
        issues.append(
            SimulationIssue(
                SimulationIssueKind.DETACHED,
                "retained target material is detached from the configured anchor",
            )
        )
    final_stock = StockVolume.from_dense(
        initial.lattice, occupied, brick_size=initial.brick_shape[0]
    )
    return VolumetricSimulationReport(
        final_stock=final_stock,
        removed=removed,
        residual=residual,
        gouge=gouge,
        connectivity=connectivity,
        issues=tuple(issues),
        collision_reports=reports,
        sample_count=required_samples,
    )


__all__ = [
    "ConnectivityStatus",
    "SimulationBudgetExceeded",
    "SimulationIssue",
    "SimulationIssueKind",
    "VolumetricSimulationError",
    "VolumetricSimulationReport",
    "VolumetricSimulationSettings",
    "simulate_volumetric_motion",
]
