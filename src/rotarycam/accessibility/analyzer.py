"""Deterministic volumetric surface accessibility analysis."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from rotarycam.accessibility.digest import digest_accessibility_context
from rotarycam.accessibility.models import (
    AccessibilityReport,
    AccessibilitySettings,
    CandidatePose,
    InaccessibilityReason,
    InaccessibleRegion,
)
from rotarycam.collisions import CollisionKind, CollisionReport, ToolComponent
from rotarycam.config import AxisLimits, MachineDefinition
from rotarycam.machine import XYZAKinematics
from rotarycam.tools import ToolAssembly
from rotarycam.volumetric import SolidVolume

BoolArray = npt.NDArray[np.bool_]
Index3 = tuple[int, int, int]
CollisionCallback = Callable[[CandidatePose, ToolAssembly], CollisionReport | None]

_NEIGHBOURS: tuple[Index3, ...] = (
    (-1, 0, 0),
    (1, 0, 0),
    (0, -1, 0),
    (0, 1, 0),
    (0, 0, -1),
    (0, 0, 1),
)


def _inside(index: Index3, shape: tuple[int, int, int]) -> bool:
    return all(0 <= index[axis] < shape[axis] for axis in range(3))


def _surface_mask(occupation: BoolArray) -> BoolArray:
    surface = np.zeros_like(occupation)
    for index_array in np.argwhere(occupation):
        index: Index3 = tuple(int(value) for value in index_array)  # type: ignore[assignment]
        for delta in _NEIGHBOURS:
            neighbour: Index3 = (
                index[0] + delta[0],
                index[1] + delta[1],
                index[2] + delta[2],
            )
            if not _inside(neighbour, occupation.shape) or not bool(
                occupation[neighbour]
            ):
                surface[index] = True
                break
    return surface


def _normal(
    occupation: BoolArray,
    index: Index3,
    spacing: tuple[float, float, float],
) -> tuple[float, float, float] | None:
    gradient: list[float] = []
    for axis in range(3):
        lower = list(index)
        upper = list(index)
        lower[axis] -= 1
        upper[axis] += 1
        lower_index: Index3 = tuple(lower)  # type: ignore[assignment]
        upper_index: Index3 = tuple(upper)  # type: ignore[assignment]
        lower_value = (
            float(occupation[lower_index])
            if _inside(lower_index, occupation.shape)
            else 0.0
        )
        upper_value = (
            float(occupation[upper_index])
            if _inside(upper_index, occupation.shape)
            else 0.0
        )
        gradient.append((lower_value - upper_value) / (2.0 * spacing[axis]))
    magnitude = math.sqrt(sum(value * value for value in gradient))
    if magnitude <= 1e-12:
        return None
    return tuple(value / magnitude for value in gradient)  # type: ignore[return-value]


def _voxel_center(target: SolidVolume, index: Index3) -> tuple[float, float, float]:
    return tuple(
        target.lattice.origin[axis] + index[axis] * target.lattice.spacing[axis]
        for axis in range(3)
    )  # type: ignore[return-value]


def _surface_point(
    target: SolidVolume,
    index: Index3,
    normal: tuple[float, float, float],
) -> tuple[float, float, float]:
    center = _voxel_center(target, index)
    face_offset = 0.5 * math.sqrt(
        sum(
            (normal[axis] * target.lattice.spacing[axis]) ** 2
            for axis in range(3)
        )
    )
    return tuple(
        center[axis] + normal[axis] * face_offset for axis in range(3)
    )  # type: ignore[return-value]


def _within(value: float, limits: AxisLimits | None, tolerance: float) -> bool:
    return limits is not None and limits.minimum - tolerance <= value <= limits.maximum + tolerance


def _collision_reasons(report: CollisionReport | None) -> set[InaccessibilityReason]:
    reasons: set[InaccessibilityReason] = set()
    if report is None:
        return reasons
    for event in report.events:
        if event.kind is CollisionKind.MACHINE:
            reasons.add(InaccessibilityReason.MACHINE_COLLISION)
        if event.kind is CollisionKind.FIXTURE:
            reasons.add(InaccessibilityReason.FIXTURE_COLLISION)
        if event.kind is CollisionKind.STOCK:
            reasons.add(InaccessibilityReason.OCCLUDED)
        if event.tool_component is not ToolComponent.CUTTER:
            reasons.add(InaccessibilityReason.NON_CUTTING_COLLISION)
    return reasons


def _approach_reasons(
    target: SolidVolume,
    occupation: BoolArray,
    index: Index3,
    angle: float,
    tool: ToolAssembly,
    kinematics: XYZAKinematics,
) -> set[InaccessibilityReason]:
    """Conservatively test the voxel column between contact and spindle."""

    spindle_axis = np.asarray(kinematics.configuration.spindle_axis, dtype=np.float64)
    spindle_axis /= np.linalg.norm(spindle_axis)
    approach = kinematics.inverse_matrix_at(angle)[:3, :3] @ -spindle_axis
    axis = int(np.argmax(np.abs(approach)))
    direction = 1 if approach[axis] >= 0.0 else -1
    cursor = list(index)
    occupied_length = 0.0
    reasons: set[InaccessibilityReason] = set()
    while True:
        cursor[axis] += direction
        probe: Index3 = tuple(cursor)  # type: ignore[assignment]
        if not _inside(probe, occupation.shape) or not bool(occupation[probe]):
            break
        reasons.add(InaccessibilityReason.OCCLUDED)
        occupied_length += target.lattice.spacing[axis]
    if occupied_length > tool.tool.cutting_length:
        reasons.add(InaccessibilityReason.CUTTER_REACH)
    return reasons


def _connected_regions(
    mask: BoolArray,
    reasons_by_index: dict[Index3, set[InaccessibilityReason]],
    error_by_index: dict[Index3, float],
) -> tuple[InaccessibleRegion, ...]:
    remaining: set[Index3] = {
        (int(item[0]), int(item[1]), int(item[2])) for item in np.argwhere(mask)
    }
    regions: list[InaccessibleRegion] = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        queue: deque[Index3] = deque((start,))
        indices: list[Index3] = []
        reasons: set[InaccessibilityReason] = set()
        max_error = 0.0
        while queue:
            index = queue.popleft()
            indices.append(index)
            reasons.update(reasons_by_index[index])
            max_error = max(max_error, error_by_index[index])
            for delta in _NEIGHBOURS:
                neighbour: Index3 = tuple(index[axis] + delta[axis] for axis in range(3))  # type: ignore[assignment]
                if neighbour in remaining:
                    remaining.remove(neighbour)
                    queue.append(neighbour)
        regions.append(
            InaccessibleRegion(tuple(indices), len(indices), max_error, tuple(reasons))
        )
    return tuple(regions)


@dataclass(frozen=True, slots=True)
class AccessibilityAnalyzer:
    """Generate valid XYZA contact poses for every occupied surface voxel."""

    machine: MachineDefinition
    settings: AccessibilitySettings
    collision_check: CollisionCallback | None = None

    def analyze(
        self,
        target: SolidVolume,
        tools: Sequence[ToolAssembly],
        *,
        plan_fingerprint: str = "",
    ) -> AccessibilityReport:
        if not tools:
            raise ValueError("at least one tool assembly is required")
        ordered_tools = tuple(sorted(tools, key=lambda item: item.tool.number))
        if len({tool.tool.number for tool in ordered_tools}) != len(ordered_tools):
            raise ValueError("tool assembly numbers must be unique")
        kinematics = XYZAKinematics.from_machine(self.machine)
        occupation = np.array(target.to_dense(), dtype=np.bool_, copy=True)
        surface = _surface_mask(occupation)
        accessible = np.zeros(target.lattice.shape, dtype=np.bool_)
        inaccessible = np.zeros(target.lattice.shape, dtype=np.bool_)
        candidates: list[CandidatePose] = []
        reasons_by_index: dict[Index3, set[InaccessibilityReason]] = {}
        error_by_index: dict[Index3, float] = {}
        quantization_error = 0.5 * math.sqrt(
            sum(component * component for component in target.lattice.spacing)
        )

        for item in np.argwhere(surface):
            index: Index3 = tuple(int(value) for value in item)  # type: ignore[assignment]
            normal = _normal(occupation, index, target.lattice.spacing)
            failures: set[InaccessibilityReason] = set()
            if normal is None:
                failures.add(InaccessibilityReason.TOPOLOGY)
            if quantization_error > self.settings.max_error:
                failures.add(InaccessibilityReason.TOPOLOGY)
            if normal is not None and not failures:
                point = _surface_point(target, index, normal)
                for tool in ordered_tools:
                    for angle in self.settings.angles_deg:
                        pose = kinematics.machine_pose_for_part_point(
                            np.asarray(point, dtype=np.float64), angle
                        )
                        attempt = CandidatePose(
                            index,
                            point,
                            normal,
                            tool.tool.number,
                            pose,
                            quantization_error,
                        )
                        attempt_failures: set[InaccessibilityReason] = set()
                        if not (
                            _within(pose.x, self.machine.x_limits, self.settings.tolerance)
                            and _within(pose.y, self.machine.y_limits, self.settings.tolerance)
                            and _within(pose.z, self.machine.z_limits, self.settings.tolerance)
                        ):
                            attempt_failures.add(InaccessibilityReason.TRAVEL)
                        attempt_failures.update(
                            _approach_reasons(target, occupation, index, angle, tool, kinematics)
                        )
                        if self.collision_check is not None:
                            attempt_failures.update(
                                _collision_reasons(self.collision_check(attempt, tool))
                            )
                        if attempt_failures:
                            failures.update(attempt_failures)
                        else:
                            candidates.append(attempt)
                            accessible[index] = True
            if not accessible[index]:
                inaccessible[index] = True
                reasons_by_index[index] = failures or {InaccessibilityReason.TOPOLOGY}
                error_by_index[index] = quantization_error

        digest = digest_accessibility_context(
            target,
            ordered_tools,
            self.machine,
            self.settings,
            plan_fingerprint,
        )
        regions = _connected_regions(inaccessible, reasons_by_index, error_by_index)
        return AccessibilityReport(
            tuple(candidates), accessible, inaccessible, regions, digest
        )


def analyze_accessibility(
    target: SolidVolume,
    tools: Sequence[ToolAssembly],
    machine: MachineDefinition,
    settings: AccessibilitySettings,
    *,
    plan_fingerprint: str = "",
    collision_check: CollisionCallback | None = None,
) -> AccessibilityReport:
    """Functional wrapper for :class:`AccessibilityAnalyzer`."""

    return AccessibilityAnalyzer(machine, settings, collision_check).analyze(
        target, tools, plan_fingerprint=plan_fingerprint
    )


__all__ = ["AccessibilityAnalyzer", "CollisionCallback", "analyze_accessibility"]
