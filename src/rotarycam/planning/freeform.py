"""Deterministic freeform planning from volumetric accessibility candidates."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import numpy.typing as npt

from rotarycam.accessibility import AccessibilityReport, CandidatePose
from rotarycam.config import MachineDefinition
from rotarycam.machine import XYZAKinematics
from rotarycam.motion import MachinePose, MotionBlock, MotionKind
from rotarycam.tools import ToolAssembly, ToolType
from rotarycam.volumetric import SolidVolume, StockVolume

Index3 = tuple[int, int, int]
_NEIGHBOURS: tuple[Index3, ...] = (
    (-1, 0, 0),
    (1, 0, 0),
    (0, -1, 0),
    (0, 1, 0),
    (0, 0, -1),
    (0, 0, 1),
)


class FreeformOperationKind(StrEnum):
    ROUGHING = "roughing"
    FINISHING = "finishing"
    REST = "rest"


@dataclass(frozen=True, slots=True)
class FreeformPlannerSettings:
    safe_clearance: float = 5.0
    roughing_allowance: float = 0.5
    max_link_distance: float = 10.0
    min_duration_s: float = 0.001

    def __post_init__(self) -> None:
        for name, value in (
            ("safe_clearance", self.safe_clearance),
            ("max_link_distance", self.max_link_distance),
            ("min_duration_s", self.min_duration_s),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and greater than zero")
        if not math.isfinite(self.roughing_allowance) or self.roughing_allowance < 0.0:
            raise ValueError("roughing_allowance must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class PlannedPass:
    kind: FreeformOperationKind
    tool_number: int
    surface_indices: tuple[Index3, ...]
    start_pose: MachinePose
    blocks: tuple[MotionBlock, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.kind, FreeformOperationKind):
            raise TypeError("kind must be a FreeformOperationKind")
        if self.tool_number <= 0:
            raise ValueError("tool_number must be greater than zero")
        if not self.surface_indices:
            raise ValueError("surface_indices must not be empty")
        if not isinstance(self.start_pose, MachinePose):
            raise TypeError("start_pose must be a MachinePose")
        if not self.blocks:
            raise ValueError("blocks must not be empty")


@dataclass(frozen=True, slots=True)
class FreeformPlan:
    passes: tuple[PlannedPass, ...]
    accessibility: AccessibilityReport

    @property
    def blocks(self) -> tuple[MotionBlock, ...]:
        return tuple(block for planned_pass in self.passes for block in planned_pass.blocks)

    @property
    def inaccessible_digest(self) -> str:
        return self.accessibility.digest

    @property
    def operation_kinds(self) -> tuple[FreeformOperationKind, ...]:
        return tuple(dict.fromkeys(planned_pass.kind for planned_pass in self.passes))


def unwrap_angle(previous_deg: float, candidate_deg: float) -> float:
    """Choose the equivalent candidate angle nearest an unwrapped predecessor."""

    if not math.isfinite(previous_deg) or not math.isfinite(candidate_deg):
        raise ValueError("angles must be finite")
    normalized = candidate_deg % 360.0
    base_turn = math.floor((previous_deg - normalized) / 360.0)
    choices = tuple(normalized + 360.0 * (base_turn + offset) for offset in (0, 1))
    return min(choices, key=lambda value: (abs(value - previous_deg), value))


def _index_paths(indices: Iterable[Index3]) -> tuple[tuple[Index3, ...], ...]:
    remaining = set(indices)
    paths: list[tuple[Index3, ...]] = []
    while remaining:
        current = min(remaining)
        remaining.remove(current)
        path = [current]
        while True:
            neighbours: list[Index3] = []
            for delta in _NEIGHBOURS:
                candidate: Index3 = tuple(
                    current[axis] + delta[axis] for axis in range(3)
                )  # type: ignore[assignment]
                if candidate in remaining:
                    neighbours.append(candidate)
            if not neighbours:
                break
            current = min(neighbours)
            remaining.remove(current)
            path.append(current)
        paths.append(tuple(path))
    return tuple(paths)


@dataclass(frozen=True, slots=True)
class _PoseState:
    cost: float
    poses: tuple[MachinePose, ...]
    candidates: tuple[CandidatePose, ...]


def _pose_distance(left: MachinePose, right: MachinePose) -> float:
    return math.dist((left.x, left.y, left.z), (right.x, right.y, right.z))


def _with_angle(pose: MachinePose, angle: float) -> MachinePose:
    return MachinePose(pose.x, pose.y, pose.z, angle)


def _optimize_pose_sequence(
    candidate_groups: Sequence[tuple[CandidatePose, ...]],
    reference: MachinePose | None,
) -> tuple[tuple[CandidatePose, MachinePose], ...]:
    """Dynamic-program candidate orientations with stable winding tie-breaks."""

    states: list[_PoseState] = []
    for candidate in candidate_groups[0]:
        if reference is None:
            pose = _with_angle(candidate.pose, candidate.pose.a % 360.0)
            cost = pose.a
        else:
            pose = _with_angle(candidate.pose, unwrap_angle(reference.a, candidate.pose.a))
            cost = abs(pose.a - reference.a) + _pose_distance(reference, pose)
        states.append(_PoseState(cost, (pose,), (candidate,)))
    for group in candidate_groups[1:]:
        next_states: list[_PoseState] = []
        for candidate in group:
            choices: list[_PoseState] = []
            for state in states:
                previous = state.poses[-1]
                pose = _with_angle(candidate.pose, unwrap_angle(previous.a, candidate.pose.a))
                cost = state.cost + abs(pose.a - previous.a) + _pose_distance(previous, pose)
                choices.append(
                    _PoseState(cost, (*state.poses, pose), (*state.candidates, candidate))
                )
            next_states.append(
                min(
                    choices,
                    key=lambda state: (
                        state.cost,
                        tuple(pose.a for pose in state.poses),
                        tuple(item.tool_number for item in state.candidates),
                    ),
                )
            )
        states = next_states
    best = min(
        states,
        key=lambda state: (
            state.cost,
            tuple(pose.a for pose in state.poses),
            tuple(item.tool_number for item in state.candidates),
        ),
    )
    return tuple(zip(best.candidates, best.poses, strict=True))


def _split_by_distance(
    selected: tuple[tuple[CandidatePose, MachinePose], ...], maximum: float
) -> tuple[tuple[tuple[CandidatePose, MachinePose], ...], ...]:
    paths: list[list[tuple[CandidatePose, MachinePose]]] = [[selected[0]]]
    for item in selected[1:]:
        if _pose_distance(paths[-1][-1][1], item[1]) > maximum:
            paths.append([item])
        else:
            paths[-1].append(item)
    return tuple(tuple(path) for path in paths)


def _retracted(
    pose: MachinePose, machine: MachineDefinition, clearance: float
) -> MachinePose:
    configuration = machine.xyza_configuration
    if configuration is None:
        raise ValueError("machine profile has no XYZA configuration")
    axis = np.asarray(configuration.spindle_axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    displacement = -axis * clearance
    return MachinePose(
        pose.x + float(displacement[0]),
        pose.y + float(displacement[1]),
        pose.z + float(displacement[2]),
        pose.a,
    )


def _duration(
    start: MachinePose, end: MachinePose, feed: float, minimum: float
) -> float:
    return max(minimum, 60.0 * _pose_distance(start, end) / feed)


def _roughing_pose(
    candidate: CandidatePose,
    pose: MachinePose,
    allowance: float,
    kinematics: XYZAKinematics,
) -> MachinePose:
    if allowance == 0.0:
        return pose
    point = np.asarray(candidate.part_point, dtype=np.float64)
    normal = np.asarray(candidate.normal, dtype=np.float64)
    transformed = kinematics.part_to_g54(point + allowance * normal, pose.a)
    return MachinePose(
        float(transformed[0]), float(transformed[1]), float(transformed[2]), pose.a
    )


def _candidate_map(
    report: AccessibilityReport,
) -> dict[Index3, tuple[CandidatePose, ...]]:
    values: dict[Index3, list[CandidatePose]] = {}
    for candidate in report.candidates:
        values.setdefault(candidate.surface_index, []).append(candidate)
    return {
        index: tuple(
            sorted(
                candidates,
                key=lambda item: (
                    item.tool_number,
                    item.pose.a,
                    item.pose.x,
                    item.pose.y,
                    item.pose.z,
                ),
            )
        )
        for index, candidates in values.items()
    }


def _nearby_roughing_indices(
    indices: set[Index3],
    residual: npt.NDArray[np.bool_],
    target: SolidVolume,
    tool: ToolAssembly,
    allowance: float,
) -> set[Index3]:
    residual_indices = np.argwhere(residual)
    if residual_indices.size == 0:
        return set()
    spacing = np.asarray(target.lattice.spacing, dtype=np.float64)
    reach = tool.tool.cutting_length + allowance
    return {
        index
        for index in indices
        if float(
            np.min(
                np.linalg.norm(
                    (residual_indices - np.asarray(index, dtype=np.int64)) * spacing,
                    axis=1,
                )
            )
        )
        <= reach
    }


def _require_pose_in_travel(pose: MachinePose, machine: MachineDefinition) -> None:
    limits = (machine.x_limits, machine.y_limits, machine.z_limits)
    for axis, value, axis_limits in zip("XYZ", (pose.x, pose.y, pose.z), limits, strict=True):
        if axis_limits is None or not axis_limits.minimum <= value <= axis_limits.maximum:
            raise ValueError(f"generated {axis} coordinate lies outside machine travel")


def plan_freeform(
    initial: StockVolume,
    target: SolidVolume,
    accessibility: AccessibilityReport,
    tools: Sequence[ToolAssembly],
    machine: MachineDefinition,
    settings: FreeformPlannerSettings | None = None,
) -> FreeformPlan:
    """Create stock-aware roughing, finishing, and rest passes."""

    selected_settings = settings or FreeformPlannerSettings()
    if initial.lattice != target.lattice:
        raise ValueError("initial and target must use the same lattice")
    if accessibility.accessible_mask.shape != target.lattice.shape:
        raise ValueError("accessibility masks must match the target lattice")
    initial_mask = initial.to_dense()
    target_mask = target.to_dense()
    if np.any(target_mask & ~initial_mask):
        raise ValueError("target must be contained in the initial stock")
    ordered_tools = tuple(
        sorted(tools, key=lambda item: (-item.tool.diameter, item.tool.number))
    )
    if not ordered_tools:
        raise ValueError("at least one tool assembly is required")
    if len({tool.tool.number for tool in ordered_tools}) != len(ordered_tools):
        raise ValueError("tool numbers must be unique")
    supported = {ToolType.FLAT, ToolType.BALL, ToolType.TAPERED}
    if any(tool.tool.tool_type not in supported for tool in ordered_tools):
        raise ValueError("unsupported tool type")
    tool_by_number = {tool.tool.number: tool for tool in ordered_tools}
    candidates = _candidate_map(accessibility)
    unknown_tools = {
        item.tool_number for group in candidates.values() for item in group
    } - set(tool_by_number)
    if unknown_tools:
        raise ValueError("accessibility candidates reference unknown tools")

    primary_number = ordered_tools[0].tool.number
    primary_indices = {
        index
        for index, group in candidates.items()
        if any(item.tool_number == primary_number for item in group)
    }
    rest_indices = set(candidates) - primary_indices
    roughing_indices = _nearby_roughing_indices(
        primary_indices,
        np.asarray(initial_mask & ~target_mask, dtype=np.bool_),
        target,
        ordered_tools[0],
        selected_settings.roughing_allowance,
    )
    phases: list[tuple[FreeformOperationKind, set[Index3], int | None]] = []
    if roughing_indices:
        phases.append((FreeformOperationKind.ROUGHING, roughing_indices, primary_number))
    if primary_indices:
        phases.append((FreeformOperationKind.FINISHING, primary_indices, primary_number))
    if rest_indices:
        phases.append((FreeformOperationKind.REST, rest_indices, None))

    kinematics = XYZAKinematics.from_machine(machine)
    passes: list[PlannedPass] = []
    current_safe: MachinePose | None = None
    for kind, indices, required_tool in phases:
        for index_path in _index_paths(indices):
            candidate_groups: list[tuple[CandidatePose, ...]] = []
            for index in index_path:
                group = candidates[index]
                if required_tool is not None:
                    group = tuple(item for item in group if item.tool_number == required_tool)
                else:
                    best_number = min(
                        (item.tool_number for item in group),
                        key=lambda number: (-tool_by_number[number].tool.diameter, number),
                    )
                    group = tuple(item for item in group if item.tool_number == best_number)
                candidate_groups.append(group)
            selected = _optimize_pose_sequence(candidate_groups, current_safe)
            if kind is FreeformOperationKind.ROUGHING:
                selected = tuple(
                    (
                        candidate,
                        _roughing_pose(
                            candidate,
                            pose,
                            selected_settings.roughing_allowance,
                            kinematics,
                        ),
                    )
                    for candidate, pose in selected
                )
            for selected_path in _split_by_distance(
                selected, selected_settings.max_link_distance
            ):
                tool_number = selected_path[0][0].tool_number
                tool = tool_by_number[tool_number].tool
                first_pose = selected_path[0][1]
                first_safe = _retracted(first_pose, machine, selected_settings.safe_clearance)
                _require_pose_in_travel(first_pose, machine)
                _require_pose_in_travel(first_safe, machine)
                start_pose = current_safe or first_safe
                blocks: list[MotionBlock] = []
                if start_pose != first_safe:
                    blocks.append(
                        MotionBlock(
                            first_safe,
                            MotionKind.RAPID,
                            _duration(
                                start_pose,
                                first_safe,
                                tool.feed,
                                selected_settings.min_duration_s,
                            ),
                        )
                    )
                previous = first_safe
                for _, pose in selected_path:
                    _require_pose_in_travel(pose, machine)
                    blocks.append(
                        MotionBlock(
                            pose,
                            MotionKind.LINEAR,
                            _duration(
                                previous,
                                pose,
                                tool.feed,
                                selected_settings.min_duration_s,
                            ),
                            tool.feed,
                        )
                    )
                    previous = pose
                final_safe = _retracted(
                    selected_path[-1][1], machine, selected_settings.safe_clearance
                )
                _require_pose_in_travel(final_safe, machine)
                blocks.append(
                    MotionBlock(
                        final_safe,
                        MotionKind.RAPID,
                        _duration(
                            previous,
                            final_safe,
                            tool.feed,
                            selected_settings.min_duration_s,
                        ),
                    )
                )
                passes.append(
                    PlannedPass(
                        kind,
                        tool_number,
                        tuple(item[0].surface_index for item in selected_path),
                        start_pose,
                        tuple(blocks),
                    )
                )
                current_safe = final_safe
    return FreeformPlan(tuple(passes), accessibility)


__all__ = [
    "FreeformOperationKind",
    "FreeformPlan",
    "FreeformPlannerSettings",
    "PlannedPass",
    "plan_freeform",
    "unwrap_angle",
]
