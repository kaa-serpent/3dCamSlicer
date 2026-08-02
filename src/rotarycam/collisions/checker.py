"""Conservative continuous validation of one XYZA motion segment."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from rotarycam.collisions.models import (
    CollisionBudgetExceeded,
    CollisionEvent,
    CollisionKind,
    CollisionReport,
    SweptCollisionSettings,
    ToolComponent,
)
from rotarycam.collisions.obstacles import CollisionObstacle
from rotarycam.machine import FrameKind
from rotarycam.motion import MachinePose, MotionBlock, MotionKind
from rotarycam.tools import ToolAssembly


def _interpolate(start: MachinePose, end: MachinePose, fraction: float) -> MachinePose:
    return MachinePose(
        start.x + fraction * (end.x - start.x),
        start.y + fraction * (end.y - start.y),
        start.z + fraction * (end.z - start.z),
        start.a + fraction * (end.a - start.a),
    )


@dataclass(frozen=True, slots=True)
class _ToolEnvelope:
    component: ToolComponent
    center_z: float
    radius: float


def _sphere_for_interval(
    component: ToolComponent, z_min: float, z_max: float, radial_radius: float
) -> _ToolEnvelope:
    half_length = 0.5 * (z_max - z_min)
    return _ToolEnvelope(
        component,
        0.5 * (z_min + z_max),
        math.hypot(half_length, radial_radius),
    )


def _tool_envelopes(tool: ToolAssembly) -> tuple[_ToolEnvelope, ...]:
    """Return conservative component spheres without conflating cutting contact."""

    definition = tool.tool
    tool_radius = 0.5 * definition.diameter
    values = [
        _sphere_for_interval(
            ToolComponent.CUTTER, 0.0, definition.cutting_length, tool_radius
        )
    ]
    if definition.flute_length > definition.cutting_length:
        values.append(
            _sphere_for_interval(
                ToolComponent.NON_CUTTING,
                definition.cutting_length,
                definition.flute_length,
                tool_radius,
            )
        )
    exposed = definition.stickout or definition.overall_length
    if exposed > definition.flute_length:
        values.append(
            _sphere_for_interval(
                ToolComponent.SHANK,
                definition.flute_length,
                exposed,
                0.5 * definition.shank_diameter,
            )
        )
    if definition.holder is not None and definition.stickout is not None:
        values.append(
            _sphere_for_interval(
                ToolComponent.HOLDER,
                exposed,
                exposed + definition.holder.length,
                0.5 * definition.holder.diameter,
            )
        )
    return tuple(values)


@dataclass(slots=True)
class _SearchBudget:
    maximum: int
    used: int = 0

    def take(self, obstacle: str) -> None:
        if self.used >= self.maximum:
            raise CollisionBudgetExceeded(
                f"cannot prove clearance from {obstacle!r} within "
                f"{self.maximum} subdivisions"
            )
        self.used += 1


class SweptCollisionChecker:
    """Check the full tool envelope over linearly interpolated XYZA poses.

    Signed-distance lower bounds prove clear intervals. Ambiguous intervals are
    bisected until their maximum possible relative motion is no greater than
    ``settings.tolerance``; such an interval is reported conservatively rather
    than silently accepted.
    """

    def __init__(
        self,
        tool: ToolAssembly,
        obstacles: tuple[CollisionObstacle, ...] | list[CollisionObstacle],
        settings: SweptCollisionSettings | None = None,
    ) -> None:
        if not isinstance(tool, ToolAssembly):
            raise TypeError("tool must be a ToolAssembly")
        values = tuple(obstacles)
        if not all(isinstance(obstacle, CollisionObstacle) for obstacle in values):
            raise TypeError("obstacles must contain CollisionObstacle values")
        names = [obstacle.name for obstacle in values]
        if len(names) != len(set(names)):
            raise ValueError("obstacle names must be unique")
        self._tool = tool
        self._obstacles = values
        self._settings = settings or SweptCollisionSettings()
        self._tool_envelopes = _tool_envelopes(tool)

    @property
    def tool_envelope_radius(self) -> float:
        return max(
            envelope.center_z + envelope.radius for envelope in self._tool_envelopes
        )

    def _clearance(
        self,
        obstacle: CollisionObstacle,
        pose: MachinePose,
        envelope: _ToolEnvelope,
    ) -> float:
        center = np.asarray(
            ((pose.x, pose.y, pose.z + envelope.center_z),), dtype=np.float64
        )
        raw = np.asarray(obstacle.signed_distance(center, pose), dtype=np.float64)
        if raw.shape != (1,):
            raise ValueError("obstacle signed_distance must preserve the point-array shape")
        value = float(raw[0])
        if math.isnan(value) or value == -math.inf:
            raise ValueError("obstacle signed_distance must not return NaN or -infinity")
        return value - envelope.radius - self._settings.clearance

    def _motion_bound(
        self,
        start: MachinePose,
        end: MachinePose,
        obstacle: CollisionObstacle,
        envelope: _ToolEnvelope,
    ) -> float:
        translation = math.dist(
            (start.x, start.y, start.z),
            (end.x, end.y, end.z),
        )
        if obstacle.frame is not FrameKind.ROTARY:
            return translation
        pivot_y, pivot_z = obstacle.rotary_pivot_yz
        start_radial = math.hypot(
            start.y - pivot_y, start.z + envelope.center_z - pivot_z
        )
        end_radial = math.hypot(
            end.y - pivot_y, end.z + envelope.center_z - pivot_z
        )
        rotation_radius = (
            max(start_radial, end_radial)
            + translation
            + envelope.radius
            + obstacle.envelope_radius
        )
        rotation = abs(math.radians(end.a - start.a)) * rotation_radius
        return translation + rotation

    def _event(
        self,
        obstacle: CollisionObstacle,
        envelope: _ToolEnvelope,
        start: MachinePose,
        end: MachinePose,
        fraction: float,
        clearance: float,
        *,
        conservative: bool,
    ) -> CollisionEvent:
        return CollisionEvent(
            obstacle.kind,
            obstacle.name,
            envelope.component,
            fraction,
            _interpolate(start, end, fraction),
            clearance,
            conservative,
        )

    def _search_obstacle(
        self,
        start: MachinePose,
        end: MachinePose,
        obstacle: CollisionObstacle,
        envelope: _ToolEnvelope,
        budget: _SearchBudget,
    ) -> CollisionEvent | None:
        total_bound = self._motion_bound(start, end, obstacle, envelope)
        start_clearance = self._clearance(obstacle, start, envelope)
        end_clearance = self._clearance(obstacle, end, envelope)
        if start_clearance <= 0.0:
            return self._event(
                obstacle,
                envelope,
                start,
                end,
                0.0,
                start_clearance,
                conservative=False,
            )

        def search(
            lower_fraction: float,
            lower_clearance: float,
            upper_fraction: float,
            upper_clearance: float,
        ) -> CollisionEvent | None:
            interval_bound = total_bound * (upper_fraction - lower_fraction)
            if min(lower_clearance, upper_clearance) - interval_bound > 0.0:
                return None
            if interval_bound <= self._settings.tolerance:
                if lower_clearance <= upper_clearance:
                    fraction, clearance = lower_fraction, lower_clearance
                else:
                    fraction, clearance = upper_fraction, upper_clearance
                return self._event(
                    obstacle,
                    envelope,
                    start,
                    end,
                    fraction,
                    clearance,
                    conservative=clearance > 0.0,
                )

            budget.take(obstacle.name)
            middle_fraction = 0.5 * (lower_fraction + upper_fraction)
            middle_pose = _interpolate(start, end, middle_fraction)
            middle_clearance = self._clearance(obstacle, middle_pose, envelope)
            left = search(
                lower_fraction,
                lower_clearance,
                middle_fraction,
                middle_clearance,
            )
            if left is not None:
                return left
            return search(
                middle_fraction,
                middle_clearance,
                upper_fraction,
                upper_clearance,
            )

        return search(0.0, start_clearance, 1.0, end_clearance)

    def check_segment(self, start: MachinePose, block: MotionBlock) -> CollisionReport:
        """Validate one segment, allowing only cutter/stock contact during cutting."""

        if not isinstance(start, MachinePose):
            raise TypeError("start must be a MachinePose")
        if not isinstance(block, MotionBlock):
            raise TypeError("block must be a MotionBlock")
        events: list[CollisionEvent] = []
        budget = _SearchBudget(self._settings.max_subdivisions)
        for obstacle in self._obstacles:
            for envelope in self._tool_envelopes:
                if (
                    obstacle.kind is CollisionKind.STOCK
                    and block.kind is not MotionKind.RAPID
                    and envelope.component is ToolComponent.CUTTER
                ):
                    continue
                event = self._search_obstacle(
                    start, block.pose, obstacle, envelope, budget
                )
                if event is not None:
                    events.append(event)
        events.sort(
            key=lambda event: (
                event.fraction,
                event.kind.value,
                event.obstacle,
                event.tool_component.value,
            )
        )
        return CollisionReport(tuple(events))

    def check(self, start: MachinePose, block: MotionBlock) -> CollisionReport:
        """Alias for :meth:`check_segment`."""

        return self.check_segment(start, block)


__all__ = ["SweptCollisionChecker"]
