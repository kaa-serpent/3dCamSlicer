"""Conservative time parameterization for coordinated XYZA motion."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import replace

from rotarycam.config import MachineDefinition
from rotarycam.errors import ToolpathValidationError
from rotarycam.machine.assemblies import AxisDynamics
from rotarycam.motion.models import MachinePose, MotionBlock, MotionKind

_AXES = ("X", "Y", "Z", "A")


def _complete_dynamics(machine: MachineDefinition) -> Mapping[str, AxisDynamics]:
    dynamics = machine.dynamics
    if dynamics is None:
        raise ToolpathValidationError("XYZA timing requires dynamics for X, Y, Z, and A")
    missing = [axis for axis in _AXES if axis not in dynamics]
    if missing:
        raise ToolpathValidationError(
            "XYZA timing requires dynamics for " + ", ".join(missing)
        )
    return dynamics


def _rest_to_rest_duration(distance: float, dynamics: AxisDynamics) -> float:
    """Return a conservative rest-to-rest duration in seconds for one axis."""

    if distance == 0.0:
        return 0.0
    velocity_per_second = dynamics.max_velocity / 60.0
    acceleration = dynamics.max_acceleration
    transition_distance = velocity_per_second * velocity_per_second / acceleration
    if distance <= transition_distance:
        return 2.0 * math.sqrt(distance / acceleration)
    return distance / velocity_per_second + velocity_per_second / acceleration


def time_parameterize(
    start: MachinePose,
    blocks: Sequence[MotionBlock],
    machine: MachineDefinition,
) -> tuple[MotionBlock, ...]:
    """Assign a safe duration to each block using measured XYZA dynamics.

    Axis limits assume every block starts and ends at rest.  This is deliberately
    conservative and deterministic until a look-ahead model is controller-verified.
    Linear blocks additionally respect their requested Cartesian TCP feed.
    """

    if not isinstance(start, MachinePose):
        raise TypeError("start must be a MachinePose")
    dynamics = _complete_dynamics(machine)
    previous = start
    parameterized: list[MotionBlock] = []

    for block in blocks:
        if not isinstance(block, MotionBlock):
            raise TypeError("blocks must contain only MotionBlock records")
        deltas = (
            abs(block.pose.x - previous.x),
            abs(block.pose.y - previous.y),
            abs(block.pose.z - previous.z),
            abs(block.pose.a - previous.a),
        )
        if not any(delta > 0.0 for delta in deltas):
            raise ToolpathValidationError("zero-distance motion blocks are not allowed")

        duration_s = max(
            _rest_to_rest_duration(delta, dynamics[axis])
            for axis, delta in zip(_AXES, deltas, strict=True)
        )
        if block.kind is MotionKind.LINEAR:
            if block.feed is None:
                raise ToolpathValidationError("linear motion requires a TCP feed")
            tcp_distance = math.sqrt(sum(delta * delta for delta in deltas[:3]))
            duration_s = max(duration_s, 60.0 * tcp_distance / block.feed)

        if not math.isfinite(duration_s) or duration_s <= 0.0:
            raise ToolpathValidationError("motion duration must be finite and positive")
        parameterized.append(replace(block, duration_s=duration_s))
        previous = block.pose

    return tuple(parameterized)
