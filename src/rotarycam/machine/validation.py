"""Static safety validation for X/Z/A operations."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from rotarycam.config import MachineDefinition
from rotarycam.planning.operation import MachiningOperation
from rotarycam.toolpath.models import ToolpathPoint

_ENVELOPE_TOLERANCE_MM = 1e-9


@dataclass(frozen=True, slots=True)
class ToolpathValidationReport:
    """Errors and warnings found before post-processing."""

    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        """Return whether no blocking error was found."""

        return not self.errors


def validate_operations(
    operations: Sequence[MachiningOperation],
    machine: MachineDefinition,
    *,
    stock_length: float | None = None,
    stock_max_radius: float | None = None,
) -> ToolpathValidationReport:
    """Validate coordinates, modal feeds, travel and conservative rapid safety.

    Stock-envelope values may be omitted for non-exportable previews.  When
    supplied, they must be finite and positive, must fit any declared rotary
    envelope, and the machine must define a safe radius outside the stock.
    """

    errors: list[str] = []
    warnings: list[str] = []
    safe_radius = machine.safe_radius
    if not operations:
        errors.append("At least one machining operation is required.")
    if stock_length is not None:
        if not math.isfinite(stock_length) or stock_length <= 0.0:
            errors.append("Stock length must be finite and positive.")
        elif (
            machine.max_rotary_stock_length is not None
            and stock_length
            > machine.max_rotary_stock_length + _ENVELOPE_TOLERANCE_MM
        ):
            errors.append("Stock length exceeds the machine rotary envelope.")
    if stock_max_radius is not None:
        if not math.isfinite(stock_max_radius) or stock_max_radius <= 0.0:
            errors.append("Stock maximum radius must be finite and positive.")
        else:
            if (
                machine.max_rotary_stock_radius is not None
                and stock_max_radius
                > machine.max_rotary_stock_radius + _ENVELOPE_TOLERANCE_MM
            ):
                errors.append("Stock radius exceeds the machine rotary envelope.")
            if safe_radius is None or safe_radius <= stock_max_radius:
                errors.append("Machine safe radius must be greater than the stock radius.")

    for operation in operations:
        modal_feed: float | None = None
        contains_rotary_motion = False
        if not operation.toolpaths:
            errors.append(f"{operation.name}: operation contains no toolpaths.")
        if (
            machine.max_spindle_rpm is not None
            and operation.tool.spindle_rpm > machine.max_spindle_rpm
        ):
            errors.append(f"{operation.name}: spindle speed exceeds the machine limit.")
        for path_index, toolpath in enumerate(operation.toolpaths):
            if not toolpath.points:
                errors.append(f"{operation.name} path {path_index}: toolpath contains no points.")
            previous: ToolpathPoint | None = None
            for point_index, point in enumerate(toolpath.points):
                label = f"{operation.name} path {path_index} point {point_index}"
                if not all(math.isfinite(value) for value in (point.x, point.z, point.a)):
                    errors.append(f"{label}: coordinate is not finite.")
                    continue
                if not machine.x_limits.minimum <= point.x <= machine.x_limits.maximum:
                    errors.append(f"{label}: X is outside machine travel.")
                if not machine.z_limits.minimum <= point.z <= machine.z_limits.maximum:
                    errors.append(f"{label}: Z is outside machine travel.")
                if point.feed is not None and (not math.isfinite(point.feed) or point.feed <= 0.0):
                    errors.append(f"{label}: feed must be finite and positive.")
                if not point.rapid:
                    if point.feed is None and modal_feed is None:
                        errors.append(f"{label}: cutting move has no modal feed.")
                    elif point.feed is not None:
                        modal_feed = point.feed
                        if (
                            machine.max_linear_speed_mm_min is not None
                            and point.feed > machine.max_linear_speed_mm_min
                        ):
                            errors.append(
                                f"{label}: cutting feed exceeds the machine linear speed limit."
                            )
                if point.rapid and stock_max_radius is not None and point.z <= stock_max_radius:
                    errors.append(f"{label}: rapid move intersects the stock envelope.")
                if previous is not None:
                    angular_jump = abs(point.a - previous.a)
                    contains_rotary_motion = contains_rotary_motion or angular_jump > 0.0
                    if angular_jump > 180.0:
                        warnings.append(f"{label}: angular jump exceeds 180 degrees.")
                    if (
                        angular_jump > 5.0
                        and safe_radius is not None
                        and min(point.z, previous.z) < safe_radius
                        and (point.rapid or previous.rapid)
                    ):
                        errors.append(f"{label}: rotary repositioning occurs below safe radius.")
                previous = point
        rotary_speed_limit = machine.rotary_axis.max_speed_deg_per_min
        if contains_rotary_motion and rotary_speed_limit is not None:
            warnings.append(
                f"{operation.name}: A-axis speed cannot be derived from linear feed values; "
                f"verify controller motion stays at or below {rotary_speed_limit:g} deg/min."
            )
    return ToolpathValidationReport(tuple(dict.fromkeys(errors)), tuple(dict.fromkeys(warnings)))
