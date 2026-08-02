"""Controller-neutral inverse-time G-code emission for measured XYZA machines."""

from __future__ import annotations

import math
from collections.abc import Sequence

from rotarycam.config import MachineDefinition
from rotarycam.errors import ToolpathValidationError, UnverifiedMachineProfileError
from rotarycam.motion import MachinePose, MotionBlock, MotionKind, time_parameterize


def _require_export_profile(machine: MachineDefinition) -> None:
    if not machine.profile_verified:
        raise UnverifiedMachineProfileError(
            "XYZA G-code export is blocked until the machine profile is verified"
        )
    capabilities = machine.capabilities
    if capabilities is None:
        raise ToolpathValidationError("XYZA export requires measured controller capabilities")
    if not capabilities.simultaneous_xyza:
        raise ToolpathValidationError("simultaneous XYZA support is not confirmed")
    if not capabilities.inverse_time_feed_g93:
        raise ToolpathValidationError("inverse-time G93 support is not confirmed")
    if machine.rotary_axis.axis_letter != "A":
        raise ToolpathValidationError("XYZA export requires the rotary axis letter A")
    if machine.y_limits is None or machine.xyza_configuration is None:
        raise ToolpathValidationError("XYZA export requires complete Y and kinematics data")
    if machine.assembly is None or not machine.assembly.is_complete:
        raise ToolpathValidationError("XYZA export requires a complete machine assembly")
    if machine.dynamics is None or any(
        axis not in machine.dynamics for axis in ("X", "Y", "Z", "A")
    ):
        raise ToolpathValidationError("XYZA export requires dynamics for X, Y, Z, and A")


def _validate_dynamic_durations(
    start: MachinePose,
    blocks: Sequence[MotionBlock],
    machine: MachineDefinition,
) -> None:
    minimum_blocks = time_parameterize(start, blocks, machine)
    for index, (block, minimum) in enumerate(
        zip(blocks, minimum_blocks, strict=True), start=1
    ):
        if block.duration_s + 1e-12 < minimum.duration_s:
            raise ToolpathValidationError(
                f"motion block {index} duration violates feed or XYZA dynamics"
            )


def _changed_axis_words(
    previous: MachinePose,
    current: MachinePose,
    machine: MachineDefinition,
) -> list[str]:
    precision = machine.coordinate_precision
    words: list[str] = []
    for letter, before, after in (
        ("X", previous.x, current.x),
        ("Y", previous.y, current.y),
        ("Z", previous.z, current.z),
        (machine.rotary_axis.axis_letter, previous.a, current.a),
    ):
        if after != before:
            rendered_before = f"{before:.{precision}f}"
            rendered_after = f"{after:.{precision}f}"
            if rendered_after == rendered_before:
                raise ToolpathValidationError(
                    f"coordinate precision would erase {letter} motion"
                )
            words.append(f"{letter}{rendered_after}")
    if not words:
        raise ToolpathValidationError("zero-distance motion blocks are not allowed")
    return words


def _render_xyza_gcode(
    start: MachinePose,
    blocks: Sequence[MotionBlock],
    machine: MachineDefinition,
    *,
    preview: bool,
    tool_number: int | None = None,
    spindle_rpm: int | None = None,
    tool_schedule: Sequence[tuple[int, int, int]] | None = None,
) -> str:
    if not isinstance(start, MachinePose):
        raise TypeError("start must be a MachinePose")

    lines: list[str] = []
    if preview:
        lines.append("(RotaryCAM XYZA PREVIEW - NOT FOR MACHINE EXECUTION)")
    lines.extend(("G21", "G90"))
    lines.extend(machine.program_header)
    if tool_schedule is not None and (tool_number is not None or spindle_rpm is not None):
        raise ToolpathValidationError(
            "use either a tool schedule or one tool number/spindle speed pair"
        )
    if (tool_number is None) != (spindle_rpm is None):
        raise ToolpathValidationError(
            "tool number and spindle speed must be provided together"
        )
    schedule = (
        tuple(tool_schedule)
        if tool_schedule is not None
        else (
            ((0, tool_number, spindle_rpm),)
            if tool_number is not None and spindle_rpm is not None
            else ()
        )
    )
    schedule_by_offset: dict[int, tuple[int, int]] = {}
    for offset, scheduled_tool, scheduled_rpm in schedule:
        if offset < 0 or offset >= len(blocks):
            raise ToolpathValidationError("tool-change block offset lies outside the program")
        if offset in schedule_by_offset:
            raise ToolpathValidationError("tool-change block offsets must be unique")
        if scheduled_tool <= 0 or scheduled_rpm <= 0:
            raise ToolpathValidationError(
                "tool number and spindle speed must be greater than zero"
            )
        if machine.max_spindle_rpm is not None and scheduled_rpm > machine.max_spindle_rpm:
            raise ToolpathValidationError("requested spindle speed exceeds machine limit")
        schedule_by_offset[offset] = (scheduled_tool, scheduled_rpm)

    previous = start
    inverse_time_active = False
    spindle_active = False
    for block_index, block in enumerate(blocks):
        if not isinstance(block, MotionBlock):
            raise TypeError("blocks must contain only MotionBlock records")
        scheduled = schedule_by_offset.get(block_index)
        if scheduled is not None:
            if spindle_active:
                lines.append("M5")
            if inverse_time_active:
                lines.append("G94")
                inverse_time_active = False
            scheduled_tool, scheduled_rpm = scheduled
            lines.extend((f"T{scheduled_tool} M6", f"S{scheduled_rpm} M3"))
            spindle_active = True
        axis_words = _changed_axis_words(previous, block.pose, machine)
        if block.kind is MotionKind.RAPID:
            lines.append(" ".join(("G0", *axis_words)))
        else:
            if not inverse_time_active:
                lines.append("G93")
                inverse_time_active = True
            inverse_time_feed = 60.0 / block.duration_s
            if not math.isfinite(inverse_time_feed) or inverse_time_feed <= 0.0:
                raise ToolpathValidationError("inverse-time feed must be finite and positive")
            rendered_feed = f"{inverse_time_feed:.{machine.coordinate_precision}f}"
            if float(rendered_feed) <= 0.0:
                raise ToolpathValidationError(
                    "coordinate precision would render inverse-time feed as zero"
                )
            lines.append(" ".join(("G1", *axis_words, f"F{rendered_feed}")))
        previous = block.pose

    if spindle_active:
        lines.append("M5")
    lines.append("G94")
    lines.extend(machine.program_footer)
    return "\n".join(lines) + "\n"


def generate_xyza_preview(
    start: MachinePose,
    blocks: Sequence[MotionBlock],
    machine: MachineDefinition,
) -> str:
    """Render review-only XYZA text without claiming export readiness."""

    return _render_xyza_gcode(start, blocks, machine, preview=True)


def generate_xyza_gcode(
    start: MachinePose,
    blocks: Sequence[MotionBlock],
    machine: MachineDefinition,
    *,
    tool_number: int | None = None,
    spindle_rpm: int | None = None,
    tool_schedule: Sequence[tuple[int, int, int]] | None = None,
) -> str:
    """Generate validated G93 XYZA G-code for a verified capable profile."""

    _require_export_profile(machine)
    _validate_dynamic_durations(start, blocks, machine)
    return _render_xyza_gcode(
        start,
        blocks,
        machine,
        preview=False,
        tool_number=tool_number,
        spindle_rpm=spindle_rpm,
        tool_schedule=tool_schedule,
    )


__all__ = ["generate_xyza_gcode", "generate_xyza_preview"]
