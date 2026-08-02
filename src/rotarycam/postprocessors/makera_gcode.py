"""Conservative Makera Z1 community-derived X/Z/A post-processor.

Only the small common subset described in this module is emitted.  The output
is intentionally not advertised as controller-compatible until the exact
machine profile and emitted program have been verified.
"""

from __future__ import annotations

from collections.abc import Sequence

from rotarycam.config import MachineDefinition
from rotarycam.errors import ToolpathValidationError, UnverifiedMachineProfileError
from rotarycam.machine.validation import validate_operations
from rotarycam.planning.operation import MachiningOperation


class MakeraZ1PostProcessor:
    """Emit a deterministic conservative subset for Makera Z1 review."""

    @staticmethod
    def _emit(
        operations: Sequence[MachiningOperation],
        machine: MachineDefinition,
        *,
        preview: bool,
    ) -> str:
        """Render operations that have already passed the applicable validation."""

        precision = machine.coordinate_precision
        rotary_axis = machine.rotary_axis
        lines = []
        if preview:
            lines.append(
                "(RotaryCAM Makera Z1 community-derived UNVERIFIED preview - "
                "verify before machining)"
            )
        lines.extend(("G21", "G90", "G94", "G17"))
        lines.extend(machine.program_header)
        previous_tool_number: int | None = None
        for operation in operations:
            tool = operation.tool
            lines.append(f"({operation.name})")
            if tool.number != previous_tool_number:
                lines.append(f"M6 T{tool.number}")
                previous_tool_number = tool.number
            lines.append(f"S{tool.spindle_rpm} M3")
            last_feed: float | None = None
            for toolpath in operation.toolpaths:
                for point in toolpath.points:
                    code = "G0" if point.rapid else "G1"
                    a_value = (
                        rotary_axis.direction
                        * point.a
                        * rotary_axis.degrees_per_revolution
                        / 360.0
                    )
                    words = [
                        code,
                        f"X{point.x:.{precision}f}",
                        f"Z{point.z:.{precision}f}",
                        f"{rotary_axis.axis_letter}{a_value:.{precision}f}",
                    ]
                    if not point.rapid and point.feed is not None and point.feed != last_feed:
                        words.append(f"F{point.feed:.{precision}f}")
                        last_feed = point.feed
                    lines.append(" ".join(words))
            if machine.safe_radius is not None:
                lines.append(f"G0 Z{machine.safe_radius:.{precision}f}")
            lines.append("M5")
        lines.extend(machine.program_footer)
        lines.append("M30")
        return "\n".join(lines) + "\n"

    def generate_preview(
        self,
        operations: Sequence[MachiningOperation],
        machine: MachineDefinition,
    ) -> str:
        """Generate a non-exportable preview, still enforcing static validation."""

        report = validate_operations(operations, machine)
        if not report.valid:
            raise ToolpathValidationError("; ".join(report.errors))
        return self._emit(operations, machine, preview=True)

    def generate(
        self,
        operations: Sequence[MachiningOperation],
        machine: MachineDefinition,
        *,
        stock_length: float,
        stock_max_radius: float,
    ) -> str:
        """Generate exportable text for a verified profile and known stock envelope."""

        if not machine.profile_verified:
            raise UnverifiedMachineProfileError(
                "G-code export is blocked until the machine profile is verified."
            )
        if stock_length is None or stock_max_radius is None:
            raise ToolpathValidationError(
                "G-code export requires explicit finite stock length and maximum radius."
            )
        report = validate_operations(
            operations,
            machine,
            stock_length=stock_length,
            stock_max_radius=stock_max_radius,
        )
        if not report.valid:
            raise ToolpathValidationError("; ".join(report.errors))
        return self._emit(operations, machine, preview=False)


# Compatibility with the public name used before the Z1-only profile was introduced.
MakeraPostProcessor = MakeraZ1PostProcessor
