"""Post-processor interface."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from rotarycam.config import MachineDefinition
from rotarycam.planning.operation import MachiningOperation


class PostProcessor(Protocol):
    """Convert validated machining operations into controller text."""

    def generate(
        self,
        operations: Sequence[MachiningOperation],
        machine: MachineDefinition,
        *,
        stock_length: float,
        stock_max_radius: float,
    ) -> str:
        """Generate exportable controller text."""

        ...
