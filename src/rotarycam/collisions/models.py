"""Typed, immutable results for continuous swept-collision validation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from rotarycam.motion import MachinePose


class CollisionKind(StrEnum):
    """Safety-significant class of obstacle."""

    MACHINE = "machine"
    FIXTURE = "fixture"
    STOCK = "stock"


class ToolComponent(StrEnum):
    """Tool-assembly component participating in a collision pair."""

    CUTTER = "cutter"
    NON_CUTTING = "non_cutting"
    SHANK = "shank"
    HOLDER = "holder"


class CollisionBudgetExceeded(RuntimeError):
    """Raised when the configured subdivision budget cannot prove safety."""


@dataclass(frozen=True, slots=True)
class SweptCollisionSettings:
    """Spatial proof tolerance and hard adaptive-search budget."""

    tolerance: float = 0.05
    clearance: float = 0.0
    max_subdivisions: int = 4096

    def __post_init__(self) -> None:
        if not math.isfinite(self.tolerance) or self.tolerance <= 0.0:
            raise ValueError("tolerance must be finite and greater than zero")
        if not math.isfinite(self.clearance) or self.clearance < 0.0:
            raise ValueError("clearance must be finite and non-negative")
        if (
            isinstance(self.max_subdivisions, bool)
            or not isinstance(self.max_subdivisions, int)
            or self.max_subdivisions < 0
        ):
            raise ValueError("max_subdivisions must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class CollisionEvent:
    """A collision or a tolerance-wide conservative contact interval."""

    kind: CollisionKind
    obstacle: str
    tool_component: ToolComponent
    fraction: float
    pose: MachinePose
    clearance: float
    conservative: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.kind, CollisionKind):
            raise TypeError("kind must be a CollisionKind")
        if not self.obstacle:
            raise ValueError("obstacle must not be empty")
        if not isinstance(self.tool_component, ToolComponent):
            raise TypeError("tool_component must be a ToolComponent")
        if not math.isfinite(self.fraction) or not 0.0 <= self.fraction <= 1.0:
            raise ValueError("fraction must be finite and within [0, 1]")
        if not isinstance(self.pose, MachinePose):
            raise TypeError("pose must be a MachinePose")
        if not math.isfinite(self.clearance):
            raise ValueError("clearance must be finite")


@dataclass(frozen=True, slots=True)
class CollisionReport:
    """Deterministically ordered collision events for one motion block."""

    events: tuple[CollisionEvent, ...] = ()

    @property
    def collision_free(self) -> bool:
        return not self.events

    @property
    def first_event(self) -> CollisionEvent | None:
        return self.events[0] if self.events else None


__all__ = [
    "CollisionBudgetExceeded",
    "CollisionEvent",
    "CollisionKind",
    "CollisionReport",
    "SweptCollisionSettings",
    "ToolComponent",
]
