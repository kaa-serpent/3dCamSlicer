"""Immutable, presentation-independent machine motion records."""

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite


@dataclass(frozen=True, slots=True)
class MachinePose:
    """One machine pose in G54 coordinates.

    Linear coordinates are millimetres and ``a`` is an unwrapped angle in
    degrees.
    """

    x: float
    y: float
    z: float
    a: float

    def __post_init__(self) -> None:
        for field_name, value in (
            ("x", self.x),
            ("y", self.y),
            ("z", self.z),
            ("a", self.a),
        ):
            if not isfinite(value):
                raise ValueError(f"{field_name} must be finite")


class MotionKind(StrEnum):
    """Controller-independent interpolation kinds."""

    RAPID = "rapid"
    LINEAR = "linear"


@dataclass(frozen=True, slots=True)
class MotionBlock:
    """A time-parameterized move ending at ``pose``.

    ``duration_s`` is always required so later validation can enforce linear
    and rotary dynamics. ``feed`` is the requested TCP feed in mm/min when the
    block has one; rapid blocks may leave it unset.
    """

    pose: MachinePose
    kind: MotionKind
    duration_s: float
    feed: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.pose, MachinePose):
            raise TypeError("pose must be a MachinePose")
        if not isinstance(self.kind, MotionKind):
            raise TypeError("kind must be a MotionKind")
        if not isfinite(self.duration_s) or self.duration_s <= 0.0:
            raise ValueError("duration_s must be finite and greater than zero")
        if self.feed is not None and (not isfinite(self.feed) or self.feed <= 0.0):
            raise ValueError("feed must be finite and greater than zero when specified")
